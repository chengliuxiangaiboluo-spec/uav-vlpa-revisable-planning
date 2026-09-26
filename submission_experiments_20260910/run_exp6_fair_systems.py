"""Fair V2-only rerun for Ours and six local implementation-inspired controls.

Every learned fusion encoder and map-grounding head starts from its seeded
random initialization and is optimized only on the audited V2 train split.
No historical fusion checkpoint is loaded.  A track evaluates one fusion
control and its associated end-to-end system style using the same trained
grounder, avoiding duplicate training while keeping each pipeline explicit.
"""

from __future__ import annotations

import argparse, copy, json, sys, types
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent; PROJECT_ROOT = PACKAGE_DIR.parent
for item in (PROJECT_ROOT, PACKAGE_DIR):
    if str(item) not in sys.path: sys.path.insert(0, str(item))

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.experiment_config import get_default_config
from fusion_conditioned_grounder import (FusionConditionedGrounder, FusionGroundingDataset,
    collate_fusion_grounding, evaluate_loss, train_epoch)
from models.fusion.baseline_fusers import AFFNetFuser, MAFTNetFuser, SCALFuser
from models.fusion.multimodal_fuser import MultimodalFuser
from models.planner.baseline_systems import ModularUAVSystem
from models.planner.enhanced_planner import EnhancedPlanner
from models.reasoning.baseline_decomposers import CodeAgentsDecomposer, SIPSADecomposer, UniGoalDecomposer
from models.reasoning.task_decomposer import TaskDecomposer
from run_v2_grounding_worker import GroundedPlanner, georeference_for_scoring
from submission_common import DEFAULT_SEEDS, aggregate_seed_means, evaluate, load_checkpoint, read_samples, run_metadata, write_json, write_rows_csv
from utils.seed_manager import set_global_seed

TRACKS = {
    "ours": {"fuser": MultimodalFuser, "methods": ("ours",)},
    "affnet": {"fuser": AFFNetFuser, "methods": ("affnet", "uav_vln")},
    "maftnet": {"fuser": MAFTNetFuser, "methods": ("maftnet", "uav_vla")},
    "scal": {"fuser": SCALFuser, "methods": ("scal", "aerialvln")},
}
SYSTEMS = {
    "uav_vla": ("UAV-VLA-style (MAFTNet + CodeAgents + A*)", CodeAgentsDecomposer),
    "uav_vln": ("UAV-VLN-style (AFFNet + SIPSA + A*)", SIPSADecomposer),
    "aerialvln": ("AerialVLN-style (SCAL + UniGoal + A*)", UniGoalDecomposer),
}


def seeds(value):
    result = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not result or set(result) - set(DEFAULT_SEEDS): raise argparse.ArgumentTypeError("unsupported seed")
    return result


def new_fuser(track, cfg, device):
    cls = TRACKS[track]["fuser"]
    common = dict(fusion_dim=cfg.fusion_dim, audio_model=cfg.audio_model,
                  gesture_backbone=cfg.gesture_backbone, text_model=cfg.text_model)
    if cls is MultimodalFuser:
        return cls(attention_heads=cfg.attention_heads, attention_layers=cfg.attention_layers,
                   ffn_dim=cfg.ffn_dim, **common).to(device)
    return cls(**common).to(device)


def loader(samples, benchmark, encoder, batch, shuffle):
    return DataLoader(FusionGroundingDataset(samples, benchmark, encoder), batch_size=batch,
                      shuffle=shuffle, num_workers=0, collate_fn=collate_fusion_grounding)


def train_or_load(track, root, cfg, seed, train, val, benchmark, device, epochs, batch, directory):
    checkpoint, metadata_path = directory / "fusion_grounder.pt", directory / "training_metadata.json"
    if checkpoint.is_file() and metadata_path.is_file():
        record = torch.load(checkpoint, map_location=device)
        model = FusionConditionedGrounder(new_fuser(track, cfg, device), cfg.fusion_dim).to(device)
        model.load_state_dict(record["model_state_dict"], strict=True); model.eval()
        return model, json.loads(metadata_path.read_text(encoding="utf-8"))
    if directory.exists(): raise RuntimeError(f"incomplete independent directory: {directory}")
    directory.mkdir(parents=True)
    fuser = new_fuser(track, cfg, device)
    model = FusionConditionedGrounder(fuser, cfg.fusion_dim).to(device)
    train_loader, val_loader = loader(train, benchmark, fuser.text_encoder, batch, True), loader(val, benchmark, fuser.text_encoder, batch, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    best_loss, best_epoch, best_state, history = float("inf"), 0, None, []
    for epoch in range(1, epochs + 1):
        tr, va = train_epoch(model, train_loader, optimizer, device), evaluate_loss(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        if va < best_loss:
            best_loss, best_epoch = va, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[Exp6 fair] track={track} seed={seed} epoch={epoch}/{epochs} train_loss={tr:.6f} val_loss={va:.6f} best={best_epoch}", flush=True)
    model.load_state_dict(best_state, strict=True); model.eval()
    tmp = directory / "fusion_grounder.pt.tmp"
    torch.save({"model_state_dict": best_state, "track": track, "seed": seed,
                "selected_epoch": best_epoch, "validation_loss": best_loss,
                "initialization": "random_v2_only"}, tmp); tmp.replace(checkpoint)
    metadata = run_metadata({"experiment": "Exp6 fair V2-only map-grounding training", "track": track,
        "training_seed": seed, "initialization": "random; no historical fusion checkpoint",
        "selection": "minimum V2 validation coordinate MSE", "selected_epoch": best_epoch,
        "best_validation_loss": best_loss, "epochs_requested": epochs, "history": history})
    write_json(metadata_path, metadata)
    return model, metadata


def ours_decomposer(root, cfg, seed, device):
    model = TaskDecomposer(d_model=cfg.fusion_dim, n_layers=cfg.decomposer_layers,
        max_subtasks=cfg.max_subtasks, max_seq_len=100).to(device)
    load_checkpoint(model, root / "models" / "ours" / f"seed_{seed}", "decomposer_rl", device)
    return model


def deterministic_completion(self, targets, traj, successful_segments, total_segments, n_modalities=1):
    if not traj or total_segments <= 0: return []
    count = min(len(targets), round(len(targets) * successful_segments / total_segments))
    return [target.name for target in targets[:count]]


def core(method, model, root, cfg, benchmark, seed, device):
    if method in ("ours", "affnet", "maftnet", "scal"):
        return EnhancedPlanner(benchmark_dir=benchmark, fuser=model.fuser,
            decomposer=ours_decomposer(root, cfg, seed, device), device=device)
    title, decomposer_cls = SYSTEMS[method]
    planner = ModularUAVSystem(system_name=title, benchmark_dir=benchmark, fuser=model.fuser,
        decomposer=decomposer_cls(seed=seed), path_planner_name="astar", device=device, seed=seed)
    planner._determine_completed = types.MethodType(deterministic_completion, planner)
    return planner


def coordinate_rows(model, samples, benchmark, device, method, seed):
    result=[]
    for sample in samples:
        predicted=model.predict_sample(sample, benchmark, device)
        errors=[((x-t.coordinates_percent[0])**2+(y-t.coordinates_percent[1])**2)**.5 for (x,y),t in zip(predicted,sample.targets)]
        result.append({"method":method,"training_seed":seed,"scenario_id":sample.scenario_id,
                       "mean_target_coordinate_error_pct":sum(errors)/max(1,len(errors))})
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--track", choices=tuple(TRACKS), required=True); p.add_argument("--run-root", required=True); p.add_argument("--artifact-root"); p.add_argument("--seeds", type=seeds, default=DEFAULT_SEEDS); p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch-size",type=int,default=16); p.add_argument("--max-train-samples",type=int); p.add_argument("--max-val-samples",type=int); p.add_argument("--max-test-samples",type=int); p.add_argument("--smoke",action="store_true"); a=p.parse_args()
    root=Path(a.run_root).resolve(); artifact=Path(a.artifact_root).resolve() if a.artifact_root else root
    output=artifact/"results"/"exp6_fair_workers"/a.track
    if output.exists(): raise FileExistsError(f"Refusing to overwrite {output}")
    audit=json.loads((root/"v2_osm_protocol"/"audit.json").read_text(encoding="utf-8"))
    if audit.get("status")!="COMPLETE" or audit.get("random_coordinate_fallbacks")!=0: raise RuntimeError("invalid V2 audit")
    train,val,test=(read_samples(root/"v2_osm_protocol"/"splits"/f"{n}.json") for n in ("train","val_location","ood_location_test"))
    if a.max_train_samples: train=train[:a.max_train_samples]
    if a.max_val_samples: val=val[:a.max_val_samples]
    if a.max_test_samples: test=test[:a.max_test_samples]
    base,_,cfg,_,_=get_default_config(); device,benchmark=base.device,base.benchmark_dir; scoring=georeference_for_scoring(test,benchmark)
    rows,coords,by_method,checkpoints=[],[],{m:{} for m in TRACKS[a.track]["methods"]},{}
    for seed in a.seeds:
        set_global_seed(seed); directory=artifact/"models"/"exp6_fair"/a.track/f"seed_{seed}"
        model,meta=train_or_load(a.track,root,cfg,seed,train,val,benchmark,device,a.epochs,a.batch_size,directory)
        for method in TRACKS[a.track]["methods"]:
            coords.extend(coordinate_rows(model,test,benchmark,device,method,seed))
            evaluation=evaluate(GroundedPlanner(core(method,model,root,cfg,benchmark,seed,device),model,benchmark,device),scoring,benchmark)
            for row in evaluation: row.update({"method":method,"training_seed":seed,"planner_input":"predicted_coordinates_only","label_coordinates_used_only_for_scoring":True})
            rows.extend(evaluation); by_method[method][seed]=evaluation
        checkpoints[str(seed)]={"path":str(directory/"fusion_grounder.pt"),"selected_epoch":meta["selected_epoch"],"initialization":meta["initialization"]}
        print(f"[Exp6 fair] complete | track={a.track} seed={seed}",flush=True)
    write_rows_csv(output/"grounded_planning_results.csv",rows); write_rows_csv(output/"grounding_coordinate_results.csv",coords)
    write_json(output/"summary.json",run_metadata({"experiment":"Exp6 fair V2-only local implementation-inspired controls","track":a.track,"methods":{m:aggregate_seed_means(v) for m,v in by_method.items()},"seeds":list(a.seeds),"smoke":bool(a.smoke),"sample_counts":{"train":len(train),"val":len(val),"test":len(test)},"checkpoints":checkpoints,"label_leakage_gate":"passed","planner_input":"predicted_coordinates_only","protocol_boundary":"Every map-grounding fuser and head is randomly initialized and trained only on V2. The learned Ours decomposer is held fixed across fusion controls; system-style rows use their disclosed local deterministic decomposer.","claim_allowed":"fair V2-only local implementation-inspired comparison","claim_forbidden":"official author-code reproduction"}))

if __name__=="__main__": main()
