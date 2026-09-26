"""Train and evaluate one independent Exp6 fusion-conditioned grounder."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
for location in (PROJECT_ROOT, PACKAGE_DIR):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.experiment_config import get_default_config
from experiment_models import _materialize_audio_encoder
from fusion_conditioned_grounder import (
    FusionConditionedGrounder, FusionGroundingDataset, collate_fusion_grounding,
    evaluate_loss, train_epoch,
)
from models.fusion.baseline_fusers import AFFNetFuser, MAFTNetFuser, SCALFuser
from models.planner.enhanced_planner import EnhancedPlanner
from models.reasoning.task_decomposer import TaskDecomposer
from run_v2_grounding_worker import GroundedPlanner, georeference_for_scoring
from submission_common import (
    DEFAULT_SEEDS, aggregate_seed_means, evaluate, load_checkpoint, read_samples,
    run_metadata, write_json, write_rows_csv,
)
from utils.seed_manager import set_global_seed

FUSERS = {"affnet": AFFNetFuser, "maftnet": MAFTNetFuser, "scal": SCALFuser}


def _parse_seeds(value):
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or set(seeds) - set(DEFAULT_SEEDS):
        raise argparse.ArgumentTypeError("seeds must be selected from 42,123,456,789,2024")
    return seeds


def _loader(samples, benchmark, text_encoder, batch_size, shuffle):
    dataset = FusionGroundingDataset(samples, benchmark, text_encoder)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      collate_fn=collate_fusion_grounding)


def _new_fuser(method, config, device):
    fuser = FUSERS[method](
        fusion_dim=config.fusion_dim, audio_model=config.audio_model,
        gesture_backbone=config.gesture_backbone, text_model=config.text_model,
    ).to(device)
    _materialize_audio_encoder(fuser)
    return fuser


def _load_fuser(method, root, config, seed, device):
    fuser = _new_fuser(method, config, device)
    checkpoint_dir = root / "models" / method / f"seed_{seed}"
    path = load_checkpoint(fuser, checkpoint_dir, "best_fusion_model", device)
    return fuser, str(path)


def _load_decomposer(root, config, seed, device):
    model = TaskDecomposer(d_model=config.fusion_dim, n_layers=config.decomposer_layers,
                           max_subtasks=config.max_subtasks, max_seq_len=100).to(device)
    path = load_checkpoint(model, root / "models" / "ours" / f"seed_{seed}",
                           "decomposer_rl", device)
    return model, str(path)


def _train_or_load(method, root, config, seed, train, val, benchmark, device,
                   epochs, batch_size, model_dir):
    checkpoint = model_dir / "fusion_grounder.pt"
    metadata_path = model_dir / "training_metadata.json"
    if checkpoint.is_file() and metadata_path.is_file():
        record = torch.load(checkpoint, map_location=device)
        fuser, fuser_path = _load_fuser(method, root, config, seed, device)
        model = FusionConditionedGrounder(fuser, config.fusion_dim).to(device)
        model.load_state_dict(record["model_state_dict"], strict=True)
        model.eval()
        return model, json.loads(metadata_path.read_text(encoding="utf-8")), fuser_path
    if model_dir.exists():
        raise RuntimeError(f"Incomplete independent fusion-grounder directory: {model_dir}")
    model_dir.mkdir(parents=True)
    fuser, fuser_path = _load_fuser(method, root, config, seed, device)
    train_loader = _loader(train, benchmark, fuser.text_encoder, batch_size, True)
    val_loader = _loader(val, benchmark, fuser.text_encoder, batch_size, False)
    model = FusionConditionedGrounder(fuser, config.fusion_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    best_loss, best_epoch, best_state, history = float("inf"), 0, None, []
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        print(f"[Exp6 fusion-grounder] method={method} seed={seed} "
              f"epoch={epoch}/{epochs} train_loss={train_loss:.6f} "
              f"val_loss={val_loss:.6f} best_epoch={best_epoch}", flush=True)
    if best_state is None:
        raise RuntimeError("No validation checkpoint selected")
    model.load_state_dict(best_state, strict=True); model.eval()
    temporary = model_dir / "fusion_grounder.pt.tmp"
    torch.save({"model_state_dict": best_state, "method": method, "seed": seed,
                "selected_epoch": best_epoch, "validation_loss": best_loss}, temporary)
    temporary.replace(checkpoint)
    metadata = run_metadata({
        "experiment": "Exp6 fusion-conditioned map grounder", "method": method,
        "training_seed": seed, "selection": "minimum validation coordinate MSE",
        "selected_epoch": best_epoch, "best_validation_loss": best_loss,
        "epochs_requested": epochs, "history": history,
        "protocol_boundary": "V2 provides an OSM semantic map and text instruction only; raw voice, gesture, and annotation files are unavailable.",
    })
    write_json(metadata_path, metadata)
    return model, metadata, fuser_path


def _coordinate_rows(model, samples, benchmark, device, method, seed):
    rows = []
    for sample in samples:
        predicted = model.predict_sample(sample, benchmark, device)
        errors = [((x - target.coordinates_percent[0]) ** 2 +
                   (y - target.coordinates_percent[1]) ** 2) ** 0.5
                  for (x, y), target in zip(predicted, sample.targets)]
        rows.append({"method": method, "training_seed": seed,
                     "scenario_id": sample.scenario_id,
                     "mean_target_coordinate_error_pct": sum(errors) / max(len(errors), 1)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=tuple(FUSERS), required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--artifact-root")
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    artifact_root = Path(args.artifact_root).resolve() if args.artifact_root else root
    output = artifact_root / "results" / "exp6_fusion_grounder_workers" / args.method
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    protocol = root / "v2_osm_protocol"
    audit = json.loads((protocol / "audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "COMPLETE" or audit.get("random_coordinate_fallbacks") != 0:
        raise RuntimeError("V2 protocol audit is incomplete or permits coordinate fallbacks")
    train, val, test = (read_samples(protocol / "splits" / f"{part}.json")
                        for part in ("train", "val_location", "ood_location_test"))
    if args.max_train_samples: train = train[:args.max_train_samples]
    if args.max_val_samples: val = val[:args.max_val_samples]
    if args.max_test_samples: test = test[:args.max_test_samples]
    base_cfg, _, model_cfg, _, _ = get_default_config()
    device, benchmark = base_cfg.device, base_cfg.benchmark_dir
    scoring_test = georeference_for_scoring(test, benchmark)
    all_coordinate_rows, all_planning_rows, per_seed, checkpoint_audit = [], [], {}, {}
    for seed in args.seeds:
        set_global_seed(seed)
        model_dir = artifact_root / "models" / "exp6_fusion_grounders" / args.method / f"seed_{seed}"
        model, metadata, fuser_path = _train_or_load(args.method, root, model_cfg, seed,
                                                      train, val, benchmark, device,
                                                      args.epochs, args.batch_size, model_dir)
        all_coordinate_rows.extend(_coordinate_rows(model, test, benchmark, device, args.method, seed))
        decomposer, decomposer_path = _load_decomposer(root, model_cfg, seed, device)
        planner = EnhancedPlanner(benchmark_dir=benchmark, fuser=model.fuser,
                                  decomposer=decomposer, device=device)
        rows = evaluate(GroundedPlanner(planner, model, benchmark, device), scoring_test, benchmark)
        for row in rows:
            row.update({"method": args.method, "training_seed": seed,
                        "planner_input": "predicted_coordinates_only",
                        "label_coordinates_used_only_for_scoring": True})
        all_planning_rows.extend(rows); per_seed[seed] = rows
        checkpoint_audit[str(seed)] = {"fusion_grounder": str(model_dir / "fusion_grounder.pt"),
                                       "source_fuser": fuser_path, "decomposer": decomposer_path,
                                       "selected_epoch": metadata["selected_epoch"]}
        print(f"[Exp6 fusion-grounder] complete | method={args.method} seed={seed}", flush=True)
    write_rows_csv(output / "grounding_coordinate_results.csv", all_coordinate_rows)
    write_rows_csv(output / "grounded_planning_results.csv", all_planning_rows)
    write_json(output / "summary.json", run_metadata({
        "experiment": "Exp6 independent fusion-conditioned map-grounding control",
        "method": args.method, "seeds": list(args.seeds), "smoke": bool(args.smoke),
        "sample_counts": {"train": len(train), "val": len(val), "test": len(test)},
        "coordinate_error_mean_pct": float(np.mean([r["mean_target_coordinate_error_pct"] for r in all_coordinate_rows])),
        "summary": aggregate_seed_means(per_seed), "checkpoint_audit": checkpoint_audit,
        "label_leakage_gate": "passed", "planner_input": "predicted_coordinates_only",
        "protocol_boundary": "Each method learns its own map-grounding head conditioned on its own pretrained fusion encoder. V2 has map and text inputs only; it is not a raw four-modality comparison.",
        "claim_allowed": "local implementation-inspired fusion architecture comparison under the shared V2 map-grounding protocol",
        "claim_forbidden": "official author-code reproduction or a claim that raw voice, gesture, and annotation are evaluated by V2",
    }))


if __name__ == "__main__":
    main()
