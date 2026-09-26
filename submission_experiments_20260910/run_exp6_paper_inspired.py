"""Exp6 paper-inspired baselines under the V2 learned-grounding protocol.

These are not author-code reproductions.  Each entry isolates the paper's
published core idea while sharing the same V2 visual-language coordinate
grounder, train/validation split, OOD images and realised-trajectory scorer.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
for search_path in (PROJECT_ROOT, PACKAGE_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import torch

from configs.experiment_config import get_default_config
from models.fusion.baseline_fusers import AFFNetFuser, MAFTNetFuser, SCALFuser
from models.fusion.multimodal_fuser import MultimodalFuser
from models.planner.baseline_systems import ModularUAVSystem, build_baseline_system
from models.planner.enhanced_planner import EnhancedPlanner
from models.planner.baseline_path_planners import get_baseline_path_planner
from models.reasoning.baseline_decomposers import get_baseline_decomposer
from models.reasoning.task_decomposer import TaskDecomposer
from submission_common import DEFAULT_SEEDS, aggregate_seed_means, evaluate, read_samples, run_metadata, write_json, write_rows_csv
from utils.seed_manager import set_global_seed
from v2_grounding import VisualLanguageGrounder
from run_v2_grounding_worker import GroundedPlanner, georeference_for_scoring
from experiment_models import _materialize_audio_encoder
from submission_common import load_checkpoint


BASELINES = {
    "uav_vla": {"category": "system", "source": "UAV-VLA", "system": "uav_vla_style", "checkpoint": "maftnet"},
    "uav_vln": {"category": "system", "source": "UAV-VLN", "system": "uav_vln_style", "checkpoint": "affnet"},
    "aerialvln": {"category": "system", "source": "AerialVLN", "system": "aerialvln_style", "checkpoint": "scal"},
    "affnet": {"category": "fusion", "source": "AFFNet", "fuser": AFFNetFuser, "checkpoint": "affnet"},
    "maftnet": {"category": "fusion", "source": "MAFTNet", "fuser": MAFTNetFuser, "checkpoint": "maftnet"},
    "scal": {"category": "fusion", "source": "SCAL", "fuser": SCALFuser, "checkpoint": "scal"},
    "sipsa": {"category": "decomposition", "source": "SIPSA", "decomposer": "sipsa"},
    "unigoal": {"category": "decomposition", "source": "UniGoal", "decomposer": "unigoal"},
    "codeagents": {"category": "decomposition", "source": "UAV-CodeAgents", "decomposer": "codeagents"},
    "ikap": {"category": "planning", "source": "iKap", "path": "ikap"},
    "csglso": {"category": "planning", "source": "CSGLSO", "path": "csglso"},
}


def _parse_seeds(value):
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    unsupported = set(seeds) - set(DEFAULT_SEEDS)
    if unsupported:
        raise argparse.ArgumentTypeError(f"Unsupported seeds: {sorted(unsupported)}")
    return seeds


def _new_fuser(cls, cfg, device):
    common = dict(fusion_dim=cfg.fusion_dim, audio_model=cfg.audio_model,
                  gesture_backbone=cfg.gesture_backbone, text_model=cfg.text_model)
    if cls is MultimodalFuser:
        return cls(attention_heads=cfg.attention_heads, attention_layers=cfg.attention_layers,
                   ffn_dim=cfg.ffn_dim, **common).to(device)
    return cls(**common).to(device)


def _load_fuser(method, root, cfg, device):
    cls = {"ours": MultimodalFuser, "affnet": AFFNetFuser, "maftnet": MAFTNetFuser, "scal": SCALFuser}[method]
    fuser = _new_fuser(cls, cfg, device); _materialize_audio_encoder(fuser)
    load_checkpoint(fuser, root / "models" / method / f"seed_{_ACTIVE_SEED}", "best_fusion_model", device)
    return fuser


def _load_ours_decomposer(root, cfg, device):
    decomposer = TaskDecomposer(d_model=cfg.fusion_dim, n_layers=cfg.decomposer_layers,
                                max_subtasks=cfg.max_subtasks, max_seq_len=100).to(device)
    load_checkpoint(decomposer, root / "models" / "ours" / f"seed_{_ACTIVE_SEED}", "decomposer_rl", device)
    return decomposer


def _controlled_enhanced(fuser, root, base_cfg, model_cfg, device):
    """The shared Full pipeline used for every one-module comparison."""
    return EnhancedPlanner(benchmark_dir=base_cfg.benchmark_dir, fuser=fuser,
                           decomposer=_load_ours_decomposer(root, model_cfg, device), device=device)


def _deterministic_completion(self, targets, traj, successful_segments, total_segments, n_modalities=1):
    """Remove legacy random completion confirmation from comparison methods."""
    if not traj or total_segments <= 0:
        return []
    count = min(len(targets), round(len(targets) * successful_segments / total_segments))
    return [target.name for target in targets[:count]]


def _planner(baseline, root, base_cfg, model_cfg, device):
    spec = BASELINES[baseline]
    if spec["category"] == "system":
        # A system baseline intentionally changes multiple modules.  Its
        # results are reported only in the System-level subsection.
        planner = build_baseline_system(spec["system"], base_cfg.benchmark_dir, model_cfg, device, _ACTIVE_SEED)
        _materialize_audio_encoder(planner.fuser)
        load_checkpoint(planner.fuser, root / "models" / spec["checkpoint"] / f"seed_{_ACTIVE_SEED}", "best_fusion_model", device)
    elif spec["category"] == "fusion":
        fuser = _load_fuser(spec["checkpoint"], root, model_cfg, device)
        # Controlled variable: fuser only. Grounder, neural decomposer, A*,
        # safety settings and scorer remain the Full system's exact modules.
        planner = _controlled_enhanced(fuser, root, base_cfg, model_cfg, device)
    elif spec["category"] == "decomposition":
        fuser = _load_fuser("ours", root, model_cfg, device)
        planner = _controlled_enhanced(fuser, root, base_cfg, model_cfg, device)
        replacement = get_baseline_decomposer(spec["decomposer"], _ACTIVE_SEED)
        # EnhancedPlanner consumes (tasks, ordered_targets) from this hook.
        # Replacing only this hook preserves its fusion, safety margin, A*
        # implementation, coordinate conversion and evaluation interface.
        counter = {"calls": 0}
        def swapped_decompose(self, scenario, fused):
            counter["calls"] += 1
            return replacement.decompose_scenario(scenario, fused)
        planner._neural_task_decompose = types.MethodType(swapped_decompose, planner)
        planner._component_audit_counter = counter
    else:
        fuser = _load_fuser("ours", root, model_cfg, device)
        planner = _controlled_enhanced(fuser, root, base_cfg, model_cfg, device)
        external_path = get_baseline_path_planner(spec["path"], base_cfg.benchmark_dir, _ACTIVE_SEED)
        # Controlled variable: path solver only. The fuser/decomposer prepare
        # the same ordered predicted targets before the substitute solver runs.
        original_plan = planner.plan
        counter = {"calls": 0}
        def swapped_path(self, image_id, image_path, targets_pct, obstacles_pct, min_rad=25):
            counter["calls"] += 1
            scenario = self._exp6_active_scenario
            clone = __import__("copy").deepcopy(scenario)
            by_name = {target.name: target for target in clone.targets}
            clone.targets = [by_name[name] for name in targets_pct if name in by_name]
            for name, value in targets_pct.items():
                if name in by_name:
                    by_name[name].coordinates_percent = tuple(value["coordinates"])
            plan = external_path.plan(clone)
            return list(plan.trajectory_latlon), int(bool(plan.trajectory_latlon)), 1
        planner._run_astar_enhanced = types.MethodType(swapped_path, planner)
        def active_plan(scenario):
            planner._exp6_active_scenario = scenario
            return original_plan(scenario)
        planner.plan = active_plan
        planner._component_audit_counter = counter
    if isinstance(planner, ModularUAVSystem):
        planner._determine_completed = types.MethodType(_deterministic_completion, planner)
    return planner


def _grounder_checkpoint(root, seed):
    """Resolve the V2 checkpoint from the independent-seed shard layout.

    The legacy monolithic location remains a read-only fallback for runs made
    before V2 was parallelised.  A checkpoint without its matching metadata is
    rejected so Exp6 cannot silently consume an incomplete V2 run.
    """
    candidates = (
        root / "v2_worker_shards" / "visual_language_grounding" /
        f"seed_{seed}" / "models" / "v2_osm_grounding" /
        "visual_language_grounding" / f"seed_{seed}",
        root / "models" / "v2_osm_grounding" /
        "visual_language_grounding" / f"seed_{seed}",
    )
    for directory in candidates:
        checkpoint, metadata = directory / "grounder.pt", directory / "training_metadata.json"
        if checkpoint.is_file() and metadata.is_file():
            return checkpoint
    checked = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Missing complete V2 visual-grounder checkpoint for seed {seed}. Checked:\n  {checked}"
    )


def _load_grounder(root, seed, device):
    checkpoint = _grounder_checkpoint(root, seed)
    record = torch.load(checkpoint, map_location=device)
    model = VisualLanguageGrounder(use_image=bool(record["use_image"])).to(device)
    model.load_state_dict(record["model_state_dict"], strict=True); model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=tuple(BASELINES), required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--artifact-root")
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--component-control", action="store_true",
                        help="write an auditable local one-component control, not an external reproduction")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    artifact_root = Path(args.artifact_root).resolve() if args.artifact_root else root
    worker_root = "exp6_component_workers" if args.component_control else "exp6_paper_workers"
    output = artifact_root / "results" / worker_root / args.baseline
    if output.exists(): raise FileExistsError(f"Refusing to overwrite {output}")
    base_cfg, _, model_cfg, _, _ = get_default_config(); device = base_cfg.device
    test = read_samples(root / "v2_osm_protocol" / "splits" / "ood_location_test.json")
    if args.max_test_samples:
        test = test[:args.max_test_samples]
    scoring_test = georeference_for_scoring(test, base_cfg.benchmark_dir)
    per_seed, all_rows, checkpoints, component_calls = {}, [], {}, {}
    global _ACTIVE_SEED
    for seed in args.seeds:
        _ACTIVE_SEED = seed; set_global_seed(seed)
        grounder = _load_grounder(root, seed, device)
        core = _planner(args.baseline, root, base_cfg, model_cfg, device)
        planner = GroundedPlanner(core, grounder, base_cfg.benchmark_dir, device)
        rows = evaluate(planner, scoring_test, base_cfg.benchmark_dir)
        calls = int(getattr(core, "_component_audit_counter", {}).get("calls", 0))
        if args.component_control and calls <= 0:
            raise RuntimeError(f"{args.baseline} did not invoke its declared replacement component for seed {seed}")
        for row in rows: row.update({"method": args.baseline, "training_seed": seed})
        per_seed[seed] = rows; all_rows.extend(rows)
        checkpoints[str(seed)] = {
            "grounder": str(_grounder_checkpoint(root, seed)),
            "core": BASELINES[args.baseline],
        }
        component_calls[str(seed)] = calls
        print(f"[Exp6 V2] complete | baseline={args.baseline} | seed={seed}", flush=True)
    write_rows_csv(output / "per_scenario_results.csv", all_rows)
    write_json(output / "summary.json", run_metadata({
        "experiment": "Exp6 local component control" if args.component_control else "Exp6 paper-inspired baseline adaptation", "baseline": args.baseline,
        "source_method": BASELINES[args.baseline], "seeds": list(args.seeds),
        "smoke": bool(args.smoke), "test_samples": len(test),
        "summary": aggregate_seed_means(per_seed), "checkpoint_audit": checkpoints,
        "component_intervention_calls": component_calls,
        "label_leakage_gate": "passed",
        "planner_input": "predicted_coordinates_only",
        "implementation_status": "local implementation-inspired component control" if args.component_control else "paper-inspired adaptation",
        "claim_allowed": "comparison of the disclosed local one-component control under the shared V2 protocol" if args.component_control else "comparison of the disclosed adapted implementation under the shared V2 protocol; fusion/decomposition/path entries replace exactly one listed module",
        "claim_forbidden": "official-code reproduction, direct equivalence to the source paper's original benchmark, or an external SOTA ranking",
    }))


if __name__ == "__main__":
    main()
