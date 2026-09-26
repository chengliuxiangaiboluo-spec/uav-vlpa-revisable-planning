"""Static and artifact checks for the three independent Exp6 fusion controls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SEEDS = (42, 123, 456, 789, 2024)
METHODS = ("affnet", "maftnet", "scal")


def complete_checkpoint(directory: Path, stem: str):
    return directory.is_dir() and any(directory.glob(f"{stem}*.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--stage", choices=("before_train", "before_merge"), required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    errors = []
    # Refuse a submission if an upload omitted any part of this independent
    # implementation.  This is deliberately a static check, so it works on a
    # login node without importing CUDA/PyTorch.
    package = Path(__file__).resolve().parent
    for name in (
        "fusion_conditioned_grounder.py", "run_exp6_fusion_grounders.py",
        "preflight_exp6_fusion_grounders.py", "run_v2_grounding_worker.py",
    ):
        if not (package / name).is_file():
            errors.append(f"missing required Exp6 source file: {package / name}")
    audit_path = root / "v2_osm_protocol" / "audit.json"
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "COMPLETE": errors.append("V2 protocol audit is not COMPLETE")
        if audit.get("random_coordinate_fallbacks") != 0: errors.append("V2 audit permits random coordinate fallbacks")
    except Exception as exc:
        errors.append(f"invalid V2 protocol audit: {exc}")
    for split in ("train.json", "val_location.json", "ood_location_test.json"):
        if not (root / "v2_osm_protocol" / "splits" / split).is_file():
            errors.append(f"missing V2 split: {split}")
    for method in METHODS:
        for seed in SEEDS:
            source = root / "models" / method / f"seed_{seed}"
            if not (source / "training_metadata.json").is_file() or not complete_checkpoint(source, "best_fusion_model"):
                errors.append(f"missing complete source fuser: {method}/seed_{seed}")
            ours = root / "models" / "ours" / f"seed_{seed}"
            if not complete_checkpoint(ours, "decomposer_rl"):
                errors.append(f"missing Ours decomposer: seed_{seed}")
    # This directory is intentionally checked rather than downloaded: server
    # experiments must never silently attempt a Hugging Face network request.
    weights = Path(os.environ.get("UAV_VLPA_WEIGHTS_DIR", Path(__file__).resolve().parents[1] / "Weights")) / "all-MiniLM-L6-v2"
    if not (weights / "config.json").is_file():
        errors.append(f"missing offline sentence encoder config: {weights / 'config.json'}")
    if args.stage == "before_merge":
        for method in METHODS:
            summary = root / "results" / "exp6_fusion_grounder_workers" / method / "summary.json"
            rows = root / "results" / "exp6_fusion_grounder_workers" / method / "grounded_planning_results.csv"
            try:
                record = json.loads(summary.read_text(encoding="utf-8"))
                if record.get("smoke"): errors.append(f"{method} is a smoke result")
                if record.get("label_leakage_gate") != "passed": errors.append(f"{method} leakage gate failed")
                if record.get("planner_input") != "predicted_coordinates_only": errors.append(f"{method} planner input is invalid")
                if set(record.get("seeds", [])) != set(SEEDS): errors.append(f"{method} does not contain all five seeds")
                if not rows.is_file() or rows.stat().st_size == 0: errors.append(f"{method} has no per-scenario planning rows")
            except Exception as exc:
                errors.append(f"invalid completed {method} result: {exc}")
    if errors:
        print("STRICT EXP6 FUSION-GROUNDER PREFLIGHT FAILED")
        print("\n".join(f"- {item}" for item in errors))
        raise SystemExit(1)
    print("STRICT EXP6 FUSION-GROUNDER PREFLIGHT PASSED")
    print("Each independent method will learn its own map-grounding head; labels remain evaluator-only.")


if __name__ == "__main__":
    main()
