"""Cheap file-level checks before submitting the main-baseline GPU rerun."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from submission_common import DEFAULT_SEEDS


def has_checkpoint(directory: Path, stem: str) -> bool:
    return (directory / f"{stem}.pt").exists() or any(directory.glob(f"{stem}_epoch*.pt"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--stage", choices=("before_training", "before_evaluation", "before_merge"), required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    problems = []
    for filename in ("train.json", "val_location.json", "ood_location_test.json", "split_audit.json"):
        if not (root / "splits" / filename).is_file():
            problems.append(f"Missing split file: splits/{filename}")
    audit_file = root / "splits" / "split_audit.json"
    if audit_file.is_file():
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
        if audit.get("random_coordinate_fallbacks") not in (0, None):
            problems.append("Split audit reports random coordinate fallbacks")
    if args.stage in {"before_evaluation", "before_merge"}:
        for seed in DEFAULT_SEEDS:
            directory = root / "models" / "lpanet" / f"seed_{seed}"
            if not (directory / "training_metadata.json").is_file() or not has_checkpoint(directory, "best_fusion_model"):
                problems.append(f"Incomplete LPANet checkpoint for seed {seed}")
    if args.stage == "before_merge":
        for baseline in ("uav_vlpa", "lpanet", "citynav"):
            directory = root / "results" / "main_baseline_workers" / baseline
            if not (directory / "summary.json").is_file() or not (directory / "per_scenario_results.csv").is_file():
                problems.append(f"Missing evaluation output for {baseline}")
    if problems:
        print("MAIN-BASELINE PREFLIGHT FAILED")
        print("\n".join(f"- {item}" for item in problems))
        raise SystemExit(1)
    print(f"MAIN-BASELINE PREFLIGHT PASSED | stage={args.stage} | root={root}")


if __name__ == "__main__":
    main()
