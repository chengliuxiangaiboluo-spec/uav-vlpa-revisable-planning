"""Fail closed before strict main-baseline jobs are submitted or merged."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Keep this file-only preflight runnable on a login node with no CUDA/CuDNN
# modules.  Importing ``submission_common`` would import PyTorch unnecessarily.
DEFAULT_SEEDS = (42, 123, 456, 789, 2024)


def _checkpoint(directory: Path, stem: str) -> bool:
    return (directory / f"{stem}.pt").is_file() or any(directory.glob(f"{stem}_epoch*.pt"))


def _grounder(root: Path, condition: str, seed: int) -> bool:
    candidates = (
        root / "v2_worker_shards" / condition / f"seed_{seed}" / "models" /
        "v2_osm_grounding" / condition / f"seed_{seed}",
        root / "models" / "v2_osm_grounding" / condition / f"seed_{seed}",
    )
    return any((path / "grounder.pt").is_file() and (path / "training_metadata.json").is_file() for path in candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--stage", choices=("before_training", "before_evaluation", "before_merge"), required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    protocol = root / "v2_osm_protocol"
    errors = []
    for name in ("train.json", "val_location.json", "ood_location_test.json"):
        if not (protocol / "splits" / name).is_file():
            errors.append(f"missing V2 split: {name}")
    # ``prepare_v2_osm_protocol.py`` writes the canonical V2 record as
    # ``audit.json``.  Do not silently accept an un-audited protocol.
    audit_path = protocol / "audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "COMPLETE":
            errors.append("V2 protocol audit is not COMPLETE")
        if audit.get("random_coordinate_fallbacks") != 0:
            errors.append("V2 audit reports random coordinate fallback")
    else:
        errors.append("missing V2 protocol audit")

    for seed in DEFAULT_SEEDS:
        for condition in ("text_only_grounding", "visual_language_grounding"):
            if not _grounder(root, condition, seed):
                errors.append(f"missing complete V2 {condition} grounder for seed {seed}")
    if args.stage in {"before_evaluation", "before_merge"}:
        for seed in DEFAULT_SEEDS:
            folder = root / "models" / "lpanet_v2_strict" / f"seed_{seed}"
            if not (folder / "training_metadata.json").is_file() or not _checkpoint(folder, "best_fusion_model"):
                errors.append(f"missing strict LPANet checkpoint for seed {seed}")
                continue
            metadata = json.loads((folder / "training_metadata.json").read_text(encoding="utf-8"))
            if metadata.get("method") != "lpanet" or metadata.get("output_model_id") != "lpanet_v2_strict":
                errors.append(f"seed {seed}: wrong model identity in training metadata")
            if metadata.get("training_seed") != seed:
                errors.append(f"seed {seed}: training metadata seed mismatch")
            if Path(str(metadata.get("train_file", ""))).resolve() != (protocol / "splits" / "train.json").resolve():
                errors.append(f"seed {seed}: training did not use the V2 train split")
            if Path(str(metadata.get("val_file", ""))).resolve() != (protocol / "splits" / "val_location.json").resolve():
                errors.append(f"seed {seed}: training did not use the V2 validation split")
            if not metadata.get("fusion_history"):
                errors.append(f"seed {seed}: fusion training history is missing")
    if args.stage == "before_merge":
        for baseline in ("uav_vlpa", "lpanet", "citynav"):
            folder = root / "results" / "main_baseline_strict_workers" / baseline
            for name in ("summary.json", "per_scenario_results.csv"):
                if not (folder / name).is_file():
                    errors.append(f"missing strict {baseline} output: {name}")
            summary_path = folder / "summary.json"
            if summary_path.is_file() and json.loads(summary_path.read_text(encoding="utf-8")).get("smoke"):
                errors.append(f"strict {baseline} output is a smoke run")
    if errors:
        print("STRICT MAIN-BASELINE PREFLIGHT FAILED")
        print("\n".join(f"- {item}" for item in errors))
        raise SystemExit(1)
    print(f"STRICT MAIN-BASELINE PREFLIGHT PASSED | stage={args.stage} | root={root}")


if __name__ == "__main__":
    main()
