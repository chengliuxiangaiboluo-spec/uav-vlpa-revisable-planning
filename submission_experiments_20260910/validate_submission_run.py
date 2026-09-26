"""Fail closed when a submission run lacks the evidence required for reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from submission_common import DEFAULT_SEEDS


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a complete submission experiment run")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    failures: list[str] = []
    audit_path = root / "splits" / "split_audit.json"
    require(audit_path.exists(), "Missing split_audit.json", failures)
    if audit_path.exists():
        audit = load_json(audit_path)
        sets = audit.get("sets", {})
        require(all(name in sets and sets[name].get("n_scenarios", 0) > 0
                    for name in ("train", "val_location", "id_test", "ood_location_test")),
                "Split audit lacks a non-empty train/val/ID/OOD set", failures)
        train_images = set(sets.get("train", {}).get("image_ids", []))
        id_images = set(sets.get("id_test", {}).get("image_ids", []))
        ood_images = set(sets.get("ood_location_test", {}).get("image_ids", []))
        require(id_images <= train_images, "ID test is not restricted to training locations", failures)
        require(not (ood_images & train_images), "OOD test shares locations with training", failures)

    for method in ("ours", "affnet", "maftnet", "scal"):
        for seed in DEFAULT_SEEDS:
            metadata = root / "models" / method / f"seed_{seed}" / "training_metadata.json"
            require(metadata.exists(), f"Missing independent training metadata: {method}/seed_{seed}", failures)
            if metadata.exists():
                record = load_json(metadata)
                require(record.get("training_seed") == seed,
                        f"Seed metadata mismatch: {method}/seed_{seed}", failures)
                require(record.get("independent_initialization") is True,
                        f"Independent initialization not recorded: {method}/seed_{seed}", failures)
        if method == "ours":
            for seed in DEFAULT_SEEDS:
                folder = root / "models" / method / f"seed_{seed}"
                require(any(folder.glob("decomposer_rl_epoch*.pt")),
                        f"Missing decomposer checkpoint: ours/seed_{seed}", failures)

    required_results = (
        root / "results" / "exp1_fair_baselines" / "summary.json",
        root / "results" / "exp2_input_interventions" / "summary.json",
        root / "results" / "exp3_robustness" / "summary.json",
        root / "results" / "exp4_strict_generalization" / "summary.json",
    )
    for path in required_results:
        require(path.exists(), f"Missing required experiment summary: {path.relative_to(root)}", failures)
        if path.exists():
            record = load_json(path)
            require(record.get("protocol_version") == "submission-2026-09-10",
                    f"Wrong/missing protocol version: {path.relative_to(root)}", failures)
            require(record.get("seeds") == list(DEFAULT_SEEDS),
                    f"Result does not record exactly the prescribed five seeds: {path.relative_to(root)}", failures)

    if failures:
        print("SUBMISSION RUN NOT READY")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("SUBMISSION RUN AUDIT PASSED")
    print("This confirms protocol completeness only; it does not guarantee acceptance or effect direction.")


if __name__ == "__main__":
    main()
