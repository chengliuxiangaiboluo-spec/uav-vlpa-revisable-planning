"""Create the only data partition permitted by the submission experiment suite."""

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from pathlib import Path

from submission_common import (
    assert_no_scenario_overlap, image_ids, read_samples, run_metadata,
    scenario_ids, sha256_file, write_json, write_samples,
)


def split_by_location_then_scenario(samples, seed: int, id_holdout_ratio: float):
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.image_id].append(sample)
    if len(groups) < 10:
        raise ValueError("Need at least 10 distinct image_id locations for a three-way location split")

    locations = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(locations)
    # Retains the historical 21/4/5 location allocation for a 30-map corpus.
    n_ood = max(1, round(len(locations) * (5 / 30)))
    n_val = max(1, round(len(locations) * (4 / 30)))
    n_train_locations = len(locations) - n_val - n_ood
    if n_train_locations < 1:
        raise ValueError("No location remains for training")

    train_locations = locations[:n_train_locations]
    val_locations = locations[n_train_locations:n_train_locations + n_val]
    ood_locations = locations[n_train_locations + n_val:]

    train, id_test = [], []
    for location in train_locations:
        within_location = sorted(groups[location], key=lambda s: str(s.scenario_id))
        local_rng = random.Random(f"{seed}:{location}")
        local_rng.shuffle(within_location)
        n_id = max(1, round(len(within_location) * id_holdout_ratio))
        id_test.extend(within_location[:n_id])
        train.extend(within_location[n_id:])

    val = [s for location in val_locations for s in groups[location]]
    ood = [s for location in ood_locations for s in groups[location]]
    return train, val, id_test, ood, {
        "train_locations": sorted(train_locations),
        "validation_locations": sorted(val_locations),
        "ood_locations": sorted(ood_locations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create audited train/val/ID/OOD submission splits")
    parser.add_argument("--dataset", help="Single source dataset JSON")
    parser.add_argument("--source-files", help="Comma-separated source split JSON files to combine")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-holdout-ratio", type=float, default=0.20)
    args = parser.parse_args()
    started_at = time.monotonic()
    if not 0.05 <= args.id_holdout_ratio <= 0.40:
        raise ValueError("id-holdout-ratio must be between 0.05 and 0.40")

    if bool(args.dataset) == bool(args.source_files):
        raise ValueError("Provide exactly one of --dataset or --source-files")
    output = Path(args.output_dir).resolve()
    if args.dataset:
        source_paths = [Path(args.dataset).resolve()]
    else:
        source_paths = [Path(part.strip()).resolve() for part in args.source_files.split(",") if part.strip()]
    if not source_paths or any(not path.exists() for path in source_paths):
        raise FileNotFoundError(f"Missing source dataset file(s): {source_paths}")
    samples = [sample for path in source_paths for sample in read_samples(path)]
    print(f"[Split] source loaded | files={len(source_paths)} | samples={len(samples)} | seed={args.seed}", flush=True)
    train, val, id_test, ood, locations = split_by_location_then_scenario(
        samples, args.seed, args.id_holdout_ratio
    )
    named = {"train": train, "val_location": val, "id_test": id_test, "ood_location_test": ood}
    assert_no_scenario_overlap(named)

    # ID is same-location but unseen-scenario; OOD is unseen-location.
    if not image_ids(id_test) <= image_ids(train):
        raise AssertionError("ID test must only use training locations")
    if image_ids(ood) & (image_ids(train) | image_ids(val) | image_ids(id_test)):
        raise AssertionError("OOD location leakage detected")
    if image_ids(val) & (image_ids(train) | image_ids(id_test) | image_ids(ood)):
        raise AssertionError("Validation location leakage detected")

    for name, subset in named.items():
        write_samples(output / f"{name}.json", subset)
        print(f"[Split] wrote | subset={name} | samples={len(subset)} | locations={len(image_ids(subset))}", flush=True)

    audit = run_metadata({
        "protocol_version": "submission-2026-09-10",
        "source_dataset_files": [str(path) for path in source_paths],
        "source_dataset_sha256": {str(path): sha256_file(path) for path in source_paths},
        "split_seed": args.seed,
        "id_holdout_ratio": args.id_holdout_ratio,
        "location_allocation": locations,
        "sets": {
            name: {
                "n_scenarios": len(subset),
                "n_locations": len(image_ids(subset)),
                "image_ids": sorted(image_ids(subset)),
                "scenario_ids_sha256": __import__("hashlib").sha256(
                    "\\n".join(sorted(scenario_ids(subset))).encode("utf-8")
                ).hexdigest(),
            }
            for name, subset in named.items()
        },
        "claims_allowed": {
            "id_test": "unseen scenarios at locations represented in training",
            "ood_location_test": "unseen satellite-map locations from the same synthetic generator",
        },
        "claims_forbidden": [
            "real-world validation",
            "public-map OOD without an independently collected public-map dataset",
        ],
    })
    write_json(output / "split_audit.json", audit)
    print(f"[Split] complete | elapsed={(time.monotonic() - started_at) / 60.0:.1f} min | output={output}", flush=True)


if __name__ == "__main__":
    main()
