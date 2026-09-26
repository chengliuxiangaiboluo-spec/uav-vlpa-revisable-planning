"""Validate and merge the three independent main-table baseline workers."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from submission_common import DEFAULT_SEEDS, METRICS, aggregate_seed_means, run_metadata, write_json, write_rows_csv


EXPECTED = {"uav_vlpa", "lpanet", "citynav"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    workers = root / "results" / "main_baseline_workers"
    rows, grouped, worker_audit = [], defaultdict(lambda: defaultdict(list)), {}
    for baseline_id in sorted(EXPECTED):
        directory = workers / baseline_id
        with (directory / "summary.json").open(encoding="utf-8") as handle:
            worker_audit[baseline_id] = json.load(handle)
        with (directory / "per_scenario_results.csv").open(encoding="utf-8", newline="") as handle:
            current = list(csv.DictReader(handle))
        if len(current) != 5 * 500:
            raise ValueError(f"{baseline_id} must contain exactly 2,500 rows, found {len(current)}")
        for row in current:
            if row["method"] != baseline_id:
                raise ValueError(f"Unexpected method in {baseline_id} worker")
            for metric in METRICS:
                row[metric] = float(row[metric])
            grouped[baseline_id][int(row["training_seed"])].append(row)
            rows.append(row)
        if set(grouped[baseline_id]) != set(DEFAULT_SEEDS):
            raise ValueError(f"{baseline_id} does not contain all five seeds")
    output = root / "results" / "main_baseline_comparison"
    output.mkdir(parents=True, exist_ok=False)
    write_rows_csv(output / "per_scenario_results.csv", rows)
    write_json(output / "summary.json", run_metadata({
        "experiment": "Main-table original baselines",
        "methods": {
            "uav_vlpa": "UAV-VLPA (text-only)",
            "lpanet": "LPANet (local unified-interface implementation; not author-code reproduction)",
            "citynav": "CityNav (local unified-interface implementation; not author-code reproduction)",
        },
        "seeds": list(DEFAULT_SEEDS),
        "summary": {name: aggregate_seed_means(seed_rows) for name, seed_rows in grouped.items()},
        "worker_audit": worker_audit,
        "protocol_boundary": "No synthetic coordinate noise or random completion sampling is used by these three reruns.",
    }))
    print(f"Merged main baselines: {output}")


if __name__ == "__main__":
    main()
