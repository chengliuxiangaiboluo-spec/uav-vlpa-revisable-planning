"""Merge the three corrected, independent Exp6 fusion-grounder controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from submission_common import aggregate_seed_means, run_metadata, write_json, write_rows_csv

SEEDS = {42, 123, 456, 789, 2024}
METHODS = ("affnet", "maftnet", "scal")


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    destination = root / "results" / "exp6_fusion_grounder_comparison"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    all_rows, by_method, source = [], {}, {}
    for method in METHODS:
        worker = root / "results" / "exp6_fusion_grounder_workers" / method
        record = json.loads((worker / "summary.json").read_text(encoding="utf-8"))
        rows = read_rows(worker / "grounded_planning_results.csv")
        if record.get("smoke") or record.get("label_leakage_gate") != "passed" or record.get("planner_input") != "predicted_coordinates_only":
            raise RuntimeError(f"invalid protocol gate: {method}")
        if set(record.get("seeds", [])) != SEEDS:
            raise RuntimeError(f"incomplete five-seed result: {method}")
        if any(str(row.get("planner_input")) != "predicted_coordinates_only" or str(row.get("label_coordinates_used_only_for_scoring")).lower() != "true" for row in rows):
            raise RuntimeError(f"planner/label audit failed in {method}")
        grouped = {seed: [] for seed in SEEDS}
        for row in rows: grouped[int(row["training_seed"])].append(row)
        if any(not value for value in grouped.values()): raise RuntimeError(f"missing rows for a seed in {method}")
        by_method[method] = aggregate_seed_means(grouped)
        all_rows.extend(rows)
        source[method] = str(worker)
    write_rows_csv(destination / "per_scenario_results.csv", all_rows)
    write_json(destination / "summary.json", run_metadata({
        "experiment": "Exp6 corrected independent fusion-conditioned map-grounding controls",
        "methods": by_method, "seeds": sorted(SEEDS), "source_workers": source,
        "label_leakage_gate": "passed", "planner_input": "predicted_coordinates_only",
        "protocol_boundary": "The V2 OSM protocol supplies a semantic map and text instruction, not raw voice, gesture, or annotation files. Each method has its own fine-tuned fusion-conditioned grounding head and shares only the split, evaluation planner, and scorer.",
        "claim_allowed": "local implementation-inspired map-grounding fusion architecture comparison",
        "claim_forbidden": "official author-code reproduction or raw four-modality fusion claim",
    }))
    print(f"Merged corrected Exp6 fusion controls: {destination}")


if __name__ == "__main__":
    main()
