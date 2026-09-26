"""Merge only audited strict main-baseline outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from submission_common import run_metadata, write_json, write_rows_csv


BASELINES = ("uav_vlpa", "lpanet", "citynav")


def _rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    output = root / "results" / "main_baseline_strict_comparison"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite strict merged results: {output}")
    all_rows, methods = [], {}
    for baseline in BASELINES:
        folder = root / "results" / "main_baseline_strict_workers" / baseline
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        rows = _rows(folder / "per_scenario_results.csv")
        if summary.get("label_leakage_gate") != "passed" or summary.get("planner_input") != "predicted_coordinates_only":
            raise RuntimeError(f"Protocol gate failed for {baseline}")
        if not rows or any(str(row.get("planner_input")) != "predicted_coordinates_only" or str(row.get("label_coordinates_used_only_for_scoring")).lower() != "true" for row in rows):
            raise RuntimeError(f"Per-row label/planner audit failed for {baseline}")
        methods[baseline] = {
            "display_name": summary["display_name"],
            "summary": summary["summary"],
            "checkpoint_and_protocol_audit": summary["checkpoint_and_protocol_audit"],
        }
        all_rows.extend(rows)
    output.mkdir(parents=True)
    write_rows_csv(output / "per_scenario_results.csv", all_rows)
    write_json(output / "summary.json", run_metadata({
        "experiment": "Strict original main-table baseline comparison",
        "methods": methods,
        "label_leakage_gate": "passed",
        "planner_input": "predicted_coordinates_only",
        "protocol_boundary": "V2 grounder labels supervise training and score held-out predictions only; no target coordinate reaches a planner.",
        "claim_allowed": "held-out shared-V2 learned-grounding comparison of disclosed local adaptations",
        "claim_forbidden": "official implementation reproduction or oracle-coordinate planner comparison",
    }))
    print(f"[Strict main baseline merge] output={output}", flush=True)


if __name__ == "__main__":
    main()
