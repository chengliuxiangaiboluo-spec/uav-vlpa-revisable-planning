"""Merge LPANet and CityNav for the Molmo-UAV-VLPA* V2 main table.

The frozen Molmo result is intentionally not merged here because it has one
deterministic zero-shot run rather than five independently trained seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from submission_common import DEFAULT_SEEDS, run_metadata, write_json, write_rows_csv


BASELINES = ("lpanet", "citynav")


def _read_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    worker_root = root / "results" / "main_baseline_molmo_strict_workers"
    output = root / "results" / "main_baseline_molmo_strict_comparison"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite strict Molmo-main baseline merge: {output}")

    all_rows, methods = [], {}
    expected = len(DEFAULT_SEEDS) * 500
    for baseline in BASELINES:
        folder = worker_root / baseline
        summary_path, rows_path = folder / "summary.json", folder / "per_scenario_results.csv"
        if not summary_path.is_file() or not rows_path.is_file():
            raise FileNotFoundError(f"Missing formal {baseline} output under {folder}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = _read_rows(rows_path)
        if summary.get("smoke") or tuple(summary.get("seeds", ())) != DEFAULT_SEEDS:
            raise RuntimeError(f"{baseline}: invalid seed or smoke metadata")
        if summary.get("n_test_scenarios") != 500 or len(rows) != expected:
            raise RuntimeError(f"{baseline}: expected 500 scenarios and {expected} rows, got {summary.get('n_test_scenarios')} / {len(rows)}")
        if summary.get("label_leakage_gate") != "passed" or summary.get("planner_input") != "predicted_coordinates_only":
            raise RuntimeError(f"{baseline}: protocol gate failed")
        if any(
            row.get("planner_input") != "predicted_coordinates_only"
            or row.get("label_coordinates_used_only_for_scoring", "").lower() != "true"
            for row in rows
        ):
            raise RuntimeError(f"{baseline}: per-row label/planner audit failed")
        methods[baseline] = {
            "display_name": summary["display_name"],
            "summary": summary["summary"],
            "checkpoint_and_protocol_audit": summary["checkpoint_and_protocol_audit"],
        }
        all_rows.extend(rows)

    output.mkdir(parents=True)
    write_rows_csv(output / "per_scenario_results.csv", all_rows)
    write_json(output / "summary.json", run_metadata({
        "experiment": "Strict LPANet and CityNav comparison for the frozen Molmo UAV-VLPA* V2 main table",
        "methods": methods,
        "seeds": list(DEFAULT_SEEDS),
        "label_leakage_gate": "passed",
        "planner_input": "predicted_coordinates_only",
        "protocol_boundary": "LPANet and CityNav use held-out V2 samples and five learned-grounder seeds. UAV-VLPA* is added separately as a frozen zero-shot n=1 reference.",
    }))
    print(f"[Strict Molmo-main baseline merge] output={output}", flush=True)


if __name__ == "__main__":
    main()
