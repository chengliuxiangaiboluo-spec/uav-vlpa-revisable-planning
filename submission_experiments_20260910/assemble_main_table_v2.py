"""Assemble the four-method, same-protocol V2 main-table comparison.

This is a data-provenance assembly step.  It never recomputes model outputs
and refuses to mix the old oracle-coordinate main table with V2 results.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from submission_common import DEFAULT_SEEDS, run_metadata, write_json, write_rows_csv


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _require_v2_protocol(root: Path):
    audit = _read_json(root / "v2_osm_protocol" / "audit.json")
    if audit.get("status") != "COMPLETE" or audit.get("random_coordinate_fallbacks") != 0:
        raise RuntimeError("V2 protocol audit is incomplete or contains random coordinate fallbacks")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    output = root / "results" / "main_table_v2_strict"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite assembled V2 table: {output}")
    _require_v2_protocol(root)

    baseline_root = root / "results" / "main_baseline_strict_comparison"
    baseline_summary = _read_json(baseline_root / "summary.json")
    if baseline_summary.get("label_leakage_gate") != "passed" or baseline_summary.get("planner_input") != "predicted_coordinates_only":
        raise RuntimeError("Strict baseline gate is absent or failed")
    baseline_rows = _read_rows(baseline_root / "per_scenario_results.csv")
    expected = len(DEFAULT_SEEDS) * 500
    methods, rows = {}, []
    for method in ("uav_vlpa", "lpanet", "citynav"):
        selected = [row for row in baseline_rows if row.get("method") == method]
        if len(selected) != expected:
            raise RuntimeError(f"{method}: expected {expected} strict rows, got {len(selected)}")
        if any(row.get("planner_input") != "predicted_coordinates_only" or row.get("label_coordinates_used_only_for_scoring", "").lower() != "true" for row in selected):
            raise RuntimeError(f"{method}: per-row protocol audit failed")
        methods[method] = baseline_summary["methods"][method]
        rows.extend(selected)

    v2_root = root / "results" / "v2_osm_grounding_replan"
    v2_summary = _read_json(v2_root / "summary.json")
    if tuple(sorted(v2_summary.get("seeds", []))) != tuple(sorted(DEFAULT_SEEDS)):
        raise RuntimeError("V2 final summary does not contain exactly five seeds")
    visual_summary = v2_summary.get("planning_summary", {}).get("visual_language_grounding")
    if not visual_summary:
        raise RuntimeError("V2 visual-language Ours summary is missing")
    visual_rows = [
        row for row in _read_rows(v2_root / "grounded_planning_results.csv")
        if row.get("condition") == "visual_language_grounding"
    ]
    if len(visual_rows) != expected:
        raise RuntimeError(f"V2 Ours: expected {expected} visual-language rows, got {len(visual_rows)}")
    if {int(row["training_seed"]) for row in visual_rows} != set(DEFAULT_SEEDS):
        raise RuntimeError("V2 Ours rows do not cover the declared seeds")
    for row in visual_rows:
        row.update({
            "method": "ours_v2",
            "display_name": "Ours (visual-language grounding + enhanced planner)",
            "test_set": "v2_osm_ood_location_test",
            "planner_input": "predicted_coordinates_only",
            "label_coordinates_used_only_for_scoring": True,
        })
    methods["ours_v2"] = {
        "display_name": "Ours (visual-language grounding + enhanced planner)",
        "summary": visual_summary,
        "source": "results/v2_osm_grounding_replan/grounded_planning_results.csv",
        "source_condition": "visual_language_grounding",
        "planner_input": "predicted_coordinates_only",
        "label_coordinates_used_only_for_scoring": True,
    }
    rows.extend(visual_rows)

    output.mkdir(parents=True)
    write_rows_csv(output / "per_scenario_results.csv", rows)
    write_json(output / "summary.json", run_metadata({
        "experiment": "Four-method main comparison under shared V2 learned-grounding protocol",
        "methods": methods,
        "seeds": list(DEFAULT_SEEDS),
        "test_set": "v2_osm_ood_location_test",
        "n_rows_per_method": expected,
        "label_leakage_gate": "passed",
        "planner_input": "predicted_coordinates_only",
        "protocol_boundary": "All four methods use held-out V2 samples. Coordinates are predicted before planning; labels supervise fitting and score outputs only.",
        "text_only_naming": {
            "main_baseline": "UAV-VLPA (text-only grounder + A* planner)",
            "v2_ablation": "Ours with text-only coordinate grounding (enhanced planner retained)",
            "rule": "These are distinct controlled conditions and are not interchangeable numerical rows.",
        },
    }))
    print(f"[V2 main table] complete | output={output}", flush=True)


if __name__ == "__main__":
    main()
