"""Assemble the intended V2 main table: Molmo UAV-VLPA*, LPANet, CityNav, Ours.

The first row is a frozen deterministic zero-shot reference.  It is retained
as a 500-row audit trail but is explicitly excluded from five-seed statistics.
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


def _require_v2_protocol(root: Path) -> None:
    audit = _read_json(root / "v2_osm_protocol" / "audit.json")
    if audit.get("status") != "COMPLETE" or audit.get("random_coordinate_fallbacks") != 0:
        raise RuntimeError("V2 protocol audit is incomplete or contains random coordinate fallbacks")


def _molmo_method(result_dir: Path):
    summary = _read_json(result_dir / "summary.json")
    rows = _read_rows(result_dir / "per_scenario_results.csv")
    required = {
        "smoke": False,
        "n_test_scenarios": 500,
        "evaluation_type": "deterministic frozen zero-shot inference",
        "training_seed": "frozen_zero_shot",
        "prompt_version": "uav-vlpa-molmo-targetwise-point-v4",
        "parser_valid_rows": 500,
        "parser_exact_match_rows": 500,
        "parser_failure_rows": 0,
        "coordinate_fallback_rows": 0,
        "label_leakage_gate": "passed",
        "planner_input": "predicted_coordinates_only",
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise RuntimeError(f"Molmo UAV-VLPA*: expected {key}={expected!r}, got {summary.get(key)!r}")
    if len(rows) != 500:
        raise RuntimeError(f"Molmo UAV-VLPA*: expected 500 rows, got {len(rows)}")
    if any(
        row.get("planner_input") != "predicted_coordinates_only"
        or row.get("label_coordinates_used_only_for_scoring", "").lower() != "true"
        for row in rows
    ):
        raise RuntimeError("Molmo UAV-VLPA*: per-row label/planner audit failed")
    for row in rows:
        row.update({
            "method": "uav_vlpa_molmo_star",
            "display_name": "UAV-VLPA* (frozen zero-shot, target-wise Molmo adaptation)",
            "evaluation_unit": "single deterministic frozen zero-shot run",
        })
    metric_summary = {
        metric: {"mean": value, "std": None, "n_independent_training_seeds": 0}
        for metric, value in summary["metrics"].items()
    }
    return rows, {
        "display_name": "UAV-VLPA* (frozen zero-shot, target-wise Molmo adaptation)",
        "summary": metric_summary,
        "source": str(result_dir),
        "evaluation_unit": "single deterministic frozen zero-shot run",
        "n_test_scenarios": 500,
        "statistical_test_eligibility": "excluded from seed-level tests",
        "planner_input": "predicted_coordinates_only",
        "label_coordinates_used_only_for_scoring": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--molmo-result-dir", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    molmo_dir = Path(args.molmo_result_dir).resolve()
    output = root / "results" / "main_table_v2_molmo_strict"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Molmo V2 main table: {output}")
    _require_v2_protocol(root)

    baseline_root = root / "results" / "main_baseline_molmo_strict_comparison"
    baseline_summary = _read_json(baseline_root / "summary.json")
    baseline_rows = _read_rows(baseline_root / "per_scenario_results.csv")
    expected = len(DEFAULT_SEEDS) * 500
    methods, rows = {}, []
    for method in ("lpanet", "citynav"):
        selected = [row for row in baseline_rows if row.get("method") == method]
        if len(selected) != expected:
            raise RuntimeError(f"{method}: expected {expected} rows, got {len(selected)}")
        methods[method] = baseline_summary["methods"][method]
        rows.extend(selected)

    v2_root = root / "results" / "v2_osm_grounding_replan"
    v2_summary = _read_json(v2_root / "summary.json")
    if tuple(v2_summary.get("seeds", ())) != DEFAULT_SEEDS:
        raise RuntimeError("V2 Ours summary does not contain exactly the declared five seeds")
    visual_summary = v2_summary.get("planning_summary", {}).get("visual_language_grounding")
    visual_rows = [row for row in _read_rows(v2_root / "grounded_planning_results.csv") if row.get("condition") == "visual_language_grounding"]
    if not visual_summary or len(visual_rows) != expected:
        raise RuntimeError("V2 Ours visual-language result is incomplete")
    for row in visual_rows:
        row.update({
            "method": "ours_v2",
            "display_name": "Ours (visual-language grounding + enhanced planner)",
            "evaluation_unit": "five independent learned-grounder seeds",
        })
    methods["ours_v2"] = {
        "display_name": "Ours (visual-language grounding + enhanced planner)",
        "summary": visual_summary,
        "source": "results/v2_osm_grounding_replan/grounded_planning_results.csv",
        "evaluation_unit": "five independent learned-grounder seeds",
        "planner_input": "predicted_coordinates_only",
        "label_coordinates_used_only_for_scoring": True,
    }
    rows.extend(visual_rows)

    molmo_rows, molmo_metadata = _molmo_method(molmo_dir)
    methods["uav_vlpa_molmo_star"] = molmo_metadata
    rows.extend(molmo_rows)

    output.mkdir(parents=True)
    write_rows_csv(output / "per_scenario_results.csv", rows)
    write_json(output / "summary.json", run_metadata({
        "experiment": "V2 main comparison: frozen Molmo UAV-VLPA*, LPANet, CityNav, and Ours",
        "methods": methods,
        "five_seed_methods": ["lpanet", "citynav", "ours_v2"],
        "single_run_reference": "uav_vlpa_molmo_star",
        "test_set": "v2_osm_ood_location_test",
        "label_leakage_gate": "passed",
        "planner_input": "predicted_coordinates_only",
        "protocol_boundary": "LPANet, CityNav and Ours are five-seed learned-grounding results. UAV-VLPA* is an audited deterministic frozen zero-shot Molmo reference on the same 500 held-out scenarios and is excluded from seed-level statistical tests.",
    }))
    print(f"[Molmo V2 main table] output={output}", flush=True)


if __name__ == "__main__":
    main()
