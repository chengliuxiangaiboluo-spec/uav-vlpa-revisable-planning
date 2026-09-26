"""Merge the four strict real-flight validation shards without altering them.

The merge fails closed if the four workers used different strict cohorts, if a
worker reports a failed plan, or if an enhanced condition lacks an input audit.
The primary multimodal contrasts are paired against text_only_enhanced, not the
separate system-level baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

PART_COMBOS = {
    "part1": {"text_only_baseline", "text_only_enhanced"},
    "part2": {"text_voice", "text_gesture"},
    "part3": {"text_annotation"},
    "part4": {"full_modal"},
}
ENHANCED = {"text_only_enhanced", "text_voice", "text_gesture", "text_annotation", "full_modal"}
PRIMARY_METRICS = ("rmse_m", "dtw_m", "sequential_tcr", "reachability_tcr")


def _load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _paired_test(a, b):
    """Two-sided paired t test and paired standardized mean difference."""
    delta = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    n = len(delta)
    mean = float(delta.mean())
    sd = float(delta.std(ddof=1)) if n > 1 else 0.0
    dz = mean / sd if sd > 0 else 0.0
    if n < 2 or sd == 0:
        return {"n": n, "mean_delta": mean, "t": 0.0, "p_two_sided": 1.0, "cohens_dz": dz}
    t = mean / (sd / math.sqrt(n))
    try:
        from scipy.stats import t as student_t
        p = float(2.0 * student_t.sf(abs(t), n - 1))
    except Exception:
        p = None
    return {"n": n, "mean_delta": mean, "t": float(t), "p_two_sided": p, "cohens_dz": float(dz)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    workers = Path(args.workers_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    manifests, results = {}, {}
    reference_ids = None
    combined = {}
    for part, expected in PART_COMBOS.items():
        part_dir = workers / part
        result = _load_json(part_dir / "realdata_validation.json")
        manifest = _load_json(part_dir / "strict_cohort_manifest.json")
        if set(result.get("modality_combos", [])) != expected:
            raise RuntimeError(f"{part} modality mismatch: {result.get('modality_combos')}")
        included = sorted(item["scenario_id"] for item in manifest.get("records", []) if item.get("included"))
        if not included:
            raise RuntimeError(f"{part} has an empty strict cohort")
        if reference_ids is None:
            reference_ids = included
        elif included != reference_ids:
            raise RuntimeError(f"{part} strict cohort differs from part1")
        records = {item["scenario_id"]: item for item in result.get("per_scenario_results", [])}
        if sorted(records) != reference_ids:
            raise RuntimeError(f"{part} result IDs differ from strict cohort")
        for scenario_id, record in records.items():
            combined.setdefault(scenario_id, {
                "scenario_id": scenario_id,
                "complexity": record.get("complexity"),
                "n_targets": record.get("n_targets"),
                "n_obstacles": record.get("n_obstacles"),
                "actual_length_km": record.get("actual_length_km"),
                "home_pct": record.get("home_pct"),
                "modalities": {},
            })["modalities"].update(record.get("modalities", {}))
        manifests[part] = manifest
        results[part] = result

    ordered = [combined[key] for key in reference_ids]
    expected_combos = set().union(*PART_COMBOS.values())
    for record in ordered:
        if set(record["modalities"]) != expected_combos:
            raise RuntimeError(f"Incomplete modality rows for {record['scenario_id']}")
        for combo, row in record["modalities"].items():
            if row.get("planning_failed"):
                raise RuntimeError(f"Planning failed: {record['scenario_id']} / {combo}: {row.get('failure_reason')}")
            if combo in ENHANCED:
                audit = row.get("input_audit") or {}
                if set(audit.get("requested_modalities", [])) != set(audit.get("loaded_modalities", [])) or audit.get("missing_modalities"):
                    raise RuntimeError(f"Input audit failed: {record['scenario_id']} / {combo}: {audit}")

    global_means, global_stds = {}, {}
    for combo in sorted(expected_combos):
        global_means[combo], global_stds[combo] = {}, {}
        for metric in PRIMARY_METRICS + ("task_completion_rate", "approach_score", "target_proximity_score"):
            values = np.asarray([row["modalities"][combo][metric] for row in ordered], dtype=float)
            global_means[combo][metric] = float(values.mean())
            global_stds[combo][metric] = float(values.std(ddof=1)) if len(values) > 1 else 0.0

    paired = {}
    control = "text_only_enhanced"
    for combo in ("text_voice", "text_gesture", "text_annotation", "full_modal"):
        paired[combo] = {}
        for metric in PRIMARY_METRICS:
            control_values = [row["modalities"][control][metric] for row in ordered]
            treatment_values = [row["modalities"][combo][metric] for row in ordered]
            paired[combo][metric] = _paired_test(control_values, treatment_values)

    merged = {
        "protocol": "strict_real_flight_waypoint_conditioned_v1",
        "n_scenarios": len(ordered),
        "modality_combos": sorted(expected_combos),
        "global_means": global_means,
        "global_stds": global_stds,
        "paired_input_effect_tests": paired,
        "per_scenario_results": ordered,
        "cohort_manifest": manifests["part1"],
        "claim_allowed": "real-flight-trajectory comparison under a disclosed waypoint-conditioned multimodal input protocol",
        "claim_forbidden": "end-to-end visual target/obstacle grounding from raw screenshots",
        "legacy_results_replaced": False,
    }
    (output / "realdata_strict_merged.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "strict_paired_tests.json").write_text(json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "realdata_strict_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["modality_combo", "n", "rmse_m_mean", "rmse_m_std", "dtw_m_mean", "sequential_tcr_mean", "reachability_tcr_mean"])
        for combo in sorted(expected_combos):
            writer.writerow([combo, len(ordered), global_means[combo]["rmse_m"], global_stds[combo]["rmse_m"], global_means[combo]["dtw_m"], global_means[combo]["sequential_tcr"], global_means[combo]["reachability_tcr"]])
    print(f"STRICT REALDATA MERGE PASSED | n={len(ordered)} | output={output}", flush=True)


if __name__ == "__main__":
    main()
