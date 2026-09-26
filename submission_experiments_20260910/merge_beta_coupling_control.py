"""Merge the five independently evaluated availability-signal workers.

The training seed is the independent experimental unit.  Per-scenario rows
are retained only as an audit trail and are never treated as independent
replicates for inferential statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from beta_coupling_protocol import (
    AVAILABILITY_BINS,
    BETA_ARMS,
    PRIMARY_AVAILABILITY_CONDITIONS,
    PROTOCOL_VERSION,
    SEEDS,
)
from submission_common import (
    METRICS,
    aggregate_seed_means,
    mean_metrics,
    paired_seed_test,
    run_metadata,
    write_json,
    write_rows_csv,
)


def _read_workers(worker_root: Path) -> tuple[list[Dict[str, Any]], Dict[int, Mapping[str, Any]]]:
    rows: list[Dict[str, Any]] = []
    summaries: Dict[int, Mapping[str, Any]] = {}
    for seed in SEEDS:
        directory = worker_root / f"seed_{seed}"
        summary_path, csv_path = directory / "summary.json", directory / "per_scenario_results.csv"
        if not summary_path.is_file() or not csv_path.is_file():
            raise FileNotFoundError(f"Missing beta-control worker artifacts for seed {seed}: {directory}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(
                f"Worker {directory} uses protocol {summary.get('protocol_version')!r}; "
                f"expected {PROTOCOL_VERSION!r}"
            )
        if summary.get("smoke") is True or int(summary.get("training_seed", -1)) != seed:
            raise RuntimeError(f"Worker {directory} is a smoke result or has an incorrect seed")
        if "same planner inputs" not in summary.get("label_leakage_gate", ""):
            raise RuntimeError(f"Worker {directory} failed the matched-input protocol gate")
        if tuple(summary.get("beta_arms", ())) != BETA_ARMS:
            raise RuntimeError(f"Worker {directory} has an incomplete or changed beta-arm set")
        audit = summary.get("implementation_audit", {})
        if not (audit.get("fuser_beta_interceptor_required") is True
                and audit.get("initial_astar_radius_must_equal_beta_derived_radius") is True
                and audit.get("rng_reset_before_each_arm") is True):
            raise RuntimeError(f"Worker {directory} lacks the v3 fail-closed implementation audit")
        summaries[seed] = summary
        with csv_path.open(encoding="utf-8", newline="") as handle:
            current = list(csv.DictReader(handle))
        if not current:
            raise RuntimeError(f"Worker {directory} has an empty CSV")
        for row in current:
            for metric in METRICS + (
                "learned_beta", "beta_used", "n_modalities_delivered",
                "expected_initial_min_obstacle_radius_px", "min_obstacle_radius_px",
                "initial_min_obstacle_radius_px", "final_min_obstacle_radius_px",
            ):
                row[metric] = float(row[metric])
            if not np.isfinite(row["learned_beta"]):
                raise RuntimeError(f"Worker {directory} contains an unrecorded learned beta")
            count = int(row["n_modalities_delivered"])
            expected_radius = 25 if count == 1 else int(25 + 8 * min(row["beta_used"] + 0.3, 1.0))
            if (int(row["expected_initial_min_obstacle_radius_px"]) != expected_radius
                    or int(row["initial_min_obstacle_radius_px"]) != expected_radius):
                raise RuntimeError(f"Worker {directory} has a beta/radius audit mismatch")
            row["training_seed"] = int(row["training_seed"])
            row["astar_fallback_to_radius25"] = str(row["astar_fallback_to_radius25"]).lower() == "true"
            rows.append(row)
        random_active_radii = {
            int(row["initial_min_obstacle_radius_px"])
            for row in current
            if row["beta_arm"] == "random_hash" and int(row["n_modalities_delivered"]) >= 2
        }
        if len(random_active_radii) < 2:
            raise RuntimeError(f"Worker {directory} random arm did not vary actual beta-active radii")
    return rows, summaries


def _verify_m1_negative_control(rows: Sequence[Mapping[str, Any]]) -> None:
    groups: Dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["availability_condition"] == "M1":
            groups[(int(row["training_seed"]), str(row["scenario_id"]))].append(row)
    fields = METRICS + ("initial_min_obstacle_radius_px", "final_min_obstacle_radius_px")
    for key, group in groups.items():
        if len(group) != len(BETA_ARMS):
            raise RuntimeError(f"M1 negative-control arms are incomplete for {key}")
        for field in fields:
            values = [float(row[field]) for row in group]
            if max(values) - min(values) > 1e-12:
                raise RuntimeError(f"M1 negative-control output differs across arms for {key}: {field}")


def _group_rows(rows: Sequence[Mapping[str, Any]], expected_bins: Sequence[str]):
    groups: Dict[tuple[str, str], Dict[int, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    seen: set[tuple[str, str, int, str]] = set()
    for row in rows:
        key = (str(row["availability_condition"]), str(row["beta_arm"]))
        unique = (*key, int(row["training_seed"]), str(row["scenario_id"]))
        if unique in seen:
            raise RuntimeError(f"Duplicate beta-control measurement: {unique}")
        seen.add(unique)
        groups[key][int(row["training_seed"])].append(row)
    expected = {(condition, arm) for condition in expected_bins for arm in BETA_ARMS}
    if set(groups) != expected:
        raise RuntimeError(f"Unexpected beta-control groups: found {sorted(groups)}, expected {sorted(expected)}")
    for key, per_seed in groups.items():
        if set(per_seed) != set(SEEDS):
            raise RuntimeError(f"{key} has seeds {sorted(per_seed)}, expected {list(SEEDS)}")
        reference_ids: set[str] | None = None
        for seed, measurements in per_seed.items():
            identifiers = {str(row["scenario_id"]) for row in measurements}
            if len(identifiers) != len(measurements):
                raise RuntimeError(f"{key}, seed {seed} has duplicate scenario IDs")
            if reference_ids is None:
                reference_ids = identifiers
            elif identifiers != reference_ids:
                raise RuntimeError(f"{key} scenario panel differs across training seeds")
    # Pairing within an availability condition must use the same panel for each arm.
    for condition in expected_bins:
        panels = [{str(row["scenario_id"]) for row in groups[(condition, arm)][SEEDS[0]]}
                  for arm in BETA_ARMS]
        if any(panel != panels[0] for panel in panels[1:]):
            raise RuntimeError(f"{condition} has a different test panel across beta arms")
    return groups


def _diagnostic_summary(per_seed: Mapping[int, Sequence[Mapping[str, Any]]]) -> Dict[str, Dict[str, float]]:
    output: Dict[str, Dict[str, float]] = {}
    for field in ("beta_used", "learned_beta", "min_obstacle_radius_px", "initial_min_obstacle_radius_px"):
        values = np.asarray([
            float(np.mean([float(row[field]) for row in per_seed[seed]]))
            for seed in SEEDS
        ])
        output[field] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "n_independent_training_seeds": len(values),
        }
    fallbacks = np.asarray([
        sum(bool(row["astar_fallback_to_radius25"]) for row in per_seed[seed])
        for seed in SEEDS
    ], dtype=float)
    output["astar_fallback_rows"] = {
        "mean": float(fallbacks.mean()),
        "std": float(fallbacks.std(ddof=1)),
        "n_independent_training_seeds": len(fallbacks),
    }
    return output


def _bonferroni_tcr_tests(groups, primary_bins: Sequence[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Correct one contrast per present M2-M4 stratum and control arm."""
    family_size = len(primary_bins) * (len(BETA_ARMS) - 1)
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for condition in primary_bins:
        learned = groups[(condition, "learned")]
        results[condition] = {}
        for arm in BETA_ARMS:
            if arm == "learned":
                continue
            raw = paired_seed_test(learned, groups[(condition, arm)])["task_completion_rate"]
            p_value = float(raw["p_value"])
            results[condition][f"learned_minus_{arm}"] = {
                **raw,
                "test": "two-sided paired one-sample t-test of seed-level learned-minus-control TCR deltas",
                "bonferroni_family_size": family_size,
                "p_value_bonferroni": min(1.0, p_value * family_size) if math.isfinite(p_value) else float("nan"),
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge beta-coupling control workers")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    worker_root = root / "results" / "beta_coupling_workers"
    final = root / "results" / "beta_coupling_controls"
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite beta-control merged results: {final}")
    rows, worker_summaries = _read_workers(worker_root)
    _verify_m1_negative_control(rows)
    first_counts = worker_summaries[SEEDS[0]].get("availability_conditions", {})
    if set(first_counts) != set(AVAILABILITY_BINS):
        raise RuntimeError("Worker availability-count summary does not contain M1-M4")
    for seed, metadata in worker_summaries.items():
        if metadata.get("availability_conditions") != first_counts:
            raise RuntimeError(f"Seed {seed} has a different usable-modality distribution")
    present_bins = [name for name in AVAILABILITY_BINS if int(first_counts[name]) > 0]
    primary_bins = [name for name in PRIMARY_AVAILABILITY_CONDITIONS if name in present_bins]
    groups = _group_rows(rows, present_bins)
    final.mkdir(parents=True)
    write_rows_csv(final / "per_scenario_results.csv", rows)

    summary = {
        condition: {
            arm: {
                "metrics": aggregate_seed_means(groups[(condition, arm)]),
                "calibration_diagnostics": _diagnostic_summary(groups[(condition, arm)]),
            }
            for arm in BETA_ARMS
        }
        for condition in present_bins
    }
    write_json(final / "summary.json", run_metadata({
        "experiment": "Five-seed availability-signal planner-control intervention",
        "protocol_version": PROTOCOL_VERSION,
        "seeds": list(SEEDS),
        "source_worker_directory": str(worker_root),
        "worker_count": len(worker_summaries),
        "availability_conditions": first_counts,
        "primary_availability_conditions": primary_bins,
        "m1_role": "implementation negative control; beta does not change its fixed 25 px B1 safety-radius branch",
        "beta_arms": list(BETA_ARMS),
        "summary": summary,
        "primary_tcr_tests": _bonferroni_tcr_tests(groups, primary_bins),
        "secondary_metric_tests": "not family-corrected; retain only as descriptive audit outputs in per-scenario CSV",
        "independent_unit": "one previously trained full model per seed; scenarios are paired repeated measurements only",
        "claim_allowed": "evidence on the B1 planner-calibration path under the declared intervention panel",
        "claim_forbidden": "replacement for Exp1 or evidence for unimplemented replan-scope effects",
    }))
    print(f"Merged beta-coupling controls: {final}")


if __name__ == "__main__":
    main()
