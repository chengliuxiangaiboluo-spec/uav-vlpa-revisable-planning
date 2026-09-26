"""Validate and merge the two formal V2 grounding workers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
for search_path in (PROJECT_ROOT, PACKAGE_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import numpy as np

from submission_common import (
    DEFAULT_SEEDS, METRICS, aggregate_seed_means, paired_seed_test,
    run_metadata, write_json, write_rows_csv,
)


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty result: {path}")
    numeric = METRICS + (
        "mean_target_coordinate_error_pct", "mean_target_coordinate_error_m",
        "target_precision_25m", "target_precision_50m", "n_targets",
        "evaluable_event",
        "event_success", "post_event_collision", "minimum_clearance_pct",
        "response_latency_ms", "trigger_fraction", "obstacle_fraction",
        "post_event_path_length_pct", "path_length_ratio_vs_retained",
        "post_event_route_nontrivial", "post_event_route_point_count",
    )
    for row in rows:
        row["training_seed"] = int(row["training_seed"])
        for key in numeric:
            if key in row and row[key] != "":
                row[key] = float(row[key])
    return rows


def _by_seed(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["training_seed"])].append(row)
    return dict(grouped)


def _paired_event_test(rows, metric):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = float(row.get(metric, float("nan")))
        if np.isfinite(value) and float(row.get("evaluable_event", 1)) == 1:
            grouped[int(row["training_seed"])][row["condition"]].append(value)
    left, right = [], []
    for seed in sorted(grouped):
        with_values = grouped[seed]["visual_language_with_replan"]
        without_values = grouped[seed]["visual_language_without_replan"]
        if with_values and without_values:
            left.append(float(np.mean(with_values)))
            right.append(float(np.mean(without_values)))
    delta = np.asarray(left) - np.asarray(right)
    try:
        from scipy.stats import ttest_1samp
        p_value = float(ttest_1samp(delta, 0.0).pvalue) if len(delta) >= 3 else float("nan")
    except Exception:
        p_value = float("nan")
    return {
        "n_independent_training_seeds": len(delta),
        "with_replan_mean": float(np.mean(left)) if left else float("nan"),
        "without_replan_mean": float(np.mean(right)) if right else float("nan"),
        "mean_delta": float(np.mean(delta)) if len(delta) else float("nan"),
        "p_value": p_value,
    }


def _paired_grounding_test(rows, metric):
    grouped = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        value = float(row.get(metric, float("nan")))
        if np.isfinite(value):
            grouped[int(row["training_seed"])][row["condition"]][
                row["scenario_id"]
            ] = value
    visual_seed_means, text_seed_means = [], []
    paired_scenarios = 0
    for seed in sorted(grouped):
        visual = grouped[seed]["visual_language_grounding"]
        text = grouped[seed]["text_only_grounding"]
        common = sorted(set(visual) & set(text))
        if common:
            visual_seed_means.append(float(np.mean([visual[key] for key in common])))
            text_seed_means.append(float(np.mean([text[key] for key in common])))
            paired_scenarios += len(common)
    delta = np.asarray(visual_seed_means) - np.asarray(text_seed_means)
    try:
        from scipy.stats import ttest_1samp
        p_value = float(ttest_1samp(delta, 0.0).pvalue) if len(delta) >= 3 else float("nan")
    except Exception:
        p_value = float("nan")
    return {
        "n_independent_training_seeds": len(delta),
        "n_paired_scenario_seed_rows": paired_scenarios,
        "visual_mean": float(np.mean(visual_seed_means)) if visual_seed_means else float("nan"),
        "text_only_mean": float(np.mean(text_seed_means)) if text_seed_means else float("nan"),
        "visual_minus_text_mean_delta": float(np.mean(delta)) if len(delta) else float("nan"),
        "p_value": p_value,
    }


def _materialise_seed_shards(root, worker_root):
    """Combine isolated one-seed V2 jobs without allowing concurrent writes."""
    shard_root = root / "v2_worker_shards"
    for condition in ("visual_language_grounding", "text_only_grounding"):
        destination = worker_root / condition
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite consolidated V2 worker output: {destination}"
            )
        coordinate_rows, planning_rows, event_rows, shard_summaries = [], [], [], {}
        for seed in DEFAULT_SEEDS:
            source = (
                shard_root / condition / f"seed_{seed}" / "results" /
                "v2_osm_grounding_workers" / condition
            )
            summary_path = source / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"Missing V2 shard summary: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("smoke") or tuple(summary.get("seeds", ())) != (seed,):
                raise ValueError(f"Invalid formal V2 shard: {summary_path}")
            shard_summaries[str(seed)] = summary
            coordinate_rows.extend(_read_csv(source / "grounding_coordinate_results.csv"))
            planning_rows.extend(_read_csv(source / "grounded_planning_results.csv"))
            event_path = source / "replan_event_results.csv"
            if condition == "visual_language_grounding":
                event_rows.extend(_read_csv(event_path))
            elif event_path.exists():
                raise ValueError(f"Text-only V2 shard must not emit replan events: {event_path}")
        destination.mkdir(parents=True)
        write_rows_csv(destination / "grounding_coordinate_results.csv", coordinate_rows)
        write_rows_csv(destination / "grounded_planning_results.csv", planning_rows)
        if event_rows:
            write_rows_csv(destination / "replan_event_results.csv", event_rows)
        write_json(destination / "summary.json", run_metadata({
            "experiment": "V2 consolidated independent seed shards",
            "condition": condition,
            "seeds": list(DEFAULT_SEEDS),
            "smoke": False,
            "shard_summaries": shard_summaries,
            "event_success_definition": (
                "a non-degenerate collision-free route with clearance at least "
                "the event radius; route feasibility, not task-completion rate"
                if event_rows else None
            ),
        }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--from-seed-shards", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    worker_root = root / "results" / "v2_osm_grounding_workers"
    final = root / "results" / "v2_osm_grounding_replan"
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite {final}")
    if args.from_seed_shards:
        _materialise_seed_shards(root, worker_root)

    summaries, coordinate_rows, planning_rows, planning = {}, [], [], {}
    for condition in ("visual_language_grounding", "text_only_grounding"):
        worker = worker_root / condition
        summary = json.loads((worker / "summary.json").read_text(encoding="utf-8"))
        seeds = tuple(sorted(int(seed) for seed in summary["seeds"]))
        if seeds != tuple(sorted(DEFAULT_SEEDS)) or summary.get("smoke"):
            raise ValueError(f"Incomplete/formal check failed for {condition}: {seeds}")
        summaries[condition] = summary
        condition_coordinates = _read_csv(worker / "grounding_coordinate_results.csv")
        condition_planning = _read_csv(worker / "grounded_planning_results.csv")
        coordinate_rows.extend(condition_coordinates)
        planning_rows.extend(condition_planning)
        planning[condition] = _by_seed(condition_planning)

    event_rows = _read_csv(
        worker_root / "visual_language_grounding" / "replan_event_results.csv"
    )
    coordinate_pairs = defaultdict(set)
    for row in coordinate_rows:
        coordinate_pairs[(row["training_seed"], row["scenario_id"])].add(
            row["condition"]
        )
    expected_grounding_conditions = {
        "visual_language_grounding", "text_only_grounding",
    }
    if not coordinate_pairs or any(
        value != expected_grounding_conditions
        for value in coordinate_pairs.values()
    ):
        raise ValueError(
            "Grounding coordinate results are not paired for every seed/scenario"
        )
    if set(seed for seed, _ in coordinate_pairs) != set(DEFAULT_SEEDS):
        raise ValueError("Grounding results do not contain all five seeds")
    paired = defaultdict(set)
    for row in event_rows:
        paired[(row["training_seed"], row["scenario_id"])].add(row["condition"])
    expected_conditions = {
        "visual_language_without_replan", "visual_language_with_replan",
    }
    if not paired or any(value != expected_conditions for value in paired.values()):
        raise ValueError("Replan result is not paired for every seed/scenario")
    if set(seed for seed, _ in paired) != set(DEFAULT_SEEDS):
        raise ValueError("Replan result does not contain all five seeds")

    final.mkdir(parents=True)
    write_rows_csv(final / "grounding_coordinate_results.csv", coordinate_rows)
    write_rows_csv(final / "grounded_planning_results.csv", planning_rows)
    write_rows_csv(final / "replan_event_results.csv", event_rows)
    coordinate_summary = {}
    for condition in planning:
        selected = [row for row in coordinate_rows if row["condition"] == condition]
        coordinate_summary[condition] = {metric: float(np.nanmean([
            row[metric] for row in selected
        ])) for metric in (
            "mean_target_coordinate_error_pct", "mean_target_coordinate_error_m",
            "target_precision_25m", "target_precision_50m",
        )}
        coordinate_summary[condition]["n_rows"] = len(selected)
    write_json(final / "summary.json", run_metadata({
        "protocol": "V2 OSM-assisted semantic-map grounding and paired simulated map-event replanning",
        "seeds": list(DEFAULT_SEEDS),
        "coordinate_summary": coordinate_summary,
        "grounding_paired_tests": {
            metric: _paired_grounding_test(coordinate_rows, metric)
            for metric in (
                "mean_target_coordinate_error_pct",
                "mean_target_coordinate_error_m",
                "target_precision_25m", "target_precision_50m",
            )
        },
        "planning_summary": {
            condition: aggregate_seed_means(rows)
            for condition, rows in planning.items()
        },
        "visual_vs_text_paired_tests": paired_seed_test(
            planning["visual_language_grounding"],
            planning["text_only_grounding"],
        ),
        "replan_paired_tests": {
            metric: _paired_event_test(event_rows, metric)
            for metric in (
                "event_success", "post_event_collision", "minimum_clearance_pct",
                "post_event_path_length_pct", "path_length_ratio_vs_retained",
            )
        },
        "worker_summaries": summaries,
        "claim_allowed": "simulated event-driven map-replanning stress test",
        "event_success_definition": (
            "a non-degenerate (at least two points and non-zero length) "
            "post-event route whose minimum clearance meets the event radius; "
            "this is route feasibility, not task-completion rate"
        ),
        "claim_forbidden": "real-flight dynamic-event validation",
    }))
    print(f"V2 MERGE PASSED | output={final}", flush=True)


if __name__ == "__main__":
    main()
