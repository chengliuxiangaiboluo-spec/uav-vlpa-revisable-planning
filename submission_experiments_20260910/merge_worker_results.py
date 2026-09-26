"""Merge disjoint SLURM worker outputs and rebuild final seed-level statistics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from submission_common import (
    DEFAULT_SEEDS, METRICS, aggregate_seed_means, paired_seed_test,
    run_metadata, write_json, write_rows_csv,
)


def read_worker_rows(workers: Path):
    rows = []
    files = sorted(workers.glob("*/per_scenario_results.csv"))
    if not files:
        raise FileNotFoundError(f"No worker CSV files under {workers}")
    for path in files:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for metric in METRICS:
                    if metric in row:
                        row[metric] = float(row[metric])
                row["_worker"] = path.parent.name
                rows.append(row)
    return rows


def group_rows(rows, keys):
    grouped = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in rows:
        key = tuple(row[k] for k in keys)
        seed = int(row["training_seed"])
        unique = (key, seed, row["scenario_id"])
        if unique in seen:
            raise ValueError(f"Duplicate worker measurement: {unique}")
        seen.add(unique)
        grouped[key][seed].append(row)
    return grouped


def ensure_five_seeds(grouped):
    expected = set(DEFAULT_SEEDS)
    for key, values in grouped.items():
        found = set(values)
        if found != expected:
            raise ValueError(f"Group {key} has seeds {sorted(found)}, expected {sorted(expected)}")


def exp3_worker_protocol(workers: Path):
    """Recover valid within-subset Exp3 comparisons from worker metadata.

    File-backed corruptions are evaluated only on scenarios where the respective
    modality exists.  Their clean references are intentionally not duplicated
    in the worker CSV, so rebuilding a test against the all-scenario `clean`
    rows would be statistically invalid.
    """
    tests, references, coverage = {}, {}, {}
    for path in sorted(workers.glob("*/summary.json")):
        with path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        for condition, value in metadata.get("paired_tests_vs_clean", {}).items():
            if condition in tests:
                raise ValueError(f"duplicate Exp3 paired test for {condition}")
            tests[condition] = value
        for condition, value in metadata.get("paired_clean_reference_summary", {}).items():
            if condition in references:
                raise ValueError(f"duplicate Exp3 clean reference for {condition}")
            references[condition] = value
        for condition, value in metadata.get("perturbation_audit", {}).items():
            if condition != "clean":
                if condition in coverage:
                    raise ValueError(f"duplicate Exp3 coverage audit for {condition}")
                coverage[condition] = value
    expected = {"voice_snr_10db", "gesture_jitter_6px", "annotation_offset_10px", "text_ambiguity"}
    if set(tests) != expected or set(references) != expected or set(coverage) != expected:
        raise ValueError("Exp3 workers are missing a modality-specific paired reference, test, or coverage audit")
    return tests, references, coverage


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge submission experiment workers")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--experiment", choices=("exp1", "exp2", "exp3", "exp4", "exp5", "exp6"), required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    specs = {
        "exp1": ("exp1_fair_baselines", "exp1_workers", ("method",)),
        "exp2": ("exp2_input_interventions", "exp2_workers_iafix", ("condition",)),
        "exp3": ("exp3_robustness", "exp3_workers", ("condition",)),
        "exp4": ("exp4_strict_generalization", "exp4_workers", ("method", "test_set")),
        "exp5": ("exp5_architecture_ablations", "exp5_workers", ("variant",)),
        "exp6": ("exp6_paper_inspired_baselines", "exp6_paper_workers", ("method",)),
    }
    final_name, worker_name, keys = specs[args.experiment]
    rows = read_worker_rows(root / "results" / worker_name)
    grouped = group_rows(rows, keys)
    ensure_five_seeds(grouped)
    final = root / "results" / final_name
    final.mkdir(parents=True, exist_ok=False)
    write_rows_csv(final / "per_scenario_results.csv", rows)

    if args.experiment == "exp1":
        summary = {key[0]: aggregate_seed_means(values) for key, values in grouped.items()}
        if set(summary) != {"ours", "affnet", "maftnet", "scal"}:
            raise ValueError("Exp1 workers do not cover all four planned methods")
        tests = {name: paired_seed_test(grouped[("ours",)], values)
                 for (name,), values in grouped.items() if name != "ours"}
    elif args.experiment == "exp2":
        summary = {key[0]: aggregate_seed_means(values) for key, values in grouped.items()}
        expected = {"full_available", "text_only", "minus_voice", "minus_gesture", "minus_annotation"}
        if set(summary) != expected:
            raise ValueError(f"{args.experiment} workers do not cover every planned condition")
        tests = {name: paired_seed_test(grouped[("full_available",)], values)
                 for (name,), values in grouped.items() if name != "full_available"}
    elif args.experiment == "exp3":
        summary = {key[0]: aggregate_seed_means(values) for key, values in grouped.items()}
        expected = {"clean", "voice_snr_10db", "gesture_jitter_6px", "annotation_offset_10px", "text_ambiguity"}
        if set(summary) != expected:
            raise ValueError("Exp3 workers do not cover every planned condition")
        tests, exp3_references, exp3_coverage = exp3_worker_protocol(root / "results" / worker_name)
    elif args.experiment == "exp4":
        summary = {method: {test_set: aggregate_seed_means(values)
                            for (m, test_set), values in grouped.items() if m == method}
                   for method in ("ours", "scal")}
        if set(grouped) != {("ours", "id_unseen_scenarios"), ("ours", "ood_unseen_locations"),
                            ("scal", "id_unseen_scenarios"), ("scal", "ood_unseen_locations")}:
            raise ValueError("Exp4 workers do not cover both methods on both strict test sets")
        tests = {test_set: paired_seed_test(grouped[("ours", test_set)], grouped[("scal", test_set)])
                 for test_set in ("id_unseen_scenarios", "ood_unseen_locations")}
    elif args.experiment == "exp5":
        summary = {key[0]: aggregate_seed_means(values) for key, values in grouped.items()}
        expected = {
            "minus_cross_modal_attention", "minus_data_driven_calibration",
            "minus_task_decomposition", "minus_semantic_constraint",
        }
        if set(summary) != expected:
            raise ValueError("Exp5 workers do not cover every valid architecture-ablation variant")
        # The full-model reference is Exp1/ours, which is deliberately kept
        # separate so this merger never duplicates the full-model measurements.
        tests = {}
    else:
        summary = {key[0]: aggregate_seed_means(values) for key, values in grouped.items()}
        expected = {"uav_vla", "uav_vln", "aerialvln", "affnet", "maftnet", "scal",
                    "sipsa", "unigoal", "codeagents", "ikap", "csglso"}
        if set(summary) != expected:
            raise ValueError("Exp6 workers do not cover all eleven disclosed paper-inspired baselines")
        tests = {}

    metadata = {
        "protocol_version": "submission-2026-09-10",
        "experiment": args.experiment,
        "seeds": list(DEFAULT_SEEDS),
        "worker_count": len({row["_worker"] for row in rows}),
        "summary": summary,
        "paired_tests": tests,
        "source_worker_directory": str(root / "results" / worker_name),
    }
    if args.experiment == "exp3":
        metadata.update({
            "paired_clean_reference_summary": exp3_references,
            "perturbation_coverage": exp3_coverage,
            "comparison_note": "Each corruption is paired with clean execution on its own modality-available scenario subset; do not compare its raw aggregate directly with the all-scenario clean aggregate.",
        })
    write_json(final / "summary.json", run_metadata(metadata))
    print(f"Merged {args.experiment}: {final}")


if __name__ == "__main__":
    main()
