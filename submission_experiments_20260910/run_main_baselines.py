"""Evaluate one of the manuscript's original three main baselines over five seeds."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from main_baseline_models import build_main_baseline
from submission_common import DEFAULT_SEEDS, aggregate_seed_means, evaluate, read_samples, run_metadata, write_json, write_rows_csv


LABELS = {"uav_vlpa": "UAV-VLPA (text-only)", "lpanet": "LPANet", "citynav": "CityNav"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Five-seed evaluation of one main-table control")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--baseline-id", choices=tuple(LABELS), required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    samples = read_samples(args.test_file)
    if not samples:
        raise ValueError("Test set is empty")
    started = time.monotonic()
    per_seed, all_rows, audit = {}, [], {}
    for seed in DEFAULT_SEEDS:
        print(f"[Main baselines] evaluation started | baseline={args.baseline_id} | seed={seed}", flush=True)
        planner, metadata = build_main_baseline(args.baseline_id, args.run_root, seed)
        rows = evaluate(planner, samples, planner.benchmark_dir)
        for row in rows:
            row.update({"method": args.baseline_id, "display_name": LABELS[args.baseline_id], "training_seed": seed, "test_set": "ood_location_test"})
        per_seed[seed] = rows
        audit[str(seed)] = metadata
        all_rows.extend(rows)
        print(f"[Main baselines] evaluation complete | baseline={args.baseline_id} | seed={seed} | rows={len(rows)}", flush=True)
    write_rows_csv(output / "per_scenario_results.csv", all_rows)
    write_json(output / "summary.json", run_metadata({
        "experiment": "Main-table original baseline rerun",
        "baseline_id": args.baseline_id,
        "display_name": LABELS[args.baseline_id],
        "test_file": str(Path(args.test_file).resolve()),
        "seeds": list(DEFAULT_SEEDS),
        "independent_unit": "one independently initialized or fixed implementation run per declared seed",
        "checkpoint_and_protocol_audit": audit,
        "summary": aggregate_seed_means(per_seed),
        "elapsed_minutes": (time.monotonic() - started) / 60.0,
    }))
    print(f"[Main baselines] complete | baseline={args.baseline_id} | output={output}", flush=True)


if __name__ == "__main__":
    main()
