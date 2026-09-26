"""Exp1: independently trained, implementation-defined fair baselines."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from experiment_models import build_evaluation_planner
from submission_common import (
    DEFAULT_SEEDS, aggregate_seed_means, evaluate, paired_seed_test, read_samples,
    run_metadata, write_json, write_rows_csv,
)


def parse_seeds(value: str):
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("A submission run requires exactly five distinct training seeds")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp1: strict fair baseline comparison")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--test-file", required=True, help="Only the untouched OOD location test file")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--methods", default="ours,affnet,maftnet,scal")
    parser.add_argument("--output-dir", help="Worker-specific output directory; must not already exist")
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    test = read_samples(args.test_file)
    if not test:
        raise ValueError("Test set is empty")
    methods = tuple(part.strip() for part in args.methods.split(",") if part.strip())
    if not methods or any(method not in {"ours", "affnet", "maftnet", "scal"} for method in methods):
        raise ValueError("methods must be drawn from ours,affnet,maftnet,scal")
    output = Path(args.output_dir).resolve() if args.output_dir else Path(args.run_root).resolve() / "results" / "exp1_fair_baselines"
    output.mkdir(parents=True, exist_ok=False)
    started_at = time.monotonic()
    print(f"[Exp1] start | test_samples={len(test)} | methods={methods} | seeds={seeds}", flush=True)

    per_method = {}
    checkpoint_audit = {}
    all_rows = []
    for method in methods:
        per_method[method] = {}
        checkpoint_audit[method] = {}
        for seed in seeds:
            print(f"[Exp1] evaluation started | method={method} | seed={seed}", flush=True)
            planner, checkpoints = build_evaluation_planner(method, args.run_root, seed)
            rows = evaluate(planner, test, planner.benchmark_dir)
            for row in rows:
                row.update({"method": method, "training_seed": seed, "test_set": "ood_location_test"})
            per_method[method][seed] = rows
            checkpoint_audit[method][str(seed)] = checkpoints
            all_rows.extend(rows)
            print(f"[Exp1] evaluation complete | method={method} | seed={seed} | rows={len(rows)}", flush=True)

    summary = {method: aggregate_seed_means(seed_rows) for method, seed_rows in per_method.items()}
    tests = ({method: paired_seed_test(per_method["ours"], data)
              for method, data in per_method.items() if method != "ours"}
             if "ours" in per_method else {})
    write_rows_csv(output / "per_scenario_results.csv", all_rows)
    write_json(output / "summary.json", run_metadata({
        "experiment": "Exp1 fair baseline comparison",
        "test_file": str(Path(args.test_file).resolve()),
        "seeds": list(seeds),
        "worker_methods": list(methods),
        "independent_unit": "one separately trained model per seed; test scenarios are retained as raw measurements only",
        "methods": {
            "ours": "UAV-VLPA implementation",
            "affnet": "local implementation-inspired fusion control; do not claim a reproduced external method",
            "maftnet": "local implementation-inspired fusion control; do not claim a reproduced external method",
            "scal": "local implementation-inspired fusion control; do not claim a reproduced external method",
        },
        "checkpoint_audit": checkpoint_audit,
        "summary": summary,
        "paired_tests_vs_ours": tests,
    }))
    print(f"[Exp1] complete | elapsed={(time.monotonic() - started_at) / 60.0:.1f} min | output={output}", flush=True)


if __name__ == "__main__":
    main()
