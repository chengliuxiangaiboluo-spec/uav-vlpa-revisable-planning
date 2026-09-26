"""Exp4: strict unseen-scenario ID and unseen-location OOD evaluation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from experiment_models import build_evaluation_planner
from submission_common import (
    DEFAULT_SEEDS, aggregate_seed_means, assert_no_scenario_overlap, evaluate,
    image_ids, paired_seed_test, read_samples, run_metadata, write_json,
    write_rows_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp4: strict ID/OOD generalization")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--id-test-file", required=True)
    parser.add_argument("--ood-test-file", required=True)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--methods", default="ours,scal")
    parser.add_argument("--test-sets", default="id_unseen_scenarios,ood_unseen_locations")
    parser.add_argument("--output-dir", help="Worker-specific output directory; must not already exist")
    parser.add_argument("--max-samples", type=int,
                        help="Use only the first N samples from each selected test set; smoke test only.")
    parser.add_argument("--smoke", action="store_true",
                        help="Permit one seed with --max-samples for pre-submission runtime validation.")
    args = parser.parse_args()
    seeds = tuple(int(part) for part in args.seeds.split(","))
    if not args.smoke and (len(seeds) != 5 or len(set(seeds)) != 5):
        raise ValueError("Exactly five distinct seeds are required")
    if args.smoke and (len(seeds) != 1 or args.max_samples is None or args.max_samples < 2):
        raise ValueError("A smoke test requires exactly one seed and --max-samples >= 2")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")

    train, id_test, ood_test = (read_samples(args.train_file), read_samples(args.id_test_file),
                                read_samples(args.ood_test_file))
    assert_no_scenario_overlap({"train": train, "id_test": id_test, "ood_test": ood_test})
    if not image_ids(id_test) <= image_ids(train):
        raise ValueError("ID-test must use locations represented in training")
    if image_ids(ood_test) & image_ids(train):
        raise ValueError("OOD-test shares locations with training")
    if args.max_samples is not None:
        id_test, ood_test = id_test[:args.max_samples], ood_test[:args.max_samples]

    methods = tuple(part.strip() for part in args.methods.split(",") if part.strip())
    selected_sets = tuple(part.strip() for part in args.test_sets.split(",") if part.strip())
    if not methods or any(method not in {"ours", "scal"} for method in methods):
        raise ValueError("methods must be ours and/or scal")
    if not selected_sets or any(name not in {"id_unseen_scenarios", "ood_unseen_locations"} for name in selected_sets):
        raise ValueError("test-sets must be id_unseen_scenarios and/or ood_unseen_locations")
    output = Path(args.output_dir).resolve() if args.output_dir else Path(args.run_root).resolve() / "results" / "exp4_strict_generalization"
    output.mkdir(parents=True, exist_ok=False)
    started_at = time.monotonic()
    print(
        f"[Exp4] start | train={len(train)} | id_test={len(id_test)} | ood_test={len(ood_test)} | "
        f"methods={methods} | test_sets={selected_sets} | seeds={seeds}",
        flush=True,
    )
    test_sets = {"id_unseen_scenarios": id_test, "ood_unseen_locations": ood_test}
    per_method_set, all_rows, checkpoint_audit = {}, [], {}
    for method in methods:
        per_method_set[method] = {}
        checkpoint_audit[method] = {}
        for set_name in selected_sets:
            samples = test_sets[set_name]
            per_method_set[method][set_name] = {}
            for seed in seeds:
                print(f"[Exp4] evaluation started | method={method} | test_set={set_name} | seed={seed}", flush=True)
                planner, checkpoints = build_evaluation_planner(method, args.run_root, seed)
                checkpoint_audit[method][str(seed)] = checkpoints
                rows = evaluate(planner, samples, planner.benchmark_dir)
                for row in rows:
                    row.update({"method": method, "test_set": set_name, "training_seed": seed})
                per_method_set[method][set_name][seed] = rows
                all_rows.extend(rows)
                print(f"[Exp4] evaluation complete | method={method} | test_set={set_name} | seed={seed} | rows={len(rows)}", flush=True)

    summary = {method: {set_name: aggregate_seed_means(seed_rows)
                        for set_name, seed_rows in sets.items()}
               for method, sets in per_method_set.items()}
    tests = ({set_name: paired_seed_test(per_method_set["ours"][set_name], per_method_set["scal"][set_name])
              for set_name in selected_sets}
             if "ours" in per_method_set and "scal" in per_method_set else {})
    write_rows_csv(output / "per_scenario_results.csv", all_rows)
    write_json(output / "summary.json", run_metadata({
        "experiment": "Exp4 strict generalization",
        "seeds": list(seeds),
        "smoke_test": args.smoke,
        "max_samples": args.max_samples,
        "worker_methods": list(methods),
        "worker_test_sets": list(selected_sets),
        "sets": {
            "id_unseen_scenarios": {
                "n": len(id_test), "location_relation": "same train locations, disjoint scenario IDs",
            },
            "ood_unseen_locations": {
                "n": len(ood_test), "location_relation": "image_id disjoint from train",
            },
        },
        "checkpoint_audit": checkpoint_audit,
        "summary": summary,
        "paired_tests_ours_vs_scal": tests,
        "claim_allowed": "generalization to held-out synthetic scenarios and held-out synthetic satellite-map locations",
        "claim_forbidden": "real-world OOD or public-map OOD without separately collected data",
    }))
    print(f"[Exp4] complete | elapsed={(time.monotonic() - started_at) / 60.0:.1f} min | output={output}", flush=True)


if __name__ == "__main__":
    main()
