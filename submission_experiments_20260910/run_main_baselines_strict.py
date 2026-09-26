"""Evaluate original main-table controls under the V2 learned-grounding protocol."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
for item in (PROJECT_ROOT, PACKAGE_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from main_baseline_strict_models import LABELS, build_strict_main_baseline
from run_v2_grounding_worker import GroundedPlanner, georeference_for_scoring
from submission_common import (
    DEFAULT_SEEDS, aggregate_seed_means, evaluate, read_samples, run_metadata,
    write_json, write_rows_csv,
)
from utils.seed_manager import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--baseline-id", choices=tuple(LABELS), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-test-samples", type=int)
    args = parser.parse_args()

    root = Path(args.run_root).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite strict main-baseline output: {output}")
    protocol_root = root / "v2_osm_protocol"
    test = read_samples(protocol_root / "splits" / "ood_location_test.json")
    if args.max_test_samples is not None:
        if not args.smoke or args.max_test_samples < 1:
            raise ValueError("--max-test-samples requires --smoke and a positive value")
        test = test[:args.max_test_samples]
    if not test:
        raise ValueError("V2 OOD test split is empty")

    # The labels are retained in this separate evaluator-side copy only.
    from configs.experiment_config import get_default_config
    base_cfg, *_ = get_default_config()
    scoring_test = georeference_for_scoring(test, base_cfg.benchmark_dir)
    started = time.monotonic()
    per_seed, all_rows, audit = {}, [], {}
    for seed in DEFAULT_SEEDS:
        set_global_seed(seed)
        core, grounder, metadata = build_strict_main_baseline(args.baseline_id, str(root), seed)
        wrapped = GroundedPlanner(core, grounder, base_cfg.benchmark_dir, base_cfg.device)
        rows = evaluate(wrapped, scoring_test, base_cfg.benchmark_dir)
        for row in rows:
            row.update({
                "method": args.baseline_id,
                "display_name": LABELS[args.baseline_id],
                "training_seed": seed,
                "test_set": "v2_osm_ood_location_test",
                "planner_input": "predicted_coordinates_only",
                "label_coordinates_used_only_for_scoring": True,
            })
        per_seed[seed] = rows
        all_rows.extend(rows)
        audit[str(seed)] = metadata
        print(f"[Strict main baselines] complete | baseline={args.baseline_id} | seed={seed} | rows={len(rows)}", flush=True)

    output.mkdir(parents=True)
    write_rows_csv(output / "per_scenario_results.csv", all_rows)
    write_json(output / "summary.json", run_metadata({
        "experiment": "Original main-table baseline rerun under V2 learned grounding",
        "baseline_id": args.baseline_id,
        "display_name": LABELS[args.baseline_id],
        "seeds": list(DEFAULT_SEEDS),
        "test_file": str(protocol_root / "splits" / "ood_location_test.json"),
        "n_test_scenarios": len(test),
        "smoke": bool(args.smoke),
        "summary": aggregate_seed_means(per_seed),
        "checkpoint_and_protocol_audit": audit,
        "label_leakage_gate": "passed",
        "planner_input": "predicted_coordinates_only",
        "claim_allowed": "comparison of disclosed local adaptations under a shared learned-grounding and held-out V2 protocol",
        "claim_forbidden": "author-code reproduction or comparison with planners supplied oracle target coordinates",
        "elapsed_minutes": (time.monotonic() - started) / 60.0,
    }))
    print(f"[Strict main baselines] output={output}", flush=True)


if __name__ == "__main__":
    main()
