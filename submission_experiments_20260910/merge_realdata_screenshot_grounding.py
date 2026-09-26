"""Fail-closed merger for the five-seed screenshot-grounding experiment."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

SEEDS = {42, 123, 456, 789, 2024}
CONDITIONS = {"text_only", "text_screenshot"}


def paired(a, b):
    delta = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    n = len(delta); mean = float(delta.mean()); sd = float(delta.std(ddof=1)) if n > 1 else 0.0
    t = mean / (sd / math.sqrt(n)) if sd else 0.0
    try:
        from scipy.stats import t as student_t
        p = float(2 * student_t.sf(abs(t), n - 1)) if sd else 1.0
    except Exception:
        p = None
    return {"n_training_seeds": n, "mean_delta": mean, "t": float(t), "p_two_sided": p,
            "cohens_dz": float(mean / sd) if sd else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers-root", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); workers, output = Path(args.workers_root), Path(args.output_dir)
    rows, test_ids, seed_means = [], None, defaultdict(lambda: defaultdict(list))
    seen_seeds = set()
    for folder in sorted(workers.glob("seed_*")):
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        seed = int(summary["seed"]); seen_seeds.add(seed)
        if summary.get("smoke") or summary.get("label_leakage_gate") != "passed":
            raise RuntimeError(f"Invalid worker {folder}: smoke/leakage gate")
        if summary.get("nonidentical_prediction_scenarios", 0) == 0:
            raise RuntimeError(f"Degenerate worker {folder}: identical predictions")
        worker_rows = json.loads((folder / "per_scenario_results.json").read_text(encoding="utf-8"))
        grouped = defaultdict(list)
        for row in worker_rows: grouped[row["condition"]].append(row)
        if set(grouped) != CONDITIONS:
            raise RuntimeError(f"{folder} conditions mismatch: {set(grouped)}")
        ids = {row["scenario_id"] for row in grouped["text_only"]}
        if ids != {row["scenario_id"] for row in grouped["text_screenshot"]}:
            raise RuntimeError(f"{folder} conditions have different test rows")
        if test_ids is None: test_ids = ids
        elif ids != test_ids: raise RuntimeError(f"{folder} test cohort differs across seeds")
        for condition, condition_rows in grouped.items():
            seed_means[condition][seed] = {
                "coordinate_mae_m": float(np.mean([r["coordinate_mae_m"] for r in condition_rows])),
                "coordinate_rmse_m": float(np.mean([r["coordinate_rmse_m"] for r in condition_rows])),
            }
        rows.extend(worker_rows)
    if seen_seeds != SEEDS:
        raise RuntimeError(f"Seeds={sorted(seen_seeds)}, expected={sorted(SEEDS)}")
    summary = {}
    for condition in sorted(CONDITIONS):
        summary[condition] = {}
        for metric in ("coordinate_mae_m", "coordinate_rmse_m"):
            values = [seed_means[condition][seed][metric] for seed in sorted(SEEDS)]
            summary[condition][metric] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "n_training_seeds": 5}
    tests = {metric: paired(
        [seed_means["text_only"][seed][metric] for seed in sorted(SEEDS)],
        [seed_means["text_screenshot"][seed][metric] for seed in sorted(SEEDS)],
    ) for metric in ("coordinate_mae_m", "coordinate_rmse_m")}
    output.mkdir(parents=True, exist_ok=False)
    merged = {"protocol": "real_trajectory_screenshot_coordinate_grounding_v1", "test_n": len(test_ids),
              "seeds": sorted(SEEDS), "summary": summary, "paired_seed_tests_text_vs_screenshot": tests,
              "rows": rows,
              "claim_allowed": "held-out trajectory-screenshot-assisted coordinate grounding",
              "claim_forbidden": "raw-camera perception, open-world VLM grounding, or end-to-end flight validation"}
    (output / "realdata_screenshot_grounding_merged.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REAL SCREENSHOT GROUNDING MERGE PASSED | test_n={len(test_ids)}", flush=True)


if __name__ == "__main__": main()
