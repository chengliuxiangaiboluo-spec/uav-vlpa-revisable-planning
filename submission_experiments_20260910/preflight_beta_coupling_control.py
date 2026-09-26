"""Protocol gates for the availability-signal intervention experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from beta_coupling_protocol import AVAILABILITY_BINS, BETA_ARMS, PROTOCOL_VERSION, SEEDS

METRIC_COLUMNS = (
    "task_completion_rate", "instruction_accuracy", "dtw_rmse", "knn_rmse",
    "sequential_rmse", "trajectory_length_km", "efficiency_ratio",
)


def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_model_artifacts(root: Path, seeds: tuple[int, ...]) -> None:
    missing: list[str] = []
    for seed in seeds:
        directory = root / "models" / "ours" / f"seed_{seed}"
        for stem in ("best_fusion_model", "decomposer_rl"):
            if not (directory / f"{stem}.pt").is_file() and not list(directory.glob(f"{stem}_epoch*.pt")):
                missing.append(f"{directory}/{stem}.pt")
    if missing:
        raise FileNotFoundError("Missing strictly required trained checkpoint(s): " + "; ".join(missing[:10]))


def before_worker(run_root: Path, seed: int) -> None:
    _verify_model_artifacts(run_root, (seed,))
    test_file = run_root / "splits" / "ood_location_test.json"
    if not test_file.is_file():
        raise FileNotFoundError(f"Missing fixed OOD test split: {test_file}")
    print(f"[Beta-control preflight] worker prerequisites passed | seed={seed}")


def after_smoke(output: Path) -> None:
    summary = _read_json(output / "summary.json")
    csv_path = output / "per_scenario_results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    n_scenarios = int(summary.get("n_test_scenarios", 0))
    expected = n_scenarios * len(BETA_ARMS)
    def finite(field: str, row: dict[str, str]) -> bool:
        try:
            return math.isfinite(float(row[field]))
        except (KeyError, ValueError):
            return False
    def expected_radius(row: dict[str, str]) -> int:
        count = int(float(row["n_modalities_delivered"]))
        beta = float(row["beta_used"])
        return 25 if count == 1 else int(25 + 8 * min(beta + 0.3, 1.0))
    m1_groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("availability_condition") == "M1":
            m1_groups.setdefault((row["training_seed"], row["scenario_id"]), []).append(row)
    def identical_m1_group(group: list[dict[str, str]]) -> bool:
        fields = METRIC_COLUMNS + ("initial_min_obstacle_radius_px", "final_min_obstacle_radius_px")
        if len(group) != len(BETA_ARMS) or any(field not in row for row in group for field in fields):
            return False
        return all(
            max(float(row[field]) for row in group) - min(float(row[field]) for row in group) <= 1e-12
            for field in fields
        )
    m1_identical = bool(m1_groups) and all(identical_m1_group(group) for group in m1_groups.values())
    random_active_radii = {
        int(float(row["initial_min_obstacle_radius_px"]))
        for row in rows
        if row.get("beta_arm") == "random_hash" and int(float(row["n_modalities_delivered"])) >= 2
    }
    checks = {
        "current protocol version": summary.get("protocol_version") == PROTOCOL_VERSION,
        "marked smoke": summary.get("smoke") is True,
        "seed 42": summary.get("training_seed") == 42,
        "five scenarios": summary.get("n_test_scenarios") == 5,
        "all modality-count bins recorded": set(summary.get("availability_conditions", {})) == set(AVAILABILITY_BINS),
        "beta-active modality bin present": any(
            int(summary.get("availability_conditions", {}).get(f"M{count}", 0)) > 0
            for count in range(2, 5)
        ),
        "all beta arms": set(summary.get("beta_arms", ())) == set(BETA_ARMS),
        "fail-closed implementation audit": summary.get("implementation_audit", {}).get("fuser_beta_interceptor_required") is True
            and summary.get("implementation_audit", {}).get("initial_astar_radius_must_equal_beta_derived_radius") is True
            and summary.get("implementation_audit", {}).get("rng_reset_before_each_arm") is True,
        "matched planner input": summary.get("planner_input") == "shared ScenarioSample target-coordinate interface across all beta arms",
        "coordinate-inference scope disclosed": "does not audit coordinate inference" in summary.get("label_leakage_gate", ""),
        "complete audit rows": len(rows) == expected,
        "finite learned beta recorded": all(finite("learned_beta", row) for row in rows),
        "all arms recorded": set(r.get("beta_arm") for r in rows) == set(BETA_ARMS),
        "modality-count labels match": all(
            r.get("availability_condition") == f"M{int(float(r['n_modalities_delivered']))}"
            for r in rows
        ),
        "fixed arm exactly 0.5": all(abs(float(r["beta_used"]) - 0.5) < 1e-9
                                      for r in rows if r["beta_arm"] == "fixed_0_5"),
        "learned arm equals archived learned beta": all(
            abs(float(r["beta_used"]) - float(r["learned_beta"])) < 1e-9
            for r in rows if r["beta_arm"] == "learned"
        ),
        "count arm equals M/4": all(abs(float(r["beta_used"]) - float(r["n_modalities_delivered"]) / 4.0) < 1e-9
                                     for r in rows if r["beta_arm"] == "observed_count"),
        "actual initial radius equals beta-derived radius": all(
            "expected_initial_min_obstacle_radius_px" in row
            and int(float(row["initial_min_obstacle_radius_px"])) == expected_radius(row)
            and int(float(row["expected_initial_min_obstacle_radius_px"])) == expected_radius(row)
            for row in rows
        ),
        "random arm changes actual beta-active radii": len(random_active_radii) >= 2,
        "M1 negative control is identical across arms": m1_identical,
    }
    for label, passed in checks.items():
        print(("PASS" if passed else "FAIL") + f" | {label}")
    if not all(checks.values()):
        raise SystemExit("Beta-control smoke failed; do not submit the five formal workers")
    print("[Beta-control preflight] smoke passed; formal five-seed submission is permitted")


def before_merge(output_root: Path, model_run_root: Path) -> None:
    _verify_model_artifacts(model_run_root, SEEDS)
    workers = output_root / "results" / "beta_coupling_workers"
    expected = {f"seed_{seed}" for seed in SEEDS}
    found = {path.name for path in workers.glob("seed_*") if path.is_dir()} if workers.is_dir() else set()
    if found != expected:
        raise RuntimeError(f"Expected worker directories {sorted(expected)}, found {sorted(found)}")
    print("[Beta-control preflight] merge prerequisites passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Beta-control protocol gates")
    parser.add_argument("--stage", choices=("before_worker", "after_smoke", "before_merge"), required=True)
    parser.add_argument("--run-root")
    parser.add_argument("--model-run-root", help="Optional separate root containing models/ours/seed_*")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke-output")
    args = parser.parse_args()
    if args.stage == "after_smoke":
        if not args.smoke_output:
            raise ValueError("--smoke-output is required for after_smoke")
        after_smoke(Path(args.smoke_output).resolve())
        return
    if not args.run_root:
        raise ValueError("--run-root is required for this stage")
    root = Path(args.run_root).resolve()
    if args.stage == "before_worker":
        if args.seed not in SEEDS:
            raise ValueError(f"--seed must be one of {SEEDS}")
        before_worker(root, args.seed)
    else:
        model_root = Path(args.model_run_root).resolve() if args.model_run_root else root
        before_merge(root, model_root)


if __name__ == "__main__":
    main()
