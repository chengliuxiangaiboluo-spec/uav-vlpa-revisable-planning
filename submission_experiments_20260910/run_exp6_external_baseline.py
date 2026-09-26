"""Strict adapter for a genuine external baseline implementation.

This wrapper intentionally has no fallback to the repository's legacy
``*-style`` classes.  Those classes are implementation-inspired controls, not
reproductions of the cited methods.  A baseline is accepted only if its own
implementation writes a complete, auditable per-scenario result CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

from submission_common import METRICS, read_samples, run_metadata, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one registered external baseline adapter")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--test-file", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    registry_path = Path(__file__).with_name("exp6_external_baselines.json")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    spec = registry.get(args.baseline)
    if spec is None:
        raise ValueError(f"Unknown Exp6 baseline {args.baseline}; choose from {', '.join(registry)}")
    if not spec.get("ready", False):
        raise RuntimeError(
            f"{args.baseline} is blocked: no verified external implementation is configured. "
            "Set ready=true only after recording repository URL, immutable commit, environment, "
            "conversion code, and an executable command in exp6_external_baselines.json."
        )
    external_dir = root / "external_baselines" / args.baseline
    command = spec.get("command")
    if not isinstance(command, list) or not command:
        raise ValueError(f"{args.baseline}: command must be a non-empty argv list")
    output = root / "results" / "exp6_workers" / args.baseline
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    env = dict(os.environ, RUN_ROOT=str(root), EXP6_OUTPUT_DIR=str(output),
               EXP6_TEST_FILE=str(Path(args.test_file).resolve()), EXP6_BASELINE=args.baseline)
    print(f"[Exp6] external baseline started | baseline={args.baseline}", flush=True)
    subprocess.run(command, cwd=external_dir, env=env, check=True)
    result_csv = output / "per_scenario_results.csv"
    provenance = output / "provenance.json"
    if not result_csv.is_file() or not provenance.is_file():
        raise FileNotFoundError("External adapter must write per_scenario_results.csv and provenance.json")
    expected = {str(s.scenario_id) for s in read_samples(args.test_file)}
    with result_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(row.get("scenario_id", "") for row in rows) != expected:
        raise ValueError("External result must contain exactly one row for every fixed test scenario")
    missing = set(METRICS) - set(rows[0])
    if missing:
        raise ValueError(f"External result misses required metrics: {sorted(missing)}")
    supplied = json.loads(provenance.read_text(encoding="utf-8"))
    required = {"repository_url", "commit", "license", "environment", "metric_conversion"}
    if not required <= set(supplied):
        raise ValueError(f"provenance.json misses {sorted(required - set(supplied))}")
    write_json(output / "adapter_audit.json", run_metadata({
        "experiment": "Exp6 verified external baseline adapter", "baseline": args.baseline,
        "registry_spec": spec, "n_rows": len(rows), "provenance": supplied,
    }))
    print(f"[Exp6] complete | baseline={args.baseline} | rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
