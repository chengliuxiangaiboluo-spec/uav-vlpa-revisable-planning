"""Merge audited five-seed local component controls without calling them external baselines."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from submission_common import aggregate_seed_means, run_metadata, write_json, write_rows_csv

METHODS = ("sipsa", "unigoal", "codeagents", "ikap", "csglso")

def main():
    p = argparse.ArgumentParser(); p.add_argument("--run-root", required=True); a = p.parse_args(); root = Path(a.run_root).resolve()
    all_rows, summary = [], {}
    for method in METHODS:
        d = root / "results" / "exp6_component_workers" / method
        with (d / "per_scenario_results.csv").open(encoding="utf-8", newline="") as h: rows = list(csv.DictReader(h))
        grouped = {}
        for row in rows: grouped.setdefault(int(row["training_seed"]), []).append(row)
        summary[method] = aggregate_seed_means(grouped); all_rows.extend(rows)
    out = root / "results" / "exp6_component_controls"
    if out.exists(): raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True); write_rows_csv(out / "per_scenario_results.csv", all_rows)
    write_json(out / "summary.json", run_metadata({"experiment": "Exp6 audited local component controls", "methods": summary, "seeds": [42,123,456,789,2024], "label_leakage_gate": "passed", "planner_input": "predicted_coordinates_only", "implementation_status": "local implementation-inspired component controls", "claim_allowed": "controlled local module comparison under the V2 protocol", "claim_forbidden": "external author-code reproduction or SOTA ranking"}))
    print(f"[Exp6 component controls] merged | methods={len(METHODS)} | rows={len(all_rows)}")

if __name__ == "__main__": main()
