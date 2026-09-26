"""Static and completion checks for local Exp6 component controls."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

METHODS = ("sipsa", "unigoal", "codeagents", "ikap", "csglso")
SEEDS = {42, 123, 456, 789, 2024}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", required=True)
    p.add_argument("--stage", choices=("before_run", "before_merge"), required=True)
    a = p.parse_args(); root = Path(a.run_root).resolve(); package = Path(__file__).resolve().parent
    errors = []
    for name in ("run_exp6_paper_inspired.py", "preflight_exp6_component_controls.py", "merge_exp6_component_controls.py"):
        if not (package / name).is_file(): errors.append(f"missing source: {name}")
    try:
        audit = json.loads((root / "v2_osm_protocol" / "audit.json").read_text(encoding="utf-8"))
        if audit.get("status") != "COMPLETE" or audit.get("random_coordinate_fallbacks") != 0:
            errors.append("V2 protocol audit is not complete and strict")
    except Exception as exc: errors.append(f"cannot read V2 audit: {exc}")
    for seed in SEEDS:
        ours = root / "models" / "ours" / f"seed_{seed}"
        if not any(ours.glob("best_fusion_model*.pt")): errors.append(f"missing Ours fusion checkpoint: seed_{seed}")
        if not any(ours.glob("decomposer_rl*.pt")): errors.append(f"missing Ours decomposer checkpoint: seed_{seed}")
        candidates = (root / "v2_worker_shards" / "visual_language_grounding" / f"seed_{seed}" / "models" / "v2_osm_grounding" / "visual_language_grounding" / f"seed_{seed}", root / "models" / "v2_osm_grounding" / "visual_language_grounding" / f"seed_{seed}")
        if not any((d / "grounder.pt").is_file() and (d / "training_metadata.json").is_file() for d in candidates): errors.append(f"missing V2 visual grounder: seed_{seed}")
    if a.stage == "before_merge":
        signatures = {}
        for method in METHODS:
            d = root / "results" / "exp6_component_workers" / method
            try:
                summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
                if summary.get("implementation_status") != "local implementation-inspired component control": errors.append(f"{method}: wrong implementation status")
                if set(summary.get("seeds", [])) != SEEDS or summary.get("smoke"): errors.append(f"{method}: incomplete seed set or smoke artifact")
                calls = summary.get("component_intervention_calls", {})
                if set(map(int, calls)) != SEEDS or any(int(v) <= 0 for v in calls.values()): errors.append(f"{method}: replacement-call audit failed")
                with (d / "per_scenario_results.csv").open(encoding="utf-8", newline="") as h: rows = list(csv.DictReader(h))
                if len(rows) != 2500: errors.append(f"{method}: expected 2500 rows, found {len(rows)}")
                fields = ("scenario_id", "training_seed", "task_completion_rate", "instruction_accuracy", "dtw_rmse", "knn_rmse", "efficiency_ratio")
                signatures[method] = tuple(tuple(row.get(f, "") for f in fields) for row in sorted(rows, key=lambda x: (x.get("training_seed", ""), x.get("scenario_id", ""))))
            except Exception as exc: errors.append(f"{method}: cannot validate output: {exc}")
        for left in METHODS:
            for right in METHODS:
                if left < right and left in signatures and right in signatures and signatures[left] == signatures[right]: errors.append(f"{left} and {right}: identical complete result signatures")
    if errors:
        print("EXP6 COMPONENT-CONTROL PREFLIGHT FAILED"); print("\n".join("- " + x for x in errors)); raise SystemExit(1)
    print("EXP6 COMPONENT-CONTROL PREFLIGHT PASSED")
    print("Each run is a local one-component control, not an author-code reproduction.")

if __name__ == "__main__": main()
