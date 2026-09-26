"""No-Torch preflight for the fair Exp6 V2-only rerun."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

SEEDS={42,123,456,789,2024}; TRACKS=("ours","affnet","maftnet","scal")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-root",required=True); p.add_argument("--stage",choices=("before_train","before_merge"),required=True); a=p.parse_args(); root=Path(a.run_root).resolve(); errors=[]
    package=Path(__file__).resolve().parent
    for name in ("fusion_conditioned_grounder.py","run_exp6_fair_systems.py","preflight_exp6_fair_systems.py","merge_exp6_fair_systems.py","run_v2_grounding_worker.py"):
        if not (package/name).is_file(): errors.append(f"missing source: {name}")
    try:
        audit=json.loads((root/"v2_osm_protocol"/"audit.json").read_text(encoding="utf-8"))
        if audit.get("status")!="COMPLETE" or audit.get("random_coordinate_fallbacks")!=0: errors.append("V2 audit is not strict and complete")
    except Exception as exc: errors.append(f"bad V2 audit: {exc}")
    for name in ("train.json","val_location.json","ood_location_test.json"):
        if not (root/"v2_osm_protocol"/"splits"/name).is_file(): errors.append(f"missing split: {name}")
    weights = Path(os.environ.get("UAV_VLPA_WEIGHTS_DIR", Path(__file__).resolve().parents[1] / "Weights")) / "all-MiniLM-L6-v2"
    if not (weights / "config.json").is_file(): errors.append(f"offline MiniLM model missing: {weights}")
    for seed in SEEDS:
        source=root/"models"/"ours"/f"seed_{seed}"
        if not any(source.glob("decomposer_rl*.pt")): errors.append(f"missing shared Ours decomposer: seed_{seed}")
    if a.stage=="before_merge":
        for track in TRACKS:
            pth=root/"results"/"exp6_fair_workers"/track; summary=pth/"summary.json"; rows=pth/"grounded_planning_results.csv"
            try:
                d=json.loads(summary.read_text(encoding="utf-8"))
                if d.get("smoke") or set(d.get("seeds",[]))!=SEEDS or d.get("label_leakage_gate")!="passed" or d.get("planner_input")!="predicted_coordinates_only": errors.append(f"invalid completed track: {track}")
                if not rows.is_file() or rows.stat().st_size==0: errors.append(f"missing rows: {track}")
            except Exception as exc: errors.append(f"cannot read {track}: {exc}")
    if errors:
        print("STRICT FAIR EXP6 PREFLIGHT FAILED"); print("\n".join("- "+x for x in errors)); raise SystemExit(1)
    print("STRICT FAIR EXP6 PREFLIGHT PASSED")
    print("All learned fusion grounders will start from random V2-only initialization.")
if __name__=="__main__": main()
