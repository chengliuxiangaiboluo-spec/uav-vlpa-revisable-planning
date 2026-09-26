"""Fail-closed five-seed merger for the leakage-free real multimodal protocol."""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

SEEDS = (42,123,456,789,2024)
CONDITIONS = ("text_only","text_voice","text_gesture","text_annotation","text_screenshot","full_modal")
METRICS = ("coordinate_mae_m","coordinate_rmse_m","tcr_at_5m","sequential_tcr_at_5m","ia_at_5m","tcr_at_20m","sequential_tcr_at_20m","ia_at_20m","tcr_at_40m","sequential_tcr_at_40m","ia_at_40m","trajectory_rmse_m","trajectory_mae_m","dtw_m")

def paired(left,right):
    delta=np.asarray(right,dtype=float)-np.asarray(left,dtype=float); n=len(delta); mean=float(delta.mean()); sd=float(delta.std(ddof=1)); t=mean/(sd/math.sqrt(n)) if sd else 0.0
    try:
        from scipy.stats import t as student_t
        p=float(2*student_t.sf(abs(t),n-1)) if sd else 1.0
    except Exception: p=None
    return {"n_training_seeds":n,"mean_delta":mean,"t":float(t),"p_two_sided":p,"cohens_dz":float(mean/sd) if sd else 0.0}

def main():
    p=argparse.ArgumentParser();p.add_argument("--workers-root",required=True);p.add_argument("--output-dir",required=True);a=p.parse_args(); workers=Path(a.workers_root); out=Path(a.output_dir)
    seed_values=defaultdict(dict); all_rows=[]; test_ids=None; found=[]
    for seed in SEEDS:
        folder=workers/f"seed_{seed}"; summary_path, rows_path=folder/"summary.json",folder/"per_scenario_results.json"
        if not summary_path.is_file() or not rows_path.is_file(): raise RuntimeError(f"missing worker output: {folder}")
        summary=json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("smoke") or summary.get("label_leakage_gate")!="passed" or summary.get("planner_input")!="predicted_coordinates_only": raise RuntimeError(f"invalid protocol gate: {folder}")
        changes=summary.get("changed_vs_text_only",{})
        if set(changes)!=set(CONDITIONS[1:]) or any(v<=0 for v in changes.values()): raise RuntimeError(f"degenerate modality predictions: {folder}")
        rows=json.loads(rows_path.read_text(encoding="utf-8")); grouped=defaultdict(list)
        for row in rows:
            if not row.get("label_coordinates_used_only_for_scoring") or row.get("planner_input")!="predicted_coordinates_only": raise RuntimeError(f"label/planner audit failed: {folder}")
            grouped[row["condition"]].append(row)
        if set(grouped)!=set(CONDITIONS): raise RuntimeError(f"condition set mismatch: {folder}")
        ids={c:{r["scenario_id"] for r in grouped[c]} for c in CONDITIONS}
        if len({tuple(sorted(x)) for x in ids.values()})!=1: raise RuntimeError(f"within-seed test cohort mismatch: {folder}")
        if test_ids is None: test_ids=ids["text_only"]
        elif test_ids!=ids["text_only"]: raise RuntimeError(f"across-seed test cohort mismatch: {folder}")
        for c in CONDITIONS:
            seed_values[c][seed]={m:float(np.mean([r[m] for r in grouped[c]])) for m in METRICS}
        all_rows.extend(rows); found.append(seed)
    if out.exists(): raise RuntimeError(f"refusing to overwrite {out}")
    summary={c:{m:{"mean":float(np.mean([seed_values[c][s][m] for s in SEEDS])),"std":float(np.std([seed_values[c][s][m] for s in SEEDS],ddof=1)),"n_training_seeds":5} for m in METRICS} for c in CONDITIONS}
    tests={c:{m:paired([seed_values["text_only"][s][m] for s in SEEDS],[seed_values[c][s][m] for s in SEEDS]) for m in METRICS} for c in CONDITIONS[1:]}
    out.mkdir(parents=True); payload={"protocol":"real_multimodal_grounded_open_loop_waypoint_v1","test_n":len(test_ids),"seeds":found,"summary":summary,"paired_seed_tests_vs_text_only":tests,"rows":all_rows,"claim_allowed":"held-out real interaction-input grounded open-loop waypoint planning","claim_forbidden":"obstacle avoidance, raw-camera perception, or end-to-end autonomous flight validation"}
    (out/"realdata_multimodal_grounded_planning_merged.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"REAL MM PLAN MERGE PASSED | test_n={len(test_ids)} seeds={found}",flush=True)
if __name__=="__main__": main()
