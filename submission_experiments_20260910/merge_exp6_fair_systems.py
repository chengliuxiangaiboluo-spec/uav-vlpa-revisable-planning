"""Merge the four fair Exp6 tracks into seven declared local controls."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from submission_common import aggregate_seed_means,run_metadata,write_json,write_rows_csv
SEEDS={42,123,456,789,2024}; TRACKS=("ours","affnet","maftnet","scal"); METHODS={"ours","affnet","maftnet","scal","uav_vla","uav_vln","aerialvln"}
def read(path):
    with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def main():
 p=argparse.ArgumentParser();p.add_argument("--run-root",required=True);a=p.parse_args();root=Path(a.run_root).resolve();out=root/"results"/"exp6_fair_comparison"
 if out.exists():raise FileExistsError(f"Refusing to overwrite {out}")
 rows=[]; summaries={}
 for track in TRACKS:
  worker=root/"results"/"exp6_fair_workers"/track; record=json.loads((worker/"summary.json").read_text(encoding="utf-8")); current=read(worker/"grounded_planning_results.csv")
  if record.get("smoke") or set(record.get("seeds",[]))!=SEEDS or record.get("label_leakage_gate")!="passed":raise RuntimeError(f"invalid track: {track}")
  rows+=current
 for method in METHODS:
  selected=[x for x in rows if x["method"]==method]
  if any(str(x.get("planner_input"))!="predicted_coordinates_only" or str(x.get("label_coordinates_used_only_for_scoring")).lower()!="true" for x in selected):raise RuntimeError(f"protocol row failure: {method}")
  grouped={seed:[] for seed in SEEDS}
  for row in selected: grouped[int(row["training_seed"])].append(row)
  if any(not x for x in grouped.values()):raise RuntimeError(f"incomplete seeds: {method}")
  summaries[method]=aggregate_seed_means(grouped)
 write_rows_csv(out/"per_scenario_results.csv",rows)
 write_json(out/"summary.json",run_metadata({"experiment":"Exp6 fair V2-only local implementation-inspired system and fusion controls","methods":summaries,"seeds":sorted(SEEDS),"label_leakage_gate":"passed","planner_input":"predicted_coordinates_only","protocol_boundary":"All map-grounding fusers and heads are seeded random initializations trained only on V2 train/validation. Ours uses its declared learned decomposer; system-style variants use their declared deterministic local decomposer. No author-code reproduction is claimed."}))
 print(f"Merged fair Exp6 result: {out}")
if __name__=="__main__":main()
