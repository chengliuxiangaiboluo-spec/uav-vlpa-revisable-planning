from __future__ import annotations
import argparse,json,os
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--run-root",required=True);a=p.parse_args();root=Path(a.run_root).resolve();errors=[]
 for x in ("run_uav_vlpa_molmo_v2.py","preflight_uav_vlpa_molmo_v2.py","archive_uav_vlpa_molmo_manifest.py","run_v2_grounding_worker.py","main_baseline_strict_models.py"):
  if not (Path(__file__).parent/x).is_file():errors.append(f"missing source: {x}")
 try:
  d=json.loads((root/"v2_osm_protocol"/"audit.json").read_text());
  if d.get("status")!="COMPLETE" or d.get("random_coordinate_fallbacks")!=0:errors.append("V2 audit invalid")
 except Exception as e:errors.append(f"bad V2 audit: {e}")
 for x in ("train.json","val_location.json","ood_location_test.json"):
  if not (root/"v2_osm_protocol"/"splits"/x).is_file():errors.append(f"missing split: {x}")
 weights=Path(os.environ.get("UAV_VLPA_WEIGHTS_DIR",Path(__file__).resolve().parents[1]/"Weights"))
 model=weights/"molmo-7B-O-bnb-4bit"
 if not (model/"config.json").is_file():errors.append(f"missing Molmo config: {model}")
 minilm=weights/"all-MiniLM-L6-v2"
 if not (minilm/"config.json").is_file():errors.append(f"missing offline MiniLM config: {minilm}")
 try:
  import sentence_transformers  # noqa: F401
 except Exception as e:
  errors.append(f"sentence-transformers unavailable: {e}")
 if errors:print("UAV-VLPA MOLMO PREFLIGHT FAILED");print("\n".join("- "+x for x in errors));raise SystemExit(1)
 print("UAV-VLPA MOLMO PREFLIGHT PASSED")
if __name__=="__main__":main()
