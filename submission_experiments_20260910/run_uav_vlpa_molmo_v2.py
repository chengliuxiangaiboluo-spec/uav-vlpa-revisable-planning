"""Offline public-UAV-VLPA Molmo + nearest-neighbour TSP + A* adaptation."""
from __future__ import annotations
import argparse, copy, hashlib, json, random, re, sys, time
from pathlib import Path
PACKAGE=Path(__file__).resolve().parent; ROOT=PACKAGE.parent
for p in (ROOT,PACKAGE):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
import numpy as np, torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from main_baseline_strict_models import StrictModularPlanner
from run_v2_grounding_worker import georeference_for_scoring
from submission_common import evaluate,read_samples,run_metadata,write_json,write_rows_csv
from data.scenario_schema import ModalityType
from configs.experiment_config import get_default_config

COORDINATE_PATTERN = re.compile(
    r'\bx\d+\s*=\s*["\']?(-?\d+(?:\.\d+)?)["\']?\s+'
    r'y\d+\s*=\s*["\']?(-?\d+(?:\.\d+)?)["\']?',
    re.I,
)
TARGET_TERMS = (
    "parking lot", "building", "stadium", "school", "hospital",
    "warehouse", "bridge", "crossroad", "church", "factory",
)
TARGET_TERM_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term in TARGET_TERMS) + r")\b",
    re.I,
)
MARKER_PATTERN = re.compile(r"\bT([1-9]\d*)\b", re.I)
MARKER_DESCRIPTOR_PATTERN = re.compile(r"\b(T[1-9]\d*)\s*\(([^)]+)\)", re.I)
POINT_TAG_PATTERN = re.compile(r"<point\b([^>]*)>.*?\\?</point>", re.I | re.S)
POINT_ATTR_PATTERN = re.compile(r"\b([xy])\s*=\s*[\"']?(-?\d+(?:\.\d+)?)[\"']?", re.I)
PROMPT_VERSION = "uav-vlpa-molmo-targetwise-point-v4"


def instruction_target_slots(instruction: str) -> int:
    """Count requested landmarks from the instruction alone.

    The benchmark templates place the mission targets before the first action
    constraint (e.g. ``avoid ...`` or ``Phase 2``).  This intentionally never
    reads target coordinates or target metadata, because the count supplied to
    Molmo is part of the textual input contract rather than a label.
    """
    marker_indices = [int(value) for value in MARKER_PATTERN.findall(instruction)]
    if marker_indices:
        unique_indices = sorted(set(marker_indices))
        expected_indices = list(range(1, len(unique_indices) + 1))
        if unique_indices != expected_indices or len(marker_indices) != len(unique_indices):
            raise ValueError(
                "malformed target-marker sequence in instruction: "
                f"found T{marker_indices}, expected T1..Tn exactly once"
            )
        return len(unique_indices)

    clause = instruction.lower()
    for boundary in (
        "phase 2:", " but stay", ", avoid", " in priority order",
        " in order", " and photograph", " and return", " then return",
        " once and", " from above", ". if ",
    ):
        clause = clause.split(boundary, 1)[0]
    slots = len(TARGET_TERM_PATTERN.findall(clause))
    if slots <= 0:
        raise ValueError(f"cannot derive a target count from instruction: {instruction!r}")
    return slots


def instruction_target_markers(instruction: str, target_slots: int):
    """Return one instruction-visible marker label per requested target."""
    matched = [(name.upper(), description.strip())
               for name, description in MARKER_DESCRIPTOR_PATTERN.findall(instruction)]
    if len(matched) != target_slots:
        # The V2 contract currently uses T1..Tn marker labels.  Do not invent
        # a label from reference metadata if a future template drops them.
        raise ValueError(
            f"cannot derive {target_slots} unique marker descriptions from instruction: {instruction!r}"
        )
    indices = [int(name[1:]) for name, _ in matched]
    if indices != list(range(1, target_slots + 1)):
        raise ValueError(f"malformed marker descriptions in instruction: {matched!r}")
    return matched


def build_prompt(instruction: str, marker_name: str, marker_description: str) -> str:
    """Frozen target-wise point-localization prompt using Molmo's native form."""
    return (
        "You are locating exactly one named destination on a map for a UAV. "
        f"Locate only marker {marker_name} ({marker_description}). "
        "Do not locate any other marker, obstacle, route, path, waypoint, or return base. "
        "Return exactly one closed XML element in this exact native point format: "
        '<point x="X" y="Y" alt="MARKER">MARKER</point>. '
        "Replace X and Y with one map position in the range 0 to 100. "
        "Do not emit <points>, x1/y1, x2/y2, additional point elements, explanation, "
        "or any other coordinates. "
        f"Full instruction context: {instruction}"
    )


def parse_single_point(text):
    """Parse exactly one native Molmo ``<point x y>`` element.

    The parser never truncates output or synthesizes a coordinate.  It rejects
    plural point tags and duplicate, missing, or out-of-range point attributes.
    """
    tags = POINT_TAG_PATTERN.findall(text)
    raw_pairs = COORDINATE_PATTERN.findall(text)
    if len(tags) != 1:
        return None, len(raw_pairs), 0
    attributes = POINT_ATTR_PATTERN.findall(tags[0])
    values = {}
    duplicate_attributes = set()
    for name, value in attributes:
        name = name.lower()
        if name in values:
            duplicate_attributes.add(name)
        values[name] = float(value)
    expected = {"x", "y"}
    coordinate_names = set(values)
    if coordinate_names != expected or duplicate_attributes or "<points" in text.lower():
        return None, len(raw_pairs), 0
    x, y = values["x"], values["y"]
    if not all(0 <= value <= 100 for value in (x, y)):
        return None, len(raw_pairs), 1
    return (x, y), 1, 0

class MolmoGroundedPlanner:
    def __init__(self, core, processor, model, benchmark, device, generation_config, decoding_seed):
        self.core,self.processor,self.model,self.benchmark,self.device=core,processor,model,benchmark,device
        self.generation_config,self.decoding_seed=generation_config,decoding_seed
        self.last_tasks,self.audit=[],[]
    def _predict(self,sample,marker_name,marker_description):
        path=Path(sample.metadata.get("grounding_image_path", ""))
        if not path.is_file(): raise FileNotFoundError(f"missing audited V2 map: {path}")
        prompt=build_prompt(sample.text_instruction,marker_name,marker_description)
        inputs=self.processor.process(images=[Image.open(path).convert("RGB")],text=prompt)
        inputs={k:v.to(self.model.device).unsqueeze(0) for k,v in inputs.items()}
        generation_config=copy.deepcopy(self.generation_config)
        # A native single-point response needs far fewer tokens than a flight
        # path.  The fixed bound prevents route-like continuation while leaving
        # room for one closed <point> element.
        generation_config.max_new_tokens=64
        with torch.no_grad(): out=self.model.generate_from_batch(inputs,generation_config,tokenizer=self.processor.tokenizer)
        text=self.processor.tokenizer.decode(out[0,inputs["input_ids"].size(1):],skip_special_tokens=True)
        point, raw_pair_count, invalid_coordinate_count = parse_single_point(text)
        return prompt, text, point, raw_pair_count, invalid_coordinate_count
    def plan(self,sample):
        requested_target_slots = instruction_target_slots(sample.text_instruction)
        markers = instruction_target_markers(sample.text_instruction, requested_target_slots)
        target_audits, points = [], []
        for marker_name, marker_description in markers:
            raw_prompt, raw, point, raw_pair_count, invalid_coordinate_count = self._predict(
                sample, marker_name, marker_description
            )
            target_audits.append({
                "marker": marker_name,
                "description": marker_description,
                "prompt": raw_prompt,
                "prompt_sha256": hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest(),
                "raw_response": raw,
                "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "raw_coordinate_pair_count": raw_pair_count,
                "invalid_coordinate_count": invalid_coordinate_count,
                "parsed_point": point,
                "valid": point is not None and invalid_coordinate_count == 0,
            })
            if point is not None and invalid_coordinate_count == 0:
                points.append(point)
        parser_exact_match = len(target_audits) == requested_target_slots and all(x["valid"] for x in target_audits)
        audit_row = {
            "scenario_id": sample.scenario_id,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(
                json.dumps([x["prompt"] for x in target_audits], sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "raw_molmo_response": json.dumps(target_audits, ensure_ascii=False, sort_keys=True),
            "raw_molmo_response_sha256": hashlib.sha256(json.dumps(target_audits, sort_keys=True).encode("utf-8")).hexdigest(),
            "raw_response_length": sum(len(x["raw_response"]) for x in target_audits),
            "raw_coordinate_pair_count": sum(x["raw_coordinate_pair_count"] for x in target_audits),
            "parsed_coordinate_count": len(points),
            "requested_target_slots": requested_target_slots,
            "invalid_coordinate_count": sum(x["invalid_coordinate_count"] for x in target_audits),
            "target_query_audit": json.dumps(target_audits, ensure_ascii=False, sort_keys=True),
            "parser_exact_match": parser_exact_match,
            "parser_valid": parser_exact_match,
            "planner_input": "predicted_coordinates_only",
            "label_coordinates_used_only_for_scoring": True,
            "coordinate_fallback_used": False,
            "decoding_seed": self.decoding_seed,
        }
        self.audit.append(audit_row)
        if not parser_exact_match:
            raise RuntimeError(
                "strict UAV-VLPA coordinate audit failed for "
                f"{sample.scenario_id}: requested={requested_target_slots}, "
                f"queried={len(target_audits)}, parsed={len(points)}, "
                f"invalid={audit_row['invalid_coordinate_count']}"
            )

        grounded=copy.deepcopy(sample)
        # Mark this as visual input so the shared modular A* core applies its
        # nearest-neighbour TSP ordering. No image-derived label is supplied.
        grounded.modalities=[ModalityType.TEXT,ModalityType.ANNOTATION]
        for i,target in enumerate(grounded.targets):
            target.coordinates_percent=points[i]
            target.coordinates_latlon=(0.0,0.0)
        result=self.core.plan(grounded); self.last_tasks=getattr(self.core,"last_tasks",[])
        return result

def main():
 p=argparse.ArgumentParser();p.add_argument("--run-root",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--smoke",action="store_true");p.add_argument("--max-test-samples",type=int);p.add_argument("--decoding-seed",type=int);p.add_argument("--temperature",type=float,default=0.2);p.add_argument("--top-p",type=float,default=0.95);a=p.parse_args();root=Path(a.run_root).resolve();out=Path(a.output_dir).resolve()
 if out.exists():raise FileExistsError(f"Refusing to overwrite {out}")
 out.mkdir(parents=True)
 audit=json.loads((root/"v2_osm_protocol"/"audit.json").read_text(encoding="utf-8"));
 if audit.get("status")!="COMPLETE" or audit.get("random_coordinate_fallbacks")!=0:raise RuntimeError("invalid strict V2 audit")
 test=read_samples(root/"v2_osm_protocol"/"splits"/"ood_location_test.json")
 if a.max_test_samples:
  if not a.smoke:raise ValueError("sample cap is smoke-only")
  test=test[:a.max_test_samples]
 # Validate the instruction-only counter against the frozen benchmark before
 # model loading.  The reference target list is never supplied to Molmo or the
 # planner; it is used here only to reject an ambiguous benchmark template.
 cardinality_errors=[]
 for sample in test:
  try: inferred_slots=instruction_target_slots(sample.text_instruction)
  except ValueError as exc: cardinality_errors.append(f"{sample.scenario_id}: {exc}"); continue
  if inferred_slots != len(sample.targets):
   cardinality_errors.append(f"{sample.scenario_id}: instruction={inferred_slots}, reference={len(sample.targets)}")
 if cardinality_errors:
  raise RuntimeError("instruction target-count audit failed; " + "; ".join(cardinality_errors[:10]))
 if a.decoding_seed is not None:
  if not (0.0 < a.temperature <= 1.0): raise ValueError("temperature must be in (0, 1]")
  if not (0.0 < a.top_p <= 1.0): raise ValueError("top-p must be in (0, 1]")
  random.seed(a.decoding_seed); np.random.seed(a.decoding_seed); torch.manual_seed(a.decoding_seed)
  if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.decoding_seed)
 base,*_=get_default_config(); model_path=ROOT/"Weights"/"molmo-7B-O-bnb-4bit"
 if not (model_path/"config.json").is_file():raise FileNotFoundError(f"missing local Molmo model: {model_path}")
 processor=AutoProcessor.from_pretrained(str(model_path),trust_remote_code=True,local_files_only=True,torch_dtype="auto")
 model=AutoModelForCausalLM.from_pretrained(str(model_path),trust_remote_code=True,local_files_only=True,torch_dtype="auto",device_map="auto");model.eval()
 generation_config=GenerationConfig(max_new_tokens=256,stop_strings="<|endoftext|>")
 if a.decoding_seed is not None:
  generation_config.do_sample=True; generation_config.temperature=a.temperature; generation_config.top_p=a.top_p
 core=StrictModularPlanner(system_name="UAV-VLPA open-source Molmo adaptation",benchmark_dir=base.benchmark_dir,fuser=None,decomposer=None,path_planner_name="astar",device=base.device,seed=42)
 decoding_label="deterministic_zero_shot" if a.decoding_seed is None else str(a.decoding_seed)
 wrapped=MolmoGroundedPlanner(core,processor,model,base.benchmark_dir,base.device,generation_config,decoding_label); started=time.monotonic(); rows=evaluate(wrapped,georeference_for_scoring(test,base.benchmark_dir),base.benchmark_dir)
 for r in rows:r.update({"method":"uav_vlpa_molmo","display_name":"UAV-VLPA (open-source Molmo adaptation)","training_seed":"frozen_zero_shot","decoding_seed":decoding_label,"test_set":"v2_osm_ood_location_test","planner_input":"predicted_coordinates_only","label_coordinates_used_only_for_scoring":True})
 # Preserve raw model responses even when the strict audit rejects the run.
 # This prevents a failed smoke test from becoming an unauditable black box
 # and lets the response grammar be revised from observed evidence.
 if wrapped.audit:
  write_rows_csv(out/"molmo_response_audit.csv",wrapped.audit)
 if len(rows)!=len(test) or len(wrapped.audit)!=len(test) or not all(x["parser_exact_match"] for x in wrapped.audit):
  raise RuntimeError("incomplete or strict-invalid UAV-VLPA audit; see molmo_response_audit.csv")
 write_rows_csv(out/"per_scenario_results.csv",rows)
 metrics={k:float(np.mean([float(r[k]) for r in rows])) for k in ("task_completion_rate","instruction_accuracy","dtw_rmse","knn_rmse","efficiency_ratio")}
 evaluation_type="deterministic frozen zero-shot inference" if a.decoding_seed is None else "stochastic frozen-model decoding evaluation"
 write_json(out/"summary.json",run_metadata({"experiment":"offline public-UAV-VLPA Molmo VLM adaptation under strict V2","smoke":bool(a.smoke),"n_test_scenarios":len(test),"model_path":str(model_path),"evaluation_type":evaluation_type,"training_seed":"frozen_zero_shot","decoding_seed":a.decoding_seed,"prompt_version":PROMPT_VERSION,"generation":{"do_sample":a.decoding_seed is not None,"temperature":a.temperature if a.decoding_seed is not None else None,"top_p":a.top_p if a.decoding_seed is not None else None,"max_new_tokens":64,"stop_strings":["<|endoftext|>"]},"metrics":metrics,"parser_valid_rows":sum(x["parser_valid"] for x in wrapped.audit),"parser_exact_match_rows":sum(x["parser_exact_match"] for x in wrapped.audit),"parser_failure_rows":sum(not x["parser_valid"] for x in wrapped.audit),"coordinate_fallback_rows":sum(bool(x["coordinate_fallback_used"]) for x in wrapped.audit),"label_leakage_gate":"passed","planner_input":"predicted_coordinates_only","claim_allowed":"target-wise frozen zero-shot public-Molmo localization followed by the shared TSP/A* core; when decoding_seed is set, stochastic decoding variability only","claim_forbidden":"author-environment reproduction or five-seed training statistic","elapsed_minutes":(time.monotonic()-started)/60}))
if __name__=="__main__":main()
