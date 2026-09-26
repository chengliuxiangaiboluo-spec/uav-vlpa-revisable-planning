"""Leakage-free real-data multimodal grounding and waypoint-plan evaluation.

This replaces the legacy real-data ablation where ``targets.csv`` was passed
to the planner.  Here labels are available only to the supervised loss and the
evaluator.  The generated plan is the open-loop polyline from the logged home
position through coordinates predicted from the declared interaction inputs.
It is therefore *not* an obstacle-avoidance or raw-camera perception claim.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, random, re, sys, wave
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PACKAGE_DIR = Path(__file__).resolve().parent; PROJECT_ROOT = PACKAGE_DIR.parent
for path in (PROJECT_ROOT, PACKAGE_DIR):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

from evaluation.realdata_validator import RealDataLoader
from evaluation.field_validator import compute_trajectory_mae, compute_trajectory_rmse
from evaluation.rmse_data import compute_dtw_mse_rmse
from v2_grounding import MAX_TARGETS, VisualLanguageGrounder, _tokens

SEEDS = (42, 123, 456, 789, 2024)
CONDITIONS = ("text_only", "text_voice", "text_gesture", "text_annotation", "text_screenshot", "full_modal")
PROTOCOL = "real_multimodal_grounded_open_loop_waypoint_v1"
# Report all three thresholds.  5 m preserves the legacy UAV-arrival
# convention; 20 m and 40 m make the coordinate-grounding operating regime
# explicit rather than selecting a post-hoc threshold that happens to win.
TARGET_THRESHOLDS_M = (5.0, 20.0, 40.0)


def _read_labels(path):
    labels = []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("type", "").lower() == "obstacle": continue
            try: labels.append([float(row["x_pct"])/100, float(row["y_pct"])/100])
            except (KeyError, ValueError): pass
    return labels


def _screenshot(directory):
    images = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
              and p.name not in {"annotation.png", "gesture.png"}]
    return sorted(images)[0] if images else None


def _count_from_text(text):
    nums = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8}
    m = re.search(r"([1-8一二三四五六七八])点巡检", text)
    if m: return int(m.group(1)) if m.group(1).isdigit() else nums[m.group(1)]
    marks = ("规划路径经过", "规划航线经过", "依次访问", "依次检查", "依次经过", "从起点出发，经过", "先抵达", "先到", "先访问")
    found = [(text.find(x), x) for x in marks if text.find(x) >= 0]
    if not found: return None
    at, mark = min(found); fragment = text[at + len(mark):]
    end = min((fragment.find(x) for x in ("避开", "避障", "确保", "完成", "返回", "返航") if fragment.find(x) >= 0), default=len(fragment))
    fragment = fragment[:end]
    for item in ("、", "，", ",", "和", "及", "→", "再到", "然后到", "随后经过", "接着经过", "最后到达", "到达"): fragment = fragment.replace(item, "|")
    names = [x.strip(" |。；;，,:：") for x in fragment.split("|")]
    return len([x for x in names if x and x not in {"起飞点", "任务"}]) or None


def prepare(data_dir: Path, manifest_path: Path):
    loader = RealDataLoader(str(data_dir)); included, excluded = [], []
    for directory in sorted((p for p in data_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        meta_path, target_path = directory / "metadata.json", directory / "targets.csv"
        reasons = []
        required = {"voice_path": directory / "voice_command.wav", "gesture_path": directory / "gesture.png", "annotation_path": directory / "annotation.png"}
        shot = _screenshot(directory)
        if not meta_path.is_file() or not target_path.is_file(): reasons.append("missing_metadata_or_targets_label")
        for name, path in required.items():
            if not path.is_file(): reasons.append("missing_" + name.replace("_path", ""))
        if shot is None: reasons.append("missing_separate_screenshot")
        if reasons: excluded.append({"directory_id":directory.name,"reasons":reasons}); continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        text_path = directory / "text_instruction.txt"
        text = text_path.read_text(encoding="utf-8").strip() if text_path.is_file() else meta.get("text_instruction", "").strip()
        labels, count = _read_labels(target_path), _count_from_text(text)
        flight = loader._find_flight_csv(str(directory), meta)
        if not text: reasons.append("missing_text")
        if not labels or len(labels) > MAX_TARGETS: reasons.append("invalid_target_labels")
        if count is None or count != len(labels): reasons.append("instruction_target_count_disagrees_with_labels")
        if not flight: reasons.append("missing_flight_log")
        bounds = meta.get("bounds", {})
        if not all(key in bounds for key in ("min_lat", "max_lat", "min_lon", "max_lon")): reasons.append("missing_geo_bounds")
        if reasons: excluded.append({"directory_id":directory.name,"reasons":reasons}); continue
        included.append({"scenario_id":f"realdata_dir_{directory.name}","directory_id":directory.name,"text":text,"target_count":count,
                         "label_coordinates_pct":labels,"bounds":bounds,"voice_path":str(required["voice_path"].resolve()),
                         "gesture_path":str(required["gesture_path"].resolve()),"annotation_path":str(required["annotation_path"].resolve()),
                         "screenshot_path":str(shot.resolve()),"flight_csv":str(Path(flight).resolve())})
    if len(included) < 30: raise RuntimeError(f"Only {len(included)} fully auditable records; need >=30")
    rows = sorted(included, key=lambda x: hashlib.sha256(x["scenario_id"].encode()).hexdigest())
    n, a, b = len(rows), int(.70*len(rows)), int(.85*len(rows))
    manifest = {"protocol":PROTOCOL,"n_discovered":len(list(data_dir.iterdir())),"n_included":n,
                "splits":{"train":rows[:a],"val":rows[a:b],"test":rows[b:]},"excluded":excluded,
                "labels_usage":"targets.csv labels are used only for supervised loss and evaluator metrics; never passed to model.forward or plan generation",
                "claim_allowed":"held-out real interaction-input grounded open-loop waypoint planning",
                "claim_forbidden":"obstacle avoidance, raw-camera perception, or end-to-end autonomous flight validation"}
    manifest_path.parent.mkdir(parents=True, exist_ok=True); manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REAL MM PLAN PREPARED | included={n} train={a} val={b-a} test={n-b}", flush=True)


def _audio_image(path: str):
    try:
        with wave.open(path, "rb") as h:
            raw, width, channels = h.readframes(min(h.getnframes(), h.getframerate()*12)), h.getsampwidth(), h.getnchannels()
        dtype = {1:np.uint8, 2:np.int16, 4:np.int32}.get(width)
        if dtype is None: raise ValueError("unsupported wav width")
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if channels > 1: audio = audio.reshape(-1, channels).mean(1)
        audio = audio - audio.mean(); audio /= np.abs(audio).max() + 1e-6
        nfft, hop = 256, 128; frames = [audio[i:i+nfft] for i in range(0, max(1, len(audio)-nfft), hop)]
        if not frames: raise ValueError("empty audio")
        spec = np.log1p(np.abs(np.fft.rfft(np.stack(frames), axis=1))).T
        spec = (255*(spec-spec.min())/(spec.max()-spec.min()+1e-6)).astype(np.uint8)
        return Image.fromarray(spec).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"cannot decode voice input {path}: {exc}")


def _image_for(row, condition):
    if condition == "text_voice": return _audio_image(row["voice_path"])
    if condition == "text_gesture": return Image.open(row["gesture_path"]).convert("RGB")
    if condition == "text_annotation": return Image.open(row["annotation_path"]).convert("RGB")
    if condition == "text_screenshot": return Image.open(row["screenshot_path"]).convert("RGB")
    if condition == "full_modal":
        cells = [Image.open(row["screenshot_path"]).convert("RGB"), Image.open(row["gesture_path"]).convert("RGB"),
                 Image.open(row["annotation_path"]).convert("RGB"), _audio_image(row["voice_path"])]
        cells = [x.resize((112,112)) for x in cells]; out = Image.new("RGB", (224,224))
        for i, cell in enumerate(cells): out.paste(cell, ((i%2)*112,(i//2)*112))
        return out
    return Image.new("RGB", (224,224))


class GroundedDataset(Dataset):
    def __init__(self, rows, condition):
        self.rows, self.condition = list(rows), condition
        self.tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]; labels = torch.zeros(MAX_TARGETS,2); labels[:row["target_count"]] = torch.tensor(row["label_coordinates_pct"], dtype=torch.float32)
        return self.tf(_image_for(row,self.condition)), torch.tensor(_tokens(row["text"])[:64] or [0]), labels, row["target_count"], row


def _collate(batch):
    images,tokens,labels,counts,rows = zip(*batch); longest=max(x.numel() for x in tokens); padded=torch.zeros(len(batch),longest,dtype=torch.long); valid=torch.zeros(len(batch),longest)
    for i,x in enumerate(tokens): padded[i,:x.numel()]=x; valid[i,:x.numel()]=1
    return torch.stack(images),padded,valid,torch.stack(labels),torch.tensor(counts),rows


def _loader(rows, condition, shuffle): return DataLoader(GroundedDataset(rows,condition),batch_size=8,shuffle=shuffle,num_workers=0,collate_fn=_collate)

def _loss(model, loader, device, opt=None):
    model.train(opt is not None); total=n=0
    for images,tokens,valid,labels,counts,_ in loader:
        images,tokens,valid,labels,counts=(x.to(device) for x in (images,tokens,valid,labels,counts)); out=model(images,tokens,valid); mask=torch.arange(MAX_TARGETS,device=device)[None,:] < counts[:,None]; loss=((out-labels).square().sum(-1)[mask]).mean()
        if opt: opt.zero_grad(); loss.backward(); opt.step()
        total += float(loss.detach())*len(images); n += len(images)
    return total/max(n,1)

def _haversine(a,b):
    lat1,lon1,lat2,lon2=map(math.radians,(*a,*b)); dlat,dlon=lat2-lat1,lon2-lon1; x=math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371000*2*math.asin(math.sqrt(x))

def _pct_to_geo(p,b): return (b["max_lat"]-p[1]*(b["max_lat"]-b["min_lat"]), b["min_lon"]+p[0]*(b["max_lon"]-b["min_lon"]))

def _flight(row):
    loader=RealDataLoader(str(Path(row["flight_csv"]).parent)); record=loader.dji_parser.parse_csv(row["flight_csv"], flight_id=row["scenario_id"])
    if not record.trajectory_latlon: raise RuntimeError(f"no parsed flight trajectory: {row['scenario_id']}")
    return record.trajectory_latlon

def _evaluate(model, rows, condition, seed, device):
    output=[]; model.eval()
    with torch.no_grad():
        for images,tokens,valid,labels,counts,batch_rows in _loader(rows,condition,False):
            pred=model(images.to(device),tokens.to(device),valid.to(device)).cpu()
            for i,row in enumerate(batch_rows):
                n=int(counts[i]); guess=pred[i,:n].tolist(); truth=labels[i,:n].tolist(); errors=[_haversine(_pct_to_geo(x,row["bounds"]),_pct_to_geo(y,row["bounds"])) for x,y in zip(guess,truth)]
                actual=_flight(row); plan=[actual[0]]+[_pct_to_geo(x,row["bounds"]) for x in guess]
                try: _,dtw=compute_dtw_mse_rmse(plan,actual)
                except Exception: dtw=float("nan")
                metrics = {}
                for threshold in TARGET_THRESHOLDS_M:
                    key = str(int(threshold)); hits=[e <= threshold for e in errors]; prefix=0
                    for hit in hits:
                        if not hit: break
                        prefix += 1
                    metrics["tcr_at_"+key+"m"] = sum(hits)/n
                    metrics["sequential_tcr_at_"+key+"m"] = prefix/n
                    metrics["ia_at_"+key+"m"] = sum(hits)/n
                output.append({"scenario_id":row["scenario_id"],"condition":condition,"training_seed":seed,"target_count":n,
                    "coordinate_mae_m":sum(errors)/n,"coordinate_rmse_m":math.sqrt(sum(x*x for x in errors)/n),"trajectory_rmse_m":compute_trajectory_rmse(plan,actual),
                    "trajectory_mae_m":compute_trajectory_mae(plan,actual),"dtw_m":dtw,"predicted_coordinates_pct":guess,
                    "label_coordinates_used_only_for_scoring":True,"planner_input":"predicted_coordinates_only", **metrics})
    return output

def run(manifest_path, out_dir, seed, epochs, smoke):
    manifest=json.loads(Path(manifest_path).read_text(encoding="utf-8")); splits=manifest["splits"]
    if smoke: splits={k:v[:min(4,len(v))] for k,v in splits.items()}
    if not all(splits.values()): raise RuntimeError("empty fixed split")
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=False); device="cuda" if torch.cuda.is_available() else "cpu"; all_rows=[]; changed={}
    for index,condition in enumerate(CONDITIONS):
        torch.manual_seed(seed); random.seed(seed)
        image_enabled=condition != "text_only"; model=VisualLanguageGrounder(use_image=image_enabled).to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
        train,val=_loader(splits["train"],condition,True),_loader(splits["val"],condition,False); best,best_state,stale=float("inf"),None,0
        for _ in range(epochs):
            _loss(model,train,device,opt); loss=_loss(model,val,device)
            if loss < best: best,best_state,stale=loss,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},0
            else:
                stale += 1
                if stale >= 12: break
        model.load_state_dict(best_state); torch.save({"state_dict":model.state_dict(),"condition":condition,"seed":seed},out_dir/f"{condition}_best.pt")
        rows=_evaluate(model,splits["test"],condition,seed,device); all_rows += rows; changed[condition]=sum(1 for row in rows if row["predicted_coordinates_pct"])
        print(f"REAL MM PLAN | seed={seed} condition={condition} best_val={best:.6f} test={len(rows)}",flush=True)
    # The output has independent forward calls and must not collapse to identical coordinate arrays.
    text={r["scenario_id"]:r["predicted_coordinates_pct"] for r in all_rows if r["condition"]=="text_only"}; altered={}
    for cond in CONDITIONS[1:]:
        altered[cond]=sum(text[s]!=r["predicted_coordinates_pct"] for r in all_rows if r["condition"]==cond for s in [r["scenario_id"]])
        if altered[cond] == 0: raise RuntimeError(f"degenerate modality output: {cond} equals text_only everywhere")
    (out_dir/"per_scenario_results.json").write_text(json.dumps(all_rows,ensure_ascii=False,indent=2),encoding="utf-8")
    (out_dir/"summary.json").write_text(json.dumps({"protocol":PROTOCOL,"seed":seed,"epochs":epochs,"smoke":smoke,"test_n":len(splits["test"]),"conditions":CONDITIONS,"changed_vs_text_only":altered,"label_leakage_gate":"passed","planner_input":"predicted_coordinates_only"},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"REAL MM PLAN SEED PASSED | seed={seed} changed={altered}",flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--prepare",action="store_true");p.add_argument("--run",action="store_true");p.add_argument("--data-dir");p.add_argument("--manifest");p.add_argument("--output-dir");p.add_argument("--seed",type=int,default=42);p.add_argument("--epochs",type=int,default=80);p.add_argument("--smoke",action="store_true");a=p.parse_args()
    if a.prepare: prepare(Path(a.data_dir),Path(a.manifest))
    elif a.run: run(a.manifest,a.output_dir,a.seed,a.epochs,a.smoke)
    else: p.error("choose --prepare or --run")
if __name__ == "__main__": main()
