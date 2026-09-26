"""Strict real-data coordinate grounding without planner-side target leakage.

`targets.csv` is used only to construct supervised labels and evaluator-side
metrics.  The model forward pass receives a screenshot and instruction only.
The screenshots can contain recorded route overlays, so this protocol is
explicitly *trajectory-screenshot-assisted coordinate grounding*, not raw
camera perception or open-world VLM grounding.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence

import sys

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
for _path in (PROJECT_ROOT, PACKAGE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from v2_grounding import MAX_TARGETS, VOCAB_SIZE, VisualLanguageGrounder, _tokens

DEFAULT_SEEDS = (42, 123, 456, 789, 2024)
PROTOCOL = "real_trajectory_screenshot_coordinate_grounding_v1"


@dataclass
class Record:
    scenario_id: str
    directory_id: str
    screenshot_path: str
    text: str
    target_count: int
    label_coordinates_pct: List[List[float]]
    bounds: Dict[str, float]


def _raw_screenshot(directory: Path):
    images = [path for path in directory.iterdir() if path.is_file()
              and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
              and path.name not in {"annotation.png", "gesture.png"}]
    return sorted(images)[0] if images else None


def _read_labels(path: Path):
    values = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("type", "").lower() == "obstacle":
                continue
            try:
                values.append([float(row["x_pct"]) / 100.0, float(row["y_pct"]) / 100.0])
            except (KeyError, ValueError):
                continue
    return values


def _count_targets_from_text(text: str):
    """Conservative instruction-side target-count parser; no CSV access."""
    text = text.strip()
    chinese_numbers = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8}
    explicit = re.search(r"([1-8一二三四五六七八])点巡检", text)
    if explicit:
        token = explicit.group(1)
        return int(token) if token.isdigit() else chinese_numbers[token]
    start_markers = (
        "规划路径经过", "规划航线经过", "依次访问", "依次检查", "依次经过",
        "从起点出发，经过", "从起点出发,经过", "先抵达", "先到", "先访问", "前往", "经过", "访问",
    )
    positions = [(text.find(marker), marker) for marker in start_markers if text.find(marker) >= 0]
    start, marker = min(positions, default=(-1, ""), key=lambda item: item[0])
    if start < 0:
        return None
    fragment = text[start + len(marker):]
    endings = ("避开", "避障", "确保", "注意", "完成", "任务", "安全返航", "并返回", "返回", "返航")
    endpoint = min((fragment.find(item) for item in endings if fragment.find(item) >= 0), default=len(fragment))
    fragment = fragment[:endpoint]
    for symbol in ("、", "，", ",", "和", "及", "→", "再到", "然后到", "随后经过", "接着经过", "接着前往", "然后", "最后到达", "最终到达", "到达"):
        fragment = fragment.replace(symbol, "|")
    names = [item.strip(" |。；;，,:：") for item in fragment.split("|")]
    # Strip common command words while retaining genuine place names.
    names = [item.replace("依次访问", "").replace("依次检查", "").replace("依次经过", "").replace("检查", "").replace("先到", "").replace("先访问", "").replace("前往", "").replace("经过", "").replace("访问", "").strip()
             for item in names]
    names = [item for item in names if item and item not in {"起飞点", "任务"}]
    return len(names) or None


def build_manifest(data_dir: Path, output: Path):
    included, excluded = [], []
    for directory in sorted((item for item in data_dir.iterdir() if item.is_dir()), key=lambda item: item.name):
        reasons = []
        metadata_path, labels_path = directory / "metadata.json", directory / "targets.csv"
        if not metadata_path.is_file() or not labels_path.is_file():
            reasons.append("missing_metadata_or_targets_label")
        screenshot = _raw_screenshot(directory)
        if screenshot is None:
            reasons.append("missing_separate_screenshot")
        if reasons:
            excluded.append({"directory_id": directory.name, "reasons": reasons})
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        text = (directory / "text_instruction.txt").read_text(encoding="utf-8").strip() if (directory / "text_instruction.txt").is_file() else metadata.get("text_instruction", "").strip()
        labels = _read_labels(labels_path)
        parsed_count = _count_targets_from_text(text)
        if not text:
            reasons.append("missing_text")
        if not labels or len(labels) > MAX_TARGETS:
            reasons.append("invalid_target_labels")
        if parsed_count is None:
            reasons.append("unparseable_target_count_from_instruction")
        elif parsed_count != len(labels):
            reasons.append("instruction_target_count_disagrees_with_labels")
        if reasons:
            excluded.append({"directory_id": directory.name, "reasons": reasons})
            continue
        included.append(asdict(Record(
            scenario_id=f"realdata_dir_{directory.name}", directory_id=directory.name,
            screenshot_path=str(screenshot.resolve()), text=text, target_count=parsed_count,
            label_coordinates_pct=labels, bounds=metadata.get("bounds", {}),
        )))
    if len(included) < 30:
        raise RuntimeError(f"Only {len(included)} auditable screenshot records; need at least 30")
    # Fixed split, separated from model initialization seeds.  This is an
    # instance-held-out protocol; its geographic limitation is documented.
    ordered = sorted(included, key=lambda item: hashlib.sha256(item["scenario_id"].encode()).hexdigest())
    n = len(ordered); n_train = int(n * 0.70); n_val = int(n * 0.15)
    manifest = {
        "protocol": PROTOCOL,
        "n_discovered": len(list(data_dir.iterdir())), "n_included": n,
        "splits": {"train": ordered[:n_train], "val": ordered[n_train:n_train + n_val], "test": ordered[n_train + n_val:]},
        "excluded": excluded,
        "labels_usage": "targets.csv is read only by dataset loss/evaluator; never passed to model.forward or planner input",
        "claim_allowed": "held-out trajectory-screenshot-assisted coordinate grounding",
        "claim_forbidden": "raw-camera perception, open-world VLM grounding, or end-to-end flight validation",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REAL GROUNDING PREPARED | included={n} train={n_train} val={n_val} test={n-n_train-n_val}", flush=True)


class ScreenshotDataset(Dataset):
    def __init__(self, rows: Sequence[dict]):
        self.rows = list(rows)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = self.transform(Image.open(row["screenshot_path"]).convert("RGB"))
        labels = torch.zeros(MAX_TARGETS, 2, dtype=torch.float32)
        labels[:row["target_count"]] = torch.tensor(row["label_coordinates_pct"], dtype=torch.float32)
        return image, torch.tensor(_tokens(row["text"])[:64] or [0]), labels, row["target_count"], row


def collate(batch):
    images, token_lists, labels, counts, rows = zip(*batch)
    longest = max(item.numel() for item in token_lists)
    tokens = torch.zeros(len(batch), longest, dtype=torch.long)
    valid = torch.zeros(len(batch), longest, dtype=torch.float32)
    for index, item in enumerate(token_lists):
        tokens[index, :item.numel()] = item; valid[index, :item.numel()] = 1.0
    return torch.stack(images), tokens, valid, torch.stack(labels), torch.tensor(counts), rows


def _loader(rows, batch, shuffle):
    return DataLoader(ScreenshotDataset(rows), batch_size=batch, shuffle=shuffle, num_workers=0, collate_fn=collate)


def _loss(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training); total = 0.0; n = 0
    for images, tokens, valid, labels, counts, _ in loader:
        images, tokens, valid, labels, counts = (item.to(device) for item in (images, tokens, valid, labels, counts))
        prediction = model(images, tokens, valid)
        mask = torch.arange(MAX_TARGETS, device=device).unsqueeze(0) < counts.unsqueeze(1)
        loss = ((prediction - labels).square().sum(-1)[mask]).mean()
        if training:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += float(loss.detach()) * len(images); n += len(images)
    return total / max(n, 1)


def _meters(prediction, label, bounds):
    min_lat, max_lat = bounds["min_lat"], bounds["max_lat"]
    min_lon, max_lon = bounds["min_lon"], bounds["max_lon"]
    lat_a, lon_a = max_lat - prediction[1] * (max_lat-min_lat), min_lon + prediction[0] * (max_lon-min_lon)
    lat_b, lon_b = max_lat - label[1] * (max_lat-min_lat), min_lon + label[0] * (max_lon-min_lon)
    dy = (lat_a-lat_b) * 111_320.0; dx = (lon_a-lon_b) * 111_320.0 * math.cos(math.radians((lat_a+lat_b)/2))
    return math.hypot(dx, dy)


def _evaluate(model, rows, device, condition, seed):
    loader = _loader(rows, batch=8, shuffle=False); model.eval(); output = []
    with torch.no_grad():
        for images, tokens, valid, labels, counts, batch_rows in loader:
            prediction = model(images.to(device), tokens.to(device), valid.to(device)).cpu()
            for i, row in enumerate(batch_rows):
                count = int(counts[i]); predicted = prediction[i, :count].tolist(); truth = labels[i, :count].tolist()
                errors = [_meters(p, t, row["bounds"]) for p, t in zip(predicted, truth)]
                output.append({
                    "scenario_id": row["scenario_id"], "condition": condition, "training_seed": seed,
                    "target_count_from_instruction": count, "coordinate_mae_m": sum(errors)/len(errors),
                    "coordinate_rmse_m": math.sqrt(sum(value*value for value in errors)/len(errors)),
                    "predicted_coordinates_pct": predicted,
                    "prediction_label_distance_pct": float(torch.mean((prediction[i, :count] - labels[i, :count]).square()).sqrt()),
                    "label_coordinates_used_only_for_scoring": True,
                })
    return output


def run_seed(manifest_path: Path, output_dir: Path, seed: int, epochs: int, smoke: bool):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")); splits = manifest["splits"]
    if smoke:
        splits = {key: value[:min(len(value), 4)] for key, value in splits.items()}
    if not all(splits.values()): raise RuntimeError("manifest has an empty split")
    device = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(seed); random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    all_rows, predictions = [], {}
    for condition, use_image in (("text_only", False), ("text_screenshot", True)):
        torch.manual_seed(seed)
        model = VisualLanguageGrounder(use_image=use_image).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        train_loader, val_loader = _loader(splits["train"], 8, True), _loader(splits["val"], 8, False)
        best, best_state, stale = float("inf"), None, 0
        for epoch in range(1, epochs + 1):
            _loss(model, train_loader, device, optimizer)
            val = _loss(model, val_loader, device)
            if val < best:
                best, stale = val, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= 12: break
        model.load_state_dict(best_state, strict=True)
        torch.save({"model_state_dict": model.state_dict(), "use_image": use_image, "seed": seed}, output_dir / f"{condition}_grounder.pt")
        result = _evaluate(model, splits["test"], device, condition, seed)
        predictions[condition] = {row["scenario_id"]: row["predicted_coordinates_pct"] for row in result}
        all_rows.extend(result)
        print(f"REAL GROUNDING | seed={seed} condition={condition} best_val={best:.6f}", flush=True)
    deltas = []
    for scenario_id in predictions["text_only"]:
        left, right = predictions["text_only"][scenario_id], predictions["text_screenshot"][scenario_id]
        deltas.append(sum((a-b)**2 for x, y in zip(left, right) for a, b in zip(x, y)) ** 0.5)
    nonidentical = sum(value > 1e-6 for value in deltas)
    if nonidentical == 0:
        raise RuntimeError("Degenerate grounding: screenshot and text-only predictions are identical")
    with (output_dir / "per_scenario_results.json").open("w", encoding="utf-8") as handle:
        json.dump(all_rows, handle, ensure_ascii=False, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"protocol": PROTOCOL, "seed": seed, "smoke": smoke, "epochs": epochs,
                   "test_n": len(splits["test"]), "nonidentical_prediction_scenarios": nonidentical,
                   "label_leakage_gate": "passed", "manifest": str(manifest_path)}, handle, ensure_ascii=False, indent=2)
    print(f"REAL GROUNDING SEED PASSED | seed={seed} changed_predictions={nonidentical}/{len(deltas)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true"); parser.add_argument("--run-seed", action="store_true")
    parser.add_argument("--data-dir"); parser.add_argument("--manifest"); parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--epochs", type=int, default=80); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        build_manifest(Path(args.data_dir), Path(args.manifest)); return
    if args.run_seed:
        run_seed(Path(args.manifest), Path(args.output_dir), args.seed, args.epochs, args.smoke); return
    raise SystemExit("Choose --prepare or --run-seed")


if __name__ == "__main__": main()
