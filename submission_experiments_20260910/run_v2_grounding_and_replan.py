"""Train/evaluate V2 visual grounding and its event-driven replan ablation."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from submission_common import DEFAULT_SEEDS, aggregate_seed_means, evaluate, paired_seed_test, read_samples, run_metadata, write_json, write_rows_csv
from experiment_models import build_evaluation_planner
from utils.seed_manager import set_global_seed
from v2_grounding import GroundingDataset, VisualLanguageGrounder, _tokens, collate_grounding, train_epoch


def _loader(samples, benchmark, batch, shuffle):
    return DataLoader(GroundingDataset(samples, benchmark), batch_size=batch, shuffle=shuffle,
                      num_workers=0, collate_fn=collate_grounding)


def _predict(model, sample, benchmark, device):
    ds = GroundingDataset([sample], benchmark)
    image, tokens, _, _ = ds[0]
    token = tokens.unsqueeze(0).to(device)
    valid = torch.ones_like(token, dtype=torch.float32, device=device)
    with torch.no_grad():
        coords = model(image.unsqueeze(0).to(device), token, valid)[0]
    return [(float(x * 100.0), float(y * 100.0)) for x, y in coords[:len(sample.targets)].cpu()]


class GroundedPlanner:
    """Use predicted coordinates for planning while retaining original labels for scoring."""
    def __init__(self, planner, grounder, benchmark, device):
        self.planner, self.grounder, self.benchmark_dir, self.device = planner, grounder, benchmark, device
        self.last_tasks = []

    def plan(self, sample):
        grounded = copy.deepcopy(sample)
        for target, coordinates in zip(grounded.targets, _predict(self.grounder, sample, self.benchmark_dir, self.device)):
            target.coordinates_percent = coordinates
        result = self.planner.plan(grounded)
        self.last_tasks = getattr(self.planner, "last_tasks", [])
        return result

    def replan(self, sample, event):
        coords = _predict(self.grounder, sample, self.benchmark_dir, self.device)
        targets = {target.name: {"type": target.target_type, "coordinates": list(coord)}
                   for target, coord in zip(sample.targets, coords)}
        obstacles = {o.name: {"type": o.target_type, "coordinates": list(o.coordinates_percent)} for o in sample.obstacles}
        return self.planner.replan({"image_id": sample.image_id, "targets_pct": targets, "obstacles_pct": obstacles},
                                   {"new_obstacles": event["new_obstacles"]})


def _coordinate_rows(model, samples, benchmark, device, seed, condition):
    rows = []
    for sample in samples:
        prediction = _predict(model, sample, benchmark, device)
        errors = [((x - t.coordinates_percent[0]) ** 2 + (y - t.coordinates_percent[1]) ** 2) ** .5
                  for (x, y), t in zip(prediction, sample.targets)]
        rows.append({"scenario_id": sample.scenario_id, "training_seed": seed, "condition": condition,
                     "mean_target_coordinate_error_pct": sum(errors) / max(len(errors), 1)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    root = Path(args.run_root).resolve(); output = root / "results" / "v2_grounding_replan"
    if output.exists(): raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    train, val, ood = (read_samples(root / "splits" / f"{name}.json") for name in ("train", "val_location", "ood_location_test"))
    events = json.loads((root / "v2_protocol" / "ood_replan_events.json").read_text(encoding="utf-8"))["events"]
    event_map = {event["scenario_id"]: event for event in events}
    # Config and all models are kept on the active CUDA device when available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from configs.experiment_config import get_default_config
    base_cfg, *_ = get_default_config(); benchmark = base_cfg.benchmark_dir
    all_coords, visual_eval, text_eval, replan_rows = [], {}, {}, []
    for seed in DEFAULT_SEEDS:
        set_global_seed(seed)
        for condition, use_image in (("visual_language_grounding", True), ("text_only_grounding", False)):
            model = VisualLanguageGrounder(use_image=use_image).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
            train_loader = _loader(train, benchmark, args.batch_size, True)
            for epoch in range(args.epochs):
                loss = train_epoch(model, train_loader, opt, device)
                print(f"[V2 grounding] seed={seed} condition={condition} epoch={epoch+1}/{args.epochs} loss={loss:.5f}", flush=True)
            model_dir = root / "models" / "v2_grounding" / condition / f"seed_{seed}"; model_dir.mkdir(parents=True, exist_ok=False)
            torch.save({"model_state_dict": model.state_dict(), "use_image": use_image}, model_dir / "grounder.pt")
            coord_rows = _coordinate_rows(model, ood, benchmark, device, seed, condition)
            all_coords.extend(coord_rows)
            if use_image:
                planner, _ = build_evaluation_planner("ours", str(root), seed)
                wrapped = GroundedPlanner(planner, model, benchmark, device)
                rows = evaluate(wrapped, ood, benchmark)
                for row in rows: row.update({"training_seed": seed, "condition": condition})
                visual_eval[seed] = rows
                for sample in ood:
                    event = event_map.get(sample.scenario_id)
                    result = wrapped.replan(sample, event) if event else None
                    replan_rows.append({"scenario_id": sample.scenario_id, "training_seed": seed,
                                        "condition": "visual_language_with_replan",
                                        "replan_success": int(bool(result and result.trajectory_latlon)),
                                        "response_latency_ms": float(result.execution_time_ms) if result else float("nan")})
            else:
                planner, _ = build_evaluation_planner("ours", str(root), seed)
                wrapped = GroundedPlanner(planner, model, benchmark, device)
                rows = evaluate(wrapped, ood, benchmark)
                for row in rows: row.update({"training_seed": seed, "condition": condition})
                text_eval[seed] = rows
                for sample in ood:
                    replan_rows.append({"scenario_id": sample.scenario_id, "training_seed": seed,
                                        "condition": "text_only_no_replan", "replan_success": 0,
                                        "response_latency_ms": float("nan")})
    write_rows_csv(output / "grounding_coordinate_results.csv", all_coords)
    write_rows_csv(output / "grounded_planning_results.csv", [row for values in visual_eval.values() for row in values] + [row for values in text_eval.values() for row in values])
    write_rows_csv(output / "replan_event_results.csv", replan_rows)
    write_json(output / "summary.json", run_metadata({
        "protocol": "V2 learned visual-language grounding plus simulated event-driven map replan stress test",
        "seeds": list(DEFAULT_SEEDS), "grounding_coordinate_error_unit": "map percentage points",
        "coordinate_summary": {condition: {"mean": sum(row["mean_target_coordinate_error_pct"] for row in all_coords if row["condition"] == condition) / max(sum(row["condition"] == condition for row in all_coords), 1)} for condition in ("visual_language_grounding", "text_only_grounding")},
        "planning_summary": {"visual_language_grounding": aggregate_seed_means(visual_eval), "text_only_grounding": aggregate_seed_means(text_eval)},
        "paired_tests": paired_seed_test(visual_eval, text_eval),
        "replan_claim": "simulated map-event response only; not real-flight dynamic validation",
    }))
    print(f"[V2 grounding] complete | output={output}", flush=True)


if __name__ == "__main__":
    main()
