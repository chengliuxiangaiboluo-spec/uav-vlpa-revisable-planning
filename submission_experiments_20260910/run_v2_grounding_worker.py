"""Run one independently schedulable V2 grounding track.

The visual track also performs a paired simulated event test.  With and
without replanning share the same visual grounder, seed, initial route and
event; only the replanning action differs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
for search_path in (PROJECT_ROOT, PACKAGE_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiment_models import build_evaluation_planner
from data.recalculate_to_latlon import (
    read_coordinates_from_csv, recalculate_coordinates,
)
from submission_common import (
    DEFAULT_SEEDS, aggregate_seed_means, evaluate, read_samples, run_metadata,
    write_json, write_rows_csv,
)
from utils.seed_manager import set_global_seed
from v2_grounding import (
    GroundingDataset, VisualLanguageGrounder, collate_grounding, evaluate_loss,
    train_epoch,
)

CONDITIONS = ("visual_language_grounding", "text_only_grounding")
EVENT_FIELDS = {
    "scenario_id", "trigger_fraction", "obstacle_fraction",
    "clearance_radius_pct", "placement_rule",
}


def _loader(samples, benchmark, batch_size, shuffle):
    return DataLoader(
        GroundingDataset(samples, benchmark), batch_size=batch_size,
        shuffle=shuffle, num_workers=0, collate_fn=collate_grounding,
    )


def _validate_event_protocol(record, expected_ids):
    events = record.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("C1 V2 event protocol is empty")
    event_ids = {str(event.get("scenario_id")) for event in events}
    if event_ids != expected_ids:
        raise ValueError("C1 V2 event IDs do not exactly match OOD scenarios")
    for event in events:
        missing = EVENT_FIELDS - set(event)
        if missing:
            raise ValueError(
                "Obsolete C1 V2 protocol; rerun 09_v2_prepare_protocol.sh. "
                f"scenario={event.get('scenario_id')} missing={sorted(missing)}"
            )
        if "new_obstacles" in event:
            raise ValueError(
                "Obsolete target-derived C1 event found; the current protocol "
                "must materialise the obstacle from the pre-event route"
            )


def _predict(model, sample, benchmark, device):
    # Exp6 fusion-conditioned controls need the original instruction string to
    # obtain an audited local sentence embedding.  This hook still returns only
    # learned coordinates; labels are never passed into the planner.
    if hasattr(model, "predict_sample"):
        return model.predict_sample(sample, benchmark, device)
    image, tokens, _, _ = GroundingDataset([sample], benchmark)[0]
    tokens = tokens.unsqueeze(0).to(device)
    valid = torch.ones_like(tokens, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        coordinates = model(image.unsqueeze(0).to(device), tokens, valid)[0]
    return [
        (float(x * 100.0), float(y * 100.0))
        for x, y in coordinates[:len(sample.targets)].cpu()
    ]


class GroundedPlanner:
    """Plan on learned coordinates while leaving original labels for scoring."""

    def __init__(self, planner, grounder, benchmark, device):
        self.planner = planner
        self.grounder = grounder
        self.benchmark_dir = benchmark
        self.device = device
        self.last_tasks = []

    def plan(self, sample):
        grounded = copy.deepcopy(sample)
        prediction = _predict(
            self.grounder, sample, self.benchmark_dir, self.device,
        )
        for target, coordinates in zip(grounded.targets, prediction):
            target.coordinates_percent = coordinates
            # The scorer's true geodetic location must never reach the planner.
            target.coordinates_latlon = (0.0, 0.0)
        result = self.planner.plan(grounded)
        self.last_tasks = getattr(self.planner, "last_tasks", [])
        return result

    def replan(self, sample, event, current_position_pct):
        coordinates = _predict(
            self.grounder, sample, self.benchmark_dir, self.device,
        )
        targets = {
            target.name: {"type": target.target_type, "coordinates": list(coord)}
            for target, coord in zip(sample.targets, coordinates)
        }
        obstacles = {
            obstacle.name: {
                "type": obstacle.target_type,
                "coordinates": list(obstacle.coordinates_percent),
            }
            for obstacle in sample.obstacles
        }
        old_home = list(getattr(self.planner, "home_coordinate", [10.0, 10.0]))
        self.planner.home_coordinate = list(current_position_pct)
        try:
            return self.planner.replan(
                {"image_id": sample.image_id, "targets_pct": targets,
                 "obstacles_pct": obstacles},
                {"new_obstacles": event["new_obstacles"]},
            )
        finally:
            self.planner.home_coordinate = old_home


def georeference_for_scoring(samples, benchmark):
    """Attach true target/obstacle lat-lon only to evaluator-side samples."""
    coordinate_dict = read_coordinates_from_csv(
        str(Path(benchmark) / "parsed_coordinates.csv")
    )
    scoring_samples = copy.deepcopy(list(samples))
    for sample in scoring_samples:
        objects = list(sample.targets) + list(sample.obstacles)
        percentage = {
            f"object_{index}": {
                "type": item.target_type,
                "coordinates": list(item.coordinates_percent),
            }
            for index, item in enumerate(objects)
        }
        geodetic = recalculate_coordinates(
            percentage, int(sample.image_id), coordinate_dict,
        )
        for index, item in enumerate(objects):
            item.coordinates_latlon = tuple(
                geodetic[f"object_{index}"]["coordinates"]
            )
    return scoring_samples


def _coordinate_rows(model, samples, benchmark, device, seed, condition):
    rows = []
    coordinate_dict = read_coordinates_from_csv(
        str(Path(benchmark) / "parsed_coordinates.csv")
    )
    for sample in samples:
        prediction = _predict(model, sample, benchmark, device)
        percent_errors = [
            math.hypot(x - target.coordinates_percent[0],
                       y - target.coordinates_percent[1])
            for (x, y), target in zip(prediction, sample.targets)
        ]
        predicted_latlon = _pct_to_latlon(
            prediction, sample.image_id, coordinate_dict,
        )
        metre_errors = [
            _haversine_m(predicted, tuple(target.coordinates_latlon))
            for predicted, target in zip(predicted_latlon, sample.targets)
        ]
        rows.append({
            "scenario_id": sample.scenario_id,
            "training_seed": seed,
            "condition": condition,
            "mean_target_coordinate_error_pct": (
                sum(percent_errors) / len(percent_errors)
                if percent_errors else float("nan")
            ),
            "mean_target_coordinate_error_m": (
                sum(metre_errors) / len(metre_errors)
                if metre_errors else float("nan")
            ),
            "target_precision_25m": (
                sum(error <= 25.0 for error in metre_errors) / len(metre_errors)
                if metre_errors else float("nan")
            ),
            "target_precision_50m": (
                sum(error <= 50.0 for error in metre_errors) / len(metre_errors)
                if metre_errors else float("nan")
            ),
            "n_targets": len(metre_errors),
        })
    return rows


def _haversine_m(left, right):
    """Great-circle distance in metres for two (latitude, longitude) pairs."""
    lat1, lon1 = map(math.radians, left)
    lat2, lon2 = map(math.radians, right)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return 6371008.8 * 2.0 * math.asin(min(1.0, math.sqrt(value)))


def _pct_to_latlon(points, image_id, coordinate_dict):
    corners = coordinate_dict[int(image_id)]
    nw_lat, nw_lon = corners["NW"]
    se_lat, se_lon = corners["SE"]
    return [
        (
            nw_lat - float(y) / 100.0 * (nw_lat - se_lat),
            nw_lon + float(x) / 100.0 * (se_lon - nw_lon),
        )
        for x, y in points
    ]


def _latlon_to_pct(trajectory, image_id, coordinate_dict):
    corners = coordinate_dict[int(image_id)]
    nw_lat, nw_lon = corners["NW"]
    se_lat, se_lon = corners["SE"]
    lat_span, lon_span = nw_lat - se_lat, se_lon - nw_lon
    if not lat_span or not lon_span:
        raise ValueError(f"Degenerate georeference for image {image_id}")
    return [
        ((lon - nw_lon) / lon_span * 100.0,
         (nw_lat - lat) / lat_span * 100.0)
        for lat, lon in trajectory
    ]


def _materialise_event(spec, route_pct):
    if not spec or len(route_pct) < 3:
        return None
    last = len(route_pct) - 1
    trigger_index = min(
        max(int(round(float(spec["trigger_fraction"]) * last)), 0), last - 1,
    )
    obstacle_index = min(
        max(int(round(float(spec["obstacle_fraction"]) * last)),
            trigger_index + 1), last,
    )
    center = route_pct[obstacle_index]
    event = copy.deepcopy(spec)
    event.update({
        "trigger_index": trigger_index,
        "obstacle_index": obstacle_index,
        "current_position_pct": list(route_pct[trigger_index]),
        "new_obstacles": {
            "event_obstacle": {
                "type": "temporary_no_fly_zone",
                "coordinates": [round(center[0], 6), round(center[1], 6)],
            }
        },
    })
    return event


def _clearance(route_pct, center):
    if not route_pct:
        return float("nan")
    return min(math.hypot(x - center[0], y - center[1]) for x, y in route_pct)


def _route_length_pct(route_pct):
    return sum(
        math.hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(route_pct, route_pct[1:])
    )


def _is_nontrivial_route(route_pct):
    """Reject a one-point/zero-length replan masquerading as a success."""
    return len(route_pct) >= 2 and _route_length_pct(route_pct) > 1e-6


def _event_rows(wrapped, samples, event_map, seed):
    rows = []
    coordinate_dict = wrapped.planner.coordinates_dict
    for index, sample in enumerate(samples, 1):
        initial = wrapped.plan(sample)
        initial_pct = _latlon_to_pct(
            initial.trajectory_latlon, sample.image_id, coordinate_dict,
        ) if initial and initial.trajectory_latlon else []
        event = _materialise_event(event_map.get(sample.scenario_id), initial_pct)
        if event is None:
            for condition in (
                "visual_language_without_replan", "visual_language_with_replan",
            ):
                rows.append({
                    "scenario_id": sample.scenario_id, "training_seed": seed,
                    "condition": condition, "evaluable_event": 0,
                    "event_success": 0, "post_event_collision": float("nan"),
                    "minimum_clearance_pct": float("nan"),
                    "response_latency_ms": float("nan"),
                    "post_event_path_length_pct": float("nan"),
                    "path_length_ratio_vs_retained": float("nan"),
                    "post_event_route_nontrivial": 0,
                    "post_event_route_point_count": 0,
                })
            continue

        center = event["new_obstacles"]["event_obstacle"]["coordinates"]
        radius = float(event["clearance_radius_pct"])
        retained = initial_pct[event["trigger_index"]:]
        retained_clearance = _clearance(retained, center)
        retained_length = _route_length_pct(retained)
        retained_nontrivial = _is_nontrivial_route(retained)
        rows.append({
            "scenario_id": sample.scenario_id, "training_seed": seed,
            "condition": "visual_language_without_replan", "evaluable_event": 1,
            "event_success": int(retained_nontrivial and retained_clearance >= radius),
            "post_event_collision": int(retained_clearance < radius),
            "minimum_clearance_pct": retained_clearance,
            "response_latency_ms": 0.0,
            "post_event_path_length_pct": retained_length,
            "path_length_ratio_vs_retained": 1.0,
            "post_event_route_nontrivial": int(retained_nontrivial),
            "post_event_route_point_count": len(retained),
            "trigger_fraction": event["trigger_fraction"],
            "obstacle_fraction": event["obstacle_fraction"],
        })

        replanned = wrapped.replan(
            sample, event, event["current_position_pct"],
        )
        replanned_pct = _latlon_to_pct(
            replanned.trajectory_latlon, sample.image_id, coordinate_dict,
        ) if replanned and replanned.trajectory_latlon else []
        replanned_clearance = _clearance(replanned_pct, center)
        replanned_length = _route_length_pct(replanned_pct)
        replanned_nontrivial = _is_nontrivial_route(replanned_pct)
        rows.append({
            "scenario_id": sample.scenario_id, "training_seed": seed,
            "condition": "visual_language_with_replan", "evaluable_event": 1,
            "event_success": int(
                replanned_nontrivial and replanned_clearance >= radius
            ),
            "post_event_collision": int(
                not replanned_pct or replanned_clearance < radius
            ),
            "minimum_clearance_pct": replanned_clearance,
            "response_latency_ms": (
                float(replanned.execution_time_ms) if replanned else float("nan")
            ),
            "post_event_path_length_pct": replanned_length,
            "path_length_ratio_vs_retained": (
                replanned_length / retained_length
                if retained_length > 0.0 and replanned_pct else float("nan")
            ),
            "post_event_route_nontrivial": int(replanned_nontrivial),
            "post_event_route_point_count": len(replanned_pct),
            "trigger_fraction": event["trigger_fraction"],
            "obstacle_fraction": event["obstacle_fraction"],
        })
        if index % 25 == 0 or index == len(samples):
            print(
                f"[V2 replan] seed={seed} scenarios={index}/{len(samples)}",
                flush=True,
            )
    return rows


def _load_or_train(condition, seed, train, val, benchmark, device,
                   batch_size, epochs, model_dir):
    checkpoint = model_dir / "grounder.pt"
    metadata_path = model_dir / "training_metadata.json"
    use_image = condition == "visual_language_grounding"
    model = VisualLanguageGrounder(use_image=use_image).to(device)
    if checkpoint.exists() and metadata_path.exists():
        record = torch.load(checkpoint, map_location=device)
        if bool(record.get("use_image")) != use_image:
            raise RuntimeError(f"Checkpoint condition mismatch: {checkpoint}")
        model.load_state_dict(record["model_state_dict"], strict=True)
        model.eval()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(f"[V2 grounding] resume | {checkpoint}", flush=True)
        return model, metadata
    if model_dir.exists():
        raise RuntimeError(
            f"Incomplete seed directory: {model_dir}. Move only this directory "
            "to a backup name before retrying. Completed seeds resume automatically."
        )
    model_dir.mkdir(parents=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    train_loader = _loader(train, benchmark, batch_size, True)
    val_loader = _loader(val, benchmark, batch_size, False)
    best_loss, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, device)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
        })
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"[V2 grounding] seed={seed} condition={condition} "
            f"epoch={epoch}/{epochs} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} best_epoch={best_epoch}", flush=True,
        )
    if best_state is None:
        raise RuntimeError("No validation checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    model.to(device).eval()
    temporary = model_dir / "grounder.pt.tmp"
    torch.save({
        "model_state_dict": best_state, "use_image": use_image,
        "condition": condition, "training_seed": seed,
        "selected_epoch": best_epoch, "validation_loss": best_loss,
    }, temporary)
    temporary.replace(checkpoint)
    metadata = run_metadata({
        "condition": condition, "training_seed": seed,
        "selection": "minimum validation coordinate MSE",
        "selected_epoch": best_epoch, "best_validation_loss": best_loss,
        "epochs_requested": epochs, "history": history,
    })
    write_json(metadata_path, metadata)
    return model, metadata


def _parse_seeds(value):
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--artifact-root")
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    root = Path(args.run_root).resolve()
    artifact_root = Path(args.artifact_root).resolve() if args.artifact_root else root
    output = artifact_root / "results" / "v2_osm_grounding_workers" / args.condition
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    protocol_root = root / "v2_osm_protocol"
    train, val, test = (
        read_samples(protocol_root / "splits" / f"{name}.json")
        for name in ("train", "val_location", "ood_location_test")
    )
    protocol_test_ids = {str(sample.scenario_id) for sample in test}
    train = train[:args.max_train_samples] if args.max_train_samples else train
    val = val[:args.max_val_samples] if args.max_val_samples else val
    test = test[:args.max_test_samples] if args.max_test_samples else test
    protocol = json.loads(
        (protocol_root / "ood_replan_events.json").read_text(encoding="utf-8")
    )
    _validate_event_protocol(protocol, protocol_test_ids)
    event_map = {event["scenario_id"]: event for event in protocol["events"]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from configs.experiment_config import get_default_config
    base_cfg, *_ = get_default_config()
    benchmark = base_cfg.benchmark_dir
    scoring_test = georeference_for_scoring(test, benchmark)
    coordinate_rows, planning_rows, planning_by_seed = [], [], {}
    event_rows, checkpoint_audit = [], {}
    for seed in args.seeds:
        set_global_seed(seed)
        model_dir = (
            artifact_root / "models" / "v2_osm_grounding" / args.condition /
            f"seed_{seed}"
        )
        model, metadata = _load_or_train(
            args.condition, seed, train, val, benchmark, device,
            args.batch_size, args.epochs, model_dir,
        )
        coordinate_rows.extend(
            _coordinate_rows(model, test, benchmark, device, seed, args.condition)
        )
        planner, _ = build_evaluation_planner("ours", str(root), seed)
        wrapped = GroundedPlanner(planner, model, benchmark, device)
        rows = evaluate(wrapped, scoring_test, benchmark)
        for row in rows:
            row.update({"training_seed": seed, "condition": args.condition})
        planning_by_seed[seed] = rows
        planning_rows.extend(rows)
        if args.condition == "visual_language_grounding":
            event_rows.extend(_event_rows(wrapped, scoring_test, event_map, seed))
        checkpoint_audit[str(seed)] = {
            "path": str(model_dir / "grounder.pt"),
            "selected_epoch": metadata["selected_epoch"],
            "best_validation_loss": metadata["best_validation_loss"],
        }
        print(
            f"[V2 worker] complete | condition={args.condition} seed={seed}",
            flush=True,
        )

    write_rows_csv(output / "grounding_coordinate_results.csv", coordinate_rows)
    write_rows_csv(output / "grounded_planning_results.csv", planning_rows)
    if event_rows:
        write_rows_csv(output / "replan_event_results.csv", event_rows)
    write_json(output / "summary.json", run_metadata({
        "experiment": "V2 split grounding worker",
        "condition": args.condition, "seeds": list(args.seeds),
        "smoke": bool(args.smoke),
        "sample_counts": {"train": len(train), "val": len(val), "test": len(test)},
        "coordinate_error_mean_pct": float(np.nanmean([
            row["mean_target_coordinate_error_pct"] for row in coordinate_rows
        ])),
        "coordinate_error_mean_m": float(np.nanmean([
            row["mean_target_coordinate_error_m"] for row in coordinate_rows
        ])),
        "target_precision_25m": float(np.nanmean([
            row["target_precision_25m"] for row in coordinate_rows
        ])),
        "target_precision_50m": float(np.nanmean([
            row["target_precision_50m"] for row in coordinate_rows
        ])),
        "planning_summary": aggregate_seed_means(planning_by_seed),
        "checkpoint_audit": checkpoint_audit,
        "event_protocol_sha256": hashlib.sha256(
            (protocol_root / "ood_replan_events.json").read_bytes()
        ).hexdigest(),
        "replan_control": (
            "same visual grounder, seed, initial route and event; only the "
            "event-response replanning action differs" if event_rows else None
        ),
        "event_success_definition": (
            "a non-degenerate (at least two points and non-zero length) "
            "post-event route whose minimum clearance meets the event radius; "
            "this is route feasibility, not task-completion rate"
            if event_rows else None
        ),
        "claim_forbidden": "real-flight dynamic-event validation",
    }))
    print(f"[V2 worker] output={output}", flush=True)


if __name__ == "__main__":
    main()
