"""Build an auditable OSM-assisted semantic-map grounding protocol.

Existing random waypoints are used only to preserve scenario counts and rough
target-count difficulty.  Every V2 target/obstacle coordinate is replaced by
an OpenStreetMap element inside the georeferenced image bounds.  The rendered
mission layer is an explicit model input, not hidden ground truth.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
for search_path in (PROJECT_ROOT, PACKAGE_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from PIL import Image, ImageDraw, ImageFont

from configs.experiment_config import get_default_config
from data.ground_truth_generator import GroundTruthPathGenerator
from data.recalculate_to_latlon import read_coordinates_from_csv
from data.scenario_schema import AtomicTask, ExpertPath, ModalityType, WaypointTarget
from submission_common import read_samples, run_metadata, sha256_file, write_json, write_samples

SPLITS = ("train", "val_location", "id_test", "ood_location_test")
DEFAULT_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
TARGET_COLORS = (
    (0, 110, 255), (0, 180, 90), (255, 145, 0), (165, 70, 220),
    (0, 190, 210), (230, 70, 150), (130, 100, 40), (40, 170, 40),
)


def _atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _query(bounds):
    south, west, north, east = bounds
    bbox = f"{south:.8f},{west:.8f},{north:.8f},{east:.8f}"
    return (
        "[out:json][timeout:120];("
        f'nwr["amenity"]({bbox});'
        f'nwr["building"]({bbox});'
        f'nwr["highway"]({bbox});'
        f'nwr["leisure"]({bbox});'
        f'nwr["landuse"]({bbox});'
        f'nwr["natural"]({bbox});'
        f'nwr["waterway"]({bbox});'
        f'nwr["man_made"]({bbox});'
        f'nwr["military"]({bbox});'
        ");out center;"
    )


def _fetch_osm(bounds, endpoints, attempts=3):
    encoded = urllib.parse.urlencode({"data": _query(bounds)}).encode("utf-8")
    errors = []
    for endpoint in endpoints:
        for attempt in range(1, attempts + 1):
            try:
                request = urllib.request.Request(
                    endpoint, data=encoded,
                    headers={
                        "User-Agent": "UVA-VLPA-research-protocol/1.0",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                )
                with urllib.request.urlopen(request, timeout=180) as response:
                    record = json.loads(response.read().decode("utf-8"))
                if not isinstance(record.get("elements"), list):
                    raise ValueError("Overpass response has no elements list")
                record["retrieval"] = {
                    "endpoint": endpoint,
                    "retrieved_utc_epoch": time.time(),
                    "query": _query(bounds),
                }
                return record
            except Exception as exc:
                errors.append(f"{endpoint} attempt {attempt}: {exc}")
                time.sleep(min(4 * attempt, 12))
    raise RuntimeError("All Overpass requests failed: " + " | ".join(errors))


def _center(element):
    if "lat" in element and "lon" in element:
        return float(element["lat"]), float(element["lon"])
    if "center" in element:
        return float(element["center"]["lat"]), float(element["center"]["lon"])
    if "bounds" in element:
        bounds = element["bounds"]
        return ((float(bounds["minlat"]) + float(bounds["maxlat"])) / 2.0,
                (float(bounds["minlon"]) + float(bounds["maxlon"])) / 2.0)
    return None


def _category(tags):
    amenity = tags.get("amenity", "")
    leisure = tags.get("leisure", "")
    landuse = tags.get("landuse", "")
    natural = tags.get("natural", "")
    highway = tags.get("highway", "")
    # Safety/restriction semantics take precedence over a generic building or
    # highway tag.  Otherwise, for example, a military building would be
    # incorrectly exposed as a visit target.
    if tags.get("military") or tags.get("access") in {"no", "private"}:
        return "obstacle", "restricted area"
    if natural in {"water", "wetland"} or tags.get("waterway"):
        return "obstacle", "water"
    if natural in {"wood", "scrub"} or landuse in {"forest", "orchard"}:
        return "obstacle", "vegetation"
    if landuse in {"industrial", "quarry", "landfill"}:
        return "obstacle", "restricted area"
    if amenity in {"hospital", "clinic", "doctors"}:
        return "target", "hospital"
    if amenity in {"school", "college", "university", "kindergarten"}:
        return "target", "school"
    if amenity == "place_of_worship":
        return "target", "place of worship"
    if amenity == "parking" or tags.get("parking"):
        return "target", "parking area"
    if leisure in {"stadium", "sports_centre", "pitch", "track"}:
        return "target", "sports facility"
    if tags.get("bridge") not in {None, "no"}:
        return "target", "bridge"
    if tags.get("building") not in {None, "no"}:
        return "target", "building"
    if highway:
        return "target", "road feature"
    if tags.get("man_made") in {"works", "tower", "water_tower", "silo"}:
        return "target", "infrastructure"
    return None


def _extract_features(record, corners):
    nw_lat, nw_lon = corners["NW"]
    se_lat, se_lon = corners["SE"]
    lat_span, lon_span = nw_lat - se_lat, se_lon - nw_lon
    features, seen = [], set()
    for element in record.get("elements", []):
        position = _center(element)
        tags = element.get("tags", {})
        category = _category(tags)
        if position is None or category is None:
            continue
        lat, lon = position
        x = (lon - nw_lon) / lon_span * 100.0
        y = (nw_lat - lat) / lat_span * 100.0
        if not (4.0 <= x <= 96.0 and 4.0 <= y <= 96.0):
            continue
        if math.hypot(x - 10.0, y - 10.0) < 6.0:
            continue
        osm_key = f"{element.get('type')}:{element.get('id')}"
        if osm_key in seen:
            continue
        seen.add(osm_key)
        role, label = category
        features.append({
            "osm_id": osm_key, "role": role, "label": label,
            "name": tags.get("name") or tags.get("ref") or label,
            "lat": lat, "lon": lon,
            "coordinates_percent": [round(x, 6), round(y, 6)],
            "tags": tags,
        })
    return sorted(features, key=lambda item: item["osm_id"])


def _rotate(items, key):
    if not items:
        return []
    offset = int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(items)
    return items[offset:] + items[:offset]


def _instruction(targets, obstacles):
    visits = ", then ".join(
        f"marker T{index + 1} ({target.target_type})"
        for index, target in enumerate(targets)
    )
    text = f"Visit {visits} in order"
    if obstacles:
        avoid = ", ".join(
            f"O{index + 1} ({obstacle.target_type})"
            for index, obstacle in enumerate(obstacles)
        )
        text += f", avoid {avoid}"
    return text + ", and return to base."


def _validate_target_marker_contract(instruction, targets, scenario_id):
    """Ensure the visible language/map contract names every planned target.

    This prevents a stale or partially generated V2 split from asking the
    grounder for T1..Tk while the scorer expects a different number of targets.
    """
    indices = [int(value) for value in re.findall(r"\bT([1-9]\d*)\b", instruction)]
    expected = list(range(1, len(targets) + 1))
    if indices != expected:
        raise RuntimeError(
            f"invalid V2 target-marker contract for {scenario_id}: "
            f"instruction markers={indices}, target slots={expected}"
        )


def _tasks(targets, obstacles):
    tasks, priority = [], 1
    for target in targets:
        tasks.append(AtomicTask("fly_to", copy.deepcopy(target), priority, []))
        priority += 1
    for obstacle in obstacles:
        tasks.append(AtomicTask("avoid", copy.deepcopy(obstacle), priority, []))
        priority += 1
    tasks.append(AtomicTask("return", None, priority, []))
    return tasks


def _render(base_path, output_path, targets, obstacles):
    image = Image.open(base_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    radius = max(7, round(min(width, height) * 0.018))
    for index, target in enumerate(targets):
        x = round(target.coordinates_percent[0] / 100.0 * width)
        y = round(target.coordinates_percent[1] / 100.0 * height)
        color = TARGET_COLORS[index % len(TARGET_COLORS)]
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color,
                     outline=(255, 255, 255), width=max(2, radius // 4))
        draw.text((x+radius+2, y-radius), f"T{index+1}", fill=(255,255,255),
                  stroke_width=2, stroke_fill=(0,0,0), font=font)
    for index, obstacle in enumerate(obstacles):
        x = round(obstacle.coordinates_percent[0] / 100.0 * width)
        y = round(obstacle.coordinates_percent[1] / 100.0 * height)
        draw.line((x-radius, y-radius, x+radius, y+radius), fill=(255,30,30), width=4)
        draw.line((x-radius, y+radius, x+radius, y-radius), fill=(255,30,30), width=4)
        draw.text((x+radius+2, y-radius), f"O{index+1}", fill=(255,255,255),
                  stroke_width=2, stroke_fill=(0,0,0), font=font)
    draw.rectangle((0, height-15, min(width, 190), height), fill=(255,255,255))
    draw.text((3, height-13), "Map data © OpenStreetMap contributors",
              fill=(0,0,0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.png")
    image.save(temporary, format="PNG", optimize=True)
    temporary.replace(output_path)


def _waypoint(feature, name):
    return WaypointTarget(
        name=name, target_type=feature["label"],
        coordinates_percent=tuple(feature["coordinates_percent"]),
        coordinates_latlon=(feature["lat"], feature["lon"]),
    )


def _path_dict(points):
    return {
        point.name: {"type": point.target_type,
                     "coordinates": list(point.coordinates_percent)}
        for point in points
    }


def _event(sample):
    digest = hashlib.sha256(sample.scenario_id.encode()).digest()
    return {
        "scenario_id": sample.scenario_id,
        "trigger_fraction": round(0.30 + digest[0] / 2550.0, 6),
        "obstacle_fraction": round(0.58 + digest[1] / 2125.0, 6),
        "clearance_radius_pct": 4.0,
        "placement_rule": "future waypoint of the model pre-event route",
        "generation": "deterministic paired synthetic route-blocking event v2-osm",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--endpoint", action="append", default=[])
    parser.add_argument("--request-delay", type=float, default=1.0)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    protocol = root / "v2_osm_protocol"
    audit_path = protocol / "audit.json"
    if audit_path.exists():
        raise FileExistsError(f"Completed V2-OSM protocol already exists: {protocol}")
    protocol.mkdir(parents=True, exist_ok=True)
    raw_dir, map_dir = protocol / "raw_osm", protocol / "semantic_maps"

    base_cfg, *_ = get_default_config()
    corners_by_image = read_coordinates_from_csv(base_cfg.benchmark_csv)
    split_samples = {
        split: read_samples(root / "splits" / f"{split}.json") for split in SPLITS
    }
    image_sets = {
        split: {int(sample.image_id) for sample in samples}
        for split, samples in split_samples.items()
    }
    if image_sets["train"] & image_sets["ood_location_test"]:
        raise ValueError("Train/OOD location leakage before OSM preparation")
    image_ids = sorted(set().union(*image_sets.values()))
    endpoints = tuple(args.endpoint) or DEFAULT_ENDPOINTS
    feature_index, excluded_images = {}, set()
    for order, image_id in enumerate(image_ids, 1):
        corners = corners_by_image[image_id]
        bounds = (corners["SE"][0], corners["NW"][1],
                  corners["NW"][0], corners["SE"][1])
        raw_path = raw_dir / f"image_{image_id:02d}.json"
        if raw_path.exists():
            record = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            if args.cache_only:
                raise FileNotFoundError(f"Missing cached OSM response: {raw_path}")
            record = _fetch_osm(bounds, endpoints)
            _atomic_write(raw_path, json.dumps(record, ensure_ascii=False, indent=2))
            time.sleep(max(args.request_delay, 0.0))
        features = _extract_features(record, corners)
        targets = [item for item in features if item["role"] == "target"]
        obstacles = [item for item in features if item["role"] == "obstacle"]
        if not targets:
            # This is a coverage exclusion, never an invitation to fabricate a
            # coordinate.  All scenarios from this location are removed from
            # every split below, preserving the location-level split rule.
            excluded_images.add(image_id)
            print(
                f"[V2-OSM coverage exclusion] image={image_id} has no eligible "
                "OSM target; all scenarios from this location will be excluded",
                flush=True,
            )
        feature_index[image_id] = {"targets": targets, "obstacles": obstacles}
        print(
            f"[V2-OSM fetch] {order}/{len(image_ids)} image={image_id} "
            f"targets={len(targets)} obstacles={len(obstacles)}", flush=True,
        )

    generator = GroundTruthPathGenerator(
        base_cfg.benchmark_csv, base_cfg.benchmark_images_dir,
    )
    output_splits, reductions, osm_ids_by_split = {}, [], {}
    excluded_scenarios = {split: [] for split in SPLITS}
    for split, samples in split_samples.items():
        converted, split_osm_ids = [], set()
        for index, source in enumerate(samples, 1):
            if int(source.image_id) in excluded_images:
                excluded_scenarios[split].append(str(source.scenario_id))
                continue
            sample = copy.deepcopy(source)
            pool = feature_index[int(sample.image_id)]
            targets_pool = _rotate(pool["targets"], sample.scenario_id + ":target")
            obstacle_pool = _rotate(pool["obstacles"], sample.scenario_id + ":obstacle")
            requested_targets = min(max(len(source.targets), 1), 8)
            n_targets = min(requested_targets, len(targets_pool))
            n_obstacles = min(len(source.obstacles), len(obstacle_pool))
            if n_targets < requested_targets:
                reductions.append({
                    "scenario_id": sample.scenario_id, "image_id": sample.image_id,
                    "requested_targets": requested_targets, "available_targets": n_targets,
                })
            chosen_targets = targets_pool[:n_targets]
            chosen_obstacles = obstacle_pool[:n_obstacles]
            sample.targets = [
                _waypoint(feature, f"target_{i+1}")
                for i, feature in enumerate(chosen_targets)
            ]
            sample.obstacles = [
                _waypoint(feature, f"obstacle_{i+1}")
                for i, feature in enumerate(chosen_obstacles)
            ]
            sample.text_instruction = _instruction(sample.targets, sample.obstacles)
            _validate_target_marker_contract(
                sample.text_instruction, sample.targets, sample.scenario_id,
            )
            sample.modalities = [ModalityType.TEXT]
            sample.audio_path = sample.gesture_image_path = sample.annotation_image_path = None
            sample.expert_atomic_tasks = _tasks(sample.targets, sample.obstacles)
            overlay = map_dir / split / f"{sample.scenario_id}.png"
            _render(Path(base_cfg.benchmark_images_dir) / f"{sample.image_id}.jpg",
                    overlay, sample.targets, sample.obstacles)
            sample.metadata = {
                "protocol": "v2_osm_semantic_map_grounding",
                "grounding_image_path": str(overlay),
                "map_data_attribution": "© OpenStreetMap contributors",
                "map_data_license": "ODbL 1.0",
                "target_osm_ids": [item["osm_id"] for item in chosen_targets],
                "obstacle_osm_ids": [item["osm_id"] for item in chosen_obstacles],
                "source_template_scenario_id": source.scenario_id,
            }
            split_osm_ids.update(sample.metadata["target_osm_ids"])
            split_osm_ids.update(sample.metadata["obstacle_osm_ids"])
            if split == "ood_location_test":
                paths = generator.generate_expert_paths(
                    int(sample.image_id), _path_dict(sample.targets),
                    _path_dict(sample.obstacles), n_paths=1,
                )
                sample.ground_truth_paths = [
                    ExpertPath(
                        waypoints=copy.deepcopy(sample.targets),
                        path_coordinates_latlon=[tuple(x) for x in item["path_coordinates_latlon"]],
                        variant_label=item["variant_label"],
                    ) for item in paths
                ]
                if not sample.ground_truth_paths:
                    raise RuntimeError(f"No expert path for {sample.scenario_id}")
            else:
                sample.ground_truth_paths = []
            converted.append(sample)
            if index % 100 == 0 or index == len(samples):
                print(f"[V2-OSM build] split={split} {index}/{len(samples)}", flush=True)
        output_splits[split] = converted
        if not converted:
            raise ValueError(
                f"OSM coverage exclusion removed every sample from split {split}"
            )
        osm_ids_by_split[split] = split_osm_ids
        write_samples(protocol / "splits" / f"{split}.json", converted)

    if osm_ids_by_split["train"] & osm_ids_by_split["ood_location_test"]:
        raise ValueError("OSM element leakage between train and OOD")
    events = [_event(sample) for sample in output_splits["ood_location_test"]]
    write_json(protocol / "ood_replan_events.json", run_metadata({
        "protocol": "paired simulated event-driven map-replanning stress test on V2-OSM",
        "events": events, "prohibited_claim": "real-flight dynamic validation",
    }))
    raw_hashes = {
        path.name: sha256_file(path) for path in sorted(raw_dir.glob("*.json"))
    }
    audit = run_metadata({
        "protocol": "V2 OSM-assisted semantic-map grounding",
        "status": "COMPLETE",
        "source": "OpenStreetMap via Overpass API",
        "attribution": "© OpenStreetMap contributors",
        "license": "ODbL 1.0",
        "license_url": "https://www.openstreetmap.org/copyright",
        "split_counts": {key: len(value) for key, value in output_splits.items()},
        "original_split_counts": {
            key: len(value) for key, value in split_samples.items()
        },
        "osm_coverage_excluded_image_ids": sorted(excluded_images),
        "osm_coverage_excluded_scenario_ids": excluded_scenarios,
        "coverage_exclusion_rule": (
            "Exclude every scenario from any image whose OSM response contains "
            "no eligible target; apply consistently across all splits without "
            "coordinate fabrication or location reassignment."
        ),
        "image_ids": {key: sorted(value) for key, value in image_sets.items()},
        "raw_osm_sha256": raw_hashes,
        "target_count_reductions": reductions,
        "random_coordinate_fallbacks": 0,
        "instruction_target_marker_contract": "validated during protocol build",
        "train_ood_image_overlap": sorted(
            image_sets["train"] & image_sets["ood_location_test"]
        ),
        "train_ood_osm_id_overlap": sorted(
            osm_ids_by_split["train"] & osm_ids_by_split["ood_location_test"]
        ),
        "imagery_license_action_required": (
            "Verify and cite the independent licence/provenance of the satellite "
            "images before public redistribution; OSM attribution does not cover imagery."
        ),
    })
    write_json(audit_path, audit)
    (protocol / "OSM_ATTRIBUTION.txt").write_text(
        "Map data © OpenStreetMap contributors, available under the ODbL 1.0.\n"
        "https://www.openstreetmap.org/copyright\n", encoding="utf-8",
    )
    print(f"V2-OSM PROTOCOL PASSED | output={protocol}", flush=True)


if __name__ == "__main__":
    main()
