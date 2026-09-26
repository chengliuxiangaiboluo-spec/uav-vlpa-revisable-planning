"""Create auditable V2 labels and simulated event-driven replan cases."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from submission_common import read_samples, run_metadata, sha256_file, write_json


def event_for(sample):
    """A deterministic synthetic constraint event, disclosed as simulation.

    The event is inserted on the direct home-to-first-target corridor.  It is
    generated from held-out scenario metadata only; no train example is used.
    It must be described as a simulated map-event stress test, not real flight
    telemetry.
    """
    digest = hashlib.sha256(sample.scenario_id.encode()).digest()
    return {
        "scenario_id": sample.scenario_id,
        "trigger_fraction": round(0.30 + (digest[0] / 255.0) * 0.10, 6),
        "obstacle_fraction": round(0.58 + (digest[1] / 255.0) * 0.12, 6),
        "clearance_radius_pct": 4.0,
        "placement_rule": "future waypoint of the model pre-event route",
        "generation": "deterministic paired synthetic route-blocking event v2",
    }


def grounding_manifest(samples, split_path):
    return {
        "split_file": str(split_path),
        "split_sha256": sha256_file(split_path),
        "n_samples": len(samples),
        "supervision": "image + instruction -> ordered target coordinates",
        "labels_source": "ScenarioSample.targets benchmark annotations",
        "records": [
            {
                "scenario_id": sample.scenario_id,
                "image_id": sample.image_id,
                "targets": [
                    {
                        "name": target.name,
                        "type": target.target_type,
                        "coordinates_percent": list(target.coordinates_percent),
                    }
                    for target in sample.targets
                ],
            }
            for sample in samples
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve(); assets = root / "v2_protocol"; assets.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val_location", "id_test", "ood_location_test"):
        split_path = root / "splits" / f"{split}.json"
        samples = read_samples(split_path)
        manifest = grounding_manifest(samples, split_path)
        manifest["split"] = split
        write_json(assets / f"grounding_{split}.json", run_metadata(manifest))
    ood = read_samples(root / "splits" / "ood_location_test.json")
    events = [x for x in (event_for(sample) for sample in ood) if x is not None]
    write_json(assets / "ood_replan_events.json", run_metadata({
        "protocol": "paired simulated event-driven map-replanning stress test",
        "event_materialisation": "current position and obstacle are selected from fixed fractions of each seed model's pre-event route",
        "events": events,
        "prohibited_claim": "real-world dynamic-flight validation",
    }))
    print(f"[V2 prepare] complete | assets={assets} | events={len(events)}", flush=True)


if __name__ == "__main__":
    main()
