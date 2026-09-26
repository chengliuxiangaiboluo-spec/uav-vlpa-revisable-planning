"""Create a SHA-256 provenance manifest for the frozen UAV-VLPA* reference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, root: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--model-dir", default=str(ROOT / "Weights" / "molmo-7B-O-bnb-4bit"))
    parser.add_argument("--output")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    result_dir = Path(args.result_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    output = Path(args.output).resolve() if args.output else result_dir / "reproducibility_manifest.json"

    required_result_files = [
        result_dir / "summary.json",
        result_dir / "per_scenario_results.csv",
        result_dir / "molmo_response_audit.csv",
    ]
    for path in required_result_files:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing required formal result artifact: {path}")
    if not model_dir.is_dir():
        raise RuntimeError(f"missing model directory: {model_dir}")

    source_files = [
        PACKAGE / "run_uav_vlpa_molmo_v2.py",
        PACKAGE / "preflight_uav_vlpa_molmo_v2.py",
        PACKAGE / "archive_uav_vlpa_molmo_manifest.py",
        PACKAGE / "33a_uav_vlpa_molmo_full.sh",
        PACKAGE / "main_baseline_strict_models.py",
        PACKAGE / "run_v2_grounding_worker.py",
    ]
    protocol_files = [
        run_root / "v2_osm_protocol" / "audit.json",
        run_root / "v2_osm_protocol" / "splits" / "ood_location_test.json",
    ]
    for path in [*source_files, *protocol_files]:
        if not path.is_file():
            raise RuntimeError(f"missing provenance input: {path}")

    model_files = [path for path in sorted(model_dir.rglob("*")) if path.is_file()]
    if not model_files:
        raise RuntimeError(f"no regular files found under model directory: {model_dir}")

    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    expected_rows = int(summary.get("n_test_scenarios", -1))
    if expected_rows < 1:
        raise RuntimeError("summary has no positive n_test_scenarios")
    if summary.get("parser_exact_match_rows") != expected_rows:
        raise RuntimeError("summary does not certify exact parser matches for every scenario")
    if summary.get("parser_failure_rows") != 0 or summary.get("coordinate_fallback_rows") != 0:
        raise RuntimeError("summary reports a parser failure or coordinate fallback")
    if summary.get("label_leakage_gate") != "passed" or summary.get("planner_input") != "predicted_coordinates_only":
        raise RuntimeError("summary fails the label-leakage or planner-input gate")

    with (result_dir / "molmo_response_audit.csv").open("r", encoding="utf-8", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    if len(audit_rows) != expected_rows:
        raise RuntimeError("audit row count does not equal n_test_scenarios")
    required_audit_fields = {
        "scenario_id", "requested_target_slots", "parsed_coordinate_count",
        "parser_exact_match", "raw_molmo_response_sha256",
        "coordinate_fallback_used", "planner_input",
        "label_coordinates_used_only_for_scoring",
    }
    if not audit_rows or not required_audit_fields.issubset(audit_rows[0]):
        raise RuntimeError("audit CSV is missing required strict-provenance fields")
    for row in audit_rows:
        if row["parser_exact_match"].strip().lower() != "true":
            raise RuntimeError(f"non-exact parse in audit: {row['scenario_id']}")
        if int(row["parsed_coordinate_count"]) != int(row["requested_target_slots"]):
            raise RuntimeError(f"coordinate-count mismatch in audit: {row['scenario_id']}")
        if row["coordinate_fallback_used"].strip().lower() != "false":
            raise RuntimeError(f"coordinate fallback in audit: {row['scenario_id']}")
        if row["planner_input"] != "predicted_coordinates_only":
            raise RuntimeError(f"invalid planner input in audit: {row['scenario_id']}")
        if row["label_coordinates_used_only_for_scoring"].strip().lower() != "true":
            raise RuntimeError(f"label leakage in audit: {row['scenario_id']}")

    payload = {
        "manifest_type": "uav_vlpa_molmo_frozen_zero_shot_provenance",
        "sha256_algorithm": "SHA-256",
        "scope": {
            "run_root": str(run_root),
            "result_dir": str(result_dir),
            "model_dir": str(model_dir),
        },
        "result_artifacts": [record(path, result_dir) for path in required_result_files],
        "inference_source": [record(path, ROOT) for path in source_files],
        "v2_protocol": [record(path, run_root) for path in protocol_files],
        "model_snapshot": [record(path, model_dir) for path in model_files],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"UAV-VLPA MOLMO MANIFEST WRITTEN | files={len(model_files)} | output={output}")


if __name__ == "__main__":
    main()
