"""Fail-fast static preflight for expensive submission jobs.

This intentionally performs no training or model download.  It validates the
split, every required checkpoint identity, package syntax, Slurm-script safety
requirements and the Exp2 task-trace wrapper before GPU jobs are submitted.
"""

from __future__ import annotations

import argparse
import ast
import json
import py_compile
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# This module deliberately imports no project runtime module.  Login nodes do
# not necessarily expose CUDA/cuDNN, while a static preflight must work before
# a GPU allocation.  Keep the submission seed protocol local and inspect the
# Exp2 wrapper source structurally below.
DEFAULT_SEEDS = (42, 123, 456, 789, 2024)


def fail(errors, message):
    errors.append(message)


def main():
    parser = argparse.ArgumentParser(description="Fail-fast submission preflight")
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--target", choices=("exp2", "exp3", "exp4", "exp5", "v2", "all"),
        default="exp2",
    )
    parser.add_argument("--allow-existing-output", action="store_true")
    args = parser.parse_args()
    root, errors = Path(args.run_root).resolve(), []
    # Compile every package source: catches syntax errors before Slurm wait time.
    for source in PACKAGE.glob("*.py"):
        try:
            py_compile.compile(str(source), doraise=True)
        except Exception as exc:
            fail(errors, f"Python compile failure: {source.name}: {exc}")
    # Every operational Slurm script must request one GPU, unbuffer output, and
    # reject an omitted RUN_ROOT instead of silently falling back to old runs.
    for script in PACKAGE.glob("*.sh"):
        if script.name == "run_all_submission.sh":
            # This is a local convenience script, not a Slurm entry point.
            continue
        text = script.read_text(encoding="utf-8")
        if script.name in {
            "07_exp5_minus_dynamic_replan.sh",
            "10_v2_grounding_and_replan.sh",
        }:
            continue  # intentionally blocked; see its companion document.
        if "#SBATCH --gpus=1" not in text:
            fail(errors, f"{script.name}: missing #SBATCH --gpus=1")
        if "RUN_ROOT is required" not in text:
            fail(errors, f"{script.name}: does not fail closed when RUN_ROOT is absent")
    split_names = ("train", "val_location", "id_test", "ood_location_test")
    split_ids = {}
    for name in split_names:
        path = root / "splits" / f"{name}.json"
        if not path.is_file():
            fail(errors, f"missing split: {path}"); continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
            ids = {str(row["scenario_id"]) for row in rows}
            if not rows or len(ids) != len(rows):
                fail(errors, f"invalid/duplicate scenario IDs in {name}")
            split_ids[name] = ids
        except Exception as exc:
            fail(errors, f"unreadable split {name}: {exc}")
    if split_ids.get("train", set()) & split_ids.get("id_test", set()):
        fail(errors, "train and ID-test scenario leakage")
    if split_ids.get("train", set()) & split_ids.get("ood_location_test", set()):
        fail(errors, "train and OOD-test scenario leakage")
    # Exp4 requires the ID set to retain seen locations while OOD locations are
    # completely disjoint.  Check this from JSON without importing CUDA code.
    if args.target in {"exp4", "all"}:
        try:
            train_rows = json.loads((root / "splits" / "train.json").read_text(encoding="utf-8"))
            id_rows = json.loads((root / "splits" / "id_test.json").read_text(encoding="utf-8"))
            ood_rows = json.loads((root / "splits" / "ood_location_test.json").read_text(encoding="utf-8"))
            train_images = {int(row["image_id"]) for row in train_rows}
            id_images = {int(row["image_id"]) for row in id_rows}
            ood_images = {int(row["image_id"]) for row in ood_rows}
            if not id_images <= train_images:
                fail(errors, "Exp4 ID-test includes location(s) absent from training")
            if ood_images & train_images:
                fail(errors, "Exp4 OOD-test shares location(s) with training")
        except Exception as exc:
            fail(errors, f"cannot validate Exp4 location protocol: {exc}")
    for seed in DEFAULT_SEEDS:
        base = root / "models" / "ours" / f"seed_{seed}"
        if not (base / "training_metadata.json").is_file():
            fail(errors, f"missing Ours training metadata for seed {seed}")
        if not list(base.glob("best_fusion_model*.pt")):
            fail(errors, f"missing Ours fusion checkpoint for seed {seed}")
        if not list(base.glob("decomposer_rl_epoch*.pt")):
            fail(errors, f"missing Ours decomposer checkpoint for seed {seed}")
    # Static regression test for the exact Exp2 IA bug.  Do not import
    # submission_common here: it imports torch and fails on a CUDA-less login
    # node.  Instead require the wrapper to initialise and copy last_tasks.
    common_source = (PACKAGE / "submission_common.py").read_text(encoding="utf-8")
    required_trace_fragments = (
        "self.last_tasks = []",
        'self.last_tasks = getattr(self.planner, "last_tasks", [])',
    )
    if not all(fragment in common_source for fragment in required_trace_fragments):
        fail(errors, "Exp2 wrapper no longer propagates planner.last_tasks")
    if args.target == "exp2" and not args.allow_existing_output:
        for part in ("part1_full_text", "part2_minus_voice", "part3_minus_gesture", "part4_minus_annotation"):
            target = root / "results" / "exp2_workers_iafix" / part
            if target.exists():
                fail(errors, f"Exp2 output already exists and would make job fail: {target}")
    if args.target == "exp3":
        exp3_source = (PACKAGE / "run_exp3_robustness.py").read_text(encoding="utf-8")
        for fragment in (
            '"voice_snr_10db"', '"gesture_jitter_6px"',
            '"annotation_offset_10px"', '"text_ambiguity"',
            'changed == 0', 'args.smoke', 'args.max_samples',
        ):
            if fragment not in exp3_source:
                fail(errors, f"Exp3 safety regression: missing {fragment!r}")
        if not args.allow_existing_output:
            for part in ("part1_clean_voice", "part2_gesture", "part3_annotation", "part4_text"):
                target = root / "results" / "exp3_workers" / part
                if target.exists():
                    fail(errors, f"Exp3 output already exists and would make job fail: {target}")
    if args.target == "exp4":
        exp4_source = (PACKAGE / "run_exp4_strict_generalization.py").read_text(encoding="utf-8")
        for fragment in (
            'assert_no_scenario_overlap', 'image_ids(id_test) <= image_ids(train)',
            'image_ids(ood_test) & image_ids(train)', 'args.smoke', 'args.max_samples',
        ):
            if fragment not in exp4_source:
                fail(errors, f"Exp4 safety regression: missing {fragment!r}")
        for seed in DEFAULT_SEEDS:
            base = root / "models" / "scal" / f"seed_{seed}"
            if not (base / "training_metadata.json").is_file():
                fail(errors, f"missing SCAL training metadata for seed {seed}")
            if not list(base.glob("best_fusion_model*.pt")):
                fail(errors, f"missing SCAL fusion checkpoint for seed {seed}")
        if not args.allow_existing_output:
            for part in ("part1_ours_id", "part2_ours_ood", "part3_scal_id", "part4_scal_ood"):
                target = root / "results" / "exp4_workers" / part
                if target.exists():
                    fail(errors, f"Exp4 output already exists and would make job fail: {target}")
    if args.target == "exp5":
        exp5_source = (PACKAGE / "run_exp5_architecture_ablation.py").read_text(encoding="utf-8")
        for fragment in (
            '"minus_cross_modal_attention"', '"minus_data_driven_calibration"',
            '"minus_task_decomposition"', '"minus_semantic_constraint"',
            'args.smoke', 'args.max_train_samples', 'checkpoint_dir.exists()',
        ):
            if fragment not in exp5_source:
                fail(errors, f"Exp5 safety regression: missing {fragment!r}")
        variants = (
            "minus_cross_modal_attention", "minus_data_driven_calibration",
            "minus_task_decomposition", "minus_semantic_constraint",
        )
        if not args.allow_existing_output:
            for variant in variants:
                result_dir = root / "results" / "exp5_workers" / variant
                checkpoint_dir = root / "models" / "exp5" / variant
                if result_dir.exists():
                    fail(errors, f"Exp5 result output already exists: {result_dir}")
                if checkpoint_dir.exists():
                    fail(errors, f"Exp5 checkpoint output already exists: {checkpoint_dir}")
    if args.target in {"v2", "all"}:
        protocol_dir = root / "v2_osm_protocol"
        required_protocols = tuple(
            protocol_dir / "splits" / f"{name}.json" for name in split_names
        ) + (protocol_dir / "ood_replan_events.json", protocol_dir / "audit.json")
        for path in required_protocols:
            if not path.is_file():
                fail(errors, f"missing C1 V2-OSM protocol file: {path}")
        audit_path = protocol_dir / "audit.json"
        if audit_path.is_file():
            try:
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if audit.get("status") != "COMPLETE":
                    fail(errors, "V2-OSM audit status is not COMPLETE")
                if audit.get("random_coordinate_fallbacks") != 0:
                    fail(errors, "V2-OSM contains random-coordinate fallbacks")
                if audit.get("train_ood_image_overlap"):
                    fail(errors, "V2-OSM train/OOD image leakage")
                if audit.get("train_ood_osm_id_overlap"):
                    fail(errors, "V2-OSM train/OOD OSM-element leakage")
                if audit.get("attribution") != "© OpenStreetMap contributors":
                    fail(errors, "V2-OSM attribution is missing or altered")
            except Exception as exc:
                fail(errors, f"cannot validate V2-OSM audit: {exc}")
        protocol_split_ids = {}
        for name in split_names:
            path = protocol_dir / "splits" / f"{name}.json"
            if not path.is_file():
                continue
            try:
                rows = json.loads(path.read_text(encoding="utf-8"))
                protocol_split_ids[name] = {str(row["scenario_id"]) for row in rows}
                for row in rows:
                    metadata = row.get("metadata", {})
                    if metadata.get("protocol") != "v2_osm_semantic_map_grounding":
                        fail(errors, f"non-OSM V2 sample: {row.get('scenario_id')}")
                        break
                    if not metadata.get("target_osm_ids"):
                        fail(errors, f"V2 sample has no OSM target IDs: {row.get('scenario_id')}")
                        break
                    if not Path(metadata.get("grounding_image_path", "")).is_file():
                        fail(errors, f"missing semantic map: {row.get('scenario_id')}")
                        break
                    if any(tuple(target.get("coordinates_latlon", (0, 0))) == (0, 0)
                           for target in row.get("targets", [])):
                        fail(errors, f"V2 target lacks geodetic ground truth: {row.get('scenario_id')}")
                        break
                    if name == "ood_location_test" and not row.get("ground_truth_paths"):
                        fail(errors, f"V2 OOD sample lacks expert path: {row.get('scenario_id')}")
                        break
            except Exception as exc:
                fail(errors, f"cannot validate V2-OSM split {name}: {exc}")
        event_path = protocol_dir / "ood_replan_events.json"
        if event_path.is_file():
            try:
                event_record = json.loads(event_path.read_text(encoding="utf-8"))
                events = event_record["events"]
                event_ids = {str(event["scenario_id"]) for event in events}
                if event_ids != protocol_split_ids.get("ood_location_test", set()):
                    fail(errors, "V2 event IDs do not exactly match OOD scenarios")
                for event in events:
                    for key in (
                        "trigger_fraction", "obstacle_fraction",
                        "clearance_radius_pct", "placement_rule",
                    ):
                        if key not in event:
                            fail(errors, f"V2 event missing {key}: {event.get('scenario_id')}")
                            break
                    if "new_obstacles" in event:
                        fail(errors, "C1 must store a route-independent event rule, not a target-derived obstacle")
                        break
            except Exception as exc:
                fail(errors, f"cannot validate V2 event protocol: {exc}")
        worker_source = (PACKAGE / "run_v2_grounding_worker.py").read_text(encoding="utf-8")
        for fragment in (
            '"visual_language_grounding"', '"text_only_grounding"',
            'evaluate_loss', 'best_validation_loss', 'model.eval()',
            '"visual_language_without_replan"', '"visual_language_with_replan"',
            'old_home', 'current_position_pct', 'claim_forbidden',
            'georeference_for_scoring', 'coordinates_latlon = (0.0, 0.0)',
            'post_event_route_nontrivial',
        ):
            if fragment not in worker_source:
                fail(errors, f"V2 safety regression: missing {fragment!r}")
        if "text_only_no_replan" in worker_source:
            fail(errors, "V2 control is confounded: text_only_no_replan is forbidden")
        exp6_source = (PACKAGE / "run_exp6_paper_inspired.py").read_text(encoding="utf-8")
        if "GroundedPlanner, georeference_for_scoring" not in exp6_source:
            fail(errors, "Exp6 is not wired to the split V2 grounder")
        if not args.allow_existing_output:
            final = root / "results" / "v2_osm_grounding_replan"
            if final.exists():
                fail(errors, f"V2 merged output already exists: {final}")
            for condition in ("visual_language_grounding", "text_only_grounding"):
                result = root / "results" / "v2_osm_grounding_workers" / condition
                if result.exists():
                    fail(errors, f"V2 worker output already exists: {result}")
    if errors:
        print("PREFLIGHT FAILED")
        for error in errors: print(f"- {error}")
        raise SystemExit(1)
    print("PREFLIGHT PASSED")
    print("Static dependencies, checkpoints, split isolation and requested experiment safety checks are valid.")


if __name__ == "__main__":
    main()
