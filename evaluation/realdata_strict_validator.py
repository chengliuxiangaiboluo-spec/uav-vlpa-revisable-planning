"""Strict, auditable real-flight waypoint-conditioned validation.

This module intentionally keeps the mission waypoint manifest (targets.csv)
separate from the flight-trajectory reference.  It does *not* claim visual
grounding from the raw image; it evaluates whether supplied interaction inputs
change planning under a common waypoint-conditioned protocol.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from evaluation.realdata_validator import RealDataValidator, TempBenchmarkBuilder


class StrictRealDataValidator(RealDataValidator):
    """A paired cohort with fail-closed input loading and per-plan evidence."""

    def __init__(self, *args, max_scenarios=None, benchmark_root=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_scenarios = max_scenarios
        self.strict_manifest = {}
        # Parallel modality shards must never create/resize the same temporary
        # JPEG or coordinate CSV at the same time.
        if benchmark_root is not None:
            self.temp_builder = TempBenchmarkBuilder(str(benchmark_root))

    def _get_planner(self, use_baseline, benchmark_dir, home_pct):
        if use_baseline:
            # Retained only as a system-level reference; it is never the
            # control used for multimodal input-effect tests.
            return super()._get_planner(use_baseline, benchmark_dir, home_pct)
        if self._enhanced_planner is None:
            from models.planner.realworld_planner import RealWorldPlanner
            self._enhanced_planner = RealWorldPlanner(
                benchmark_dir=benchmark_dir, fuser=self.fuser,
                decomposer=self.decomposer, device=self.device,
                home_coordinate=home_pct,
            )
            self._enhanced_planner.strict_input_audit = True
        else:
            self._update_planner_paths(self._enhanced_planner, benchmark_dir, home_pct)
        return self._enhanced_planner

    def _cohort(self):
        included, records = [], []
        for scenario in self.loader.load_all_scenarios():
            # metadata.json contains repeated labels (for example RW_001 in
            # different numbered directories).  A paired experiment needs a
            # one-to-one observation identifier, so the immutable realdata
            # directory name is the canonical strict-protocol ID.
            metadata_scenario_id = scenario.scenario_id
            directory_id = Path(scenario.data_dir).name
            scenario.scenario_id = f"realdata_dir_{directory_id}"
            reasons = []
            if not scenario.text_instruction:
                reasons.append("missing_text")
            if not scenario.audio_path:
                reasons.append("missing_voice")
            if not scenario.gesture_path:
                reasons.append("missing_gesture")
            if not scenario.annotation_path:
                reasons.append("missing_annotation")
            if not scenario.targets:
                reasons.append("missing_targets")
            trajectory, _ = self.loader.parse_flight_trajectory(scenario)
            if len(trajectory) < 5:
                reasons.append("missing_or_unparseable_flight_trajectory")
            records.append({
                "scenario_id": scenario.scenario_id,
                "source_metadata_scenario_id": metadata_scenario_id,
                "directory_id": directory_id,
                "included": not reasons,
                "exclusion_reasons": reasons,
                "data_dir": scenario.data_dir,
            })
            if not reasons:
                included.append(scenario)
        included.sort(key=lambda item: item.scenario_id)
        if self.max_scenarios:
            included = included[:self.max_scenarios]
            keep = {item.scenario_id for item in included}
            for record in records:
                if record["included"] and record["scenario_id"] not in keep:
                    record["included"] = False
                    record["exclusion_reasons"] = ["outside_deterministic_limit"]
        self.strict_manifest = {
            "protocol": "strict_real_flight_waypoint_conditioned_v1",
            "n_discovered": len(records),
            "n_included": len(included),
            "records": records,
            "claim_allowed": "real-flight-trajectory comparison under a disclosed waypoint-conditioned multimodal input protocol",
            "claim_forbidden": "end-to-end visual target/obstacle grounding from raw images",
        }
        return included

    def evaluate(self, combo_filter=None):
        original_loader = self.loader.load_all_scenarios
        cohort = self._cohort()
        if not cohort:
            raise RuntimeError("Strict cohort is empty; inspect strict_manifest exclusions")
        self.loader.load_all_scenarios = lambda: cohort
        try:
            summary = super().evaluate(combo_filter=combo_filter)
        finally:
            self.loader.load_all_scenarios = original_loader
        # The parent validator stores requested modalities. Replace that with
        # evidence returned by RealWorldPlanner and fail closed on omissions.
        for scenario in summary.per_scenario_results:
            for combo, result in scenario["modalities"].items():
                audit = result.get("input_audit", {})
                if combo != "text_only_baseline":
                    requested = set(audit.get("requested_modalities", []))
                    loaded = set(audit.get("loaded_modalities", []))
                    if requested != loaded or audit.get("missing_modalities"):
                        raise RuntimeError(
                            f"Input audit mismatch for {scenario['scenario_id']} / {combo}: {audit}"
                        )
        summary.strict_manifest = self.strict_manifest
        return summary

    @staticmethod
    def write_manifest(summary, output_dir):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        manifest = getattr(summary, "strict_manifest", {})
        (output / "strict_cohort_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        audits = []
        for scenario in summary.per_scenario_results:
            for combo, result in scenario["modalities"].items():
                audits.append({
                    "scenario_id": scenario["scenario_id"],
                    "modality_combo": combo,
                    "planning_failed": result.get("planning_failed", False),
                    "input_audit": result.get("input_audit", {}),
                })
        (output / "strict_input_audit.json").write_text(
            json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8"
        )
