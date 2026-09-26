"""Five-seed, test-time intervention study for the learned availability signal.

This is intentionally a *planner-calibration protocol*, not a replacement for
the primary Exp1 benchmark.  Every arm reuses the same strictly loaded, trained
``ours`` fuser and task decomposer for a given seed.  Only the scalar
availability signal supplied to the B1 planner-calibration path is replaced.

The protocol tests four prespecified signal arms within four prespecified input
availability settings: learned, fixed 0.5, observed modality count M/4, and a
deterministic scenario-level random signal.  It never uses targets, expert
paths, labels, or evaluation metrics to construct a control signal.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from configs.experiment_config import get_default_config
from data.scenario_schema import ModalityType, ScenarioSample
from evaluation.ablation_config import AblationGroupID, get_ablation_config_by_id
from models.fusion.multimodal_fuser import MultimodalFuser
from models.planner.ablation_planner import AblationPlanner
from models.reasoning.task_decomposer import TaskDecomposer
from beta_coupling_protocol import (
    AVAILABILITY_BINS,
    BETA_ARMS,
    PRIMARY_AVAILABILITY_CONDITIONS,
    PROTOCOL_VERSION,
    SEEDS,
)
from submission_common import (
    METRICS,
    evaluate,
    load_checkpoint,
    mean_metrics,
    read_samples,
    run_metadata,
    sha256_file,
    write_json,
    write_rows_csv,
)
from utils.offline_config import setup_offline_environment
from utils.seed_manager import set_global_seed


def _materialize_audio_encoder(fuser: MultimodalFuser) -> None:
    encoder = getattr(fuser, "audio_encoder", None)
    if encoder is not None and hasattr(encoder, "_load_model"):
        encoder._load_model()


def _random_signal(seed: int, scenario_id: str) -> float:
    """Deterministic U[0,1) control; no global RNG or outcome is consulted."""
    payload = f"{PROTOCOL_VERSION}|{seed}|{scenario_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(2**64)


def _initial_astar_radius(beta: float, n_modalities: int) -> int:
    """The declared B1 calibration-to-safety-margin mapping."""
    return 25 if n_modalities == 1 else int(25 + 8 * min(beta + 0.3, 1.0))


class _AvailabilitySignalInterceptor:
    """Replace only the availability scalar emitted by an already loaded fuser."""

    def __init__(self, fuser: Any, beta_arm: str, seed: int) -> None:
        self._fuser = fuser
        self.beta_arm = beta_arm
        self.seed = int(seed)
        self._scenario_id: str | None = None
        self._n_modalities: int | None = None
        self.call_count = 0
        self.learned_beta = float("nan")
        self.beta_used = float("nan")
        self.expected_initial_radius: int | None = None

    def begin(self, scenario: ScenarioSample) -> None:
        self._scenario_id = str(scenario.scenario_id)
        self._n_modalities = len(scenario.modalities)
        self.call_count = 0
        self.learned_beta = float("nan")
        self.beta_used = float("nan")
        self.expected_initial_radius = None

    def _control_value(self, learned: float) -> float:
        if self._scenario_id is None or self._n_modalities is None:
            raise RuntimeError("Availability signal was requested without scenario context")
        if self.beta_arm == "learned":
            return learned
        if self.beta_arm == "fixed_0_5":
            return 0.5
        if self.beta_arm == "observed_count":
            return float(self._n_modalities) / 4.0
        if self.beta_arm == "random_hash":
            return _random_signal(self.seed, self._scenario_id)
        raise ValueError(f"Unknown beta arm: {self.beta_arm}")

    def __call__(self, *args: Any, **kwargs: Any):
        output = self._fuser(*args, **kwargs)
        if not kwargs.get("return_bias", False):
            return output
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("Fuser did not return (fused, beta) for the calibration intervention")
        fused, raw_beta = output
        learned = float(raw_beta.item() if hasattr(raw_beta, "item") else raw_beta)
        if not math.isfinite(learned) or not 0.0 <= learned <= 1.0:
            raise RuntimeError("The trained fusion availability head returned an invalid beta")
        used = self._control_value(learned)
        self.learned_beta = learned
        self.beta_used = float(used)
        if self._n_modalities is None:
            raise RuntimeError("Missing modality count for calibration intervention")
        self.expected_initial_radius = _initial_astar_radius(used, self._n_modalities)
        self.call_count += 1
        if torch.is_tensor(raw_beta):
            return fused, torch.full_like(raw_beta, float(used))
        return fused, torch.tensor([[float(used)]], device=getattr(fused, "device", None))


class BetaControlPlanner(AblationPlanner):
    """B1 planner whose only intervention is the fuser-emitted availability scalar."""

    def __init__(self, *args: Any, beta_arm: str, seed: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.beta_arm = beta_arm
        self.seed = int(seed)
        self.calibration_records: List[Dict[str, Any]] = []
        if getattr(self, "fuser", None) is None:
            raise RuntimeError("B1 planner has no fuser to intercept")
        self._availability_interceptor = _AvailabilitySignalInterceptor(
            self.fuser, beta_arm, seed
        )
        # B1 uses cross-modal attention, hence _get_active_fuser() resolves
        # to self.fuser in both the legacy and current planner implementations.
        self.fuser = self._availability_interceptor
        self._current_astar_radii: List[int] = []

    def plan(self, scenario: ScenarioSample):
        self._current_astar_radii = []
        self._availability_interceptor.begin(scenario)
        result = super().plan(scenario)
        if (self._availability_interceptor.call_count < 1
                or self._availability_interceptor.expected_initial_radius is None):
            raise RuntimeError(
                "AblationPlanner.plan() did not request the fuser beta; "
                "refusing to record a post-hoc calibration value"
            )
        if not math.isfinite(self._availability_interceptor.learned_beta):
            raise RuntimeError("The learned beta was not captured during this plan")
        if not self._current_astar_radii:
            raise RuntimeError(
                f"Planner did not invoke A* for scenario {scenario.scenario_id}; "
                "cannot audit the applied safety margin"
            )
        radii = self._current_astar_radii
        if int(radii[0]) != self._availability_interceptor.expected_initial_radius:
            raise RuntimeError(
                "The actual initial A* radius differs from the radius derived from beta; "
                "refusing to emit an invalid calibration-control record"
            )
        self.calibration_records.append({
            "scenario_id": str(scenario.scenario_id),
            "learned_beta": float(self._availability_interceptor.learned_beta),
            "beta_used": float(self._availability_interceptor.beta_used),
            "n_modalities_delivered": len(scenario.modalities),
            "expected_initial_min_obstacle_radius_px": self._availability_interceptor.expected_initial_radius,
            "min_obstacle_radius_px": int(radii[0]),
            "initial_min_obstacle_radius_px": int(radii[0]),
            "final_min_obstacle_radius_px": int(radii[-1]),
            "astar_fallback_to_radius25": bool(len(radii) > 1 and radii[-1] == 25),
        })
        return result

    def _run_astar(self, image_id, image_path, targets_pct, obstacles_pct, min_rad=25):
        """Enforce and record the beta-derived first A* safety margin."""
        if not self._current_astar_radii:
            applied_radius = self._availability_interceptor.expected_initial_radius
            if applied_radius is None:
                raise RuntimeError("A* was called before the availability scalar was captured")
        else:
            # Preserve the base planner's explicit radius-25 fallback branch.
            applied_radius = int(min_rad)
        self._current_astar_radii.append(int(applied_radius))
        return super()._run_astar(
            image_id, image_path, targets_pct, obstacles_pct, min_rad=int(applied_radius)
        )


def _parse_csv(value: str, allowed: Iterable[str], flag: str) -> tuple[str, ...]:
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    if not selected or len(set(selected)) != len(selected) or any(item not in set(allowed) for item in selected):
        raise ValueError(f"{flag} must be a unique, nonempty subset of: {','.join(allowed)}")
    return selected


def _usable_input_sample(sample: ScenarioSample) -> tuple[ScenarioSample, list[str]]:
    """Remove modalities whose declared source asset is absent or unreadable."""
    cleaned = copy.deepcopy(sample)
    available = {ModalityType.TEXT}
    missing: list[str] = []
    for modality, field in (
        (ModalityType.VOICE, "audio_path"),
        (ModalityType.GESTURE, "gesture_image_path"),
        (ModalityType.ANNOTATION, "annotation_image_path"),
    ):
        value = getattr(cleaned, field, None)
        usable = modality in cleaned.modalities and bool(value) and os.path.isfile(str(value))
        if usable:
            available.add(modality)
        else:
            setattr(cleaned, field, None)
            if modality in cleaned.modalities:
                missing.append(field)
    cleaned.modalities = [m for m in cleaned.modalities if m in available]
    if ModalityType.TEXT not in cleaned.modalities:
        cleaned.modalities.insert(0, ModalityType.TEXT)
    return cleaned, missing


def _balanced_smoke_panel(samples: Sequence[ScenarioSample], limit: int) -> list[ScenarioSample]:
    """Choose a deterministic small panel spread over observed modality counts."""
    buckets: Dict[int, list[ScenarioSample]] = {count: [] for count in range(1, 5)}
    for sample in samples:
        buckets[max(1, min(4, len(sample.modalities)))].append(sample)
    selected: list[ScenarioSample] = []
    for count in range(1, 5):
        if buckets[count] and len(selected) < limit:
            selected.append(buckets[count][0])
    used = {str(sample.scenario_id) for sample in selected}
    for sample in samples:
        if len(selected) >= limit:
            break
        if str(sample.scenario_id) not in used:
            selected.append(sample)
            used.add(str(sample.scenario_id))
    return selected


def _load_planner(model_root: Path, seed: int, beta_arm: str) -> tuple[BetaControlPlanner, Dict[str, str]]:
    base_cfg, _, model_cfg, _, _ = get_default_config()
    device = base_cfg.device
    checkpoint_dir = model_root / "models" / "ours" / f"seed_{seed}"
    fuser = MultimodalFuser(
        fusion_dim=model_cfg.fusion_dim,
        attention_heads=model_cfg.attention_heads,
        attention_layers=model_cfg.attention_layers,
        ffn_dim=model_cfg.ffn_dim,
        audio_model=model_cfg.audio_model,
        gesture_backbone=model_cfg.gesture_backbone,
        text_model=model_cfg.text_model,
    ).to(device)
    _materialize_audio_encoder(fuser)
    decomposer = TaskDecomposer(
        d_model=model_cfg.fusion_dim,
        n_layers=model_cfg.decomposer_layers,
        max_subtasks=model_cfg.max_subtasks,
        max_seq_len=100,
    ).to(device)
    fusion_path = load_checkpoint(fuser, checkpoint_dir, "best_fusion_model", device)
    decomposer_path = load_checkpoint(decomposer, checkpoint_dir, "decomposer_rl", device)
    fuser.eval()
    decomposer.eval()
    planner = BetaControlPlanner(
        benchmark_dir=base_cfg.benchmark_dir,
        fuser=fuser,
        decomposer=decomposer,
        ablation_config=get_ablation_config_by_id(AblationGroupID.B1_ENHANCED_FULL),
        device=device,
        beta_arm=beta_arm,
        seed=seed,
    )
    return planner, {
        "fusion_checkpoint": str(fusion_path),
        "decomposer_checkpoint": str(decomposer_path),
    }


def _assert_intervention(arm: str, records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise RuntimeError(f"No calibration audit records for beta arm {arm}")
    used = [float(record["beta_used"]) for record in records]
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in used):
        raise RuntimeError(f"Invalid beta output in arm {arm}")
    if any(not math.isfinite(float(record["learned_beta"])) for record in records):
        raise RuntimeError(f"The trained fusion beta was not archived for arm {arm}")
    for record in records:
        if int(record["expected_initial_min_obstacle_radius_px"]) != int(record["initial_min_obstacle_radius_px"]):
            raise RuntimeError("Expected and actual initial A* radii differ")
    if arm == "fixed_0_5" and any(abs(value - 0.5) > 1e-9 for value in used):
        raise RuntimeError("fixed_0_5 intervention did not deliver beta=0.5")
    if arm == "learned" and any(
        abs(float(record["beta_used"]) - float(record["learned_beta"])) > 1e-9
        for record in records
    ):
        raise RuntimeError("learned arm did not deliver the trained fusion beta")
    if arm == "observed_count":
        for record in records:
            expected = float(record["n_modalities_delivered"]) / 4.0
            if abs(float(record["beta_used"]) - expected) > 1e-9:
                raise RuntimeError("observed_count intervention did not deliver M/4")
    if arm == "random_hash" and len(records) > 1 and len({round(value, 12) for value in used}) < 2:
        raise RuntimeError("random_hash intervention produced no scenario variation")


def _assert_m1_negative_control(rows: Sequence[Mapping[str, Any]]) -> None:
    """M1 has a fixed 25-px branch, so all non-beta outputs must match."""
    grouped: Dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["availability_condition"] != "M1":
            continue
        grouped.setdefault((int(row["training_seed"]), str(row["scenario_id"])), []).append(row)
    fields = METRICS + ("initial_min_obstacle_radius_px", "final_min_obstacle_radius_px")
    for key, group in grouped.items():
        if len(group) != len(BETA_ARMS):
            raise RuntimeError(f"M1 negative-control arms are incomplete for {key}")
        for field in fields:
            values = [float(row[field]) for row in group]
            if max(values) - min(values) > 1e-12:
                raise RuntimeError(
                    f"M1 negative control changed {field} across beta arms for {key}; "
                    "evaluation is not isolated"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Five-seed availability-signal planner-control experiment")
    parser.add_argument("--run-root", required=True, help="Output run root; must not contain this worker output.")
    parser.add_argument("--model-run-root", help="Root that contains models/ours/seed_<seed>; defaults to --run-root.")
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--beta-arms", default=",".join(BETA_ARMS))
    parser.add_argument("--output-dir", help="Explicit empty worker directory.")
    parser.add_argument("--max-test-samples", type=int, help="Positive limit permitted only for --smoke.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--server", action="store_true")
    args = parser.parse_args()

    if args.seed not in SEEDS:
        raise ValueError(f"seed must be one of the prespecified seeds: {SEEDS}")
    if args.max_test_samples is not None and (not args.smoke or args.max_test_samples < 1):
        raise ValueError("--max-test-samples is a positive smoke-only option")
    if args.smoke and args.seed != 42:
        raise ValueError("The preflight protocol is fixed to seed 42")

    arms = _parse_csv(args.beta_arms, BETA_ARMS, "--beta-arms")
    if args.smoke and set(arms) != set(BETA_ARMS):
        raise ValueError("Smoke must exercise every beta arm")

    root = Path(args.run_root).resolve()
    model_root = Path(args.model_run_root).resolve() if args.model_run_root else root
    output = Path(args.output_dir).resolve() if args.output_dir else root / "results" / "beta_coupling_workers" / f"seed_{args.seed}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite beta-control output: {output}")
    output.mkdir(parents=True)

    setup_offline_environment(server_mode=args.server)
    set_global_seed(args.seed)
    source_test = read_samples(args.test_file)
    test, missing_assets = [], []
    for sample in source_test:
        cleaned, missing = _usable_input_sample(sample)
        test.append(cleaned)
        missing_assets.extend(f"{sample.scenario_id}:{field}" for field in missing)
    if args.smoke and args.max_test_samples:
        test = _balanced_smoke_panel(test, args.max_test_samples)
    if not test:
        raise ValueError("Test set is empty")
    availability_counts = {
        f"M{count}": sum(len(sample.modalities) == count for sample in test)
        for count in range(1, 5)
    }
    if not any(availability_counts[f"M{count}"] for count in range(2, 5)):
        raise RuntimeError("No test scenarios contain a usable non-text modality; beta-active control cannot be evaluated")
    started = time.monotonic()
    print(
        f"[Beta-control] start | seed={args.seed} | samples={len(test)} | "
        f"availability_counts={availability_counts} | arms={arms} | smoke={args.smoke}", flush=True,
    )

    all_rows: list[Dict[str, Any]] = []
    worker_summary: Dict[str, Dict[str, Any]] = {}
    checkpoint_audit: Dict[str, Dict[str, str]] = {}
    for arm in arms:
        print(f"[Beta-control] evaluating | beta_arm={arm}", flush=True)
        # Each arm starts from the same RNG state.  Thus M1 (where beta cannot
        # alter the 25-px branch) is an exact negative control rather than a
        # run-order comparison.
        set_global_seed(args.seed)
        planner, audit = _load_planner(model_root, args.seed, arm)
        checkpoint_audit[arm] = audit
        rows = evaluate(planner, test, planner.benchmark_dir)
        if len(rows) != len(planner.calibration_records):
            raise RuntimeError("Calibration audit count differs from evaluator rows")
        _assert_intervention(arm, planner.calibration_records)
        for row, record, sample in zip(rows, planner.calibration_records, test):
            if str(row["scenario_id"]) != str(record["scenario_id"]):
                raise RuntimeError("Evaluator/audit scenario ordering mismatch")
            count = len(sample.modalities)
            row.update(record)
            row.update({
                "availability_condition": f"M{count}",
                "beta_arm": arm,
                "training_seed": args.seed,
                "delivered_modalities": "+".join(sorted(modality.value for modality in sample.modalities)),
            })
            if int(row["n_modalities_delivered"]) != count:
                raise RuntimeError("Recorded delivered-modality count differs from sanitized sample")
        for count in range(1, 5):
            condition = f"M{count}"
            group_rows = [row for row in rows if row["availability_condition"] == condition]
            if not group_rows:
                continue
            worker_summary[f"{condition}/{arm}"] = {
                "metrics": mean_metrics(group_rows),
                "mean_beta_used": sum(float(row["beta_used"]) for row in group_rows) / len(group_rows),
                "mean_min_obstacle_radius_px": sum(float(row["min_obstacle_radius_px"]) for row in group_rows) / len(group_rows),
                "astar_fallback_rows": sum(bool(row["astar_fallback_to_radius25"]) for row in group_rows),
                "n_rows": len(group_rows),
            }
        all_rows.extend(rows)
        print(f"[Beta-control] complete | beta_arm={arm} | rows={len(rows)}", flush=True)

    expected_rows = len(test) * len(arms)
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, received {len(all_rows)}")
    _assert_m1_negative_control(all_rows)
    write_rows_csv(output / "per_scenario_results.csv", all_rows)
    write_json(output / "summary.json", run_metadata({
        "experiment": "Five-seed availability-signal planner-control intervention",
        "protocol_version": PROTOCOL_VERSION,
        "smoke": bool(args.smoke),
        "smoke_test": bool(args.smoke),
        "training_seed": args.seed,
        "n_test_scenarios": len(test),
        "source_test_scenarios": len(source_test),
        "test_file": str(Path(args.test_file).resolve()),
        "test_sha256": sha256_file(args.test_file),
        "model_run_root": str(model_root),
        "availability_conditions": availability_counts,
        "beta_arms": list(arms),
        "checkpoint_audit": checkpoint_audit,
        "implementation_audit": {
            "fuser_beta_interceptor_required": True,
            "initial_astar_radius_must_equal_beta_derived_radius": True,
            "rng_reset_before_each_arm": True,
            "runner_sha256": sha256_file(__file__),
            "ablation_planner_sha256": sha256_file(PROJECT_ROOT / "models" / "planner" / "ablation_planner.py"),
        },
        "worker_summary": worker_summary,
        "planner_configuration": "B1 enhanced full calibration path; same trained checkpoints for all arms within seed",
        "planner_input": "shared ScenarioSample target-coordinate interface across all beta arms",
        "coordinate_fallback_rows": "not_applicable_to_this_planner-control experiment",
        "label_leakage_gate": "same planner inputs and targets supplied to every arm; this experiment does not audit coordinate inference",
        "missing_input_assets_removed": len(missing_assets),
        "missing_input_asset_examples": missing_assets[:20],
        "beta_source_audit": {
            "learned": "trained fuser availability head only",
            "fixed_0_5": "constant 0.5; no labels or outcomes",
            "observed_count": "delivered input count M divided by 4; no labels or outcomes",
            "random_hash": "SHA-256(seed, scenario_id) deterministic U[0,1); no labels or outcomes",
        },
        "independent_unit": "one previously trained full model per seed; test scenarios are paired repeated measurements",
        "primary_analysis": "TCR contrasts learned minus each control, stratified by observed usable modality count M; M1 is an implementation negative control, and available M2-M4 strata form the prespecified corrected family",
        "claim_allowed": "whether the B1 planner's beta-to-initial-A* safety-margin path benefits from the learned availability signal under this controlled protocol",
        "claim_forbidden": "a replacement for Exp1, a naturalistic operator study, or evidence about unimplemented replan-scope behaviour",
    }))
    print(
        f"[Beta-control] completed | output={output} | "
        f"elapsed={(time.monotonic() - started) / 60.0:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
