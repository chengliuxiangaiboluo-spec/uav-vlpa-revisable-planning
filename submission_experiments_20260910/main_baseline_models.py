"""Transparent, deterministic adapters for the manuscript's three main controls.

These adapters deliberately do not claim official reproductions.  They remove
the legacy synthetic coordinate noise and random completion sampling, so every
reported outcome is determined by the implemented planner and its checkpoints.
"""

from __future__ import annotations

from typing import List

from configs.experiment_config import get_default_config
from models.fusion.baseline_fusers import LPANetFuser
from models.planner.baseline_planner import BaselinePlanner
from models.planner.baseline_systems import ModularUAVSystem
from models.reasoning.baseline_decomposers import SIPSADecomposer
from submission_common import load_checkpoint


class DeterministicTextOnlyPlanner(BaselinePlanner):
    """UAV-VLPA text-only control without synthetic coordinate perturbations."""

    def _compute_biased_targets(self, targets, n_modalities, complexity, uncertainty):
        # Text-only is an input restriction, not an excuse to add artificial error.
        return list(targets)

    def _determine_completed_targets(
        self, original_targets, biased_targets, traj, successful_segments, total_segments,
    ) -> List[str]:
        if not traj or total_segments <= 0:
            return []
        count = min(len(original_targets), int(len(original_targets) * successful_segments / total_segments))
        return [target.name for target in original_targets[:count]]


class DeterministicModularPlanner(ModularUAVSystem):
    """Modular control whose completion list follows planner segment success only."""

    def _determine_completed(self, targets, traj, successful_segments, total_segments, n_modalities=1):
        if not traj or total_segments <= 0:
            return []
        count = min(len(targets), int(len(targets) * successful_segments / total_segments))
        return [target.name for target in targets[:count]]


def _materialize_audio_encoder(fuser) -> None:
    encoder = getattr(fuser, "audio_encoder", None)
    if encoder is not None and hasattr(encoder, "_load_model"):
        encoder._load_model()


def build_main_baseline(baseline_id: str, run_root: str, seed: int):
    """Return a main-table baseline plus checkpoint/protocol audit metadata."""
    base_cfg, _, model_cfg, _, _ = get_default_config()
    device = base_cfg.device
    if baseline_id == "uav_vlpa":
        return DeterministicTextOnlyPlanner(
            benchmark_dir=base_cfg.benchmark_dir, device=device,
        ), {
            "label": "UAV-VLPA (text-only)",
            "implementation": "existing project text-only control; no trainable checkpoint",
            "synthetic_coordinate_noise": False,
            "random_completion_sampling": False,
        }

    fuser = LPANetFuser(
        fusion_dim=model_cfg.fusion_dim,
        audio_model=model_cfg.audio_model,
        gesture_backbone=model_cfg.gesture_backbone,
        text_model=model_cfg.text_model,
    ).to(device)
    _materialize_audio_encoder(fuser)
    checkpoint = load_checkpoint(
        fuser, f"{run_root}/models/lpanet/seed_{seed}", "best_fusion_model", device,
    )
    if baseline_id == "lpanet":
        planner = DeterministicModularPlanner(
            system_name="LPANet (local unified-interface implementation)",
            benchmark_dir=base_cfg.benchmark_dir, fuser=fuser, decomposer=None,
            path_planner_name="astar", device=device, seed=seed,
        )
        label = "LPANet"
        composition = "LPANet fusion + heuristic decomposition + A*"
    elif baseline_id == "citynav":
        planner = DeterministicModularPlanner(
            system_name="CityNav (local unified-interface implementation)",
            benchmark_dir=base_cfg.benchmark_dir, fuser=fuser,
            decomposer=SIPSADecomposer(seed=seed), path_planner_name="ikap",
            device=device, seed=seed,
        )
        label = "CityNav"
        composition = "LPANet fusion + SIPSA decomposition + iKap planning"
    else:
        raise ValueError(f"Unsupported main baseline: {baseline_id}")
    return planner, {
        "label": label,
        "implementation": "local implementation-inspired adapter in the project's common interface; not an author-code reproduction",
        "composition": composition,
        "fusion_checkpoint": str(checkpoint),
        "synthetic_coordinate_noise": False,
        "random_completion_sampling": False,
    }
