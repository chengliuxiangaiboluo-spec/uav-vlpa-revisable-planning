"""Leakage-free adapters for the manuscript's original three controls.

All cores receive coordinates produced by a fitted V2 grounder.  The original
``ScenarioSample.targets`` coordinates are retained only in an evaluator-side
copy.  Target count, names, and types are task-parser output; their locations
are never passed to the planner.
"""

from __future__ import annotations

import types
from pathlib import Path

import torch

from configs.experiment_config import get_default_config
from models.fusion.baseline_fusers import LPANetFuser
from models.planner.baseline_planner import BaselinePlanner
from models.planner.baseline_systems import ModularUAVSystem
from models.reasoning.baseline_decomposers import SIPSADecomposer
from submission_common import load_checkpoint
from v2_grounding import VisualLanguageGrounder


LABELS = {
    "uav_vlpa": "UAV-VLPA (text-only)",
    "lpanet": "LPANet",
    "citynav": "CityNav",
}


class StrictTextOnlyPlanner(BaselinePlanner):
    """Text-only planning core; coordinates have already been predicted."""

    def _compute_biased_targets(self, targets, n_modalities, complexity, uncertainty):
        # Synthetic coordinate perturbation is prohibited.  ``targets`` is the
        # grounder's prediction, never the evaluator's label copy.
        return list(targets)

    def _determine_completed_targets(
        self, original_targets, biased_targets, traj, successful_segments, total_segments,
    ):
        if not traj or total_segments <= 0:
            return []
        count = min(len(biased_targets), int(len(biased_targets) * successful_segments / total_segments))
        return [target.name for target in biased_targets[:count]]


class StrictModularPlanner(ModularUAVSystem):
    """Modular core with deterministic completion, not synthetic penalties."""

    def _determine_completed(self, targets, traj, successful_segments, total_segments, n_modalities=1):
        if not traj or total_segments <= 0:
            return []
        count = min(len(targets), int(len(targets) * successful_segments / total_segments))
        return [target.name for target in targets[:count]]


def _materialize_audio_encoder(fuser) -> None:
    encoder = getattr(fuser, "audio_encoder", None)
    if encoder is not None and hasattr(encoder, "_load_model"):
        encoder._load_model()


def _grounder_directory(root: Path, condition: str, seed: int) -> Path:
    candidates = (
        root / "v2_worker_shards" / condition / f"seed_{seed}" / "models" /
        "v2_osm_grounding" / condition / f"seed_{seed}",
        root / "models" / "v2_osm_grounding" / condition / f"seed_{seed}",
    )
    for directory in candidates:
        if (directory / "grounder.pt").is_file() and (directory / "training_metadata.json").is_file():
            return directory
    raise FileNotFoundError(
        f"No complete V2 {condition} grounder for seed {seed}. Checked: "
        + "; ".join(str(item) for item in candidates)
    )


def load_grounder(root: Path, condition: str, seed: int, device):
    directory = _grounder_directory(root, condition, seed)
    record = torch.load(directory / "grounder.pt", map_location=device)
    expected_image = condition == "visual_language_grounding"
    if bool(record.get("use_image")) != expected_image:
        raise RuntimeError(f"Grounder condition mismatch: {directory}")
    model = VisualLanguageGrounder(use_image=expected_image).to(device)
    model.load_state_dict(record["model_state_dict"], strict=True)
    model.eval()
    return model, directory


def build_strict_main_baseline(baseline_id: str, run_root: str, seed: int):
    """Build one core and its matching learned coordinate grounder."""
    if baseline_id not in LABELS:
        raise ValueError(f"Unsupported strict main baseline: {baseline_id}")
    root = Path(run_root).resolve()
    base_cfg, _, model_cfg, _, _ = get_default_config()
    device = base_cfg.device

    if baseline_id == "uav_vlpa":
        core = StrictTextOnlyPlanner(benchmark_dir=base_cfg.benchmark_dir, device=device)
        condition = "text_only_grounding"
        composition = "text-only V2 coordinate grounder + text-only A* planner"
        fusion_checkpoint = None
    else:
        fuser = LPANetFuser(
            fusion_dim=model_cfg.fusion_dim,
            audio_model=model_cfg.audio_model,
            gesture_backbone=model_cfg.gesture_backbone,
            text_model=model_cfg.text_model,
        ).to(device)
        # The fusion checkpoint was saved before the lazy, frozen Wav2Vec2
        # module was materialised.  Strict-load that exact architecture first;
        # then instantiate the identical local pretrained backbone for inference.
        # Loading it before the checkpoint would manufacture missing-key errors.
        fusion_checkpoint = load_checkpoint(
            fuser, root / "models" / "lpanet_v2_strict" / f"seed_{seed}",
            "best_fusion_model", device,
        )
        _materialize_audio_encoder(fuser)
        if baseline_id == "lpanet":
            core = StrictModularPlanner(
                system_name="LPANet (local unified-interface implementation)",
                benchmark_dir=base_cfg.benchmark_dir, fuser=fuser, decomposer=None,
                path_planner_name="astar", device=device, seed=seed,
            )
            composition = "visual-language V2 coordinate grounder + LPANet fusion + heuristic decomposition + A*"
        else:
            core = StrictModularPlanner(
                system_name="CityNav (local unified-interface implementation)",
                benchmark_dir=base_cfg.benchmark_dir, fuser=fuser,
                decomposer=SIPSADecomposer(seed=seed), path_planner_name="ikap",
                device=device, seed=seed,
            )
            composition = "visual-language V2 coordinate grounder + LPANet fusion + SIPSA decomposition + iKap"
        condition = "visual_language_grounding"

    grounder, grounder_dir = load_grounder(root, condition, seed, device)
    return core, grounder, {
        "label": LABELS[baseline_id],
        "implementation": "local unified-interface adaptation; not an author-code reproduction",
        "composition": composition,
        "grounder_condition": condition,
        "grounder_checkpoint": str(grounder_dir / "grounder.pt"),
        "fusion_checkpoint": str(fusion_checkpoint) if fusion_checkpoint else None,
        "planner_input": "predicted_coordinates_only",
        "label_coordinates_used_only_for_scoring": True,
        "synthetic_coordinate_noise": False,
        "random_completion_sampling": False,
    }
