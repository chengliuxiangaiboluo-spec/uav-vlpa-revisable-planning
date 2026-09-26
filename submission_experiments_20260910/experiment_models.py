"""Model construction for evaluation; checkpoint loading is strict by design."""

from __future__ import annotations

from pathlib import Path

from submission_common import load_checkpoint

from configs.experiment_config import get_default_config
from models.fusion.baseline_fusers import AFFNetFuser, MAFTNetFuser, SCALFuser
from models.fusion.multimodal_fuser import MultimodalFuser
from models.planner.baseline_systems import ModularUAVSystem
from models.planner.enhanced_planner import EnhancedPlanner
from models.reasoning.task_decomposer import TaskDecomposer


def _materialize_audio_encoder(fuser) -> None:
    """Instantiate Wav2Vec2 before strict checkpoint loading.

    AudioEncoder loads Wav2Vec2 lazily on its first forward pass.  A trained
    checkpoint therefore contains Wav2Vec2 keys that are absent from a freshly
    constructed evaluation fuser unless it is materialized first.
    """
    audio_encoder = getattr(fuser, "audio_encoder", None)
    if audio_encoder is not None and hasattr(audio_encoder, "_load_model"):
        audio_encoder._load_model()


def build_evaluation_planner(method: str, run_root: str, seed: int):
    base_cfg, _, model_cfg, _, _ = get_default_config()
    device = base_cfg.device
    checkpoint_dir = Path(run_root).resolve() / "models" / method / f"seed_{seed}"
    common_kwargs = dict(
        fusion_dim=model_cfg.fusion_dim,
        audio_model=model_cfg.audio_model,
        gesture_backbone=model_cfg.gesture_backbone,
        text_model=model_cfg.text_model,
    )
    if method == "ours":
        fuser = MultimodalFuser(
            fusion_dim=model_cfg.fusion_dim,
            attention_heads=model_cfg.attention_heads,
            attention_layers=model_cfg.attention_layers,
            ffn_dim=model_cfg.ffn_dim,
            audio_model=model_cfg.audio_model,
            text_model=model_cfg.text_model,
        ).to(device)
        decomposer = TaskDecomposer(
            d_model=model_cfg.fusion_dim,
            n_layers=model_cfg.decomposer_layers,
            max_subtasks=model_cfg.max_subtasks,
            max_seq_len=100,
        ).to(device)
        _materialize_audio_encoder(fuser)
        fuser.to(device)
        fusion_path = load_checkpoint(fuser, checkpoint_dir, "best_fusion_model", device)
        decomposer_path = load_checkpoint(decomposer, checkpoint_dir, "decomposer_rl", device)
        planner = EnhancedPlanner(
            benchmark_dir=base_cfg.benchmark_dir, fuser=fuser,
            decomposer=decomposer, device=device,
        )
        return planner, {"fusion_checkpoint": str(fusion_path), "decomposer_checkpoint": str(decomposer_path)}

    cls = {"affnet": AFFNetFuser, "maftnet": MAFTNetFuser, "scal": SCALFuser}.get(method)
    if cls is None:
        raise ValueError(f"Unsupported evaluation method: {method}")
    fuser = cls(**common_kwargs).to(device)
    _materialize_audio_encoder(fuser)
    fuser.to(device)
    fusion_path = load_checkpoint(fuser, checkpoint_dir, "best_fusion_model", device)
    planner = ModularUAVSystem(
        system_name=f"{method}-implementation", benchmark_dir=base_cfg.benchmark_dir,
        fuser=fuser, decomposer=None, path_planner_name="astar", device=device, seed=seed,
    )
    return planner, {"fusion_checkpoint": str(fusion_path)}
