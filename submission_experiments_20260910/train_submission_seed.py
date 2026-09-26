"""Train one independently initialized model for one seed and one method.

This script intentionally consumes explicit train/validation files.  It never
re-splits the data internally, which prevents a hidden split mismatch between
training and final evaluation.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

from submission_common import (
    PROJECT_ROOT, read_samples, run_metadata, sha256_file, write_json,
)

from configs.experiment_config import get_default_config
from models.fusion.baseline_fusers import AFFNetFuser, LPANetFuser, MAFTNetFuser, SCALFuser
from models.fusion.multimodal_fuser import MultimodalFuser
from models.planner.enhanced_planner import EnhancedPlanner
from models.reasoning.task_decomposer import TaskDecomposer
from training.fusion_trainer import FusionModuleTrainer
from training.reward_function import RewardFunction
from training.rl_optimizer import RLTaskDecomposerOptimizer
from utils.offline_config import setup_offline_environment
from utils.seed_manager import set_global_seed


def build_fuser(method: str, model_cfg, device):
    kwargs = dict(
        fusion_dim=model_cfg.fusion_dim,
        audio_model=model_cfg.audio_model,
        gesture_backbone=model_cfg.gesture_backbone,
        text_model=model_cfg.text_model,
    )
    if method == "ours":
        return MultimodalFuser(
            fusion_dim=model_cfg.fusion_dim,
            attention_heads=model_cfg.attention_heads,
            attention_layers=model_cfg.attention_layers,
            ffn_dim=model_cfg.ffn_dim,
            audio_model=model_cfg.audio_model,
            text_model=model_cfg.text_model,
        ).to(device)
    return {
        "affnet": AFFNetFuser,
        "lpanet": LPANetFuser,
        "maftnet": MAFTNetFuser,
        "scal": SCALFuser,
    }[method](**kwargs).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one independent submission model")
    parser.add_argument("--method", choices=("ours", "affnet", "lpanet", "maftnet", "scal"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--output-model-id",
        help=(
            "Directory name below run_root/models for this training run. "
            "Defaults to --method; use a distinct value for a protocol-specific rerun."
        ),
    )
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--skip-rl", action="store_true", help="Only valid for pilot/debug runs")
    args = parser.parse_args()
    if args.method == "ours" and args.skip_rl:
        raise ValueError("The submission full model requires RL/decomposer training; --skip-rl is not permitted")

    started_at = time.monotonic()
    print(
        f"[Train] start | method={args.method} | seed={args.seed} | run_root={Path(args.run_root).resolve()}",
        flush=True,
    )
    setup_offline_environment(server_mode=args.server)
    set_global_seed(args.seed)
    base_cfg, _, model_cfg, train_cfg, _ = get_default_config()
    device = base_cfg.device
    train_data, val_data = read_samples(args.train_file), read_samples(args.val_file)
    if not train_data or not val_data:
        raise ValueError("Training and validation files must both be non-empty")
    print(
        f"[Train] data loaded | train={len(train_data)} | val={len(val_data)} | device={device}",
        flush=True,
    )

    output_model_id = args.output_model_id or args.method
    checkpoint_dir = Path(args.run_root).resolve() / "models" / output_model_id / f"seed_{args.seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    train_cfg.checkpoint_dir = str(checkpoint_dir)

    fuser = build_fuser(args.method, model_cfg, device)
    print("[Train] fusion stage started", flush=True)
    fusion_history = FusionModuleTrainer(train_cfg, fuser, device=device).train(train_data, val_data)
    print("[Train] fusion stage complete", flush=True)
    metadata = run_metadata({
        "protocol_version": "submission-2026-09-10",
        "method": args.method,
        "output_model_id": output_model_id,
        "training_seed": args.seed,
        "train_file": str(Path(args.train_file).resolve()),
        "val_file": str(Path(args.val_file).resolve()),
        "train_sha256": sha256_file(args.train_file),
        "val_sha256": sha256_file(args.val_file),
        "n_train": len(train_data),
        "n_val": len(val_data),
        "checkpoint_dir": str(checkpoint_dir),
        "fusion_history": fusion_history,
        "independent_initialization": True,
    })

    if args.method == "ours":
        for parameter in fuser.parameters():
            parameter.requires_grad = False
        fuser.eval()
        decomposer = TaskDecomposer(
            d_model=model_cfg.fusion_dim,
            n_layers=model_cfg.decomposer_layers,
            max_subtasks=model_cfg.max_subtasks,
            max_seq_len=100,
        ).to(device)
        planner = EnhancedPlanner(
            benchmark_dir=base_cfg.benchmark_dir,
            fuser=fuser,
            decomposer=decomposer,
            device=device,
            use_neural_decompose=False,
        )
        print("[Train] RL/decomposer stage started", flush=True)
        rl_history = RLTaskDecomposerOptimizer(
            train_cfg, decomposer, fuser, planner, RewardFunction(), device=device
        ).train(train_data)
        metadata["rl_history"] = rl_history
        metadata["decomposer_checkpoint_stem"] = "decomposer_rl"

    write_json(checkpoint_dir / "training_metadata.json", metadata)
    print(
        f"[Train] complete | method={args.method} | seed={args.seed} | "
        f"elapsed={(time.monotonic() - started_at) / 60.0:.1f} min | output={checkpoint_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
