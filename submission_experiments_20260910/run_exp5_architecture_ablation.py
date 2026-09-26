"""Exp5: independently trained, system-level architecture ablations.

Each invocation runs one *actual configured system* over all five seeds.  It
does not use the legacy ablation runner because that runner applies hand-tuned
modality factors to the metrics.  The only differences below are model or
planner configuration differences, and every variant has isolated weights.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from configs.experiment_config import get_default_config
from data.scenario_schema import ModalityType
from evaluation.ablation_config import get_all_ablation_configs
from models.fusion.multimodal_fuser import MultimodalFuser
from models.fusion.simple_fuser import SimpleConcatFuser
from models.planner.ablation_planner import AblationPlanner
from models.planner.enhanced_planner import EnhancedPlanner
from models.reasoning.task_decomposer import TaskDecomposer
from training.fusion_trainer import FusionModuleTrainer
from training.reward_function import RewardFunction
from training.rl_optimizer import RLTaskDecomposerOptimizer
from utils.offline_config import setup_offline_environment
from utils.seed_manager import set_global_seed

from submission_common import (
    DEFAULT_SEEDS, aggregate_seed_means, evaluate, load_checkpoint, read_samples,
    run_metadata, sha256_file, write_json, write_rows_csv,
)


# B11 is deliberately absent: this project has no trainable VLM-grounding
# component.  The legacy version injected coordinate noise, which is not a
# valid architecture ablation and must not be reported as one.
VARIANTS = {
    "minus_cross_modal_attention": ("B6", "Remove Cross-Modal Attention"),
    "minus_data_driven_calibration": ("B7", "Remove Data-Driven Calibration"),
    "minus_task_decomposition": ("B9", "Remove Refined Task Decomposition"),
    "minus_semantic_constraint": ("B10", "Remove Semantic Constraint"),
}


def _config(variant: str):
    wanted, _ = VARIANTS[variant]
    return next(cfg for cfg in get_all_ablation_configs() if cfg.group_id.value == wanted)


def _new_fuser(variant: str, model_cfg, device):
    kwargs = dict(fusion_dim=model_cfg.fusion_dim, audio_model=model_cfg.audio_model,
                  gesture_backbone=model_cfg.gesture_backbone, text_model=model_cfg.text_model)
    cls = SimpleConcatFuser if variant == "minus_cross_modal_attention" else MultimodalFuser
    if cls is MultimodalFuser:
        return cls(attention_heads=model_cfg.attention_heads,
                   attention_layers=model_cfg.attention_layers, ffn_dim=model_cfg.ffn_dim,
                   **kwargs).to(device)
    return cls(**kwargs).to(device)


def _materialize_audio_encoder(fuser):
    encoder = getattr(fuser, "audio_encoder", None)
    if encoder is not None and hasattr(encoder, "_load_model"):
        encoder._load_model()


def _checkpoint_dir(run_root: Path, variant: str, seed: int) -> Path:
    return run_root / "models" / "exp5" / variant / f"seed_{seed}"


def _train_variant(variant: str, run_root: Path, seed: int, train, val, base_cfg, model_cfg, train_cfg):
    checkpoint_dir = _checkpoint_dir(run_root, variant, seed)
    if checkpoint_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing independent Exp5 weights: {checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True)
    train_cfg.checkpoint_dir = str(checkpoint_dir)
    device = base_cfg.device
    fuser = _new_fuser(variant, model_cfg, device)
    print(f"[Exp5] fusion train | variant={variant} | seed={seed}", flush=True)
    fusion_history = FusionModuleTrainer(train_cfg, fuser, device=device).train(train, val)
    metadata = {"fusion_history": fusion_history, "rl_trained": False}

    # B9 removes the decomposition module.  The other variants retain and
    # independently train it; B10 also disables its transition constraint in
    # training, rather than merely switching it at test time.
    decomposer = None
    if variant != "minus_task_decomposition":
        for parameter in fuser.parameters():
            parameter.requires_grad = False
        fuser.eval()
        decomposer = TaskDecomposer(d_model=model_cfg.fusion_dim,
                                   n_layers=model_cfg.decomposer_layers,
                                   max_subtasks=model_cfg.max_subtasks, max_seq_len=100).to(device)
        if variant == "minus_semantic_constraint":
            decomposer.use_transition_constraint = False
        rl_planner = EnhancedPlanner(benchmark_dir=base_cfg.benchmark_dir, fuser=fuser,
                                     decomposer=decomposer, device=device,
                                     use_neural_decompose=False)
        print(f"[Exp5] RL/decomposer train | variant={variant} | seed={seed}", flush=True)
        metadata["rl_history"] = RLTaskDecomposerOptimizer(
            train_cfg, decomposer, fuser, rl_planner, RewardFunction(), device=device
        ).train(train)
        metadata["rl_trained"] = True
    write_json(checkpoint_dir / "training_metadata.json", run_metadata(metadata))


def _load_planner(variant: str, run_root: Path, seed: int, base_cfg, model_cfg):
    config = _config(variant)
    device = base_cfg.device
    checkpoint_dir = _checkpoint_dir(run_root, variant, seed)
    fuser = _new_fuser(variant, model_cfg, device)
    _materialize_audio_encoder(fuser)
    load_checkpoint(fuser, checkpoint_dir, "best_fusion_model", device)
    decomposer = None
    checkpoint_audit = {"fusion_checkpoint": str(checkpoint_dir)}
    if variant != "minus_task_decomposition":
        decomposer = TaskDecomposer(d_model=model_cfg.fusion_dim,
                                   n_layers=model_cfg.decomposer_layers,
                                   max_subtasks=model_cfg.max_subtasks, max_seq_len=100).to(device)
        if variant == "minus_semantic_constraint":
            decomposer.use_transition_constraint = False
        checkpoint_audit["decomposer_checkpoint"] = str(load_checkpoint(
            decomposer, checkpoint_dir, "decomposer_rl", device))
    planner = AblationPlanner(benchmark_dir=base_cfg.benchmark_dir, fuser=fuser,
                              decomposer=decomposer, ablation_config=config, device=device)
    return planner, checkpoint_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp5 independently-trained architecture ablation")
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--max-train-samples", type=int,
                        help="Limit training split size for a smoke test only.")
    parser.add_argument("--max-val-samples", type=int,
                        help="Limit validation split size for a smoke test only.")
    parser.add_argument("--max-test-samples", type=int,
                        help="Limit test split size for a smoke test only.")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a reduced one-seed training/evaluation smoke test; never report its metrics.")
    args = parser.parse_args()
    seeds = tuple(int(part) for part in args.seeds.split(",") if part.strip())
    if not args.smoke and (len(seeds) != 5 or len(set(seeds)) != 5):
        raise ValueError("A formal Exp5 variant requires exactly five distinct seeds")
    limits = (args.max_train_samples, args.max_val_samples, args.max_test_samples)
    if any(limit is not None and limit < 1 for limit in limits):
        raise ValueError("All max-sample limits must be positive")
    if args.smoke and (len(seeds) != 1 or any(limit is None for limit in limits)):
        raise ValueError("Exp5 smoke test requires one seed and all three --max-*-samples limits")
    started = time.monotonic()
    run_root = Path(args.run_root).resolve()
    output = run_root / "results" / "exp5_workers" / args.variant
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Exp5 result directory: {output}")
    setup_offline_environment(server_mode=args.server)
    train, val, test = read_samples(args.train_file), read_samples(args.val_file), read_samples(args.test_file)
    base_cfg, _, model_cfg, train_cfg, _ = get_default_config()
    if args.max_train_samples is not None:
        train = train[:args.max_train_samples]
    if args.max_val_samples is not None:
        val = val[:args.max_val_samples]
    if args.max_test_samples is not None:
        test = test[:args.max_test_samples]
    if args.smoke:
        # Validate the full train -> checkpoint -> reload -> planner route at
        # minimal cost.  These settings are recorded and cannot be confused
        # with the formal 20-epoch/1500-episode experiment.
        train_cfg.fusion_epochs = 1
        train_cfg.early_stopping_patience = 1
        train_cfg.rl_episodes = 5
        train_cfg.rl_ppo_epochs = 1
        train_cfg.rl_max_steps_per_episode = min(train_cfg.rl_max_steps_per_episode, 5)
    per_seed, rows, audit = {}, [], {}
    print(f"[Exp5] start | variant={args.variant} | seeds={seeds} | test={len(test)} | smoke={args.smoke}", flush=True)
    for seed in seeds:
        set_global_seed(seed)
        _train_variant(args.variant, run_root, seed, train, val, base_cfg, model_cfg, train_cfg)
        planner, audit[str(seed)] = _load_planner(args.variant, run_root, seed, base_cfg, model_cfg)
        seed_rows = evaluate(planner, test, planner.benchmark_dir)
        for row in seed_rows:
            row.update({"variant": args.variant, "training_seed": seed})
        per_seed[seed] = seed_rows
        rows.extend(seed_rows)
        print(f"[Exp5] seed complete | variant={args.variant} | seed={seed}", flush=True)
    write_rows_csv(output / "per_scenario_results.csv", rows)
    group, label = VARIANTS[args.variant]
    write_json(output / "summary.json", run_metadata({
        "experiment": "Exp5 independently-trained architecture ablation",
        "variant": args.variant, "group": group, "label": label,
        "seeds": list(seeds), "smoke_test": args.smoke,
        "n_train_scenarios": len(train), "n_val_scenarios": len(val), "n_test_scenarios": len(test),
        "train_sha256": sha256_file(args.train_file), "val_sha256": sha256_file(args.val_file),
        "test_sha256": sha256_file(args.test_file), "checkpoint_audit": audit,
        "summary": aggregate_seed_means(per_seed),
        "claim_allowed": "performance change under this independently trained system configuration",
        "claim_forbidden": "VLM-grounding ablation or any claim based on injected performance noise",
    }))
    print(f"[Exp5] complete | variant={args.variant} | elapsed={(time.monotonic()-started)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
