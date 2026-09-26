"""Merge real convergence histories and five independently extracted attention runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from submission_common import DEFAULT_SEEDS, run_metadata, write_json


def _mean_sd(series: list[list[float]]) -> dict:
    """Aggregate complete or early-stopped histories without inventing points."""
    width = max(len(values) for values in series)
    matrix = np.full((len(series), width), np.nan, dtype=float)
    for row, values in enumerate(series):
        matrix[row, :len(values)] = values
    n_by_point = np.isfinite(matrix).sum(axis=0)
    mean = np.nanmean(matrix, axis=0)
    std = np.array([
        np.nanstd(matrix[:, col], ddof=1) if n_by_point[col] > 1 else 0.0
        for col in range(width)
    ])
    return {
        "mean": mean.tolist(), "std": std.tolist(),
        "n_seeds": int(matrix.shape[0]), "n_seeds_by_point": n_by_point.astype(int).tolist(),
    }


def _tensor_mean_sd(runs: list[list[list[float]]]) -> dict:
    """Aggregate identically shaped layer-by-modality matrices across seeds."""
    array = np.asarray(runs, dtype=float)
    if array.ndim != 3 or array.shape[0] != 5:
        raise ValueError(f"Expected five layer-by-modality attention matrices, got {array.shape}")
    return {
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0, ddof=1).tolist(),
        "n_seeds": int(array.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge training and attention diagnostics")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    seeds = tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip())
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Exactly five distinct seeds are required")
    root = Path(args.run_root).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else root / "results" / "diagnostic_merged"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite merged diagnostics: {output}")

    curve_path = root / "results" / "training_diagnostics" / "training_curves.json"
    curves = json.loads(curve_path.read_text(encoding="utf-8"))["records"]
    fusion_train = [curves[str(seed)]["fusion_train_loss"] for seed in seeds]
    fusion_val = [curves[str(seed)]["fusion_validation_loss"] for seed in seeds]
    rewards = [curves[str(seed)]["rl_episode_reward"] for seed in seeds]
    policy_losses = [curves[str(seed)]["rl_policy_loss"] for seed in seeds]
    # Fusion may stop early on different epochs.  The merged series retains
    # the actual available-seed count at every epoch instead of padding it.

    attention_runs = []
    expected_ids = None
    for seed in seeds:
        path = root / "results" / "attention_workers" / f"seed_{seed}" / "attention_seed.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("seed") != seed or record.get("smoke_test"):
            raise ValueError(f"seed {seed}: wrong or smoke attention record")
        ids = record["used_scenario_ids"]
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError("Attention workers used different scenario sets; refusing to average")
        attention_runs.append(record["layer_mean_attention"])

    output.mkdir(parents=True)
    write_json(output / "diagnostic_summary.json", run_metadata({
        "experiment": "Five-seed training and attention diagnostics",
        "seeds": list(seeds),
        "convergence": {
            "fusion_train_loss": _mean_sd(fusion_train),
            "fusion_validation_loss": _mean_sd(fusion_val),
            "rl_episode_reward": _mean_sd(rewards),
            "rl_policy_loss": _mean_sd(policy_losses),
        },
        "attention": {
            "modalities": ["Text", "Voice", "Gesture", "Annotation"],
            "layer_mean_attention": _tensor_mean_sd(attention_runs),
            "n_held_out_scenarios": len(expected_ids or []),
            "scenario_ids": expected_ids,
            "claim_boundary": "Conditional attention allocation in the fitted model; not causal modality importance.",
        },
    }))
    print(f"[Diagnostics] complete | output={output}", flush=True)


if __name__ == "__main__":
    main()
