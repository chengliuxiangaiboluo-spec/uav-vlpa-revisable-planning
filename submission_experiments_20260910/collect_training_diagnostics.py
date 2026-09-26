"""Collect measured convergence histories from completed five-seed training runs.

This utility never trains a model and never creates synthetic points.  It
copies the histories already written by the trainer to a traceable diagnostic
package that can later be plotted without reading Slurm stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from submission_common import DEFAULT_SEEDS, run_metadata, write_json


def _finite_series(value: Any, name: str, seed: int) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"seed {seed}: missing non-empty {name}")
    result = [float(item) for item in value]
    if not all(item == item and abs(item) != float("inf") for item in result):
        raise ValueError(f"seed {seed}: {name} contains NaN or infinity")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect real convergence histories from existing checkpoints")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    seeds = tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip())
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Exactly five distinct completed training seeds are required")
    run_root = Path(args.run_root).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else run_root / "results" / "training_diagnostics"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output}")

    records: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        metadata_path = run_root / "models" / "ours" / f"seed_{seed}" / "training_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"seed {seed}: missing {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fusion = metadata.get("fusion_history", {})
        rl = metadata.get("rl_history", {})
        train_loss = _finite_series(fusion.get("train_loss"), "fusion_history.train_loss", seed)
        val_loss = _finite_series(fusion.get("val_loss"), "fusion_history.val_loss", seed)
        reward = _finite_series(rl.get("episode_rewards"), "rl_history.episode_rewards", seed)
        policy_loss = _finite_series(rl.get("policy_losses"), "rl_history.policy_losses", seed)
        if len(train_loss) != len(val_loss):
            raise ValueError(f"seed {seed}: fusion train/validation history lengths differ")
        if len(reward) != len(policy_loss):
            raise ValueError(f"seed {seed}: RL reward/loss history lengths differ")
        records[str(seed)] = {
            "metadata_file": str(metadata_path),
            "fusion_train_loss": train_loss,
            "fusion_validation_loss": val_loss,
            "rl_episode_reward": reward,
            "rl_policy_loss": policy_loss,
        }
        print(
            f"[Convergence] seed={seed} | fusion_epochs={len(train_loss)} | rl_episodes={len(reward)}",
            flush=True,
        )

    output.mkdir(parents=True)
    write_json(output / "training_curves.json", run_metadata({
        "experiment": "Measured training convergence diagnostics",
        "seeds": list(seeds),
        "source": "completed training_metadata.json files",
        "synthetic_points": False,
        "records": records,
    }))
    print(f"[Convergence] complete | output={output}", flush=True)


if __name__ == "__main__":
    main()
