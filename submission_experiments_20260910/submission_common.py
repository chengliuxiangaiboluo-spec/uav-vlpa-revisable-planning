"""Shared, audit-first utilities for the submission experiment package.

These helpers deliberately treat a training seed—not an individual test
scenario—as the independent experimental unit.  They also preserve every
per-scenario measurement so a reviewer can audit reported aggregates.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.scenario_schema import ModalityType, ScenarioSample
from evaluation.baseline_comparison_runner import BaselineComparisonRunner

METRICS = (
    "task_completion_rate",
    "instruction_accuracy",
    "dtw_rmse",
    "knn_rmse",
    "sequential_rmse",
    "trajectory_length_km",
    "efficiency_ratio",
)
DEFAULT_SEEDS = (42, 123, 456, 789, 2024)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


def write_json(path: os.PathLike[str] | str, data: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=str)


def read_samples(path: os.PathLike[str] | str) -> List[ScenarioSample]:
    from data.dataset_manager import DatasetManager

    resolved = Path(path).resolve()
    return DatasetManager(str(resolved.parent)).load_dataset(resolved.name)


def write_samples(path: os.PathLike[str] | str, samples: Sequence[ScenarioSample]) -> None:
    from data.dataset_manager import DatasetManager

    resolved = Path(path).resolve()
    DatasetManager(str(resolved.parent)).save_dataset(list(samples), resolved.name)


def scenario_ids(samples: Iterable[ScenarioSample]) -> set[str]:
    return {str(s.scenario_id) for s in samples}


def image_ids(samples: Iterable[ScenarioSample]) -> set[int]:
    return {int(s.image_id) for s in samples}


def assert_no_scenario_overlap(named_sets: Mapping[str, Sequence[ScenarioSample]]) -> None:
    names = list(named_sets)
    for idx, left_name in enumerate(names):
        left = scenario_ids(named_sets[left_name])
        if len(left) != len(named_sets[left_name]):
            raise ValueError(f"Duplicate scenario_id within {left_name}")
        for right_name in names[idx + 1 :]:
            overlap = left & scenario_ids(named_sets[right_name])
            if overlap:
                raise ValueError(
                    f"Scenario leakage between {left_name} and {right_name}: "
                    f"{sorted(overlap)[:5]}"
                )


def checkpoint_path(checkpoint_dir: os.PathLike[str] | str, stem: str) -> Path:
    """Resolve a checkpoint only if its identity is unambiguous."""
    directory = Path(checkpoint_dir)
    exact = directory / f"{stem}.pt"
    if exact.exists():
        return exact
    matches = list(directory.glob(f"{stem}_epoch*.pt"))
    if not matches:
        raise FileNotFoundError(f"No checkpoint for {stem} under {directory}")
    def epoch_number(path: Path) -> int:
        try:
            return int(path.stem.rsplit("epoch", 1)[1])
        except (IndexError, ValueError):
            return -1
    return max(matches, key=epoch_number)


def load_checkpoint(model: torch.nn.Module, checkpoint_dir: os.PathLike[str] | str, stem: str,
                    device: str | torch.device) -> Path:
    path = checkpoint_path(checkpoint_dir, stem)
    state = torch.load(path, map_location=device, weights_only=False)
    state_dict = state.get("model_state_dict", state)
    # A publication run must fail on architecture mismatch; strict=False can
    # silently turn a claimed trained model into a partially random one.
    model.load_state_dict(state_dict, strict=True)
    return path


def mask_modalities(sample: ScenarioSample, active: set[ModalityType]) -> ScenarioSample:
    """Create an input intervention without mutating the source test set."""
    if ModalityType.TEXT not in active:
        raise ValueError("Text is required as the anchor modality in this implementation")
    clone = copy.deepcopy(sample)
    clone.modalities = [m for m in clone.modalities if m in active]
    if ModalityType.VOICE not in active:
        clone.audio_path = None
    if ModalityType.GESTURE not in active:
        clone.gesture_image_path = None
    if ModalityType.ANNOTATION not in active:
        clone.annotation_image_path = None
    return clone


class InputInterventionPlanner:
    """A transparent wrapper used only for test-time input interventions."""

    def __init__(self, planner: Any, active_modalities: set[ModalityType]):
        self.planner = planner
        self.active_modalities = active_modalities
        # The shared evaluator reads ``last_tasks`` to calculate instruction
        # accuracy.  Keep it on the wrapper after every intervened plan so the
        # intervention changes only inputs, not the metric plumbing.
        self.last_tasks = []
        self.last_delivered_modalities = []

    def plan(self, sample: ScenarioSample):
        masked = mask_modalities(sample, self.active_modalities)
        self.last_delivered_modalities = [m.value for m in masked.modalities]
        result = self.planner.plan(masked)
        self.last_tasks = getattr(self.planner, "last_tasks", [])
        return result


def evaluate(planner: Any, samples: Sequence[ScenarioSample], benchmark_dir: str) -> List[Dict[str, Any]]:
    started_at = time.monotonic()
    print(f"[Evaluate] start | samples={len(samples)} | benchmark_dir={benchmark_dir}", flush=True)
    runner = BaselineComparisonRunner(mode="fusion", benchmark_dir=benchmark_dir, reference_planner=None)
    metrics, _ = runner._evaluate_planner(planner, list(samples), "submission")
    rows: List[Dict[str, Any]] = []
    for sample, metric in zip(samples, metrics):
        row: Dict[str, Any] = {"scenario_id": sample.scenario_id, "image_id": sample.image_id}
        for key in METRICS:
            value = getattr(metric, key, np.nan)
            row[key] = float(value) if value is not None else float("nan")
        rows.append(row)
    if len(rows) != len(samples):
        raise RuntimeError("Evaluator did not produce exactly one row per scenario")
    print(
        f"[Evaluate] complete | rows={len(rows)} | elapsed={(time.monotonic() - started_at) / 60.0:.1f} min",
        flush=True,
    )
    return rows


def mean_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for metric in METRICS:
        values = np.asarray([row.get(metric, np.nan) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        result[metric] = float(values.mean()) if len(values) else float("nan")
    return result


def aggregate_seed_means(per_seed: Mapping[int, Sequence[Mapping[str, Any]]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for metric in METRICS:
        values = []
        for seed in sorted(per_seed):
            mean = mean_metrics(per_seed[seed])[metric]
            if np.isfinite(mean):
                values.append(mean)
        array = np.asarray(values, dtype=float)
        summary[metric] = {
            "mean": float(array.mean()) if len(array) else float("nan"),
            "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "n_independent_training_seeds": int(len(array)),
        }
    return summary


def paired_seed_test(left: Mapping[int, Sequence[Mapping[str, Any]]],
                     right: Mapping[int, Sequence[Mapping[str, Any]]]) -> Dict[str, Dict[str, float]]:
    """Paired tests across seed-level means; never across correlated scenarios."""
    common = sorted(set(left) & set(right))
    output: Dict[str, Dict[str, float]] = {}
    for metric in METRICS:
        x = np.asarray([mean_metrics(left[s])[metric] for s in common], dtype=float)
        y = np.asarray([mean_metrics(right[s])[metric] for s in common], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) < 3:
            output[metric] = {"n": int(len(x)), "mean_delta": float("nan"), "p_value": float("nan")}
            continue
        delta = x - y
        sd = float(delta.std(ddof=1))
        # scipy returns NaN for a constant all-zero paired difference.  This
        # is an exact null result, for which the conventional two-sided
        # p-value is 1 and the standardized paired effect is 0.
        if np.allclose(delta, 0.0, rtol=0.0, atol=1e-12):
            p_value = 1.0
            cohens_dz = 0.0
        else:
            try:
                from scipy.stats import ttest_1samp
                p_value = float(ttest_1samp(delta, 0.0).pvalue)
            except Exception:
                p_value = float("nan")
            cohens_dz = float(delta.mean() / sd) if sd > 0 else float("nan")
        output[metric] = {
            "n": int(len(delta)),
            "mean_delta": float(delta.mean()),
            "cohens_dz": cohens_dz,
            "p_value": p_value,
        }
    return output


def write_rows_csv(path: os.PathLike[str] | str, rows: Sequence[Mapping[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty result table: {path}")
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_metadata(extra: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_utc": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "git_revision": git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        **dict(extra),
    }
