"""Render publication figures from measured diagnostic JSON only.

The source data are the merged five-seed histories and held-out attention
extractions.  This script rejects any output marked synthetic and never fills
missing early-stopped epochs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.8,
})

BLUE, GREEN, ORANGE, GRID = "#4878CF", "#6ACC65", "#FFB347", "#D8D8D8"


def _style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _series(block):
    return (np.asarray(block["mean"], dtype=float), np.asarray(block["std"], dtype=float),
            np.asarray(block["n_seeds_by_point"], dtype=int))


def _line(ax, block, label, color):
    mean, std, n = _series(block)
    x = np.arange(1, len(mean) + 1)
    valid = n > 0
    ax.plot(x[valid], mean[valid], color=color, linewidth=1.35, label=label)
    spread = valid & (n > 1)
    ax.fill_between(x[spread], (mean - std)[spread], (mean + std)[spread], color=color, alpha=0.18, linewidth=0)


def _alignment_gate(fig, out: Path, stem: str) -> None:
    """Run the publication layout gate when its approved helper is available."""
    helper_dir = os.environ.get("NATURE_FIGURE_SKILL_SCRIPTS")
    if not helper_dir or not (Path(helper_dir) / "audit_panel_alignment.py").is_file():
        print("[Diagnostics plot] layout helper unavailable; run rendered QA after download", flush=True)
        return
    if helper_dir not in sys.path:
        sys.path.insert(0, helper_dir)
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=out / f"{stem}.alignment.json",
        overlay_svg=out / f"{stem}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )


def _save(fig, out: Path, stem: str, multi_panel: bool = False):
    out.mkdir(parents=True, exist_ok=True)
    if multi_panel:
        _alignment_gate(fig, out, stem)
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot measured convergence and attention diagnostics")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if data.get("synthetic_points"):
        raise ValueError("Refusing to plot a synthetic diagnostic package")
    out = Path(args.output_dir)
    convergence = data["convergence"]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), constrained_layout=True)
    _line(axes[0], convergence["fusion_train_loss"], "Training loss", BLUE)
    _line(axes[0], convergence["fusion_validation_loss"], "Validation loss", ORANGE)
    axes[0].set_title("a  Fusion optimization", loc="left", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="upper right", fontsize=7, frameon=False)
    _style(axes[0])
    _line(axes[1], convergence["rl_episode_reward"], "Episode reward", GREEN)
    axes[1].set_title("b  Task-decomposition policy optimization", loc="left", fontweight="bold")
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Composite reward")
    _style(axes[1])
    _save(fig, out, "fig_measured_training_convergence", multi_panel=True)

    attention = data["attention"]
    matrix = np.asarray(attention["layer_mean_attention"]["mean"], dtype=float)
    fig, ax = plt.subplots(figsize=(3.35, 2.45), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_title("a  Held-out cross-modal attention allocation", loc="left", fontweight="bold")
    ax.set_xticks(np.arange(len(attention["modalities"])), attention["modalities"])
    ax.set_yticks(np.arange(matrix.shape[0]), [f"Attention layer {index + 1}" for index in range(matrix.shape[0])])
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if value > 0.55 else "black")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.06, pad=0.03)
    colorbar.set_label("Mean attention weight", fontsize=7)
    colorbar.ax.tick_params(labelsize=7)
    _save(fig, out, "fig_measured_cross_modal_attention")
    print(f"[Diagnostics plot] complete | output={out}", flush=True)


if __name__ == "__main__":
    main()
