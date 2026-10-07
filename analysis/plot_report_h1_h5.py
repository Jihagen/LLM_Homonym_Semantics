"""Generate report figures for corrected H1 and fixed-sentinel H5 analyses."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from utils.model_registry import ALL_MODELS
from utils.visual_style import (
    ARCHITECTURE,
    GRID,
    INK,
    METHOD,
    MUTED,
    OUTCOME,
    apply_report_style,
)


WORDS = ["bank", "bark", "bat", "crane", "spring", "match", "pitch"]
DISPLAY_NAMES = {
    "answerdotai/ModernBERT-large": "ModernBERT",
    "microsoft/deberta-v3-large": "DeBERTa",
    "FacebookAI/roberta-large": "RoBERTa",
    "FacebookAI/xlm-roberta-large": "XLM-R",
    "Qwen/Qwen2.5-3B": "Qwen-3B",
    "Qwen/Qwen2.5-7B": "Qwen-7B",
    "mistralai/Mistral-Nemo-Base-2407": "Mistral-Nemo",
    "allenai/OLMo-2-1124-7B": "OLMo-7B",
}
STAGES = [
    "prime_correct_margin_norm",
    "homonym_correct_margin_norm",
    "resolution_correct_margin_norm",
]
STAGE_LABELS = ["Prime", "Through homonym", "Through resolver"]


def _safe(model):
    return model.replace("/", "_")


def _architecture(model):
    return "encoder" if model in ALL_MODELS[:4] else "decoder"


def _read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_h1(results_dir, output_path):
    estimates = []
    for model in ALL_MODELS:
        rows = _read(
            results_dir / "study" / "H1" / _safe(model) / "h1_summary.csv"
        )
        folds = np.asarray([int(row["nested_n_outer_folds"]) for row in rows])
        selected = np.average(
            [float(row["nested_frac_selected"]) for row in rows], weights=folds
        )
        last = np.average(
            [float(row["nested_frac_last"]) for row in rows], weights=folds
        )
        estimates.append((model, _architecture(model), selected, last))

    fig, ax = plt.subplots(figsize=(9.7, 5.6), constrained_layout=True)
    for y, (model, arch, selected, last) in enumerate(estimates):
        ax.plot(
            [last, selected],
            [y, y],
            color=ARCHITECTURE[arch],
            linewidth=2.0,
            alpha=0.72,
            zorder=1,
        )
        ax.scatter(
            last,
            y,
            s=70,
            marker="o",
            facecolor="white",
            edgecolor=METHOD["final"],
            linewidth=1.8,
            zorder=3,
        )
        ax.scatter(
            selected,
            y,
            s=76,
            marker="D",
            color=ARCHITECTURE[arch],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        delta = 100 * (selected - last)
        ax.text(
            max(selected, last) + 0.006,
            y,
            f"{delta:+.1f} pp",
            va="center",
            fontsize=9.2,
            color=INK,
        )

    ax.axhline(3.5, color=GRID, linewidth=2.0)
    ax.set_yticks(range(len(estimates)), [DISPLAY_NAMES[row[0]] for row in estimates])
    ax.invert_yaxis()
    ax.set_xlim(0.77, 1.025)
    ax.set_xlabel("Nested leave-one-sentence-out adequate fraction")
    ax.set_title(
        "Does selected-layer performance exceed the final layer?",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            Line2D(
                [0], [0], marker="D", linestyle="", color=color,
                label=arch.capitalize(), markersize=7,
            )
            for arch, color in ARCHITECTURE.items()
        ]
        + [
            Line2D(
                [0], [0], marker="o", linestyle="", markerfacecolor="white",
                markeredgecolor=METHOD["final"], label="Final layer", markersize=7,
            ),
            Line2D(
                [0], [0], marker="D", linestyle="", color=MUTED,
                label="Nested selected layer", markersize=7,
            ),
        ],
        loc="lower left",
        ncol=2,
    )
    _save(fig, output_path)


def _load_h5_cells(results_dir):
    rows = _read(results_dir / "study" / "H5" / "h5_sentence_level.csv")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["arch_type"], row["word"])].append(row)

    cells = []
    for (model, arch, word), cell_rows in grouped.items():
        stage_means = [
            float(np.mean([float(row[column]) for row in cell_rows]))
            for column in STAGES
        ]
        cells.append(
            {
                "model": model,
                "arch": arch,
                "word": word,
                "stages": stage_means,
                "homonym": stage_means[1],
                "resolution": stage_means[2],
                "matched_control": float(
                    np.mean(
                        [
                            float(row["matched_control_correct_margin_norm"])
                            for row in cell_rows
                        ]
                    )
                ),
                "isolated_resolver": float(
                    np.mean(
                        [
                            float(row["resolver_isolated_correct_margin_norm"])
                            for row in cell_rows
                        ]
                    )
                ),
                "delta": float(
                    np.mean(
                        [
                            float(row["delta_resolution_minus_homonym_norm"])
                            for row in cell_rows
                        ]
                    )
                ),
                "matched_cost": float(
                    np.mean(
                        [
                            float(row["garden_path_cost_vs_matched_control_norm"])
                            for row in cell_rows
                        ]
                    )
                ),
                "isolated_difference": float(
                    np.mean(
                        [
                            float(row["delta_resolution_minus_isolated_resolver_norm"])
                            for row in cell_rows
                        ]
                    )
                ),
            }
        )
    return cells


def plot_h5_trajectories(cells, output_path):
    fig, axes = plt.subplots(
        2, 4, figsize=(14.0, 7.1), sharex=True, sharey=True, constrained_layout=True
    )
    axes = axes.ravel()
    all_values = [value for cell in cells for value in cell["stages"]]
    lower = min(-0.24, min(all_values) - 0.035)
    upper = max(0.12, max(all_values) + 0.035)
    x = np.arange(3)

    for ax, word in zip(axes, WORDS):
        ax.axhspan(lower, 0, color="#F2CDC7", alpha=0.62)
        ax.axhspan(0, upper, color="#D5E8D9", alpha=0.62)
        ax.axhline(0, color=INK, linewidth=1.0)
        word_cells = [cell for cell in cells if cell["word"] == word]
        for arch in ("encoder", "decoder"):
            arch_cells = [cell for cell in word_cells if cell["arch"] == arch]
            for cell in arch_cells:
                ax.plot(
                    x,
                    cell["stages"],
                    color=ARCHITECTURE[arch],
                    alpha=0.24,
                    linewidth=1.0,
                )
            mean = np.mean([cell["stages"] for cell in arch_cells], axis=0)
            ax.plot(
                x,
                mean,
                color=ARCHITECTURE[arch],
                marker="o",
                markersize=5,
                linewidth=2.5,
                zorder=4,
            )
        ax.set_title(word, loc="left", fontweight="bold")
        ax.set_xticks(x, STAGE_LABELS, rotation=18, ha="right")
        ax.set_ylim(lower, upper)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

    axes[-1].axis("off")
    axes[-1].legend(
        handles=[
            Line2D([0], [0], marker="o", color=color, label=arch.capitalize())
            for arch, color in ARCHITECTURE.items()
        ],
        loc="center",
    )
    axes[-1].text(
        0.5,
        0.30,
        "Thin lines: model means\nThick lines: architecture means\n\nGreen background: resolved/correct side\nRed background: primed/incorrect side",
        transform=axes[-1].transAxes,
        ha="center",
        va="center",
        color=MUTED,
    )
    fig.supylabel(r"Fixed-sentinel correct-sense margin $\hat{M}$")
    fig.suptitle(
        "How the same readout changes as context is added",
        fontsize=14,
        fontweight="bold",
    )
    _save(fig, output_path)


def plot_h5_effects(cells, output_path):
    panels = [
        (
            "homonym",
            "resolution",
            "Through homonym",
            "Conflict resolver",
            "Rightward = update toward resolved sense",
        ),
        (
            "matched_control",
            "resolution",
            "Matched-control resolver",
            "Conflict resolver",
            "Leftward = conflict remains below its matched control",
        ),
        (
            "isolated_resolver",
            "resolution",
            "Resolver in isolation",
            "Conflict resolver",
            "Leftward = resolver alone scores higher",
        ),
    ]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.0, 7.0),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    arch_offsets = {"encoder": -0.16, "decoder": 0.16}

    for ax, (start_key, end_key, start_label, end_label, note) in zip(axes, panels):
        ax.axvline(0, color=INK, linewidth=1.05)
        for word_index, word in enumerate(WORDS):
            word_cells = [cell for cell in cells if cell["word"] == word]
            for arch in ("encoder", "decoder"):
                arch_cells = [cell for cell in word_cells if cell["arch"] == arch]
                y = word_index + arch_offsets[arch]
                for cell in arch_cells:
                    ax.plot(
                        [cell[start_key], cell[end_key]],
                        [y, y],
                        color=ARCHITECTURE[arch],
                        alpha=0.16,
                        linewidth=0.9,
                        zorder=1,
                    )
                start = np.mean([cell[start_key] for cell in arch_cells])
                end = np.mean([cell[end_key] for cell in arch_cells])
                ax.annotate(
                    "",
                    xy=(end, y),
                    xytext=(start, y),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": ARCHITECTURE[arch],
                        "linewidth": 2.0,
                        "shrinkA": 2,
                        "shrinkB": 2,
                    },
                    zorder=4,
                )
                ax.scatter(
                    start,
                    y,
                    s=48,
                    facecolor="white",
                    edgecolor=ARCHITECTURE[arch],
                    linewidth=1.5,
                    zorder=5,
                )
                ax.scatter(
                    end,
                    y,
                    s=48,
                    facecolor=ARCHITECTURE[arch],
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=5,
                )
        ax.set_title(f"{start_label} → {end_label}", loc="left", fontweight="bold")
        ax.set_xlabel(f"Correct-sense margin\n{note}")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

    axes[0].set_yticks(range(len(WORDS)), WORDS)
    axes[0].invert_yaxis()
    fig.legend(
        handles=[
            Line2D([0], [0], color=color, linewidth=2, label=arch.capitalize())
            for arch, color in ARCHITECTURE.items()
        ]
        + [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor="white",
                markeredgecolor=INK,
                label="Reference/start",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=INK,
                markeredgecolor="white",
                label="Conflict resolver endpoint",
            ),
        ],
        loc="outside lower center",
        ncol=4,
    )
    fig.suptitle(
        "How preceding context changes the fixed-sentinel endpoint",
        fontsize=14,
        fontweight="bold",
    )
    _save(fig, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/study/figures")
    args = parser.parse_args()
    apply_report_style()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    plot_h1(results_dir, output_dir / "h1_nested_layer_selection.svg")
    cells = _load_h5_cells(results_dir)
    plot_h5_trajectories(cells, output_dir / "h5_fixed_sentinel_trajectories.svg")
    plot_h5_effects(cells, output_dir / "h5_update_controls.svg")


if __name__ == "__main__":
    main()
