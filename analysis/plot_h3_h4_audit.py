"""Focused replacement figures for the corrected H3 and H4 estimands."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from utils.model_registry import ALL_MODELS
from utils.visual_style import ARCHITECTURE, INK, METHOD, MUTED, apply_report_style


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
WORDS = ["bank", "bark", "bat", "crane", "spring", "match", "pitch"]
WORD_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*"]


def _safe(model):
    return model.replace("/", "_")


def _read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_h3(results, output):
    rows = {_safe_row["model"]: _safe_row for _safe_row in _read(
        results / "study" / "H3" / "h3_paired_summary.csv"
    )}
    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    ax.axvline(0, color=INK, linewidth=1.1)
    ax.axhline(3.5, color="#E7E4DE", linewidth=2.0)
    for index, model in enumerate(ALL_MODELS):
        row = rows[_safe(model)]
        value = float(row["mean_delta_M_norm_L_minus_R"])
        low = float(row["ci95_low_delta_M_norm"])
        high = float(row["ci95_high_delta_M_norm"])
        arch = row["arch_type"]
        ax.errorbar(
            value,
            index,
            xerr=[[value - low], [high - value]],
            fmt="o",
            markersize=7,
            color=ARCHITECTURE[arch],
            ecolor=ARCHITECTURE[arch],
            capsize=3,
            linewidth=1.5,
        )
    ax.set_yticks(range(len(ALL_MODELS)), [DISPLAY_NAMES[m] for m in ALL_MODELS])
    ax.invert_yaxis()
    ax.set_xlabel(
        r"Paired margin difference $\hat{M}_{L}-\hat{M}_{R}$"
        "\n(horizontal line = 95% word-bootstrap interval)"
    )
    ax.set_title(
        "Effect of moving decisive context behind the homonym",
        loc="left",
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.02,
        "L: decisive clause before homonym\nR: same clause after homonym",
        transform=ax.transAxes,
        ha="left",
        color=MUTED,
    )
    ax.text(
        0.99,
        0.98,
        "Right of zero: L gives the stronger correct-sense margin\n"
        "Near zero: moving the clause has little effect",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=MUTED,
    )
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color=color, label=arch.capitalize())
            for arch, color in ARCHITECTURE.items()
        ],
        loc="lower right",
    )
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    _save(fig, output)


def _wilson(successes, total, z=1.96):
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return centre - half, centre + half


def _probability(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def plot_h4(results, output):
    aggregate = _read(results / "study" / "H4" / "h4_aggregate.csv")
    by_cell = {(row["model"], row["word"]): row for row in aggregate}
    panels = [
        (
            "p_final_adequate_given_target_inadequate",
            "n_target_inadequate_final_adequate",
            "n_target_inadequate",
            "Gain conditional on target failure",
        ),
        (
            "p_final_inadequate_given_target_adequate",
            "n_target_adequate_final_inadequate",
            "n_target_adequate",
            "Regression conditional on target success",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2), sharey=True, constrained_layout=True)
    offsets = np.linspace(-0.20, 0.20, len(WORDS))
    for ax, (probability, successes, denominator, title) in zip(axes, panels):
        for model_index, model in enumerate(ALL_MODELS):
            safe_model = _safe(model)
            model_rows = [by_cell[(safe_model, word)] for word in WORDS]
            arch = model_rows[0]["arch_type"]
            for word_index, (word, marker) in enumerate(zip(WORDS, WORD_MARKERS)):
                value = _probability(model_rows[word_index][probability])
                if math.isfinite(value):
                    ax.scatter(
                        value,
                        model_index + offsets[word_index],
                        marker=marker,
                        s=27,
                        facecolor="white",
                        edgecolor=MUTED,
                        linewidth=0.8,
                        zorder=2,
                    )
            n_success = sum(int(row[successes]) for row in model_rows)
            n_total = sum(int(row[denominator]) for row in model_rows)
            if n_total:
                pooled = n_success / n_total
                low, high = _wilson(n_success, n_total)
                ax.errorbar(
                    pooled,
                    model_index,
                    xerr=[[pooled - low], [high - pooled]],
                    fmt="D",
                    markersize=6.8,
                    color=ARCHITECTURE[arch],
                    capsize=3,
                    linewidth=1.6,
                    zorder=4,
                )
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("Conditional probability")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(ALL_MODELS)), [DISPLAY_NAMES[m] for m in ALL_MODELS])
    axes[0].invert_yaxis()
    word_handles = [
        Line2D([0], [0], marker=marker, linestyle="", markerfacecolor="white", markeredgecolor=MUTED, label=word)
        for word, marker in zip(WORDS, WORD_MARKERS)
    ]
    architecture_handles = [
        Line2D([0], [0], marker="D", linestyle="", color=color, label=f"{arch} pooled")
        for arch, color in ARCHITECTURE.items()
    ]
    fig.legend(
        handles=architecture_handles + word_handles,
        loc="outside lower center",
        ncol=5,
    )
    fig.suptitle(
        "Within-position sense decodability transitions",
        fontsize=14,
        fontweight="bold",
    )
    _save(fig, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/study/figures")
    args = parser.parse_args()
    apply_report_style()
    results = Path(args.results_dir)
    output = Path(args.output_dir)
    plot_h3(results, output / "h3_paired_context_effect.svg")
    plot_h4(results, output / "h4_conditional_transitions.svg")


if __name__ == "__main__":
    main()
