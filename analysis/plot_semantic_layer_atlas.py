"""Plot GDV/M_l layer agreement by model and homonym.

Agreement is the primary visual encoding.  Layer depth is retained as a cell
label rather than competing through a second continuous color scale.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from utils.model_registry import ALL_MODELS
from utils.visual_style import AGREEMENT, GRID, INK, apply_report_style


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


def _read_summary(path: Path):
    with open(path, newline="") as handle:
        return {row["word"]: row for row in csv.DictReader(handle)}


def _load_layers(results_dir: Path):
    shape = (len(ALL_MODELS), len(WORDS))
    adequacy = np.zeros(shape, dtype=int)
    gdv = np.zeros(shape, dtype=int)
    last = np.zeros(shape, dtype=int)
    for model_index, model_name in enumerate(ALL_MODELS):
        safe_model = model_name.replace("/", "_")
        rows = _read_summary(
            results_dir / "study" / "H1" / safe_model / "h1_summary.csv"
        )
        for word_index, word in enumerate(WORDS):
            adequacy[model_index, word_index] = int(rows[word]["best_layer_M"])
            gdv[model_index, word_index] = int(rows[word]["gdv_best_layer"])
            last[model_index, word_index] = int(rows[word]["n_layers"]) - 1
    return adequacy, gdv, last


def _agreement_panel(ax, rows, adequacy, gdv, overall_word_counts, title):
    matches = (adequacy[rows] == gdv[rows]).astype(int)
    cmap = ListedColormap([AGREEMENT["disagree"], AGREEMENT["agree"]])
    ax.imshow(matches, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(
        range(len(WORDS)),
        [f"{word}\n{overall_word_counts[j]}/8 agree" for j, word in enumerate(WORDS)],
    )
    row_counts = matches.sum(axis=1)
    ax.set_yticks(
        range(len(rows)),
        [f"{DISPLAY_NAMES[ALL_MODELS[i]]}  ·  {row_counts[k]}/8" for k, i in enumerate(rows)],
    )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xticks(np.arange(-0.5, len(WORDS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", bottom=False, left=False)

    for local_row, model_index in enumerate(rows):
        for column in range(len(WORDS)):
            m_layer = adequacy[model_index, column]
            g_layer = gdv[model_index, column]
            if m_layer == g_layer:
                label = f"L{m_layer}"
                color = "white"
                weight = "bold"
            else:
                label = f"{m_layer}→{g_layer}"
                color = INK
                weight = "normal"
            ax.text(
                column,
                local_row,
                label,
                ha="center",
                va="center",
                fontsize=10.2,
                color=color,
                fontweight=weight,
            )


def make_figure(results_dir: Path, output_path: Path):
    apply_report_style()
    adequacy, gdv, _ = _load_layers(results_dir)
    word_counts = (adequacy == gdv).sum(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.4), constrained_layout=True)
    _agreement_panel(
        axes[0],
        range(0, 4),
        adequacy,
        gdv,
        word_counts,
        "Encoders",
    )
    _agreement_panel(
        axes[1],
        range(4, 8),
        adequacy,
        gdv,
        word_counts,
        "Decoders",
    )
    axes[0].tick_params(labelbottom=False)
    axes[0].set_xticklabels([])
    fig.legend(
        handles=[
            Patch(facecolor=AGREEMENT["agree"], label="same selected layer"),
            Patch(facecolor=AGREEMENT["disagree"], edgecolor=GRID, label=r"different layers ($M_l\rightarrow$GDV)"),
        ],
        loc="outside lower center",
        ncol=2,
    )
    fig.suptitle(
        r"Do GDV and $M_l$ select the same layer?",
        fontsize=15,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--output",
        default="results/study/figures/semantic_layer_atlas.svg",
    )
    args = parser.parse_args()
    make_figure(Path(args.results_dir), Path(args.output))


if __name__ == "__main__":
    main()
