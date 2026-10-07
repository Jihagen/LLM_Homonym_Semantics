"""Updated selected-layer heatmap: H1's single homonym-position layer choice
vs. Q4's *candidate set* (layers within tolerance of that model/word's own
peak adequacy on the L-condition/homonym-position curve, and above an
absolute floor).

Q4 does not pick one layer, so a same/different-layer comparison (as H1 vs.
a single alternative would allow) doesn't apply here. Instead each cell
shows whether H1's chosen layer is itself validated as a Q4 candidate, and
if not, how far (in relative depth) the nearest candidate sits from it. The
color scale is the signed relative-depth distance from H1's layer to the
nearest candidate (0 if H1's own layer is already a candidate); text
annotates the candidate count and depth range.

Mirrors plot_semantic_layer_atlas.py's two-panel (encoders/decoders) grid
layout.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from utils.model_registry import ALL_MODELS
from utils.visual_style import DEPTH_CMAP, GRID, INK, apply_report_style

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


def _read_csv(path: Path):
    with open(path, newline="") as handle:
        return {row["word"]: row for row in csv.DictReader(handle)}


def _read_candidates(path: Path):
    """Return {word: sorted [layer, ...]} from a q4_candidate_layers.csv."""
    candidates = defaultdict(list)
    if not path.exists():
        return candidates
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            candidates[row["word"]].append(int(row["layer"]))
    for word in candidates:
        candidates[word].sort()
    return candidates


def _load_data(results_dir: Path):
    shape = (len(ALL_MODELS), len(WORDS))
    h1_layer = np.full(shape, -1, dtype=int)
    last = np.zeros(shape, dtype=int)
    have_q4 = np.zeros(shape, dtype=bool)
    candidates = {}

    for model_index, model_name in enumerate(ALL_MODELS):
        safe_model = model_name.replace("/", "_")
        h1_rows = _read_csv(results_dir / "study" / "H1" / safe_model / "h1_summary.csv")
        cand_path = results_dir / "study" / "Q4" / safe_model / "q4_candidate_layers.csv"
        model_candidates = _read_candidates(cand_path)
        for word_index, word in enumerate(WORDS):
            if word not in h1_rows:
                continue
            h1_layer[model_index, word_index] = int(h1_rows[word]["best_layer_M"])
            last[model_index, word_index] = int(h1_rows[word]["n_layers"]) - 1
            if model_candidates.get(word):
                have_q4[model_index, word_index] = True
                candidates[(model_index, word_index)] = model_candidates[word]
    return h1_layer, last, have_q4, candidates


def _nearest_candidate_shift(h1_l: int, cand_layers, last_l: int) -> float:
    if h1_l in cand_layers:
        return 0.0
    nearest = min(cand_layers, key=lambda c: abs(c - h1_l))
    return (nearest - h1_l) / last_l


def _shift_panel(ax, rows, h1_layer, last, have_q4, candidates, norm, title):
    shift = np.zeros((len(rows), len(WORDS)))
    for local_row, model_index in enumerate(rows):
        for column in range(len(WORDS)):
            if have_q4[model_index, column] and last[model_index, column] > 0:
                cand = candidates[(model_index, column)]
                shift[local_row, column] = _nearest_candidate_shift(
                    h1_layer[model_index, column], cand, last[model_index, column]
                )

    ax.imshow(shift, cmap=DEPTH_CMAP, norm=norm, aspect="auto")
    ax.set_xticks(range(len(WORDS)), WORDS)
    ax.set_yticks(
        range(len(rows)), [DISPLAY_NAMES[ALL_MODELS[i]] for i in rows]
    )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xticks(np.arange(-0.5, len(WORDS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", bottom=False, left=False)

    for local_row, model_index in enumerate(rows):
        for column in range(len(WORDS)):
            if not have_q4[model_index, column]:
                label, color = "no Q4\ndata", GRID
                weight = "normal"
                fontsize = 8.5
            else:
                h1_l = h1_layer[model_index, column]
                cand = candidates[(model_index, column)]
                in_set = h1_l in cand
                label = f"H1 L{h1_l}\ncand L{cand[0]}-{cand[-1]} (n={len(cand)})"
                color = INK if abs(shift[local_row, column]) < 0.35 else "white"
                weight = "bold" if in_set else "normal"
                fontsize = 8.5
            ax.text(
                column, local_row, label, ha="center", va="center",
                fontsize=fontsize, color=color, fontweight=weight, linespacing=1.3,
            )


def make_figure(results_dir: Path, output_path: Path):
    apply_report_style()
    h1_layer, last, have_q4, candidates = _load_data(results_dir)

    valid_shifts = []
    for model_index in range(len(ALL_MODELS)):
        for column in range(len(WORDS)):
            if have_q4[model_index, column] and last[model_index, column] > 0:
                cand = candidates[(model_index, column)]
                valid_shifts.append(
                    _nearest_candidate_shift(h1_layer[model_index, column], cand, last[model_index, column])
                )
    bound = max((abs(s) for s in valid_shifts), default=1e-6) or 1e-6
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.8), constrained_layout=True)
    _shift_panel(axes[0], range(0, 4), h1_layer, last, have_q4, candidates, norm, "Encoders")
    _shift_panel(axes[1], range(4, 8), h1_layer, last, have_q4, candidates, norm, "Decoders")
    axes[0].tick_params(labelbottom=False)
    axes[0].set_xticklabels([])

    sm = plt.cm.ScalarMappable(cmap=DEPTH_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", shrink=0.5, pad=0.03)
    cbar.set_label(
        "Relative-depth distance from H1's layer to the nearest Q4 candidate\n"
        "(0 = H1's layer is itself a Q4 candidate; positive = nearest candidate is deeper)"
    )

    fig.suptitle(
        "Is H1's homonym-position layer choice validated by the fair (L-condition) candidate set?",
        fontsize=13.5,
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
        "--output", default="results/study/figures/q4_selected_layer_shift.svg"
    )
    args = parser.parse_args()
    make_figure(Path(args.results_dir), Path(args.output))


if __name__ == "__main__":
    main()
