"""Q4 layer curves (L-condition paired sentences, read out at the homonym
position) vs. the original H1 homonym-position curves (plain profiling
sentences), one row per model, for one word and for the average across all
seven study words.

Reads:
  results/study/Q4/q4_layer_curves.csv   (written by hypotheses.q4_layer_selection.run_q4)
  results/study/H1/{model}/h1_{word}.csv (written by hypotheses.h1_layer_adequacy.run_h1;
                                           columns Layer, MeanMarginNorm, FractionAdequate, ...)

Both curves use fraction-adequate as the plotted metric (0-1, directly
comparable, the same headline quantity H1/Q4 both select layers on). Q4
does not select a single "best" layer -- it flags a *candidate set* per
model/word (every layer within tolerance of that model/word's own peak
adequacy, and above an absolute floor); every candidate is marked with an
open ring on the word-specific panel.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.model_registry import ALL_MODELS
from utils.visual_style import GRID, INK, METHOD, apply_report_style

ENCODERS = {
    "answerdotai/ModernBERT-large",
    "microsoft/deberta-v3-large",
    "FacebookAI/roberta-large",
    "FacebookAI/xlm-roberta-large",
}
DEFAULT_WORDS = ["bank", "bark", "bat", "crane", "spring", "match", "pitch"]

Q4_COLOR = METHOD["gdv"]
HOMONYM_COLOR = METHOD["adequacy"]


def _load_q4_curves(q4_csv: Path):
    """Return {(model, word): [(relative_depth, adequacy, layer, is_candidate), ...]}."""
    curves = defaultdict(list)
    with open(q4_csv, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["model"], row["homonym"])
            curves[key].append((
                float(row["relative_depth"]),
                float(row["adequacy"]),
                int(row["layer"]),
                row["is_candidate"] in ("True", "true", "1"),
            ))
    for key in curves:
        curves[key].sort(key=lambda t: t[0])
    return curves


def _load_h1_curve(results_dir: Path, safe_model: str, word: str):
    """Return (relative_depth array, fraction_adequate array) for the
    homonym-position full-profile curve, skipping layer 0 (not a selection
    candidate anywhere in this study, see adequacy_best_layer)."""
    path = results_dir / "study" / "H1" / safe_model / f"h1_{word}.csv"
    if not path.exists():
        return None, None
    layers, fracs = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            layers.append(int(row["Layer"]))
            fracs.append(float(row["FractionAdequate"]))
    last_layer = max(layers)
    depth = np.array([l / last_layer for l in layers])
    order = np.argsort(depth)
    return depth[order], np.array(fracs)[order]


def make_comparison(results_dir: Path, word: str, words, output_path: Path) -> None:
    apply_report_style()
    q4_curves = _load_q4_curves(results_dir / "study" / "Q4" / "q4_layer_curves.csv")
    models = sorted(ALL_MODELS, key=lambda m: (m not in ENCODERS,))
    n = len(models)

    fig, axes = plt.subplots(
        n, 2, figsize=(9.5, 2.0 * n), sharex=True,
        constrained_layout={"h_pad": 0.12, "hspace": 0.06},
    )

    for row, model in enumerate(models):
        safe_model = model.replace("/", "_")
        arch = "encoder" if model in ENCODERS else "decoder"
        short_name = model.split("/")[-1]
        ax_word, ax_avg = axes[row]

        # --- word-specific panel ---
        q4_pts = q4_curves.get((safe_model, word), [])
        h1_depth, h1_frac = _load_h1_curve(results_dir, safe_model, word)
        if q4_pts:
            depth, adeq, layer, is_cand = zip(*q4_pts)
            ax_word.plot(depth, adeq, color=Q4_COLOR, linewidth=1.6, marker="o", markersize=2.6, label="L-condition @ homonym (Q4)")
        if h1_depth is not None:
            ax_word.plot(h1_depth, h1_frac, color=HOMONYM_COLOR, linewidth=1.6, marker="o", markersize=2.6, label="homonym (H1)")

        # --- average-across-words panel ---
        all_depth_grids, all_q4, all_h1 = [], [], []
        for w in words:
            pts = q4_curves.get((safe_model, w), [])
            if pts:
                d, a, _, _ = zip(*pts)
                all_depth_grids.append(np.array(d))
                all_q4.append(np.array(a))
            hd, hf = _load_h1_curve(results_dir, safe_model, w)
            if hd is not None:
                all_h1.append((hd, hf))
        if all_q4:
            depth_grid = all_depth_grids[0]
            avg_q4 = np.nanmean(np.vstack(all_q4), axis=0)
            ax_avg.plot(depth_grid, avg_q4, color=Q4_COLOR, linewidth=1.6, marker="o", markersize=2.6)
        if all_h1:
            h1_depth_grid = all_h1[0][0]
            avg_h1 = np.nanmean(np.vstack([f for _, f in all_h1]), axis=0)
            ax_avg.plot(h1_depth_grid, avg_h1, color=HOMONYM_COLOR, linewidth=1.6, marker="o", markersize=2.6)

        for ax, title in ((ax_word, f'"{word}"'), (ax_avg, f"avg of {len(words)} words")):
            ax.axhline(0.5, color=INK, linewidth=0.7, linestyle=":")
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlim(-0.02, 1.02)
            ax.grid(color=GRID, linewidth=0.5)
            if row == 0:
                ax.set_title(title, fontsize=10.5, fontweight="bold")
            if row == n - 1:
                ax.set_xlabel("Relative depth (layer / final layer)")

        # mark every Q4 candidate layer (not a single winner) on the
        # word-specific panel only
        for depth_pt, adeq_pt, layer_pt, is_cand in q4_pts:
            if is_cand:
                ax_word.scatter(
                    depth_pt, adeq_pt, s=55, facecolor="none",
                    edgecolor=Q4_COLOR, linewidth=1.5, zorder=5,
                )

        ax_word.set_ylabel(f"{short_name}\n({arch})", fontsize=9.5, fontweight="bold", color=INK)

    handles = [
        plt.Line2D([0], [0], color=HOMONYM_COLOR, linewidth=1.8, marker="o", markersize=4, label="H1 homonym-position curve"),
        plt.Line2D([0], [0], color=Q4_COLOR, linewidth=1.8, marker="o", markersize=4, label="Q4 L-condition @ homonym curve"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=Q4_COLOR, markersize=8, markeredgewidth=1.5, label="Q4 candidate layer"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle(
        "Fraction adequate by depth: plain profiling sentences (H1) vs. L-condition\n"
        "paired sentences, both read at the homonym position (Q4) · dotted line = chance (0.5)",
        fontsize=12,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--word", default="bank")
    parser.add_argument("--words", nargs="*", default=DEFAULT_WORDS)
    parser.add_argument("--output", default="results/study/figures/q4_layer_vs_homonym_curves.svg")
    args = parser.parse_args()
    make_comparison(Path(args.results_dir), args.word, args.words, Path(args.output))


if __name__ == "__main__":
    main()
