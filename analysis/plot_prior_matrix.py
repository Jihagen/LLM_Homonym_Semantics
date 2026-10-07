"""Plot signed bare-word and neutral-carrier lean by model and homonym."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils.model_registry import ALL_MODELS
from utils.visual_style import GRID, INK, PRIOR_CMAP, SENSE, apply_report_style


WORDS = ["bank", "bark", "bat", "crane", "spring", "match", "pitch"]
SENSE_LABELS = {
    "bank": ("river", "financial"),
    "bark": ("dog", "tree"),
    "bat": ("sports", "animal"),
    "crane": ("bird", "machine"),
    "spring": ("season", "water"),
    "match": ("sport", "fire"),
    "light": ("illumination", "low weight"),
    "pitch": ("field", "proposal"),
}
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


def _signed_carrier_values(rows):
    if rows and "signed_M_l_carrier_norm" in rows[0]:
        return np.asarray(
            [float(row["signed_M_l_carrier_norm"]) for row in rows],
            dtype=float,
        )
    by_carrier = defaultdict(dict)
    for row in rows:
        by_carrier[row["carrier"]][int(row["sense"])] = float(row["M_l_carrier_norm"])
    values = []
    for sense_values in by_carrier.values():
        if 0 in sense_values and 1 in sense_values:
            values.append(0.5 * (sense_values[0] - sense_values[1]))
        elif 0 in sense_values:
            values.append(sense_values[0])
        else:
            values.append(-sense_values[1])
    return np.asarray(values, dtype=float)


def _load_priors(results_dir: Path):
    shape = (len(ALL_MODELS), len(WORDS))
    bare = np.full(shape, np.nan)
    carrier = np.full(shape, np.nan)
    consistency = np.full(shape, np.nan)
    for model_index, model_name in enumerate(ALL_MODELS):
        safe_model = model_name.replace("/", "_")
        for word_index, word in enumerate(WORDS):
            path = results_dir / "study" / "H0" / safe_model / f"h0_{word}.csv"
            with open(path, newline="") as handle:
                rows = list(csv.DictReader(handle))
            bare_column = (
                "signed_M_l_word_alone_norm"
                if "signed_M_l_word_alone_norm" in rows[0]
                else "M_l_word_alone_norm"
            )
            bare[model_index, word_index] = float(rows[0][bare_column])
            signed = _signed_carrier_values(rows)
            carrier[model_index, word_index] = float(signed.mean())
            nonzero = signed[np.abs(signed) > 1e-8]
            if len(nonzero):
                consistency[model_index, word_index] = max(
                    float((nonzero > 0).mean()),
                    float((nonzero < 0).mean()),
                )
            else:
                consistency[model_index, word_index] = 0.5
    return bare, carrier, consistency


def _cell_label(value, word):
    sense_index = 0 if value >= 0 else 1
    return f"{SENSE_LABELS[word][sense_index]}\n{abs(value):.2f}"


def _prior_panel(ax, values, norm, title, consistency=None):
    image = ax.imshow(values, cmap=PRIOR_CMAP, norm=norm, aspect="auto")
    ax.set_xticks(range(len(WORDS)), WORDS, fontsize=9)
    ax.set_yticks(
        range(len(ALL_MODELS)),
        [DISPLAY_NAMES[model] for model in ALL_MODELS],
    )
    ax.axhline(3.5, color="white", linewidth=3)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xticks(np.arange(-0.5, len(WORDS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ALL_MODELS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            ax.text(
                column,
                row,
                _cell_label(value, WORDS[column]),
                ha="center",
                va="center",
                fontsize=8.6,
                color=INK,
                fontweight="bold" if abs(value) >= 0.30 else "normal",
            )
            if consistency is not None:
                filled = consistency[row, column] >= 0.80
                ax.scatter(
                    column + 0.34,
                    row - 0.30,
                    s=22,
                    facecolor=INK if filled else "white",
                    edgecolor=INK,
                    linewidth=0.9,
                    zorder=4,
                )
    return image


def make_figure(results_dir: Path, output_path: Path):
    apply_report_style()
    bare, carrier, consistency = _load_priors(results_dir)
    limit = max(0.20, float(np.nanmax(np.abs(np.concatenate([bare, carrier])))))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    fig, axes = plt.subplots(2, 1, figsize=(14.5, 10.0), constrained_layout=True)
    image = _prior_panel(
        axes[0],
        bare,
        norm,
        "Bare word",
    )
    _prior_panel(
        axes[1],
        carrier,
        norm,
        "Neutral carriers · mean signed lean",
        consistency=consistency,
    )
    colorbar = fig.colorbar(image, ax=axes, fraction=0.022, pad=0.02)
    colorbar.set_label("Signed normalized lean · saturation = strength")
    colorbar.set_ticks([-limit, 0, limit])
    colorbar.set_ticklabels(["negative lean", "neutral", "positive lean"])
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=INK, markeredgecolor=INK, label="≥80% carrier-direction consistency"),
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=INK, label="<80% consistency"),
        ],
        loc="outside lower center",
        ncol=2,
    )
    fig.suptitle(
        "Bare-word and neutral-carrier sense lean",
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
        default="results/study/figures/h0_prior_matrix.svg",
    )
    args = parser.parse_args()
    make_figure(Path(args.results_dir), Path(args.output))


if __name__ == "__main__":
    main()
