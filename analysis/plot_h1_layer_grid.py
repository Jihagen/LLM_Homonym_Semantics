"""Small-multiples PCA grid across every layer for one word/model pair.

Answers a simple question the two-layer H2 figure (plot_geometry_audit.py)
cannot: do the two senses start out in the same place at layer 0 (the
pre-attention embedding output) and gradually diverge, or is separation
already present from the start? Uses the same per-word H1 profiling
activation cache as the other geometry figures (results/activations/<word>/
<model>/layer_<N>.h5); no additional forward passes are run.

--position period switches to the sentence-final/period-position cache
(results/activations_final/, same profiling sentences, built by run_h2.py)
instead of the homonym-position one -- useful for comparing, layer by
layer, how a causal decoder's point cloud at the SAME chosen layer differs
depending on which token is read out (see H4/Q4).

--shared-pca fits ONE PCA across every layer (each layer first normalised
by plot_centroid_separation.shared_layer_pca, so no single layer's raw
magnitude dominates the pooled fit) instead of refitting PCA independently
per panel. Panels then share one coordinate system, so flipping across the
grid shows real movement rather than independently-rotated snapshots; a
faint per-sense centroid trail (the same path drawn on every panel, current
layer emphasised) makes that movement visible within a single panel too.
"""

import argparse
import csv
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from experiments.gdv_experiments import compute_gdv
from analysis.plot_centroid_separation import shared_layer_pca
from utils.visual_style import GRID, INK, SELECTED_LAYER_COLOR, SENSE, apply_report_style, fit_suptitle

BEST_LAYER_COLOR = SELECTED_LAYER_COLOR


def _available_layers(results_dir: Path, word: str, safe_model: str, subdir: str = "activations"):
    word_dir = results_dir / subdir / word / safe_model
    layers = sorted(
        int(path.stem.removeprefix("layer_")) for path in word_dir.glob("layer_*.h5")
    )
    return layers


def _read_h1_best_layer(results_dir: Path, safe_model: str, word: str):
    """Full-profile H1 layer selected as best for this model--word pair
    (h1_summary.csv's best_layer_M, the same "does the probe say this layer
    decodes sense best" quantity used in the H1/H2 report figures)."""
    path = results_dir / "study" / "H1" / safe_model / "h1_summary.csv"
    if not path.exists():
        return None
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["word"] == word:
                return int(row["best_layer_M"])
    return None


def _load_layer(results_dir: Path, word: str, safe_model: str, layer: int, subdir: str = "activations"):
    path = results_dir / subdir / word / safe_model / f"layer_{layer}.h5"
    with h5py.File(path, "r") as handle:
        return handle["X"][:], handle["labels"][:]


def _layer_label(layer: int, last_layer: int, best_layer) -> str:
    if layer == 0:
        label = "L0 (embedding)"
    elif layer == last_layer:
        label = f"L{layer} (final)"
    else:
        label = f"L{layer}"
    if best_layer is not None and layer == best_layer:
        label += " ★ best"
    return label


def make_layer_grid(
    results_dir: Path,
    safe_model: str,
    word: str,
    output_path: Path,
    position: str = "homonym",
    shared_pca: bool = True,
) -> None:
    if position not in ("homonym", "period"):
        raise ValueError(f"position must be 'homonym' or 'period', got {position!r}")
    subdir = "activations" if position == "homonym" else "activations_final"

    apply_report_style()
    layers = _available_layers(results_dir, word, safe_model, subdir)
    if not layers:
        raise FileNotFoundError(
            f"no cached activations for {word}/{safe_model} under {results_dir}/{subdir}"
        )
    last_layer = layers[-1]
    # Always the H1 homonym-position selection, even on the period-position
    # grid: the point is to see how the SAME chosen layer's point cloud
    # differs depending on which token is read out (H4/Q4's question).
    best_layer = _read_h1_best_layer(results_dir, safe_model, word)

    n = len(layers)
    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.05 * n_cols, 2.05 * n_rows),
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    trail = None
    shared_xlim = shared_ylim = None
    if shared_pca:
        pca, normalized = shared_layer_pca(results_dir, word, safe_model, subdir, layers)
        # Full per-sense centroid path in the shared basis, drawn faintly on
        # every panel so the whole trajectory is visible at a glance, with
        # only the current layer's position emphasised per panel.
        trail = {
            sense: np.array([
                pca.transform(normalized[layer][0][normalized[layer][1] == sense]).mean(axis=0)
                for layer in layers
            ])
            for sense in (0, 1)
        }
        # Fixed axis limits across every panel -- otherwise matplotlib
        # autoscales each panel to its own data and the shared coordinate
        # system's whole point (comparable positions across panels) is lost.
        all_coords = np.vstack([pca.transform(normalized[layer][0]) for layer in layers])
        pad_x = 0.08 * (all_coords[:, 0].max() - all_coords[:, 0].min() or 1.0)
        pad_y = 0.08 * (all_coords[:, 1].max() - all_coords[:, 1].min() or 1.0)
        shared_xlim = (all_coords[:, 0].min() - pad_x, all_coords[:, 0].max() + pad_x)
        shared_ylim = (all_coords[:, 1].min() - pad_y, all_coords[:, 1].max() + pad_y)

    for ax, layer in zip(axes_flat, layers):
        X, labels = _load_layer(results_dir, word, safe_model, layer, subdir)
        gdv = compute_gdv(X, labels)
        if shared_pca:
            norm_X, _ = normalized[layer]
            coords = pca.transform(norm_X)
            for sense in (0, 1):
                ax.plot(
                    trail[sense][:, 0], trail[sense][:, 1], color=SENSE[sense],
                    alpha=0.25, linewidth=0.9, zorder=1,
                )
        else:
            coords = PCA(n_components=2).fit_transform(X)
        for sense in (0, 1):
            pts = coords[labels == sense]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=14,
                alpha=0.75,
                color=SENSE[sense],
                edgecolor="white",
                linewidth=0.25,
            )
            centroid = pts.mean(axis=0)
            ax.scatter(
                centroid[0],
                centroid[1],
                s=50,
                marker="X",
                color=SENSE[sense],
                edgecolor=INK,
                linewidth=0.6,
                zorder=4,
            )
        is_best = best_layer is not None and layer == best_layer
        ax.set_title(
            _layer_label(layer, last_layer, best_layer),
            fontsize=8.5,
            fontweight="bold",
            pad=2,
            color=BEST_LAYER_COLOR if is_best else INK,
        )
        ax.text(
            0.03,
            0.03,
            f"GDV {gdv:.2f}",
            transform=ax.transAxes,
            fontsize=7,
            color=INK,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(color=GRID, linewidth=0.5)
        if shared_xlim is not None:
            ax.set_xlim(shared_xlim)
            ax.set_ylim(shared_ylim)
        if is_best:
            for spine in ax.spines.values():
                spine.set_edgecolor(BEST_LAYER_COLOR)
                spine.set_linewidth(2.6)
            ax.patch.set_facecolor(BEST_LAYER_COLOR)
            ax.patch.set_alpha(0.08)

    for ax in axes_flat[n:]:
        ax.axis("off")

    model_label = safe_model.replace("_", "/")
    position_label = "homonym position" if position == "homonym" else "sentence-final (period) position"
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=SENSE[s], label=f"sense {s}")
        for s in (0, 1)
    ] + [
        plt.Line2D(
            [0], [0], marker="X", linestyle="", color=INK,
            markerfacecolor="none", label="centroid",
        ),
        plt.Line2D(
            [0], [0], marker="s", linestyle="", color=BEST_LAYER_COLOR,
            markerfacecolor="none", markeredgewidth=2.2, markersize=10,
            label="H1-selected best layer (homonym-position selection)",
        ),
    ]
    if shared_pca:
        handles.append(
            plt.Line2D([0], [0], color=INK, alpha=0.4, linewidth=1.2, label="centroid path across all layers")
        )
    fig.legend(handles=handles, loc="outside lower center", ncol=4)
    pca_note = " One shared PCA basis across all layers." if shared_pca else ""
    fig.canvas.draw()
    fit_suptitle(
        fig,
        f'Sense geometry across all layers (PCA), {model_label}, "{word}", '
        f"read out at {position_label}.{pca_note}",
        max_width_in=fig.get_size_inches()[0],
        fontsize=13,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--model",
        default="FacebookAI_roberta-large",
        help="Safe (underscore-joined) model name; defaults to one of the "
        "shallowest sampled encoders (25 cached layers) so the grid stays small.",
    )
    parser.add_argument("--word", default="bank")
    parser.add_argument(
        "--output",
        default="results/study/figures/h1_layer_grid_bank.svg",
        help="Output path; PNG is also written alongside the SVG.",
    )
    parser.add_argument(
        "--position",
        choices=["homonym", "period"],
        default="homonym",
        help="'homonym' reads results/activations/ (H1's default); 'period' reads "
        "results/activations_final/ (sentence-final token, same profiling sentences).",
    )
    parser.add_argument(
        "--independent-pca",
        action="store_true",
        help="Fit PCA independently per panel (the old default) instead of one "
        "shared basis across all layers. Panels then are NOT directly "
        "comparable to each other -- only use this if you specifically want "
        "each layer's best possible 2D view in isolation.",
    )
    args = parser.parse_args()
    make_layer_grid(
        Path(args.results_dir), args.model, args.word, Path(args.output),
        args.position, not args.independent_pca,
    )


if __name__ == "__main__":
    main()
