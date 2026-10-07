"""Normalised centroid-separation progression across layers, one row per
model, for one word and for the average across all seven study words --
plus, next to those curves, what the point cloud actually looks like at two
concrete layers: a representative "average" layer from the plateau (typical
of most layers, not cherry-picked) and the H1-selected "best" layer. The
curves show that most models plateau across many layers rather than peaking
sharply at one; these two extra panels let you see directly whether a
"merely average" layer already looks about as separated as the "best" one,
or whether "best" is doing real work.

--position period switches the curves and point clouds to the sentence-
final/period-position cache (results/activations_final/, same profiling
sentences, built by run_h2.py) instead of the homonym-position one. The
homonym-position curve is still drawn, greyed out, behind the period-
position curve in the same two curve panels, so the shift is visible
directly rather than requiring a second figure to compare against.

Separation = ||c1 - c0||_2 / mean within-class distance to own centroid,
computed in the original hidden-state space (never PCA-projected). The two
point-cloud panels are the usual 2D PCA projection of that same layer's
activations (PCA is only for display; the separation ratio itself is never
computed in PCA space).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.plot_centroid_separation import (
    ENCODERS,
    _read_h1_best_layer,
    _separation_curve,
    shared_layer_pca,
)
from utils.model_registry import ALL_MODELS
from utils.visual_style import (
    ARCHITECTURE,
    GRID,
    INK,
    SELECTED_LAYER_COLOR,
    SENSE,
    apply_report_style,
    fit_suptitle,
)

DEFAULT_WORDS = ["bank", "bark", "bat", "crane", "spring", "match", "pitch"]
BEST_LAYER_COLOR = SELECTED_LAYER_COLOR
BACKGROUND_CURVE_COLOR = "#C7C3BB"
POSITION_SUBDIR = {"homonym": "activations", "period": "activations_final"}


def _average_plateau_layer(layers, norm_ratio):
    """The layer most representative of "what the separation typically
    looks like", not the best one. Restricted to the plateau (layers whose
    separation ratio already exceeds the within-class-spread threshold,
    norm_ratio >= 1.0) when one exists, since that is the region the
    "average" is meant to characterise; falls back to all non-embedding
    layers if the curve never crosses that threshold. Within that pool,
    picks the layer whose ratio is closest to the pool's own mean -- a
    concrete, representative point rather than an arbitrary midpoint index.
    """
    layers_arr = np.array(layers)
    non_embedding = layers_arr != 0
    plateau = non_embedding & (norm_ratio >= 1.0)
    pool_mask = plateau if plateau.any() else non_embedding
    pool_idx = np.where(pool_mask)[0]
    target_mean = norm_ratio[pool_idx].mean()
    closest = pool_idx[np.argmin(np.abs(norm_ratio[pool_idx] - target_mean))]
    return int(layers_arr[closest]), int(closest)


def _pca_panel(ax, coords, labels, ratio, title, xlim, ylim, highlight=False):
    """coords must already be projected through the SAME fitted PCA basis
    (shared_layer_pca) as every other panel for this model/word, so the two
    panels are directly comparable snapshots -- same axes, same units. No
    arrow is drawn between them: average-layer and selected-layer are two
    points along a depth curve that isn't necessarily monotonic in between,
    not a before/after event, so a connecting arrow would imply a direction
    of travel that isn't actually there. (For an actual sample-level
    before/after trajectory, see plot_h5_revision_trajectory.py.)"""
    for sense in (0, 1):
        pts = coords[labels == sense]
        ax.scatter(
            pts[:, 0], pts[:, 1], s=13, alpha=0.75, color=SENSE[sense],
            edgecolor="white", linewidth=0.25,
        )
        centroid = pts.mean(axis=0)
        ax.scatter(
            centroid[0], centroid[1], s=45, marker="X",
            color=SENSE[sense], edgecolor=INK, linewidth=0.6, zorder=4,
        )

    ax.text(
        0.03, 0.03, f"sep {ratio:.2f}×", transform=ax.transAxes, fontsize=7,
        color=INK, va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
    )
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.grid(color=GRID, linewidth=0.5)
    # Only the selected (H1-best) layer panel gets a colored border; the
    # average-plateau-layer panel stays plain so the highlight color
    # unambiguously means "this is the selected layer", not "this is the
    # architecture" (which the curve color to its left already shows).
    if highlight:
        ax.set_title(title, fontsize=8, fontweight="bold", pad=2, color=BEST_LAYER_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(BEST_LAYER_COLOR)
            spine.set_linewidth(2.2)
    else:
        ax.set_title(title, fontsize=8, fontweight="bold", pad=2, color=INK)


def _model_curves(results_dir: Path, safe_model: str, words, subdir: str = "activations"):
    curves = {}
    depth_grid = None
    layers = None
    for word in words:
        depth, _, norm_ratio, word_layers, _ = _separation_curve(results_dir, word, safe_model, subdir)
        curves[word] = norm_ratio
        depth_grid = depth
        layers = word_layers
    return depth_grid, layers, curves


def make_progression(
    results_dir: Path, word: str, words, output_path: Path, position: str = "homonym",
) -> None:
    if position not in POSITION_SUBDIR:
        raise ValueError(f"position must be one of {list(POSITION_SUBDIR)}, got {position!r}")
    subdir = POSITION_SUBDIR[position]
    show_background = position != "homonym"

    apply_report_style()
    models = sorted(ALL_MODELS, key=lambda m: (m not in ENCODERS,))
    n = len(models)

    fig, axes = plt.subplots(
        n, 4, figsize=(14.5, 2.15 * n), sharex=False,
        constrained_layout={"h_pad": 0.12, "hspace": 0.06, "wspace": 0.04},
        gridspec_kw={"width_ratios": [1.25, 1.25, 1.0, 1.0]},
    )

    all_word_vals, all_avg_vals = [], []
    per_model_data = []
    for model in models:
        safe_model = model.replace("/", "_")
        depth_grid, layers, curves = _model_curves(results_dir, safe_model, words, subdir)
        word_curve = curves[word]
        avg_curve = np.nanmean(np.vstack([curves[w] for w in words]), axis=0)
        best_layer = _read_h1_best_layer(results_dir, safe_model, word)
        best_idx = layers.index(best_layer) if best_layer in layers else None
        avg_layer, avg_idx = _average_plateau_layer(layers, word_curve)

        bg_word_curve = bg_avg_curve = None
        if show_background:
            bg_depth_grid, _, bg_curves = _model_curves(results_dir, safe_model, words, "activations")
            bg_word_curve = bg_curves[word]
            bg_avg_curve = np.nanmean(np.vstack([bg_curves[w] for w in words]), axis=0)

        per_model_data.append(
            (model, safe_model, depth_grid, layers, word_curve, avg_curve,
             best_idx, avg_layer, avg_idx, bg_word_curve, bg_avg_curve)
        )
        all_word_vals.append(word_curve)
        all_avg_vals.append(avg_curve)
        if show_background:
            all_word_vals.append(bg_word_curve)
            all_avg_vals.append(bg_avg_curve)

    y_max = max(np.nanmax(np.concatenate(all_word_vals)), np.nanmax(np.concatenate(all_avg_vals)))
    y_max = float(y_max) * 1.08

    for row, (model, safe_model, depth_grid, layers, word_curve, avg_curve,
              best_idx, avg_layer, avg_idx, bg_word_curve, bg_avg_curve) in enumerate(per_model_data):
        arch = "encoder" if model in ENCODERS else "decoder"
        color = ARCHITECTURE[arch]
        short_name = model.split("/")[-1]
        ax_word, ax_avg, ax_avg_layer, ax_best_layer = axes[row]

        for ax, y, bg_y, title in (
            (ax_word, word_curve, bg_word_curve, f'"{word}"'),
            (ax_avg, avg_curve, bg_avg_curve, f"avg of {len(words)} words"),
        ):
            if bg_y is not None:
                ax.plot(depth_grid, bg_y, color=BACKGROUND_CURVE_COLOR, linewidth=1.3,
                        marker="o", markersize=1.8, zorder=1)
            ax.plot(depth_grid, y, color=color, linewidth=1.6, marker="o", markersize=2.6, zorder=3)
            ax.axhline(1.0, color=INK, linewidth=0.7, linestyle=":", zorder=2)
            ax.set_ylim(0, y_max)
            ax.set_xlim(-0.02, 1.02)
            ax.grid(color=GRID, linewidth=0.5)
            if row == 0:
                ax.set_title(title, fontsize=10.5, fontweight="bold")
            if row == n - 1:
                ax.set_xlabel("Relative depth (layer / final layer)")

        if best_idx is not None:
            ax_word.scatter(
                depth_grid[best_idx], word_curve[best_idx], s=70, facecolor="none",
                edgecolor=BEST_LAYER_COLOR, linewidth=1.8, zorder=5,
            )
        ax_word.scatter(
            depth_grid[avg_idx], word_curve[avg_idx], s=45, facecolor="none",
            edgecolor=color, linewidth=1.5, marker="D", zorder=5,
        )
        ax_word.set_ylabel(f"{short_name}\n({arch})", fontsize=9.5, fontweight="bold", color=INK)

        # Point clouds for the SAME word, at the two layers marked on ax_word:
        # a representative "average" plateau layer (diamond) and the
        # H1-selected "best" layer (open ring). Both are projected through
        # ONE shared PCA basis (fit across every cached layer, not just
        # these two -- see shared_layer_pca), so the panels are directly
        # comparable, same-scale snapshots rather than independently re-fit
        # panels.
        pca, normalized = shared_layer_pca(results_dir, word, safe_model, subdir, layers)
        all_coords = np.vstack([pca.transform(normalized[l][0]) for l in layers])
        pad_x = 0.10 * (all_coords[:, 0].max() - all_coords[:, 0].min() or 1.0)
        pad_y = 0.10 * (all_coords[:, 1].max() - all_coords[:, 1].min() or 1.0)
        panel_xlim = (all_coords[:, 0].min() - pad_x, all_coords[:, 0].max() + pad_x)
        panel_ylim = (all_coords[:, 1].min() - pad_y, all_coords[:, 1].max() + pad_y)

        avg_norm_X, avg_labels = normalized[avg_layer]
        avg_coords = pca.transform(avg_norm_X)
        _pca_panel(
            ax_avg_layer, avg_coords, avg_labels, word_curve[avg_idx],
            f"avg layer L{avg_layer}" if row == 0 else f"L{avg_layer}",
            panel_xlim, panel_ylim, highlight=False,
        )
        if best_idx is not None:
            best_layer_num = layers[best_idx]
            best_norm_X, best_labels = normalized[best_layer_num]
            best_coords = pca.transform(best_norm_X)
            _pca_panel(
                ax_best_layer, best_coords, best_labels, word_curve[best_idx],
                f"selected layer L{best_layer_num} ★" if row == 0 else f"L{best_layer_num} ★",
                panel_xlim, panel_ylim, highlight=True,
            )
        else:
            ax_best_layer.axis("off")

    position_label = "homonym position" if position == "homonym" else "period position"
    background_note = " Grey = homonym-position curve, for reference." if show_background else ""
    # Center the title on the axes grid itself, not the full figure -- the
    # row labels (model names) to the left of column 1 otherwise pull the
    # default figure-centered title visibly off-center from the actual plot.
    # fit_suptitle also word-wraps to the grid's own width so a long caption
    # can't force savefig(bbox_inches="tight") to pad the canvas out with
    # empty margin on both sides just to fit an overflowing title line.
    fig.canvas.draw()
    axes_left = min(ax.get_position().x0 for ax in axes[0])
    axes_right = max(ax.get_position().x1 for ax in axes[0])
    grid_width_in = fig.get_size_inches()[0] * (axes_right - axes_left)
    fit_suptitle(
        fig,
        f'Sense-centroid separation by depth, one row per model, "{word}" ({position_label}). '
        "One shared PCA basis per row. Diamond = avg-plateau layer, red ring = H1-selected layer, "
        f"dotted line = within-class spread.{background_note}",
        x=(axes_left + axes_right) / 2,
        max_width_in=grid_width_in,
        fontsize=11,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--word", default="bank", help="Word shown in the left column.")
    parser.add_argument("--words", nargs="*", default=DEFAULT_WORDS)
    parser.add_argument(
        "--output", default="results/study/figures/h1_centroid_progression.svg"
    )
    parser.add_argument(
        "--position", choices=["homonym", "period"], default="homonym",
        help="'homonym' reads results/activations/ (default, no background curve); "
        "'period' reads results/activations_final/ and draws the homonym-position "
        "curve greyed out behind it for comparison.",
    )
    args = parser.parse_args()
    make_progression(Path(args.results_dir), args.word, args.words, Path(args.output), args.position)


if __name__ == "__main__":
    main()
