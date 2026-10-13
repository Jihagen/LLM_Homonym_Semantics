"""Quantify how far apart the two sense centroids are, per layer, and
compare that trajectory across all sampled models for one word.

Two views are plotted side by side:

1. Raw inter-centroid Euclidean distance ||c1 - c0||_2 in the original
   hidden-state space. Shows the literal starting point (does it begin
   near zero?) but is not comparable across models, since absolute
   activation scale differs hugely by architecture and depth (especially
   for decoders, where a few "massive activation" outlier dimensions can
   dominate the norm at some layers).
2. That same distance normalised by the mean distance from each point to
   its own class centroid (a simple, model-agnostic separation ratio: >1
   means the two centroids are farther apart than the typical within-class
   spread). This is what actually supports cross-model comparison.

Both panels use relative depth (layer index / final layer index) on the
x-axis so models with different numbers of layers align. Each model's
H1-selected best layer (h1_summary.csv's best_layer_M) is marked.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from utils.model_registry import ALL_MODELS
from utils.visual_style import ARCHITECTURE, GRID, INK, apply_report_style

ENCODERS = {
    "answerdotai/ModernBERT-large",
    "microsoft/deberta-v3-large",
    "FacebookAI/roberta-large",
    "FacebookAI/xlm-roberta-large",
}

MODEL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _available_layers(results_dir: Path, word: str, safe_model: str, subdir: str = "activations"):
    word_dir = results_dir / subdir / word / safe_model
    return sorted(
        int(path.stem.removeprefix("layer_")) for path in word_dir.glob("layer_*.h5")
    )


def _load_layer(results_dir: Path, word: str, safe_model: str, layer: int, subdir: str = "activations"):
    path = results_dir / subdir / word / safe_model / f"layer_{layer}.h5"
    with h5py.File(path, "r") as handle:
        return handle["X"][:], handle["labels"][:]


def _read_h1_best_layer(results_dir: Path, safe_model: str, word: str):
    path = results_dir / "study" / "H1" / safe_model / "h1_summary.csv"
    if not path.exists():
        return None
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["word"] == word:
                return int(row["best_layer_M"])
    return None


def _separation_curve(results_dir: Path, word: str, safe_model: str, subdir: str = "activations"):
    layers = _available_layers(results_dir, word, safe_model, subdir)
    last_layer = layers[-1]
    relative_depth, raw_dist, norm_ratio = [], [], []
    for layer in layers:
        X, labels = _load_layer(results_dir, word, safe_model, layer, subdir)
        c0 = X[labels == 0].mean(axis=0)
        c1 = X[labels == 1].mean(axis=0)
        d_raw = float(np.linalg.norm(c1 - c0))
        within = np.concatenate([
            np.linalg.norm(X[labels == 0] - c0, axis=1),
            np.linalg.norm(X[labels == 1] - c1, axis=1),
        ]).mean()
        relative_depth.append(layer / last_layer)
        raw_dist.append(d_raw)
        norm_ratio.append(d_raw / within if within > 0 else np.nan)
    return (
        np.array(relative_depth),
        np.array(raw_dist),
        np.array(norm_ratio),
        layers,
        last_layer,
    )


def center_scale_params(X: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, float]:
    """(center, scale) for _normalize_layer's rule: center on the midpoint
    of the two class centroids, scale by the mean within-class distance to
    centroid, so "1 unit" means "one typical within-class spread". Exposed
    separately from _normalize_layer so the SAME params derived from a
    reliable, large-n reference set (e.g. profiling sentences) can be
    applied to a different, smaller array (e.g. a handful of trajectory
    items) that shouldn't be normalised from its own noisy few-item split.
    """
    c0 = X[labels == 0].mean(axis=0)
    c1 = X[labels == 1].mean(axis=0)
    within = np.concatenate([
        np.linalg.norm(X[labels == 0] - c0, axis=1),
        np.linalg.norm(X[labels == 1] - c1, axis=1),
    ]).mean()
    center = (c0 + c1) / 2
    scale = within if within > 0 else 1.0
    return center, scale


def _normalize_layer(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Center on the midpoint of the two class centroids and scale by the
    mean within-class distance to centroid, so "1 unit" means "one typical
    within-class spread" -- the same normalisation _separation_curve already
    uses for its cross-layer/cross-architecture norm_ratio. This is what
    makes pooling multiple layers into one PCA fit meaningful: without it, a
    layer with much larger raw activation magnitude (e.g. a decoder's
    "massive activation" outlier dimensions in late layers) would dominate
    the pooled fit purely on scale, not on how separated its senses are.
    """
    center, scale = center_scale_params(X, labels)
    return (X - center) / scale


def shared_layer_pca(
    results_dir: Path,
    word: str,
    safe_model: str,
    subdir: str = "activations",
    layers: Optional[Sequence[int]] = None,
) -> Tuple[PCA, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """Fit one PCA(2) across every requested layer's activations (each
    layer normalised per _normalize_layer first, then pooled), so a single
    shared 2D coordinate system covers every layer -- flipping between
    per-layer panels shows real movement instead of independently-refit,
    arbitrarily rotated/rescaled snapshots.

    Returns (fitted_pca, {layer: (normalized_X, labels)}); callers project a
    given layer's display coordinates with ``pca.transform(normalized_X)``.
    GDV and any other per-layer statistic should still be computed from the
    raw (un-normalised) activations loaded separately -- this normalisation
    and shared fit exist only for the 2D display projection.
    """
    available = _available_layers(results_dir, word, safe_model, subdir)
    layers = list(layers) if layers is not None else available
    normalized: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for layer in layers:
        X, labels = _load_layer(results_dir, word, safe_model, layer, subdir)
        normalized[layer] = (_normalize_layer(X, labels), labels)
    pooled = np.vstack([norm_X for norm_X, _ in normalized.values()])
    pca = PCA(n_components=2).fit(pooled)
    return pca, normalized


def make_comparison(results_dir: Path, word: str, output_path: Path) -> None:
    apply_report_style()
    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)

    for i, model in enumerate(ALL_MODELS):
        safe_model = model.replace("/", "_")
        arch = "encoder" if model in ENCODERS else "decoder"
        color = ARCHITECTURE[arch]
        marker = MODEL_MARKERS[i % len(MODEL_MARKERS)]
        linestyle = "-" if arch == "encoder" else "--"

        depth, raw_dist, norm_ratio, layers, last_layer = _separation_curve(
            results_dir, word, safe_model
        )
        short_name = model.split("/")[-1]

        ax_raw.plot(
            depth, raw_dist, color=color, linestyle=linestyle, linewidth=1.4,
            marker=marker, markersize=4, alpha=0.85, label=short_name,
        )
        ax_norm.plot(
            depth, norm_ratio, color=color, linestyle=linestyle, linewidth=1.4,
            marker=marker, markersize=4, alpha=0.85, label=short_name,
        )

        best_layer = _read_h1_best_layer(results_dir, safe_model, word)
        if best_layer is not None and best_layer in layers:
            idx = layers.index(best_layer)
            ax_norm.scatter(
                depth[idx], norm_ratio[idx], s=140, facecolor="none",
                edgecolor=color, linewidth=2.0, zorder=5,
            )

    ax_raw.set_title("Raw inter-centroid distance", loc="left", fontweight="bold")
    ax_raw.set_ylabel(r"$\|c_1-c_0\|_2$ (original hidden space)")
    ax_norm.set_title(
        "Separation relative to within-class spread", loc="left", fontweight="bold"
    )
    ax_norm.set_ylabel(r"$\|c_1-c_0\|_2$ / mean within-class distance to centroid")
    ax_norm.axhline(1.0, color=INK, linewidth=0.8, linestyle=":")
    ax_norm.text(
        0.01, 1.02, "centroid gap = typical within-class spread",
        transform=ax_norm.get_yaxis_transform(), fontsize=7.5, color=INK, va="bottom",
    )

    for ax in (ax_raw, ax_norm):
        ax.set_xlabel("Relative depth (layer / final layer)")
        ax.set_xlim(-0.02, 1.02)
        ax.grid(color=GRID, linewidth=0.6)

    handles, labels = ax_raw.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=4, fontsize=8.5,
        title="Encoder (solid) / decoder (dashed); open ring = H1-selected best layer",
        title_fontsize=8.5,
    )
    fig.suptitle(
        f"Sense-centroid separation across depth · {word}", fontsize=13, fontweight="bold"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--word", default="bank")
    parser.add_argument(
        "--output", default="results/study/figures/h1_centroid_separation_bank.svg"
    )
    args = parser.parse_args()
    make_comparison(Path(args.results_dir), args.word, Path(args.output))


if __name__ == "__main__":
    main()
