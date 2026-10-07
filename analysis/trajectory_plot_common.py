"""Shared drawing helpers for the H3/H5 sample-trajectory point-cloud
figures (plot_h5_revision_trajectory.py, plot_h3_context_trajectory.py).

Addresses specific readability problems found when reviewing the first
version of these figures:

- It was not obvious why a point "close to" a centroid in the 2D plot could
  still be marked wrong (or vice versa): the real scoring happens in the
  full-dimensional space, and a 2-component PCA projection does not
  preserve most of that distance information. draw_sense_background makes
  the 2D-only decision region visible as a soft gradient (so the reader can
  SEE what the 2D view implies) with an explained-variance annotation
  (explained_variance_caption) so the gap between "looks close here" and
  "was scored correct" is explained rather than silently confusing.
- Lines crossing with no visible direction. draw_trajectory adds
  arrowheads and colors each SEGMENT (not just the final point) by whether
  that step's real (full-dimensional) margin moved toward or away from the
  correct sense -- distinct from the dot fill color, which still encodes
  identity (which sense this item belongs to).
- "sense 0 / sense 1" is meaningless without the underlying word. sense_label
  gives the actual meaning (e.g. "river bank" / "financial bank") for the
  words currently used in these figures.
"""

import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils.visual_style import GRID, INK, OUTCOME, PRIOR_CMAP, SENSE


def load_or_extract(cache_path: Path, extract_fn: Callable[[], Dict], force: bool = False) -> Dict:
    """Cache wrapper for the GPU extraction step. extract_fn (which loads
    the model checkpoint and runs forward passes) only runs if no cache
    exists yet or force=True; otherwise the previously pickled data is
    loaded straight from disk. Every plotting/visual-design change is pure
    CPU math downstream of the raw activations, so once a model/word has
    been extracted once, no later change to arrows, colors, backgrounds,
    legends, etc. should ever need the GPU again.
    """
    if cache_path.exists() and not force:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    data = extract_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    return data

# Inferred from the actual example sentences in data/garden_path_sentences.json
# and data/paired_sentences.json (no sense-name field exists in the data
# itself; these are for figure labelling only, never used in any scoring).
SENSE_MEANINGS = {
    "bank": {0: "river bank", 1: "financial bank"},
    "bark": {0: "dog bark", 1: "tree bark"},
    "bat": {0: "sports bat", 1: "animal bat"},
    "crane": {0: "bird crane", 1: "construction crane"},
    "spring": {0: "season spring", 1: "water spring"},
    "match": {0: "sports match", 1: "fire match"},
    "pitch": {0: "sports pitch", 1: "sales pitch"},
}


def sense_label(word: str, sense: int) -> str:
    return SENSE_MEANINGS.get(word, {}).get(sense, f"sense {sense}")


def draw_sense_background(
    ax, xlim: Tuple[float, float], ylim: Tuple[float, float],
    centroid0: np.ndarray, centroid1: np.ndarray, resolution: int = 200,
):
    """Soft gradient toward each plotted centroid, using ONLY 2D distance in
    this display -- not a reconstruction of the true high-dimensional
    decision surface (that can't be recovered from a 2-component
    projection). This is deliberately the same simple heuristic a reader's
    eye applies ("which X is this point nearer"), made explicit as a
    visible field precisely so it can be caveated: see
    explained_variance_caption for why it can disagree with the actual
    (correct) high-dimensional scoring.

    Deliberately NO hard boundary line: a real decision surface, seen only
    through 2 of many dimensions, does not project down to a straight
    bisector -- it can curve, fold, or vanish into the dimensions this view
    doesn't show. Drawing a crisp line at the 2D-equidistant contour would
    assert a precision the projection doesn't have, and is exactly what
    made points on the "wrong" side of that line but correctly scored (in
    the real, full-dimensional space) look like a bug instead of the
    expected consequence of viewing ~35-45% of the variance. The continuous
    gradient communicates "roughly which zone" without a false-precision
    edge; ring/arrow color (the real, full-dimensional scoring) is the
    source of truth, not the 2D field.

    PRIOR_CMAP runs low-value=SENSE[1] (amber) -> high-value=SENSE[0]
    (violet), so the field must be POSITIVE near centroid0 (violet) and
    NEGATIVE near centroid1 (amber) -- get this backwards and the gradient
    paints each centroid's neighbourhood in the OTHER sense's color, which
    reads as "the centroids are colored wrong" even though the X markers
    themselves are fine.
    """
    xs = np.linspace(*xlim, resolution)
    ys = np.linspace(*ylim, resolution)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    d0 = np.linalg.norm(grid - centroid0, axis=1)
    d1 = np.linalg.norm(grid - centroid1, axis=1)
    denom = d0 + d1
    denom[denom < 1e-9] = 1e-9
    field = ((d1 - d0) / denom).reshape(gx.shape)
    im = ax.imshow(
        field, extent=(*xlim, *ylim), origin="lower", cmap=PRIOR_CMAP,
        vmin=-1, vmax=1, alpha=0.32, aspect="auto", zorder=0, interpolation="bilinear",
    )
    return im


def explained_variance_caption(pca) -> str:
    ratios = getattr(pca, "explained_variance_ratio_", None)
    if ratios is None or len(ratios) < 2:
        return ""
    total = (ratios[0] + ratios[1]) * 100
    return (
        f"PC1+PC2 explain {total:.0f}% of variance here -- the background gradient and "
        "on-screen distances are a 2D approximation; ring/arrow colors reflect the actual "
        "scoring in the full-dimensional space, and can disagree with what looks closest above."
    )


def segment_visual_agreement(
    points: np.ndarray, correct_2d: np.ndarray, wrong_2d: np.ndarray, margins: Sequence[float],
) -> List[bool]:
    """For each segment, does the 2D-displayed movement (toward this item's
    plotted correct-sense centroid, vs. its plotted wrong-sense centroid)
    point the same way as the true (full-dimensional) margin used to color
    the arrow? PCA to 2 components keeps only ~35-45% of the variance in
    these figures (see explained_variance_caption), so the two CAN
    legitimately disagree: the real margin can improve because of movement
    along a dimension this particular 2D view doesn't show, even while the
    on-screen position appears to drift the other way. That is not a
    scoring error -- it is what "only 2 of many dimensions are visible"
    means -- but a reader has no way to tell the difference from the arrow
    alone, so draw_trajectory renders disagreeing segments with a dotted
    (rather than solid) line when this is passed in.
    """
    proxy = [float(np.linalg.norm(p - wrong_2d) - np.linalg.norm(p - correct_2d)) for p in points]
    agrees = []
    for i in range(len(points) - 1):
        true_dir = margins[i + 1] - margins[i]
        visual_dir = proxy[i + 1] - proxy[i]
        agrees.append((true_dir > 0) == (visual_dir > 0))
    return agrees


CONFIDENT_MARGIN = 0.3  # |margin| at or beyond this is drawn as a fully bold/opaque ring


def _ring_style(margin_abs: float) -> Tuple[float, float]:
    """Map |final margin| (bounded [0, 1] by normalized_adequacy_margin) to
    (linewidth, alpha) for the outcome ring, so a razor-thin margin -- the
    exact case where a lossy 2D view can legitimately show the point on the
    "wrong" side while it is still, barely, correctly resolved in the real
    full-dimensional space -- reads visually as thin and pale rather than
    looking just as certain as a robust, unambiguous margin.
    """
    t = min(margin_abs, CONFIDENT_MARGIN) / CONFIDENT_MARGIN
    return 0.8 + t * 1.8, 0.32 + t * 0.63


def draw_trajectory(
    ax,
    points: np.ndarray,
    margins: Sequence[float],
    dot_color: str,
    stage_sizes: Sequence[float],
    final_outcome_correct: Optional[bool] = None,
    segment_alphas: Optional[Sequence[float]] = None,
    visual_agrees: Optional[Sequence[bool]] = None,
) -> None:
    """points: [n_stages, 2] display coordinates, in temporal order.
    margins: real (full-dimensional) signed margin toward the correct sense
    at each stage, same length as points -- used only to color each SEGMENT
    by whether that step moved the true margin toward (green) or away from
    (red) correct, decoupled from dot_color (which identifies the item's
    sense/group, not correctness). final_outcome_correct, if given, draws a
    ring around the last point summarising the end state, with thickness
    and opacity scaled by |margins[-1]| (see _ring_style) so a barely-over-
    the-line margin reads as thin/pale rather than as confident as a robust
    one. segment_alphas, if given (one value per segment, i.e.
    len(points)-1), lets a caller fade out earlier segments (e.g.
    prime->homonym) when a later panel/version would otherwise be dominated
    by lines that already had their own, less-cluttered panel earlier in a
    progressive figure. visual_agrees (see segment_visual_agreement), if
    given, draws a segment dotted instead of solid whenever the 2D
    on-screen movement disagrees with the true margin direction the arrow
    is colored by -- flagging exactly the segments where "the arrow color
    looks wrong for where the dot moved on screen" is a projection
    artifact, not a data problem.
    """
    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i + 1]
        moved_toward_correct = margins[i + 1] > margins[i]
        seg_color = OUTCOME["correct"] if moved_toward_correct else OUTCOME["wrong"]
        alpha = 0.75 if segment_alphas is None else segment_alphas[i]
        linestyle = "-" if visual_agrees is None or visual_agrees[i] else ":"
        ax.annotate(
            "", xy=p1, xytext=p0,
            arrowprops={
                "arrowstyle": "-|>", "color": seg_color, "alpha": alpha, "linewidth": 1.3,
                "linestyle": linestyle, "shrinkA": 5, "shrinkB": 5, "mutation_scale": 11,
            },
            zorder=3,
        )
    for point, size in zip(points, stage_sizes):
        ax.scatter(
            point[0], point[1], s=size, color=dot_color, alpha=0.9,
            edgecolor="white", linewidth=0.6, zorder=4,
        )
    if final_outcome_correct is not None:
        outcome_color = OUTCOME["correct"] if final_outcome_correct else OUTCOME["wrong"]
        ring_lw, ring_alpha = _ring_style(abs(margins[-1]))
        ax.scatter(
            points[-1, 0], points[-1, 1], s=stage_sizes[-1] + 60, facecolor="none",
            edgecolor=outcome_color, linewidth=ring_lw, alpha=ring_alpha, zorder=5,
        )


def draw_sense_legend(ax, word: str, stage_names: Sequence[str], stage_sizes: Sequence[float]) -> None:
    """Compact legend anchored inside the bottom-right of the axes (not
    floating text on top of the data) mapping color -> actual sense meaning,
    plus the stage-size and segment-direction key."""
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=SENSE[s], markersize=9,
                    label=sense_label(word, s))
        for s in (0, 1)
    ] + [
        plt.Line2D([0], [0], marker="o", linestyle="", color=GRID, markeredgecolor=INK,
                    markersize=max(4, np.sqrt(size) * 0.6), label=name)
        for name, size in zip(stage_names, stage_sizes)
    ] + [
        plt.Line2D([0], [0], color=OUTCOME["correct"], linewidth=2, label="step toward correct sense"),
        plt.Line2D([0], [0], color=OUTCOME["wrong"], linewidth=2, label="step away from correct sense"),
        plt.Line2D([0], [0], color=INK, linewidth=1.3, linestyle=":", alpha=0.7,
                    label="dotted = 2D view misrepresents decision boundary/arrow direction"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=INK,
                    markeredgewidth=2.6, markersize=9, alpha=0.95, label="bold ring = confidently resolved"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=INK,
                    markeredgewidth=1.0, markersize=9, alpha=0.4, label="thin/pale ring = barely resolved"),
    ]
    ax.legend(
        handles=handles, loc="lower right", fontsize=7.3, framealpha=0.92,
        facecolor="white", edgecolor=GRID, ncol=1,
    )
