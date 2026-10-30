"""Garden path versus coherent context as a walk across an attractor
landscape, for one matched item pair, seen from above (PC1 x PC2).

The two sense regions are drawn as basins instead of flat coloured clouds:
the depth at (PC1, PC2) is the negative kernel density of the profile
sentences there, so a dense sense cluster is a deep well. Depth is shown the
way a map shows it, by contour lines and by colour: colour saturation
follows depth, and its hue says which sense's well it is. Profile points,
centroids and both paths are plotted at their published 2D positions.

The depth is a display aid only: the density of 40 profile sentences in a
two-component projection. It is not an energy or any other model quantity
(see web_export/README.md, "Density landscape"). The margins panel on the
right carries the full-space values the study actually reports.

Everything is read from web_export/garden_path_landscapes.json, so no hidden
states, model or GPU are needed. The density is re-evaluated from the
published profile points and bandwidth on a finer grid than the 33 x 33 one
in the file, which only smooths the drawn landscape.

    python -m analysis.plot_garden_path_landscape --models FacebookAI_roberta-large --words bank
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from analysis.trajectory_plot_common import sense_label
from utils.visual_style import INK, MUTED, NEUTRAL, SENSE, apply_report_style

LANDSCAPE_PATH = Path("web_export/garden_path_landscapes.json")
META_PATH = Path("web_export/meta.json")

STAGE_SIZES = (46, 78, 120)
PATH_STYLE = {
    "conflicting": {"color": INK, "label": "garden path (conflicting opening)"},
    "coherent_control": {"color": "#1B998B", "label": "coherent context (matched control)"},
}
SURFACE_GRID = 300
CROP_DEPTH = 0.12  # the frame keeps everything at least this deep, plus both paths
CONTOUR_LEVELS = np.linspace(-0.9, -0.1, 9)


def kde(points_2d: np.ndarray, at: np.ndarray, bandwidth: float) -> np.ndarray:
    """Same isotropic Gaussian estimate as scripts.build_geometry_exports
    (integrates to 1), evaluated at arbitrary [..., 2] positions."""
    d2 = ((at[..., None, :] - points_2d) ** 2).sum(axis=-1)
    return np.exp(-d2 / (2.0 * bandwidth ** 2)).mean(axis=-1) / (2.0 * np.pi * bandwidth ** 2)


class Landscape:
    """Depth and colour of the landscape at any (PC1, PC2)."""

    def __init__(self, profile_xy: np.ndarray, senses: np.ndarray, bandwidth: float, paths_xy: np.ndarray):
        extent = np.vstack([profile_xy, paths_xy])
        self.by_sense = {s: profile_xy[senses == s] for s in (0, 1)}
        self.bandwidth = bandwidth
        pad = 3.0 * bandwidth
        self._grid((extent[:, 0].min() - pad, extent[:, 0].max() + pad),
                   (extent[:, 1].min() - pad, extent[:, 1].max() + pad))
        d0, d1 = self._densities(np.stack([self.gx, self.gy], axis=-1))
        self.peak = float((0.5 * (d0 + d1)).max())
        # Frame the wells and the two paths, not the whole plateau: a single
        # stray profile sentence would otherwise shrink the wells to a corner.
        in_well = self.depth(np.stack([self.gx, self.gy], axis=-1)) < -CROP_DEPTH
        keep = np.vstack([np.column_stack([self.gx[in_well], self.gy[in_well]]), paths_xy])
        pad = 1.3 * bandwidth
        self._grid((keep[:, 0].min() - pad, keep[:, 0].max() + pad), (keep[:, 1].min() - pad, keep[:, 1].max() + pad))
        self.gz = self.depth(np.stack([self.gx, self.gy], axis=-1))

    def _grid(self, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> None:
        self.xlim, self.ylim = xlim, ylim
        self.gx, self.gy = np.meshgrid(np.linspace(*xlim, SURFACE_GRID), np.linspace(*ylim, SURFACE_GRID))

    def contains(self, xy: np.ndarray) -> np.ndarray:
        return ((xy[:, 0] >= self.xlim[0]) & (xy[:, 0] <= self.xlim[1])
                & (xy[:, 1] >= self.ylim[0]) & (xy[:, 1] <= self.ylim[1]))

    def _densities(self, at: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return kde(self.by_sense[0], at, self.bandwidth), kde(self.by_sense[1], at, self.bandwidth)

    def depth(self, at: np.ndarray) -> np.ndarray:
        """0 on the empty plateau, -1 at the bottom of the deepest well."""
        d0, d1 = self._densities(np.asarray(at, dtype=float))
        return -0.5 * (d0 + d1) / self.peak

    def colors(self) -> np.ndarray:
        """Depth sets how far the colour departs from the neutral plateau;
        the sense that owns the density there sets the hue."""
        d0, d1 = self._densities(np.stack([self.gx, self.gy], axis=-1))
        share0 = (d0 / np.maximum(d0 + d1, 1e-12))[..., None]
        hue = share0 * np.array(to_rgb(SENSE[0])) + (1.0 - share0) * np.array(to_rgb(SENSE[1]))
        t = (-self.gz[..., None]) ** 0.6
        return (1.0 - t) * np.array(to_rgb(NEUTRAL)) + t * hue


def draw_landscape(ax, cell: Dict, item: Dict, word: str) -> Landscape:
    profile = np.asarray(cell["profile"]["points"])[:, :2]
    senses = np.asarray(cell["profile"]["senses"])
    paths = {name: np.asarray(item[name]["points"])[:, :2] for name in PATH_STYLE}
    landscape = Landscape(
        profile, senses, float(cell["landscape"]["bandwidth"]), np.vstack(list(paths.values()))
    )

    ax.imshow(
        landscape.colors(), extent=(*landscape.xlim, *landscape.ylim), origin="lower",
        interpolation="bilinear", zorder=0,
    )
    contours = ax.contour(
        landscape.gx, landscape.gy, landscape.gz, levels=CONTOUR_LEVELS,
        colors=[INK], linewidths=0.6, linestyles="solid", alpha=0.4, zorder=1,
    )
    ax.clabel(contours, levels=CONTOUR_LEVELS[::2], fmt="%.1f", fontsize=6.5, inline_spacing=2)

    for sense in (0, 1):
        pts = profile[senses == sense]
        pts = pts[landscape.contains(pts)]
        ax.scatter(
            pts[:, 0], pts[:, 1], s=15, color=SENSE[sense], edgecolor="white", linewidth=0.5, alpha=0.9,
            zorder=2, label=f"profile sentences, {sense_label(word, sense)}",
        )
        centroid = np.asarray(cell["profile"]["centroids"])[sense, :2]
        ax.scatter(
            centroid[0], centroid[1], s=170, marker="X", color=SENSE[sense], edgecolor=INK, linewidth=1.1, zorder=3,
        )

    for name, style in PATH_STYLE.items():
        line = paths[name]
        ax.plot(line[:, 0], line[:, 1], color="white", linewidth=4.4, alpha=0.9, solid_capstyle="round", zorder=4)
        ax.plot([], [], color=style["color"], linewidth=2.2, label=style["label"])
        for start, end in zip(line[:-1], line[1:]):
            ax.annotate(
                "", xy=end, xytext=start, zorder=5,
                arrowprops={"arrowstyle": "-|>", "color": style["color"], "linewidth": 2.2,
                            "shrinkA": 0, "shrinkB": 5, "mutation_scale": 13},
            )
        ax.scatter(
            line[:, 0], line[:, 1], s=STAGE_SIZES, color=style["color"], edgecolor="white", linewidth=1.0, zorder=6,
        )
        for number, point in enumerate(line, start=1):
            ax.annotate(
                str(number), point, xytext=(0, 7), textcoords="offset points", color=style["color"],
                fontsize=9, fontweight="bold", ha="center", va="bottom", zorder=7,
            )

    ax.set_xlim(landscape.xlim)
    ax.set_ylim(landscape.ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    return landscape


def draw_margins(ax, item: Dict, word: str) -> None:
    """The study's own numbers for the same two paths: the plane the
    landscape lives in keeps a minority of the variance, so on-screen
    position and full-space margin can disagree."""
    target = int(item["target_sense"])
    stages = item["conflicting"]["stage_names"]
    xs = np.arange(len(stages))
    ax.axhspan(0, 1, color=SENSE[target], alpha=0.07, linewidth=0)
    ax.axhspan(-1, 0, color=SENSE[1 - target], alpha=0.07, linewidth=0)
    ax.axhline(0, color=MUTED, linewidth=0.9)
    for name, style in PATH_STYLE.items():
        margins = item[name]["full_space_correct_margins"]
        ax.plot(xs, margins, color=style["color"], linewidth=2.0, zorder=3)
        ax.scatter(xs, margins, s=STAGE_SIZES, color=style["color"], edgecolor="white", linewidth=1.0, zorder=4)
    limit = max(0.15, 1.25 * max(abs(m) for name in PATH_STYLE for m in item[name]["full_space_correct_margins"]))
    ax.set_ylim(-limit, limit)
    ax.set_xlim(-0.35, len(stages) - 0.65)
    ax.set_xticks(xs, [f"{i + 1} {stage}" for i, stage in enumerate(stages)])
    ax.set_ylabel("full-space margin toward the final sense")
    ax.text(0.02, 0.97, f"{sense_label(word, target)} side", transform=ax.transAxes, va="top",
            fontsize=8.5, color=SENSE[target])
    ax.text(0.02, 0.03, f"{sense_label(word, 1 - target)} side", transform=ax.transAxes, va="bottom",
            fontsize=8.5, color=SENSE[1 - target])
    ax.set_title("Same two paths, full hidden space", loc="left", fontsize=10.5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


def make_figure(cell: Dict, model_label: str, word: str, output: Path, item_id: Optional[str] = None) -> str:
    apply_report_style()
    item_id = item_id or cell["default_item_id"]
    item = next(entry for entry in cell["items"] if entry["item_id"] == item_id)

    fig = plt.figure(figsize=(13.2, 7.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.0], left=0.06, right=0.97, top=0.88, bottom=0.24, wspace=0.2)
    ax = fig.add_subplot(grid[0])
    landscape = draw_landscape(ax, cell, item, word)
    n_outside = int((~landscape.contains(np.asarray(cell["profile"]["points"])[:, :2])).sum())
    draw_margins(fig.add_subplot(grid[1]), item, word)

    ax.set_title("Sense landscape seen from above; contour labels give depth", loc="left",
                 fontsize=10.5, fontweight="bold")
    ax.legend(loc="best", fontsize=8, frameon=True, framealpha=0.85, facecolor="white", edgecolor="none")
    fig.suptitle(
        f"Garden path versus coherent context on the sense landscape: {model_label}, "
        f"“{word}”, layer {cell['analysis_layer']}",
        x=0.03, ha="left", fontsize=13, fontweight="bold",
    )
    for row, name in enumerate(PATH_STYLE):
        fig.text(0.03, 0.115 - 0.035 * row, PATH_STYLE[name]["label"].split(" (")[0] + ":",
                 color=PATH_STYLE[name]["color"], fontsize=9.5, fontweight="bold")
        fig.text(0.145, 0.115 - 0.035 * row, item[f"{name}_sentence"], fontsize=9.5)
    explained = 100 * sum(cell["projection"]["explained_variance_ratio"][:2])
    fig.text(
        0.03, 0.02,
        "1 prime, 2 homonym, 3 resolver. Depth is the negative density of the 40 profile sentences in the PC1–PC2 "
        f"plane ({explained:.0f}% of profile variance), scaled so the deepest\nwell is −1: a display aid, "
        "not a model energy. X = sense centroid."
        + (f" {n_outside} isolated profile sentence(s) lie outside the frame." if n_outside else ""),
        fontsize=8.5, color=MUTED,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    # Not with_suffix: model ids contain dots ("Qwen2.5-7B") that it would cut at.
    fig.savefig(f"{output}.svg")
    fig.savefig(f"{output}.png", dpi=190)
    plt.close(fig)
    return item_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["FacebookAI_roberta-large", "Qwen_Qwen2.5-7B"],
                        help="model ids as in web_export/meta.json")
    parser.add_argument("--words", nargs="+", default=["bank"])
    parser.add_argument("--item-id", default=None, help="defaults to each cell's default_item_id")
    parser.add_argument("--landscapes", default=str(LANDSCAPE_PATH))
    parser.add_argument("--output-dir", default="figures/supplementary")
    args = parser.parse_args()

    with open(args.landscapes) as f:
        cells = json.load(f)["cells"]
    with open(META_PATH) as f:
        names = {model["id"]: model["name"] for model in json.load(f)["models"]}

    for model in args.models:
        for word in args.words:
            output = Path(args.output_dir) / f"h5_garden_path_landscape_{model}_{word}"
            item_id = make_figure(cells[model][word], names.get(model, model), word, output, args.item_id)
            print(f"[{model}/{word}] item={item_id} -> {output}.png")


if __name__ == "__main__":
    main()
