"""Shared visual roles for the study figures.

The hues are derived from the original report palette, but color is never
allowed to mean a generic "group A/group B".  Each role below keeps the same
meaning across figures; marker fill and line style carry context/position.
"""

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


SENSE = {0: "#4A3AA7", 1: "#D59A18"}  # violet, report-derived amber
ARCHITECTURE = {"encoder": "#277F70", "decoder": "#8A607C"}
METHOD = {
    "adequacy": "#303038",
    "gdv": "#8A6396",
    "oracle": "#111111",
    "final": "#898781",
    "supervised": "#277F70",
}
OUTCOME = {
    "correct": "#4F8A62",
    "ambiguous": "#A6A39B",
    "wrong": "#B65D50",
}
AGREEMENT = {"agree": "#496E5B", "disagree": "#E9E3D7"}
INK = "#242329"
MUTED = "#77746E"
GRID = "#E7E4DE"
BACKGROUND = "#FFFFFF"
NEUTRAL = "#F5F2EC"

# One selected-layer marker color, used everywhere a single chosen layer is
# highlighted against a background of other layers/candidates (never reused
# for anything else, so "red border/ring" always means exactly this).
SELECTED_LAYER_COLOR = "#C1121F"

PRIOR_CMAP = LinearSegmentedColormap.from_list(
    "sense_prior",
    [SENSE[1], "#F7F4EE", SENSE[0]],
)

# Distinct from PRIOR_CMAP's amber/violet AND from ARCHITECTURE's green/violet
# on purpose. PRIOR_CMAP always means "which sense does this lean toward"
# (H0/H1); ARCHITECTURE always means encoder-vs-decoder. Depth/shift is a
# third, unrelated quantity (e.g. how far a layer choice sits from another),
# so it gets its own hue family entirely -- grey/pink is nowhere near
# amber-violet or green-violet on the color wheel, so it can't be misread as
# either.
DEPTH_CMAP = LinearSegmentedColormap.from_list(
    "depth_shift",
    [MUTED, "#F7F4EE", "#D0698F"],
)


def fit_suptitle(fig, text, x=0.5, max_width_in=None, fontsize=11, fontweight="bold"):
    """fig.suptitle, but word-wrapped to actually fit max_width_in (defaults
    to the figure's own width) instead of overflowing past the figure edge.

    A suptitle longer than the figure width doesn't get clipped or
    auto-wrapped by matplotlib -- it just renders past the edge, and
    savefig(bbox_inches="tight") then expands the saved canvas to include
    that overflow. The result is a much wider image than the actual axes
    grid, with the grid looking small and centered in a sea of margin. This
    measures the real rendered width of each candidate line with the
    figure's own renderer (exact, not a guessed chars-per-inch constant) and
    breaks lines before they'd exceed the target width.
    """
    if max_width_in is None:
        max_width_in = fig.get_size_inches()[0]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    probe = fig.text(0, 0, "", fontsize=fontsize, fontweight=fontweight)

    def _width_in(s: str) -> float:
        probe.set_text(s)
        bbox = probe.get_window_extent(renderer=renderer)
        return bbox.width / fig.dpi

    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and _width_in(candidate) > max_width_in:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    probe.remove()

    return fig.suptitle("\n".join(lines), x=x, fontsize=fontsize, fontweight=fontweight)


def apply_report_style():
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "axes.axisbelow": True,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )
