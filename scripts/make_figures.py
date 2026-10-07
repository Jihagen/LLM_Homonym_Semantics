"""Regenerate the published figures from the released tables only.

    python -m scripts.make_figures            # or: make figures

Inputs are the files under ``data/processed/`` and ``data/``; nothing is read
from ``results/``, no model is loaded, and torch/transformers are not needed.
The plotting code in ``analysis/`` expects the pipeline's ``results/`` layout,
so the released tables are first unpacked into that layout under
``build/released_results/`` (see ``scripts.release_layout``) and the unchanged
plotting functions are pointed at it.

Outputs (SVG and PNG for each):

    figures/                 the figures of the project report, plus the two
                             companion figures of the report's figure set
    figures/supplementary/   exploratory figures that are not in the report
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

# Fixed SVG metadata so repeated runs produce identical files.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from scripts import release_layout as layout  # noqa: E402
from scripts.release_layout import GEOMETRY_EXAMPLE, ROOT, TRAJECTORY_EXAMPLES, WORDS  # noqa: E402
from utils.visual_style import apply_report_style  # noqa: E402

REPORT_FIGURES = [
    "h0_prior_matrix",
    "h1_nested_layer_selection",
    "semantic_layer_atlas",
    "h2_gdv_geometric_example_overlap_annotated",
    "h2_decision_scores",
    "h3_paired_context_effect",
    "h5_fixed_sentinel_trajectories",
    "h5_update_controls",
]
COMPANION_FIGURES = [
    "h2_gdv_geometric_example",
    "h2_layer_curves",
    "h4_conditional_transitions",
]
SUPPLEMENTARY_FIGURES = [
    "q4_layer_vs_homonym_curves",
    "q4_selected_layer_shift",
    *[f"h3_context_trajectory_{model}_{word}" for model, word in TRAJECTORY_EXAMPLES],
    *[f"h5_revision_trajectory_{model}_{word}" for model, word in TRAJECTORY_EXAMPLES],
    *[f"h5_revision_trajectory_progressive_{model}_{word}" for model, word in TRAJECTORY_EXAMPLES],
]


def _style() -> None:
    plt.rcdefaults()
    plt.rcParams["svg.hashsalt"] = "llm-homonym-semantics"
    apply_report_style()


def make_report_figures(results: Path, out: Path) -> None:
    from analysis import (
        plot_geometry_audit,
        plot_h3_h4_audit,
        plot_prior_matrix,
        plot_report_h1_h5,
        plot_semantic_layer_atlas,
    )

    _style()
    plot_prior_matrix.make_figure(results, out / "h0_prior_matrix.svg")

    _style()
    plot_report_h1_h5.plot_h1(results, out / "h1_nested_layer_selection.svg")
    cells = plot_report_h1_h5._load_h5_cells(results)
    plot_report_h1_h5.plot_h5_trajectories(cells, out / "h5_fixed_sentinel_trajectories.svg")
    plot_report_h1_h5.plot_h5_effects(cells, out / "h5_update_controls.svg")

    _style()
    plot_semantic_layer_atlas.make_figure(results, out / "semantic_layer_atlas.svg")

    example = GEOMETRY_EXAMPLE
    _style()
    plot_geometry_audit.make_figures(
        results, example["model"], example["word"], list(example["layers"]),
        out / "h2_gdv_geometric_example.svg",
    )
    plot_geometry_audit.make_figures(
        results, example["model"], example["word"], list(example["layers"]),
        out / "h2_gdv_geometric_example_overlap_annotated.svg",
        annotate_overlaps_only=True,
    )

    _style()
    plot_h3_h4_audit.plot_h3(results, out / "h3_paired_context_effect.svg")
    plot_h3_h4_audit.plot_h4(results, out / "h4_conditional_transitions.svg")


def make_supplementary_figures(results: Path, out: Path) -> None:
    from analysis import (
        plot_h3_context_trajectory,
        plot_h5_revision_trajectory,
        plot_q4_curves,
        plot_q4_selected_layer_heatmap,
    )
    from utils.model_registry import ALL_MODELS

    hf_name = {name.replace("/", "_"): name for name in ALL_MODELS}

    _style()
    plot_q4_curves.make_comparison(results, "bank", WORDS, out / "q4_layer_vs_homonym_curves.svg")
    _style()
    plot_q4_selected_layer_heatmap.make_figure(results, out / "q4_selected_layer_shift.svg")

    for model, word in TRAJECTORY_EXAMPLES:
        _style()
        plot_h3_context_trajectory.make_trajectory_figure(
            layout.load_h3_trajectory_states(model, word), hf_name[model], word,
            out / f"h3_context_trajectory_{model}_{word}.svg",
        )
        data = layout.load_h5_trajectory_states(model, word)
        plot_h5_revision_trajectory.make_trajectory_figure(
            data, hf_name[model], word, out / f"h5_revision_trajectory_{model}_{word}.svg",
        )
        plot_h5_revision_trajectory.make_trajectory_figure_progressive(
            data, hf_name[model], word, out / f"h5_revision_trajectory_progressive_{model}_{word}.svg",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(ROOT / "figures"))
    parser.add_argument("--build-dir", default=str(ROOT / "build" / "released_results"))
    parser.add_argument("--skip-supplementary", action="store_true")
    args = parser.parse_args()

    build = Path(args.build_dir)
    if build.exists():
        shutil.rmtree(build)
    results = layout.materialize_results_tree(build)

    out = Path(args.output_dir)
    make_report_figures(results, out)
    if not args.skip_supplementary:
        make_supplementary_figures(results, out / "supplementary")

    expected = [out / f"{stem}{ext}" for stem in REPORT_FIGURES + COMPANION_FIGURES for ext in (".svg", ".png")]
    if not args.skip_supplementary:
        expected += [
            out / "supplementary" / f"{stem}{ext}" for stem in SUPPLEMENTARY_FIGURES for ext in (".svg", ".png")
        ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise SystemExit("figures were not written: " + ", ".join(missing))
    print(f"wrote {len(expected)} files to {out}")


if __name__ == "__main__":
    main()
