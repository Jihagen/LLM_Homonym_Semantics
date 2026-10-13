"""Sample-level trajectory from homonym-position to sentence-final
("resolved") reading, for H3's L-condition and R-condition paired
sentences, side by side, at the model's H1-selected layer.

Same recipe as plot_h5_revision_trajectory.py: each position group is
centered+scaled from its own profiling split before one PCA is fit across
everything pooled, so the two panels -- and every point within them --
share one coordinate system, without the "which position" offset (H4's own
finding: local geometry changes with token position) swallowing the much
smaller sense-separation signal. Only the resolved/final-position profiling
cloud and its sense centroids are drawn as the background reference
geometry: that is the "ground truth" semantic layout H4 already establishes
decoders eventually reach.

Uses the shared drawing helpers in trajectory_plot_common.py: a background
gradient toward each centroid (explicitly caveated -- it is a 2D-only
approximation, not the true high-dimensional decision surface the ring
colors are actually scored against), arrowheads colored by whether that
step's REAL margin moved toward or away from the correct sense (each stage
scored fairly against ITS OWN position's centroids -- homonym vs.
homonym-position centroids, resolved vs. resolved-position centroids -- not
a homonym-vs-resolved-centroids cross-position score, which H4 already
shows is unreliable and produced arrows that disagreed with the plotted 2D
movement), and an in-plot legend using the word's actual sense meanings
instead of "sense 0/1".

Prediction this is built to check visually: L-condition trajectories should
already sit close to their correct resolved-position centroid even at the
homonym stage (context was available before the homonym) and move little;
R-condition trajectories should sit off in a non-committal region at the
homonym stage (right context not yet seen) and jump toward the correct
centroid only at the resolved stage -- dramatically more so for decoders
than encoders.

Requires a GPU (loads the model checkpoint); make_trajectory_figure itself
takes plain arrays and has no GPU dependency, exercised by a smoke test.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from experiments.adequacy import normalized_adequacy_margin
from hypotheses.h3_context_position import PAIRED_DATA_PATH, _load_paired_sentences, _select_layer
from hypotheses.h5_garden_path import PROFILING_DATA_PATH, _load_profiling_examples
from analysis.plot_centroid_separation import center_scale_params
from analysis.trajectory_plot_common import (
    draw_sense_background,
    draw_sense_legend,
    draw_trajectory,
    explained_variance_caption,
    load_or_extract,
    segment_visual_agreement,
)
from utils.visual_style import GRID, INK, SENSE, apply_report_style, fit_suptitle

CONDITIONS = ("L", "R")
CONDITION_LABELS = {"L": "L-condition (resolver before homonym)", "R": "R-condition (resolver after homonym)"}
STAGES = ("homonym", "resolved")
STAGE_SIZES = {"homonym": 36, "resolved": 88}


def extract_trajectory_data(
    model_name: str,
    word: str,
    results_dir: str = "results",
    paired_data_path: str = PAIRED_DATA_PATH,
    profiling_data_path: str = PROFILING_DATA_PATH,
    batch_size: int = 4,
) -> Dict:
    """GPU step. One forward pass covers the profiling cloud plus every
    L/R paired item, reading both the homonym token and the sentence-final
    token per text (get_dual_position_activations), at the single
    H1-selected layer H3/H4 themselves read out."""
    from models import get_dual_position_activations, load_model_and_tokenizer

    sentences, conditions, ids, senses, _carriers = _load_paired_sentences(paired_data_path, word)
    profile_sentences, profile_senses = _load_profiling_examples(profiling_data_path, word)

    model, tokenizer = load_model_and_tokenizer(model_name)
    layer = _select_layer(model_name, results_dir, word)

    all_texts = profile_sentences + sentences
    all_targets = [word] * len(all_texts)
    target_acts, final_acts = get_dual_position_activations(
        model, tokenizer, all_texts, all_targets, batch_size=batch_size, layer_indices=[layer],
    )
    homonym_H = target_acts[layer].numpy()
    resolved_H = final_acts[layer].numpy()
    del model, tokenizer

    n_profile = len(profile_sentences)
    return {
        "layer": layer,
        "profile_homonym_H": homonym_H[:n_profile],
        "profile_resolved_H": resolved_H[:n_profile],
        "profile_senses": profile_senses,
        "item_homonym_H": homonym_H[n_profile:],
        "item_resolved_H": resolved_H[n_profile:],
        "item_conditions": np.array(conditions),
        "item_senses": np.array(senses),
        "item_ids": ids,
    }


def _item_records(data: Dict) -> List[Dict]:
    """Scores each stage fairly, against ITS OWN position's centroids --
    homonym-position reading vs. homonym-position centroids (H1/H3's own
    convention), resolved-position reading vs. resolved-position centroids
    (H2/H4's convention) -- not homonym-vs-resolved-centroids cross-position
    scoring. normalized_adequacy_margin already divides by that stage's own
    inter-centroid distance, so both margins land on the same bounded
    [-1, 1] scale regardless of the raw magnitude/offset differences
    between positions (H4's finding): they are directly comparable without
    needing cross-position scoring, which H4 already shows is unreliable
    (near chance for R-condition decoders) and was producing segment-
    direction arrows that disagreed with the plotted 2D movement -- a
    homonym-stage point could sit visibly closer to its correct centroid
    while still scoring as "moved away", because the old homonym_margin was
    being measured against the wrong (resolved-position) centroids
    entirely, not because of any real revision in the underlying reading.
    """
    homonym_centroids = {
        s: data["profile_homonym_H"][data["profile_senses"] == s].mean(axis=0) for s in (0, 1)
    }
    resolved_centroids = {
        s: data["profile_resolved_H"][data["profile_senses"] == s].mean(axis=0) for s in (0, 1)
    }
    other = {0: 1, 1: 0}
    records = []
    for i, (cond, sense, item_id) in enumerate(
        zip(data["item_conditions"], data["item_senses"], data["item_ids"])
    ):
        h_correct, h_wrong = homonym_centroids[sense], homonym_centroids[other[sense]]
        r_correct, r_wrong = resolved_centroids[sense], resolved_centroids[other[sense]]
        homonym_margin = normalized_adequacy_margin(data["item_homonym_H"][i], h_correct, h_wrong)
        resolved_margin = normalized_adequacy_margin(data["item_resolved_H"][i], r_correct, r_wrong)
        records.append({
            "item_index": i,
            "item_id": item_id,
            "condition": cond,
            "sense": int(sense),
            "homonym_margin_norm": homonym_margin,
            "resolved_margin_norm": resolved_margin,
            "resolved_correct": resolved_margin > 0.0,
        })
    return records


def _shared_coords(data: Dict):
    # Homonym-position and resolved-position readouts differ enormously in
    # raw magnitude/offset (H4's own finding) -- each position group is
    # centered+scaled from its OWN reliable, large-n profiling split before
    # pooling, so "which position" stops dominating the shared PCA fit.
    homonym_center, homonym_scale = center_scale_params(data["profile_homonym_H"], data["profile_senses"])
    resolved_center, resolved_scale = center_scale_params(data["profile_resolved_H"], data["profile_senses"])

    norm_profile_resolved = (data["profile_resolved_H"] - resolved_center) / resolved_scale
    norm_item_homonym = (data["item_homonym_H"] - homonym_center) / homonym_scale
    norm_item_resolved = (data["item_resolved_H"] - resolved_center) / resolved_scale
    norm_profile_homonym = (data["profile_homonym_H"] - homonym_center) / homonym_scale

    pooled = np.vstack([norm_profile_homonym, norm_profile_resolved, norm_item_homonym, norm_item_resolved])
    pca = PCA(n_components=2).fit(pooled)
    profile_resolved_coords = pca.transform(norm_profile_resolved)
    item_homonym_coords = pca.transform(norm_item_homonym)
    item_resolved_coords = pca.transform(norm_item_resolved)

    resolved_centroids = {
        s: profile_resolved_coords[data["profile_senses"] == s].mean(axis=0) for s in (0, 1)
    }
    all_coords = np.vstack([profile_resolved_coords, item_homonym_coords, item_resolved_coords])
    pad_x = 0.08 * (all_coords[:, 0].max() - all_coords[:, 0].min() or 1.0)
    pad_y = 0.08 * (all_coords[:, 1].max() - all_coords[:, 1].min() or 1.0)
    xlim = (all_coords[:, 0].min() - pad_x, all_coords[:, 0].max() + pad_x)
    ylim = (all_coords[:, 1].min() - pad_y, all_coords[:, 1].max() + pad_y)
    return pca, profile_resolved_coords, item_homonym_coords, item_resolved_coords, resolved_centroids, xlim, ylim


def make_trajectory_figure(data: Dict, model_label: str, word: str, output: Path) -> List[Dict]:
    apply_report_style()
    records = _item_records(data)
    (pca, profile_resolved_coords, item_homonym_coords, item_resolved_coords,
     resolved_centroids, xlim, ylim) = _shared_coords(data)

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.6), constrained_layout=True)

    for ax, condition in zip(axes, CONDITIONS):
        draw_sense_background(ax, xlim, ylim, resolved_centroids[0], resolved_centroids[1])
        for sense in (0, 1):
            pts = profile_resolved_coords[data["profile_senses"] == sense]
            ax.scatter(
                pts[:, 0], pts[:, 1], s=16, color=SENSE[sense], alpha=0.14,
                edgecolor="none", zorder=1,
            )
            ax.scatter(
                resolved_centroids[sense][0], resolved_centroids[sense][1], s=260, marker="X",
                facecolor=SENSE[sense], edgecolor=INK, linewidth=1.4, zorder=6,
            )

        for record in records:
            if record["condition"] != condition:
                continue
            i = record["item_index"]
            line = np.array([item_homonym_coords[i], item_resolved_coords[i]])
            margins = [record["homonym_margin_norm"], record["resolved_margin_norm"]]
            correct_2d = resolved_centroids[record["sense"]]
            wrong_2d = resolved_centroids[1 - record["sense"]]
            visual_agrees = segment_visual_agreement(line, correct_2d, wrong_2d, margins)
            draw_trajectory(
                ax, line, margins, dot_color=SENSE[record["sense"]],
                stage_sizes=[STAGE_SIZES[s] for s in STAGES],
                final_outcome_correct=record["resolved_correct"],
                visual_agrees=visual_agrees,
            )

        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.grid(color=GRID, linewidth=0.5)
        ax.set_title(CONDITION_LABELS[condition], loc="left", fontweight="bold", fontsize=11)
        n_cond = int((data["item_conditions"] == condition).sum())
        n_correct = sum(r["resolved_correct"] for r in records if r["condition"] == condition)
        ax.text(
            0.02, 0.02, f"{n_correct}/{n_cond} resolved correct", transform=ax.transAxes,
            fontsize=8.5, color=INK, va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2},
        )

    draw_sense_legend(axes[-1], word, [f"{s} (dot size)" for s in STAGES], [STAGE_SIZES[s] for s in STAGES])
    fit_suptitle(
        fig,
        f"Homonym to resolved trajectory, L vs. R context, {model_label}, {word}, layer {data['layer']}. "
        "Dot color = true sense (fill) -- see legend for meaning. Arrows colored by whether that step's "
        f"real margin moved toward or away from the correct sense. Ring = final resolved outcome. "
        f"{explained_variance_caption(pca)}",
        max_width_in=fig.get_size_inches()[0],
        fontsize=10.5,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=190, bbox_inches="tight")
    plt.close(fig)
    return records


def run(
    model_name: str, word: str, results_dir: str, output_base: Path,
    force_extraction: bool = False, cache_base: Optional[Path] = None,
) -> None:
    safe_model = model_name.replace("/", "_")
    cache_path = (cache_base or output_base / "_extraction_cache") / f"{safe_model}_{word}.pkl"
    data = load_or_extract(
        cache_path, lambda: extract_trajectory_data(model_name, word, results_dir), force=force_extraction
    )
    out_dir = output_base / safe_model
    out_dir.mkdir(parents=True, exist_ok=True)
    records = make_trajectory_figure(data, model_name, word, out_dir / f"{word}_context_trajectory.svg")
    with open(out_dir / f"{word}_context_trajectory.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    for condition in CONDITIONS:
        n = sum(r["condition"] == condition for r in records)
        n_correct = sum(r["condition"] == condition and r["resolved_correct"] for r in records)
        print(f"[{model_name}/{word}/{condition}] layer={data['layer']} resolved_correct={n_correct}/{n}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--words", nargs="+", default=["bank"])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-base", default="results/study/h3_context_trajectory")
    parser.add_argument(
        "--force-extraction", action="store_true",
        help="Re-run the GPU extraction even if a cached copy exists for this model/word "
        "(cache lives under <output-base>/_extraction_cache/). Only ever needed after a "
        "change to extract_trajectory_data itself, not for plotting/visual-design changes.",
    )
    args = parser.parse_args()

    from utils.model_registry import MODEL_ALIASES

    models = [MODEL_ALIASES.get(m, m) for m in args.models]
    for model_name in models:
        for word in args.words:
            run(model_name, word, args.results_dir, Path(args.output_base), args.force_extraction)


if __name__ == "__main__":
    main()
