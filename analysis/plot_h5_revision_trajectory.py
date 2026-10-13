"""Point-cloud trajectory of individual garden-path items through semantic
space: prime -> homonym -> resolution, at the model's H1-selected layer.

Mirrors context_revelation_trajectory.py's solution to "the activations
don't project into the same PCA": one PCA is fit across every stage AND the
profiling cloud pooled together, so all points -- and the sentinel-position
sense centroids H5 itself scores against -- live in one fixed 2D coordinate
system. Motion between stages is therefore real motion in that shared basis,
not an artifact of re-fitting PCA per panel.

Stages are H5's own fixed-sentinel prefixes (build_incremental_prefixes):
prime+sentinel, homonym+sentinel, resolution+sentinel -- the identical
stimuli and readout position H5's margin scoring uses, so a trajectory
crossing the plotted centroid boundary is the same event as
resolved_correct=True in h5_sentence_level.csv.

Two figure styles, both built from the same extracted data, so they can be
compared directly (see trajectory_plot_common.py for the shared drawing
logic -- background gradient, segment-direction arrows, meaning-based
legend):
  make_trajectory_figure              -- one panel, everything overlaid.
  make_trajectory_figure_progressive  -- three panels (prime only / +
                                          homonym / + resolution), earlier
                                          stages fading as later ones are
                                          revealed, to cut down on
                                          all-at-once line crossings.

Requires a GPU (loads the model checkpoint); the plotting logic itself
takes plain arrays and is exercised by a smoke test with synthetic data, no
model or GPU needed for that part.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from experiments.adequacy import normalized_adequacy_margin
from hypotheses.h3_context_position import _select_layer
from hypotheses.h5_garden_path import (
    DEFAULT_SENTINEL,
    GP_DATA_PATH,
    H5_EXCLUSIONS,
    PROFILING_DATA_PATH,
    RESULTS_DIR,
    _append_sentinel,
    _load_profiling_examples,
    build_incremental_prefixes,
)
from analysis.plot_centroid_separation import center_scale_params
from analysis.trajectory_plot_common import (
    draw_sense_background,
    draw_sense_legend,
    draw_trajectory,
    explained_variance_caption,
    load_or_extract,
    segment_visual_agreement,
    sense_label,
)
from utils.visual_style import GRID, INK, SENSE, apply_report_style, fit_suptitle

STAGES = ("prime", "homonym", "resolution")
STAGE_SIZES = {"prime": 34, "homonym": 56, "resolution": 88}
PROFILE_POINT_COLOR_ALPHA = 0.14


def extract_trajectory_data(
    model_name: str,
    word: str,
    results_dir: str = RESULTS_DIR,
    gp_data_path: str = GP_DATA_PATH,
    profiling_data_path: str = PROFILING_DATA_PATH,
    sentinel: str = DEFAULT_SENTINEL,
    batch_size: int = 4,
) -> Dict:
    """GPU step. One forward pass covers the profiling cloud plus every
    item's three staged prefixes, all ending in the same sentinel, at the
    single H1-selected layer (the layer H5 itself reads out)."""
    from models import get_dual_position_activations, load_model_and_tokenizer

    if word in H5_EXCLUSIONS:
        raise ValueError(f"'{word}' is excluded from H5: {H5_EXCLUSIONS[word]}")

    with open(gp_data_path) as f:
        gp_data = json.load(f)
    items = gp_data.get(word, [])
    if not items:
        raise ValueError(f"No H5 garden-path items for '{word}'")

    profile_sentences, profile_senses = _load_profiling_examples(profiling_data_path, word)
    profile_texts = [_append_sentinel(s, sentinel) for s in profile_sentences]

    stage_texts: List[str] = []
    stage_keys: List[tuple] = []
    prefixes_by_item = []
    for i, item in enumerate(items):
        prefixes = build_incremental_prefixes(item, word, sentinel)
        prefixes_by_item.append(prefixes)
        for stage in STAGES:
            stage_texts.append(prefixes[stage])
            stage_keys.append((i, stage))

    model, tokenizer = load_model_and_tokenizer(model_name)
    layer = _select_layer(model_name, results_dir, word)
    all_texts = profile_texts + stage_texts
    _, final_acts = get_dual_position_activations(
        model, tokenizer, all_texts, [word] * len(all_texts),
        batch_size=batch_size, layer_indices=[layer],
    )
    H = final_acts[layer].numpy()
    del model, tokenizer

    n_profile = len(profile_texts)
    return {
        "layer": layer,
        "profile_H": H[:n_profile],
        "profile_senses": profile_senses,
        "stage_H": H[n_profile:],
        "stage_keys": stage_keys,
        "items": items,
    }


def _item_records(data: Dict) -> List[Dict]:
    """Per-item stage margins + resolved_correct, using the exact sentinel-
    centroid scoring rule H5 itself uses (not the homonym-position cache)."""
    profile_H, profile_senses = data["profile_H"], data["profile_senses"]
    centroids = {s: profile_H[profile_senses == s].mean(axis=0) for s in (0, 1)}
    h_by_key = {key: h for key, h in zip(data["stage_keys"], data["stage_H"])}

    records = []
    for i, item in enumerate(data["items"]):
        correct, primed = int(item["correct_sense"]), int(item["primed_sense"])
        c_correct, c_primed = centroids[correct], centroids[primed]
        stage_margins = {
            stage: normalized_adequacy_margin(h_by_key[(i, stage)], c_correct, c_primed)
            for stage in STAGES
        }
        records.append({
            "item_index": i,
            "sentence_id": item.get("id", f"{data.get('word', '')}_gp_{i}"),
            "primed_sense": primed,
            "correct_sense": correct,
            "resolved_correct": stage_margins["resolution"] > 0.0,
            **{f"{stage}_margin_norm": stage_margins[stage] for stage in STAGES},
        })
    return records


def _shared_coords(data: Dict):
    """Center+scale everything from the profiling set's own (large-n,
    reliable) sense split before pooling for PCA -- otherwise a stage whose
    raw activations happen to sit at very different magnitude/offset from
    the profiling cloud (e.g. a short prime-only prefix vs a full profiling
    sentence, both ending in the same sentinel) can dominate or get
    swallowed by the pooled fit on scale/offset alone, unrelated to sense
    separation. All three item stages share ONE reference frame here
    (unlike H3's per-position frames) because H5 itself only ever scores
    against this one profiling-derived centroid pair.
    """
    center, scale = center_scale_params(data["profile_H"], data["profile_senses"])
    norm_profile_H = (data["profile_H"] - center) / scale
    norm_stage_H = (data["stage_H"] - center) / scale

    pooled = np.vstack([norm_profile_H, norm_stage_H])
    pca = PCA(n_components=2).fit(pooled)
    profile_coords = pca.transform(norm_profile_H)
    stage_coords = pca.transform(norm_stage_H)
    coord_by_key = {key: stage_coords[idx] for idx, key in enumerate(data["stage_keys"])}

    profile_centroids = {
        s: profile_coords[data["profile_senses"] == s].mean(axis=0) for s in (0, 1)
    }
    all_coords = np.vstack([profile_coords, stage_coords])
    pad_x = 0.10 * (all_coords[:, 0].max() - all_coords[:, 0].min() or 1.0)
    pad_y = 0.10 * (all_coords[:, 1].max() - all_coords[:, 1].min() or 1.0)
    xlim = (all_coords[:, 0].min() - pad_x, all_coords[:, 0].max() + pad_x)
    ylim = (all_coords[:, 1].min() - pad_y, all_coords[:, 1].max() + pad_y)
    return pca, profile_coords, coord_by_key, profile_centroids, xlim, ylim


def _draw_profile_and_background(ax, data, profile_coords, profile_centroids, xlim, ylim, word):
    draw_sense_background(ax, xlim, ylim, profile_centroids[0], profile_centroids[1])
    for sense in (0, 1):
        pts = profile_coords[data["profile_senses"] == sense]
        ax.scatter(
            pts[:, 0], pts[:, 1], s=16, color=SENSE[sense], alpha=PROFILE_POINT_COLOR_ALPHA,
            edgecolor="none", zorder=1,
        )
        ax.scatter(
            profile_centroids[sense][0], profile_centroids[sense][1], s=260, marker="X",
            facecolor=SENSE[sense], edgecolor=INK, linewidth=1.4, zorder=6,
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(color=GRID, linewidth=0.5)


def make_trajectory_figure(data: Dict, model_label: str, word: str, output: Path) -> List[Dict]:
    """Single-panel version: every stage and every item overlaid, with a
    background gradient, per-segment direction arrows, and a compact
    in-plot legend using the word's actual sense meanings."""
    apply_report_style()
    records = _item_records(data)
    pca, profile_coords, coord_by_key, profile_centroids, xlim, ylim = _shared_coords(data)

    fig, ax = plt.subplots(figsize=(9, 7.6), constrained_layout=True)
    _draw_profile_and_background(ax, data, profile_coords, profile_centroids, xlim, ylim, word)

    for record in records:
        i = record["item_index"]
        line = np.array([coord_by_key[(i, stage)] for stage in STAGES])
        margins = [record[f"{stage}_margin_norm"] for stage in STAGES]
        correct_2d = profile_centroids[record["correct_sense"]]
        wrong_2d = profile_centroids[1 - record["correct_sense"]]
        visual_agrees = segment_visual_agreement(line, correct_2d, wrong_2d, margins)
        draw_trajectory(
            ax, line, margins, dot_color=SENSE[record["correct_sense"]],
            stage_sizes=[STAGE_SIZES[s] for s in STAGES],
            final_outcome_correct=record["resolved_correct"],
            visual_agrees=visual_agrees,
        )

    draw_sense_legend(ax, word, [f"{s} (dot size)" for s in STAGES], [STAGE_SIZES[s] for s in STAGES])
    ax.set_xlabel("PC1 (shared basis: profiling cloud + all stages)")
    ax.set_ylabel("PC2")
    fit_suptitle(
        fig,
        f"Garden-path revision trajectory, {model_label}, {word}, layer {data['layer']}. "
        "Dot color = this item's CORRECT sense (fill), not the primed one -- see legend for meaning. "
        "The prime stage often starts deep in the OTHER color's territory: that is the garden-path "
        "prime taking hold, before the sentence forces revision. Arrows show prime -> homonym -> "
        "resolution, colored by whether that step moved toward or away from the correct sense "
        f"(true high-dimensional margin). Ring = final resolved outcome. {explained_variance_caption(pca)}",
        max_width_in=fig.get_size_inches()[0],
        fontsize=10.5,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=190, bbox_inches="tight")
    plt.close(fig)
    return records


def make_trajectory_figure_progressive(data: Dict, model_label: str, word: str, output: Path) -> List[Dict]:
    """Three-panel version: prime only / + homonym / + resolution, so the
    reader can build up the picture stage by stage instead of parsing every
    crossing line at once. Earlier stages fade (but stay visible) once a
    later one is revealed."""
    apply_report_style()
    records = _item_records(data)
    pca, profile_coords, coord_by_key, profile_centroids, xlim, ylim = _shared_coords(data)

    panel_stage_sets = [("prime",), ("prime", "homonym"), ("prime", "homonym", "resolution")]
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.2), constrained_layout=True)

    for ax, stages_shown in zip(axes, panel_stage_sets):
        _draw_profile_and_background(ax, data, profile_coords, profile_centroids, xlim, ylim, word)
        newest_stage = stages_shown[-1]
        for record in records:
            i = record["item_index"]
            shown = np.array([coord_by_key[(i, stage)] for stage in stages_shown])
            margins = [record[f"{stage}_margin_norm"] for stage in stages_shown]
            is_final_panel = newest_stage == "resolution"
            # In the final (3-stage) panel, the prime->homonym segment was
            # already the whole point of the previous panel -- fade it here
            # so it recedes into context instead of competing for attention
            # with the homonym->resolution segment, which is what's new.
            n_segments = len(shown) - 1
            segment_alphas = [0.22] * (n_segments - 1) + [0.8] if is_final_panel and n_segments > 0 else None
            correct_2d = profile_centroids[record["correct_sense"]]
            wrong_2d = profile_centroids[1 - record["correct_sense"]]
            visual_agrees = segment_visual_agreement(shown, correct_2d, wrong_2d, margins) if n_segments else None
            draw_trajectory(
                ax, shown, margins, dot_color=SENSE[record["correct_sense"]],
                stage_sizes=[STAGE_SIZES[s] * (1.0 if s == newest_stage else 0.6) for s in stages_shown],
                final_outcome_correct=record["resolved_correct"] if is_final_panel else None,
                segment_alphas=segment_alphas,
                visual_agrees=visual_agrees,
            )
            # Fade every point except the newest-revealed stage, so the
            # panel reads as "here is what's new" rather than repeating
            # full-strength ink for stages already shown in a prior panel.
            for stage, point in zip(stages_shown[:-1], shown[:-1]):
                ax.scatter(point[0], point[1], s=STAGE_SIZES[stage] * 0.5,
                           color=SENSE[record["correct_sense"]], alpha=0.25, zorder=2)
        ax.set_title(f"+ {newest_stage}" if len(stages_shown) > 1 else newest_stage,
                     loc="left", fontweight="bold")
        ax.set_xlabel("PC1")

    axes[0].set_ylabel("PC2")
    draw_sense_legend(axes[-1], word, [f"{s} (dot size)" for s in STAGES], [STAGE_SIZES[s] for s in STAGES])
    fit_suptitle(
        fig,
        f"Garden-path revision trajectory (progressive), {model_label}, {word}, layer {data['layer']}. "
        "Dot color = this item's CORRECT sense, not the primed one -- the prime stage often starts deep "
        "in the OTHER color's territory; that is the prime taking hold, before the sentence forces "
        f"revision. Left to right: prime only, + homonym, + resolution. {explained_variance_caption(pca)}",
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
    records = make_trajectory_figure(data, model_name, word, out_dir / f"{word}_revision_trajectory.svg")
    make_trajectory_figure_progressive(
        data, model_name, word, out_dir / f"{word}_revision_trajectory_progressive.svg"
    )
    with open(out_dir / f"{word}_revision_trajectory.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    n_correct = sum(r["resolved_correct"] for r in records)
    print(f"[{model_name}/{word}] layer={data['layer']} resolved_correct={n_correct}/{len(records)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--words", nargs="+", default=["bank"])
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--output-base", default="results/study/h5_revision_trajectory")
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
            if word in H5_EXCLUSIONS:
                print(f"skipping {word}: {H5_EXCLUSIONS[word]}")
                continue
            run(model_name, word, args.results_dir, Path(args.output_base), args.force_extraction)


if __name__ == "__main__":
    main()
