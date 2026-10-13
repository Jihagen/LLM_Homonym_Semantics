"""Export small, browser-ready JSON files for the research website.

    python -m scripts.export_web_data          # or: make web

Everything is derived from ``data/processed/`` and ``data/stimuli/``; no
statistic is introduced that is not a plain mean, count, or ratio of released
values. The export is deterministic: fixed ordering, fixed rounding, no
timestamps, so re-running it on unchanged data rewrites identical files.

Field-by-field documentation is in web_export/README.md.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from scripts import release_layout as layout
from scripts.release_layout import DATA_VERSION, MODEL_KEYS, PROCESSED_DIR, ROOT, TRAJECTORY_EXAMPLES, WORDS

WEB_DIR = ROOT / "web_export"
STAGES = ["prime", "homonym", "resolution"]


def _table(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / f"{name}.csv", **kwargs)


def _num(value, digits: int = 4):
    """JSON-safe rounded number; NaN and missing become null."""
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    rounded = round(value, digits)
    return 0.0 if rounded == 0 else rounded


def _nums(values, digits: int = 4) -> List:
    return [_num(v, digits) for v in values]


def _true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower() == "true"


def _write(name: str, payload: Dict) -> Path:
    path = WEB_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    path.write_text(text + "\n", encoding="utf-8")
    return path


def _header(description: str, sources: List[str]) -> Dict:
    return {"data_version": DATA_VERSION, "description": description, "source_tables": sources}


# --------------------------------------------------------------------------- meta

def export_meta() -> Dict:
    models = _table("models")
    homonyms = pd.read_csv(ROOT / "data" / "stimuli" / "homonyms.csv")
    return {
        **_header(
            "Models, homonyms and sense labels shared by every other file.",
            ["data/processed/models.csv", "data/stimuli/homonyms.csv"],
        ),
        "models": [
            {
                "id": row.model,
                "hf_repo_id": row.hf_repo_id,
                "name": row.display_name,
                "architecture": row.arch_type,
                "n_hidden_states": int(row.n_hidden_states),
                "last_layer": int(row.last_layer_index),
            }
            for row in models.itertuples()
        ],
        "words": [
            {
                "word": word,
                "senses": [
                    {"sense": int(row.sense), "label": row.sense_label, "short_label": row.short_label}
                    for row in homonyms[homonyms["word"] == word].sort_values("sense").itertuples()
                ],
            }
            for word in WORDS
        ],
        "margin_convention": (
            "Normalised margins lie in [-1, 1]. For sense-labelled items, positive means closer to the "
            "centroid of the intended (or finally resolved) sense. For the prior heatmap, positive means "
            "closer to sense 0 and negative closer to sense 1."
        ),
    }


# --------------------------------------------------------------------------- 1. layer separation

def export_layer_separation() -> Dict:
    profiles = _table("h1_layer_profiles")
    summary = _table("h1_layer_selection_summary").set_index(["model", "word"])
    paired = _table("q4_layer_curves").set_index(["model", "homonym"])
    cells = {}
    for model in MODEL_KEYS:
        cells[model] = {}
        for word in WORDS:
            part = profiles[(profiles["model"] == model) & (profiles["word"] == word)].sort_values("Layer")
            row = summary.loc[(model, word)]
            last = int(row["n_layers"]) - 1
            curve = paired.loc[(model, word)].sort_values("layer")
            if list(curve["layer"]) != list(part["Layer"]):
                raise ValueError(f"layer indices differ between H1 and Q4 for {model}/{word}")
            cells[model][word] = {
                "layers": [int(v) for v in part["Layer"]],
                "relative_depth": _nums(part["Layer"] / last),
                "fraction_adequate": _nums(part["FractionAdequate"]),
                "mean_margin_norm": _nums(part["MeanMarginNorm"]),
                "gdv": _nums(part["GDV"], 5),
                "paired_context_fraction_adequate": _nums(curve["adequacy"]),
                "selected_layer_adequacy": int(row["best_layer_M"]),
                "selected_layer_gdv": int(row["gdv_best_layer"]),
                "last_layer": last,
                "nested_fraction_adequate_selected": _num(row["nested_frac_selected"]),
                "nested_fraction_adequate_last": _num(row["nested_frac_last"]),
            }
    return {
        **_header(
            "Layer-wise sense separation for every model and homonym (H1/H2), for a "
            "model/homonym/layer selector.",
            [
                "data/processed/h1_layer_profiles.csv",
                "data/processed/h1_layer_selection_summary.csv",
                "data/processed/q4_layer_curves.csv",
            ],
        ),
        "cells": cells,
    }


# --------------------------------------------------------------------------- 2. trajectories

def export_revision_trajectories() -> Dict:
    h5 = _table("h5_sentence_level", keep_default_na=False)
    stimuli = pd.read_csv(ROOT / "data" / "stimuli" / "conflict_items.csv", keep_default_na=False)
    items = {}
    for row in stimuli.itertuples():
        items[row.item_id] = {
            "word": row.word,
            "primed_sense": int(row.primed_sense),
            "correct_sense": int(row.correct_sense),
            "resolution_word": row.resolution_word,
            "conflicting_sentence": row.conflicting_sentence,
            "coherent_control_sentence": row.coherent_control_sentence,
            # Text visible to the model at each stage, without the sentinel paragraph.
            "stage_text": [
                getattr(row, f"conflicting_{stage}_prefix").rsplit("\n\n", 1)[0] for stage in STAGES
            ],
        }
    trajectories = {}
    for model in MODEL_KEYS:
        trajectories[model] = {}
        for word in WORDS:
            part = h5[(h5["model"] == model) & (h5["word"] == word)].sort_values("sentence_id")
            trajectories[model][word] = {
                "layer": int(part["layer"].iloc[0]),
                "item_ids": list(part["sentence_id"]),
                "margins": [
                    _nums([row[f"{stage}_correct_margin_norm"] for stage in STAGES])
                    for _, row in part.iterrows()
                ],
                "coherent_control_margin": _nums(part["matched_control_correct_margin_norm"]),
                "resolver_only_margin": _nums(part["resolver_isolated_correct_margin_norm"]),
                "primed_at_homonym": [bool(v) for v in _true(part["primed_at_homonym"])],
                "resolved_correct": [bool(v) for v in _true(part["resolved_correct"])],
            }
    return {
        **_header(
            "Per-item fixed-sentinel trajectories (H5): correct-sense margin after the prime, after the "
            "homonym and after the resolver, for every model and context-conflict item.",
            ["data/processed/h5_sentence_level.csv", "data/stimuli/conflict_items.csv"],
        ),
        "stages": STAGES,
        "sentinel": "probe",
        "items": items,
        "trajectories": trajectories,
    }


def export_trajectory_examples() -> Dict:
    """2-D coordinates of the sample trajectories, using the same shared
    PCA frames as the supplementary figures."""
    from analysis import plot_h3_context_trajectory as h3_plot
    from analysis import plot_h5_revision_trajectory as h5_plot

    examples = []
    for model, word in TRAJECTORY_EXAMPLES:
        data = layout.load_h5_trajectory_states(model, word)
        pca, profile_coords, coord_by_key, centroids, xlim, ylim = h5_plot._shared_coords(data)
        records = h5_plot._item_records(data)
        revision = {
            "layer": data["layer"],
            "explained_variance_ratio": _nums(pca.explained_variance_ratio_),
            "xlim": _nums(xlim, 3),
            "ylim": _nums(ylim, 3),
            "profile_points": [_nums(p, 3) for p in profile_coords],
            "profile_senses": [int(s) for s in data["profile_senses"]],
            "sense_centroids": [_nums(centroids[s], 3) for s in (0, 1)],
            "items": [
                {
                    "item_id": record["sentence_id"],
                    "primed_sense": record["primed_sense"],
                    "correct_sense": record["correct_sense"],
                    "points": [_nums(coord_by_key[(record["item_index"], stage)], 3) for stage in STAGES],
                    "margins": _nums([record[f"{stage}_margin_norm"] for stage in STAGES]),
                    "resolved_correct": bool(record["resolved_correct"]),
                }
                for record in records
            ],
        }

        data = layout.load_h3_trajectory_states(model, word)
        pca, profile_coords, homonym_coords, resolved_coords, centroids, xlim, ylim = h3_plot._shared_coords(data)
        records = h3_plot._item_records(data)
        context = {
            "layer": data["layer"],
            "explained_variance_ratio": _nums(pca.explained_variance_ratio_),
            "xlim": _nums(xlim, 3),
            "ylim": _nums(ylim, 3),
            "profile_points": [_nums(p, 3) for p in profile_coords],
            "profile_senses": [int(s) for s in data["profile_senses"]],
            "sense_centroids": [_nums(centroids[s], 3) for s in (0, 1)],
            "items": [
                {
                    "item_id": record["item_id"],
                    "condition": record["condition"],
                    "sense": record["sense"],
                    "points": [_nums(homonym_coords[i], 3), _nums(resolved_coords[i], 3)],
                    "margins": _nums([record["homonym_margin_norm"], record["resolved_margin_norm"]]),
                    "resolved_correct": bool(record["resolved_correct"]),
                }
                for i, record in enumerate(records)
            ],
        }
        examples.append({"model": model, "word": word, "revision": revision, "context": context})
    return {
        **_header(
            "Two-dimensional PCA views of sample trajectories for 'bank' in one encoder and one decoder. "
            "Positions are a projection for display; margins are computed in the full hidden space.",
            ["data/processed/example_states/*.npz", "data/stimuli/conflict_items.csv",
             "data/stimuli/paired_context_sentences.csv"],
        ),
        "revision_stages": STAGES,
        "context_stages": ["homonym_position", "sentence_final_position"],
        "examples": examples,
    }


# --------------------------------------------------------------------------- 3. coherent vs conflicting

def _revision_summary(frame: pd.DataFrame) -> Dict:
    primed = _true(frame["primed_at_homonym"])
    crossed = _true(frame["successful_primed_to_correct_transition"])
    delta = frame["delta_resolution_minus_homonym_norm"]
    n_primed = int(primed.sum())
    return {
        "n": int(len(frame)),
        "mean_margin_prime": _num(frame["prime_correct_margin_norm"].mean()),
        "mean_margin_homonym": _num(frame["homonym_correct_margin_norm"].mean()),
        "mean_margin_conflicting_resolved": _num(frame["resolution_correct_margin_norm"].mean()),
        "mean_margin_coherent_control": _num(frame["matched_control_correct_margin_norm"].mean()),
        "mean_margin_resolver_only": _num(frame["resolver_isolated_correct_margin_norm"].mean()),
        "mean_movement": _num(delta.mean()),
        "n_moved_toward_resolved": int((delta > 0).sum()),
        "fraction_moved_toward_resolved": _num((delta > 0).mean()),
        "n_primed_at_homonym": n_primed,
        "n_crossed_boundary": int(crossed.sum()),
        "crossing_rate": _num(crossed.sum() / n_primed) if n_primed else None,
        "mean_conflict_minus_control": _num(frame["garden_path_cost_vs_matched_control_norm"].mean()),
        "mean_conflict_minus_resolver_only": _num(frame["delta_resolution_minus_isolated_resolver_norm"].mean()),
    }


def export_revision_comparison() -> Dict:
    h5 = _table("h5_sentence_level", keep_default_na=False)
    by_layer = _table("h5_by_layer_aggregate")
    return {
        **_header(
            "Coherent versus conflicting context paths (H5): endpoint margins, movement and "
            "decision-boundary crossing, per model and homonym and aggregated.",
            ["data/processed/h5_sentence_level.csv", "data/processed/h5_by_layer_aggregate.csv"],
        ),
        "overall": _revision_summary(h5),
        "by_architecture": {arch: _revision_summary(part) for arch, part in h5.groupby("arch_type", sort=True)},
        "by_model": {model: _revision_summary(h5[h5["model"] == model]) for model in MODEL_KEYS},
        "by_word": {word: _revision_summary(h5[h5["word"] == word]) for word in WORDS},
        "by_direction": {direction: _revision_summary(part) for direction, part in h5.groupby("direction", sort=True)},
        "by_model_word": {
            model: {
                word: {
                    "layer": int(h5.loc[(h5["model"] == model) & (h5["word"] == word), "layer"].iloc[0]),
                    **_revision_summary(h5[(h5["model"] == model) & (h5["word"] == word)]),
                }
                for word in WORDS
            }
            for model in MODEL_KEYS
        },
        # Exploratory: the same analysis repeated at other candidate layers.
        "by_model_word_layer": {
            model: {
                word: [
                    {
                        "layer": int(row.layer_used),
                        "n": int(row.n_items),
                        "mean_margin_homonym": _num(row.mean_homonym_correct_margin_norm),
                        "mean_margin_conflicting_resolved": _num(row.mean_resolution_correct_margin_norm),
                        "mean_movement": _num(row.mean_delta_resolution_minus_homonym_norm),
                        "n_primed_at_homonym": int(row.n_primed_at_homonym),
                        "n_crossed_boundary": int(row.n_successful_primed_to_correct),
                        "mean_conflict_minus_control": _num(row.mean_garden_path_cost_vs_matched_control_norm),
                    }
                    for row in by_layer[(by_layer["model"] == model) & (by_layer["word"] == word)]
                    .sort_values("layer_used").itertuples()
                ]
                for word in WORDS
            }
            for model in MODEL_KEYS
        },
    }


# --------------------------------------------------------------------------- 4. prior heatmap

def export_prior_heatmap() -> Dict:
    summary = _table("h0_summary").set_index(["model", "word"])
    lean = _table("h0_carrier_lean")

    def matrix(column: str, digits: int = 4):
        return [[_num(summary.loc[(model, word), column], digits) for word in WORDS] for model in MODEL_KEYS]

    return {
        **_header(
            "Baseline sense lean (H0): the bare word versus the mean over five ambiguous carrier "
            "sentences, for every model and homonym.",
            ["data/processed/h0_summary.csv", "data/processed/h0_carrier_lean.csv"],
        ),
        "models": MODEL_KEYS,
        "words": WORDS,
        "sign_convention": "positive = closer to sense 0, negative = closer to sense 1",
        "bare_word_lean": matrix("signed_M_l_word_alone_norm"),
        "carrier_mean_lean": matrix("mean_signed_M_l_carrier_norm"),
        "carrier_mean_abs_lean": matrix("mean_abs_M_l_carrier_norm"),
        "carrier_direction_consistency": matrix("direction_consistency"),
        "layer": [[int(summary.loc[(model, word), "layer_used"]) for word in WORDS] for model in MODEL_KEYS],
        "carriers": {
            model: {
                word: [
                    {"carrier": row.carrier, "lean": _num(row.signed_M_l_carrier_norm)}
                    for row in lean[(lean["model"] == model) & (lean["word"] == word)].itertuples()
                ]
                for word in WORDS
            }
            for model in MODEL_KEYS
        },
    }


EXPORTS = {
    "meta.json": export_meta,
    "layer_separation.json": export_layer_separation,
    "revision_trajectories.json": export_revision_trajectories,
    "trajectory_examples.json": export_trajectory_examples,
    "revision_comparison.json": export_revision_comparison,
    "prior_heatmap.json": export_prior_heatmap,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    for name, function in EXPORTS.items():
        path = _write(name, function())
        print(f"  web_export/{name}  {path.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    main()
