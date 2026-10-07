"""Fail-fast validation for partial audit and complete H0–H5 Slurm jobs.

The preflight checks data balance plus every H1-selected target/final cache.
The postflight rejects incomplete cells, legacy schemas, and impossible
normalized margins, so a Slurm success means that the requested rerun really
produced the complete corrected output set.
"""

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path

from utils.model_registry import ALL_MODELS


WORDS = ["bank", "bark", "bat", "crane", "spring", "match", "pitch"]
RESULTS = Path("results")
_FRESH_AFTER = None


def _require_fresh(path: Path):
    if _FRESH_AFTER is not None and path.stat().st_mtime <= _FRESH_AFTER:
        raise AssertionError(f"stale output from before this job: {path}")


def _read_csv(path: Path):
    if not path.exists():
        raise AssertionError(f"missing file: {path}")
    _require_fresh(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError(f"empty file: {path}")
    return rows


def _require_columns(path: Path, rows, required):
    missing = set(required) - set(rows[0])
    if missing:
        raise AssertionError(f"{path}: missing columns {sorted(missing)}")


def _require_norm_bounds(path: Path, rows, columns):
    for row_number, row in enumerate(rows, start=2):
        for column in columns:
            value = float(row[column])
            if not math.isfinite(value) or abs(value) > 1.00001:
                raise AssertionError(
                    f"{path}:{row_number}: {column}={value} outside [-1, 1]"
                )


def _selected_layers():
    selected = {}
    for model in ALL_MODELS:
        safe = model.replace("/", "_")
        path = RESULTS / "study" / "H1" / safe / "h1_summary.csv"
        if not path.exists():
            return {}
        rows = _read_csv(path)
        by_word = {row["word"]: int(row["best_layer_M"]) for row in rows}
        if set(by_word) != set(WORDS):
            raise AssertionError(
                f"{path}: expected words {WORDS}, found {sorted(by_word)}"
            )
        for word, layer in by_word.items():
            selected[(model, word)] = layer
    return selected


def validate_preflight():
    with Path("data/paired_sentences.json").open() as handle:
        paired = json.load(handle)
    if set(paired) != set(WORDS):
        raise AssertionError(
            f"paired data must contain exactly {WORDS}; found {sorted(paired)}"
        )
    all_ids = []
    for word in WORDS:
        items = paired[word]
        if len(items) != 20:
            raise AssertionError(f"{word}: expected 20 paired items, found {len(items)}")
        condition_counts = Counter(item["condition"] for item in items)
        if condition_counts != Counter({"L": 10, "R": 10}):
            raise AssertionError(f"{word}: unbalanced conditions {condition_counts}")
        for condition in ("L", "R"):
            sense_counts = Counter(
                int(item["sense"]) for item in items if item["condition"] == condition
            )
            if sense_counts != Counter({0: 5, 1: 5}):
                raise AssertionError(
                    f"{word}/{condition}: unbalanced senses {sense_counts}"
                )
        all_ids.extend(item["id"] for item in items)
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("paired sentence IDs are not globally unique")

    selected = _selected_layers()
    for (model, word), layer in selected.items():
        safe = model.replace("/", "_")
        for root in ("activations", "activations_final"):
            path = RESULTS / root / word / safe / f"layer_{layer}.h5"
            if not path.exists():
                raise AssertionError(f"missing selected-layer cache: {path}")

    hf_home = Path(os.environ.get("HF_HOME", "hf_cache"))
    for model in ALL_MODELS:
        cache = hf_home / f"models--{model.replace('/', '--')}"
        if not cache.exists():
            raise AssertionError(f"offline model cache is missing: {cache}")

    from hypotheses.h5_garden_path import audit_h5_design

    with Path("data/garden_path_sentences.json").open() as handle:
        h5_data = json.load(handle)
    h5_rows, h5_ready = audit_h5_design(h5_data, WORDS)
    if not h5_ready:
        incomplete = [
            row["word"] for row in h5_rows
            if row["eligible_for_h5"] and not row["model_internal_ready"]
        ]
        raise AssertionError(f"H5 model-internal design is incomplete for: {incomplete}")


def validate_postflight(marker_path=None, full=False):
    global _FRESH_AFTER
    if marker_path is not None:
        marker = Path(marker_path)
        if not marker.exists():
            raise AssertionError(f"missing job marker: {marker}")
        _FRESH_AFTER = marker.stat().st_mtime
    expected_cells = {(model.replace("/", "_"), word) for model in ALL_MODELS for word in WORDS}

    if full:
        inference_path = RESULTS / "study" / "geometry_inference.csv"
        inference = _read_csv(inference_path)
        if len(inference) != 12 or any(
            row["bootstrap_unit"] != "crossed model-by-word" for row in inference
        ):
            raise AssertionError("geometry inference must contain 12 crossed-bootstrap rows")
        for model in ALL_MODELS:
            safe = model.replace("/", "_")
            h1_summary_path = RESULTS / "study" / "H1" / safe / "h1_summary.csv"
            h1_summary = _read_csv(h1_summary_path)
            if len(h1_summary) != len(WORDS):
                raise AssertionError(f"{h1_summary_path}: expected {len(WORDS)} words")
            h1_by_word = {row["word"]: row for row in h1_summary}
            for word in WORDS:
                nested_path = RESULTS / "study" / "H1" / safe / f"h1_nested_loo_{word}.csv"
                nested = _read_csv(nested_path)
                expected_folds = int(h1_by_word[word]["nested_n_outer_folds"])
                if len(nested) != expected_folds:
                    raise AssertionError(
                        f"{nested_path}: expected {expected_folds} outer folds, "
                        f"found {len(nested)}"
                    )
                gdv_path = RESULTS / f"{safe}_gdv" / f"gdv_values_{word}.csv"
                _read_csv(gdv_path)
            h2_path = RESULTS / "study" / "H2" / safe / "h2_loo.csv"
            h2_rows = _read_csv(h2_path)
            _require_columns(
                h2_path,
                h2_rows,
                {
                    "held_out_word", "fraction_adequate_gdv",
                    "fraction_regret_to_oracle_gdv", "layer_distance_to_oracle_gdv",
                    "held_out_adequacy_rank_gdv", "within_2_layers_of_oracle_gdv",
                    "fraction_adequate_supervised",
                },
            )
            if len(h2_rows) != len(WORDS):
                raise AssertionError(f"{h2_path}: expected {len(WORDS)} held-out words")
        h2_summary = _read_csv(RESULTS / "study" / "H2" / "h2_strategy_summary.csv")
        if len(h2_summary) != len(ALL_MODELS):
            raise AssertionError("H2 strategy summary must contain one row per model")

    h0_summary_path = RESULTS / "study" / "H0" / "h0_summary.csv"
    h0_summary = _read_csv(h0_summary_path)
    _require_columns(
        h0_summary_path,
        h0_summary,
        {
            "model", "word", "n_independent_carriers",
            "mean_signed_M_l_carrier_norm", "direction_consistency",
            "threshold_status",
        },
    )
    h0_cells = {(row["model"], row["word"]) for row in h0_summary}
    if h0_cells != expected_cells or len(h0_summary) != len(expected_cells):
        raise AssertionError(
            f"H0 summary does not contain exactly all {len(ALL_MODELS) * len(WORDS)} model-word cells"
        )
    for row in h0_summary:
        if int(row["n_independent_carriers"]) != 5:
            raise AssertionError(
                f"H0 {row['model']}/{row['word']}: expected 5 independent carriers"
            )
    _require_norm_bounds(h0_summary_path, h0_summary, ["mean_signed_M_l_carrier_norm"])

    for model, word in sorted(expected_cells):
        path = RESULTS / "study" / "H0" / model / f"h0_{word}.csv"
        rows = _read_csv(path)
        _require_columns(
            path,
            rows,
            {"signed_M_l_carrier_norm", "signed_M_l_word_alone_norm", "prior_direction"},
        )
        if len(rows) != 5:
            raise AssertionError(f"{path}: expected 5 independent carriers, found {len(rows)}")
        _require_norm_bounds(
            path, rows, ["signed_M_l_carrier_norm", "signed_M_l_word_alone_norm"]
        )

    h3_path = RESULTS / "study" / "H3" / "h3_aggregate.csv"
    h3_rows = _read_csv(h3_path)
    _require_columns(h3_path, h3_rows, {"model", "word", "condition", "mean_M_l_norm"})
    expected_h3 = {(model, word, condition) for model, word in expected_cells for condition in ("L", "R")}
    observed_h3 = {(row["model"], row["word"], row["condition"]) for row in h3_rows}
    if observed_h3 != expected_h3 or len(h3_rows) != len(expected_h3):
        raise AssertionError(
            f"H3 aggregate does not contain exactly all {len(ALL_MODELS) * len(WORDS) * 2} model-word-condition cells"
        )
    _require_norm_bounds(h3_path, h3_rows, ["mean_M_l_norm"])
    for model, word in sorted(expected_cells):
        path = RESULTS / "study" / "H3" / model / f"h3_{word}.csv"
        rows = _read_csv(path)
        if len(rows) != 20:
            raise AssertionError(f"{path}: expected 20 paired sentences, found {len(rows)}")
        if Counter(row["condition"] for row in rows) != Counter({"L": 10, "R": 10}):
            raise AssertionError(f"{path}: L/R rows are not balanced")
        _require_norm_bounds(path, rows, ["M_l_norm"])
    paired_summary = _read_csv(RESULTS / "study" / "H3" / "h3_paired_summary.csv")
    if len(paired_summary) != len(ALL_MODELS):
        raise AssertionError("H3 paired summary must contain one row per model")
    interaction = _read_csv(RESULTS / "study" / "H3" / "h3_architecture_interaction.csv")
    if len(interaction) != 1 or interaction[0].get("bootstrap_unit") != "crossed model-by-word":
        raise AssertionError("H3 architecture interaction is missing the crossed bootstrap")

    h4_path = RESULTS / "study" / "H4" / "h4_aggregate.csv"
    h4_rows = _read_csv(h4_path)
    _require_columns(
        h4_path,
        h4_rows,
        {
            "model", "word", "p_final_adequate_given_target_inadequate",
            "p_final_inadequate_given_target_adequate", "primary_estimand",
        },
    )
    h4_cells = {(row["model"], row["word"]) for row in h4_rows}
    if h4_cells != expected_cells or len(h4_rows) != len(expected_cells):
        raise AssertionError(
            f"H4 aggregate does not contain exactly all {len(ALL_MODELS) * len(WORDS)} model-word cells"
        )
    if any(row["primary_estimand"] != "within-position sense decodability" for row in h4_rows):
        raise AssertionError("H4 primary estimand is not the corrected local-decoding analysis")
    for model, word in sorted(expected_cells):
        path = RESULTS / "study" / "H4" / model / f"h4_{word}.csv"
        rows = _read_csv(path)
        _require_columns(path, rows, {"target_local_margin_norm", "final_local_margin_norm", "local_transition"})
        if len(rows) != 10:
            raise AssertionError(f"{path}: expected 10 R-condition sentences, found {len(rows)}")
        _require_norm_bounds(path, rows, ["target_local_margin_norm", "final_local_margin_norm"])

    h5_path = RESULTS / "study" / "H5" / "h5_design_audit.csv"
    h5_rows = _read_csv(h5_path)
    expected_h5_words = set(WORDS) | {"light"}
    if {row["word"] for row in h5_rows} != expected_h5_words or len(h5_rows) != len(expected_h5_words):
        raise AssertionError(
            "H5 audit must contain exactly the seven study homonyms plus the documented light exclusion"
        )
    light = next(row for row in h5_rows if row["word"] == "light")
    if light["eligible_for_h5"].lower() != "false" or "parts of speech" not in light["exclusion_reason"]:
        raise AssertionError("H5 light exclusion is missing or incorrectly documented")
    eligible = [row for row in h5_rows if row["eligible_for_h5"].lower() == "true"]
    if {row["word"] for row in eligible} != set(WORDS):
        raise AssertionError("H5 eligible words do not match the seven study homonyms")
    all_h5_ready = bool(eligible) and all(
        row["model_internal_ready"].lower() == "true" for row in eligible
    )
    if full:
        if not all_h5_ready:
            raise AssertionError("H5 audit is not model-internal-ready after the full run")
        with Path("data/garden_path_sentences.json").open() as handle:
            h5_data = json.load(handle)
        h5_words = [word for word in WORDS if word != "light"]
        expected_h5 = {
            (model.replace("/", "_"), word)
            for model in ALL_MODELS for word in h5_words
        }
        aggregate_path = RESULTS / "study" / "H5" / "h5_aggregate.csv"
        aggregate = _read_csv(aggregate_path)
        _require_columns(
            aggregate_path,
            aggregate,
            {
                "model", "word", "mean_delta_resolution_minus_homonym_norm",
                "mean_garden_path_cost_vs_matched_control_norm", "analysis_status",
            },
        )
        observed_h5 = {(row["model"], row["word"]) for row in aggregate}
        if observed_h5 != expected_h5 or len(aggregate) != len(expected_h5):
            raise AssertionError(
                f"H5 aggregate does not contain all {len(ALL_MODELS) * len(h5_words)} eligible model-word cells"
            )
        if any(row["analysis_status"] != "model_internal_ready" for row in aggregate):
            raise AssertionError("H5 output is not labelled model_internal_ready")
        sentence_path = RESULTS / "study" / "H5" / "h5_sentence_level.csv"
        sentence_rows = _read_csv(sentence_path)
        if len(sentence_rows) != len(ALL_MODELS) * sum(len(h5_data[word]) for word in h5_words):
            raise AssertionError("H5 sentence-level output has an incomplete row count")
        for model, word in sorted(expected_h5):
            path = RESULTS / "study" / "H5" / model / f"h5_{word}.csv"
            rows = _read_csv(path)
            if len(rows) != len(h5_data[word]):
                raise AssertionError(f"{path}: incomplete H5 item count")
            _require_norm_bounds(
                path,
                rows,
                [
                    "prime_correct_margin_norm", "homonym_correct_margin_norm",
                    "resolution_correct_margin_norm",
                    "resolver_isolated_correct_margin_norm",
                    "matched_control_correct_margin_norm",
                ],
            )
    elif all_h5_ready:
        raise AssertionError(
            "H5 is model-internal-ready; use the full-study batch instead of the partial audit"
        )

    for stem in (
        "h0_prior_matrix",
        "h3_paired_context_effect",
        "h4_conditional_transitions",
    ):
        for suffix in (".svg", ".png"):
            figure = RESULTS / "study" / "figures" / f"{stem}{suffix}"
            if not figure.exists():
                raise AssertionError(f"missing corrected audit figure: {figure}")
            _require_fresh(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--newer-than")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        validate_preflight()
        print(
            "Preflight passed: paired design, 128 selected-layer caches, "
            "and all 8 offline model caches are present."
        )
    else:
        validate_postflight(args.newer_than, full=args.full)
        scope = "H0–H5" if args.full else "H0/H3/H4"
        print(f"Postflight passed: complete corrected {scope} outputs are present.")


if __name__ == "__main__":
    main()
