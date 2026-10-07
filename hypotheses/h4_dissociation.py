"""H4 — Token-position sense decodability.

H4 compares two readout positions from the *same completed forward pass*:
the homonym and the last non-special token. It is therefore a sequence-
position analysis, not incremental reading or representational recovery.

The primary comparison scores each position in its own profiling geometry:

* target_local_margin: homonym state vs homonym-position sense centroids;
* final_local_margin: final state vs final-position sense centroids.

This asks whether sentence sense is decodable at each position. The two
margins do not establish that the sense was recovered in another basis.
For completeness, final_cross_position_margin scores the final state against
homonym centroids. Because both token identity and position change, this is
only a cross-position transfer diagnostic and is never the primary H4 score.

The headline summaries are conditional transitions rather than the raw
prevalence of one cell: P(final adequate | target inadequate) and
P(final inadequate | target adequate). Counts and Wilson intervals make
small or empty denominators visible.
"""

import csv
import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from experiments.adequacy import (
    load_centroids,
    load_final_centroids,
    symmetric_adequacy_margins,
    symmetric_normalized_adequacy_margins,
)
from hypotheses.h3_context_position import H3_MODELS, PAIRED_DATA_PATH, _select_layer
from utils.hpc import configure_hpc_runtime

configure_hpc_runtime()
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
OUTPUT_BASE = Path("results/study/H4")
DEFAULT_WORDS = ["bank", "bark", "bat", "crane", "spring", "match", "pitch"]


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a 95% Wilson interval; NaNs make an empty denominator explicit."""
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return centre - half, centre + half


def conditional_transition_summary(
    target_adequate: Sequence[bool], final_adequate: Sequence[bool]
) -> Dict[str, float | int]:
    """Summarise the full 2x2 target-to-final table and both conditional rates."""
    if len(target_adequate) != len(final_adequate):
        raise ValueError("target_adequate and final_adequate must have equal length")

    counts = {
        "n_target_inadequate_final_inadequate": 0,
        "n_target_inadequate_final_adequate": 0,
        "n_target_adequate_final_inadequate": 0,
        "n_target_adequate_final_adequate": 0,
    }
    for target_ok, final_ok in zip(target_adequate, final_adequate):
        key = (
            f"n_target_{'adequate' if target_ok else 'inadequate'}_"
            f"final_{'adequate' if final_ok else 'inadequate'}"
        )
        counts[key] += 1

    n_target_bad = (
        counts["n_target_inadequate_final_inadequate"]
        + counts["n_target_inadequate_final_adequate"]
    )
    n_target_good = (
        counts["n_target_adequate_final_inadequate"]
        + counts["n_target_adequate_final_adequate"]
    )
    n_gain = counts["n_target_inadequate_final_adequate"]
    n_loss = counts["n_target_adequate_final_inadequate"]
    gain_lo, gain_hi = _wilson_interval(n_gain, n_target_bad)
    loss_lo, loss_hi = _wilson_interval(n_loss, n_target_good)

    return {
        **counts,
        "n_target_inadequate": n_target_bad,
        "p_final_adequate_given_target_inadequate": (
            n_gain / n_target_bad if n_target_bad else math.nan
        ),
        "p_final_adequate_given_target_inadequate_ci_low": gain_lo,
        "p_final_adequate_given_target_inadequate_ci_high": gain_hi,
        "n_target_adequate": n_target_good,
        "p_final_inadequate_given_target_adequate": (
            n_loss / n_target_good if n_target_good else math.nan
        ),
        "p_final_inadequate_given_target_adequate_ci_low": loss_lo,
        "p_final_inadequate_given_target_adequate_ci_high": loss_hi,
    }


def _round_or_nan(value: float, digits: int = 4) -> float:
    return round(float(value), digits) if math.isfinite(float(value)) else math.nan


def _transition_label(target_ok: bool, final_ok: bool) -> str:
    if not target_ok and final_ok:
        return "inadequate_to_adequate"
    if target_ok and not final_ok:
        return "adequate_to_inadequate"
    if target_ok and final_ok:
        return "adequate_at_both"
    return "inadequate_at_both"


def _last_surface_unit(sentence: str) -> str:
    """Human-readable final word/punctuation; model tokenization may subdivide it."""
    units = re.findall(r"\w+(?:[-']\w+)*|[^\w\s]", sentence, flags=re.UNICODE)
    return units[-1] if units else ""


def run_h4(
    model_names: Optional[List[str]] = None,
    words: Optional[List[str]] = None,
    results_dir: str = RESULTS_DIR,
    paired_data_path: str = PAIRED_DATA_PATH,
    epsilon: float = 0.0,
) -> None:
    """Run H4 as a descriptive token-position decodability analysis."""
    model_names = model_names or H3_MODELS
    words = words or DEFAULT_WORDS
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    if not Path(paired_data_path).exists():
        logger.error("[H4] Paired sentence dataset not found at %s.", paired_data_path)
        return

    with open(paired_data_path) as f:
        paired_data = json.load(f)

    aggregate_rows = []
    all_sentence_rows = []

    for model_name in model_names:
        safe_model = model_name.replace("/", "_")
        model_out = OUTPUT_BASE / safe_model
        model_out.mkdir(parents=True, exist_ok=True)

        from models import get_dual_position_activations, is_decoder_only, load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(model_name)
        arch_type = "decoder" if is_decoder_only(model) else "encoder"
        logger.info("[H4] %s (%s)", model_name, arch_type)

        for word in words:
            if word not in paired_data:
                logger.warning("[H4] Word '%s' not in paired data - skipping.", word)
                continue
            items = [item for item in paired_data[word] if item["condition"] == "R"]
            if not items:
                logger.warning("[H4] No R-condition sentences for '%s'.", word)
                continue

            sentences = [item["sentence"] for item in items]
            sent_ids = [item.get("id", f"{word}_R_{i}") for i, item in enumerate(items)]
            senses = np.asarray([item["sense"] for item in items])

            try:
                target_centroids = load_centroids(results_dir, model_name, word)
                final_centroids = load_final_centroids(results_dir, model_name, word)
                layer_idx = _select_layer(model_name, results_dir, word)
            except FileNotFoundError as exc:
                logger.warning("[H4] %s", exc)
                continue
            if layer_idx not in target_centroids or layer_idx not in final_centroids:
                logger.warning("[H4] Missing layer %s for %s/%s", layer_idx, model_name, word)
                continue

            c0, c1 = target_centroids[layer_idx][0], target_centroids[layer_idx][1]
            fc0, fc1 = final_centroids[layer_idx][0], final_centroids[layer_idx][1]
            target_acts, final_acts = get_dual_position_activations(
                model, tokenizer, sentences, [word] * len(sentences),
                batch_size=4, layer_indices=[layer_idx],
            )
            h_target = target_acts[layer_idx].numpy()
            h_final = final_acts[layer_idx].numpy()

            target_local_raw = symmetric_adequacy_margins(h_target, senses, c0, c1)
            target_local_norm = symmetric_normalized_adequacy_margins(h_target, senses, c0, c1)
            final_local_raw = symmetric_adequacy_margins(h_final, senses, fc0, fc1)
            final_local_norm = symmetric_normalized_adequacy_margins(h_final, senses, fc0, fc1)
            final_cross_raw = symmetric_adequacy_margins(h_final, senses, c0, c1)
            final_cross_norm = symmetric_normalized_adequacy_margins(h_final, senses, c0, c1)

            target_ok = target_local_raw > epsilon
            final_ok = final_local_raw > epsilon
            transition = conditional_transition_summary(target_ok, final_ok)
            csv_rows = []
            for i, item in enumerate(items):
                row = {
                    "model": safe_model,
                    "arch_type": arch_type,
                    "word": word,
                    "sentence_id": sent_ids[i],
                    "sentence": sentences[i],
                    "carrier": item.get("carrier", ""),
                    "sense": int(senses[i]),
                    "final_surface_unit": _last_surface_unit(sentences[i]),
                    "target_local_margin_raw": round(float(target_local_raw[i]), 4),
                    "target_local_margin_norm": round(float(target_local_norm[i]), 4),
                    "final_local_margin_raw": round(float(final_local_raw[i]), 4),
                    "final_local_margin_norm": round(float(final_local_norm[i]), 4),
                    "final_cross_position_margin_raw": round(float(final_cross_raw[i]), 4),
                    "final_cross_position_margin_norm": round(float(final_cross_norm[i]), 4),
                    "target_local_adequate": bool(target_ok[i]),
                    "final_local_adequate": bool(final_ok[i]),
                    "final_cross_position_adequate": bool(final_cross_raw[i] > epsilon),
                    "local_transition": _transition_label(bool(target_ok[i]), bool(final_ok[i])),
                    "layer": layer_idx,
                }
                csv_rows.append(row)
                all_sentence_rows.append(row)

            with open(model_out / f"h4_{word}.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)

            aggregate_rows.append({
                "model": safe_model,
                "arch_type": arch_type,
                "word": word,
                "n_R_sentences": len(csv_rows),
                "mean_target_local_margin_raw": round(float(target_local_raw.mean()), 4),
                "mean_target_local_margin_norm": round(float(target_local_norm.mean()), 4),
                "mean_final_local_margin_raw": round(float(final_local_raw.mean()), 4),
                "mean_final_local_margin_norm": round(float(final_local_norm.mean()), 4),
                "mean_final_cross_position_margin_raw": round(float(final_cross_raw.mean()), 4),
                "mean_final_cross_position_margin_norm": round(float(final_cross_norm.mean()), 4),
                "frac_target_local_adequate": round(float(target_ok.mean()), 3),
                "frac_final_local_adequate": round(float(final_ok.mean()), 3),
                "frac_final_cross_position_adequate": round(float((final_cross_raw > epsilon).mean()), 3),
                **{key: _round_or_nan(value, 3) if key.startswith("p_") else value
                   for key, value in transition.items()},
                "layer_used": layer_idx,
                "primary_estimand": "within-position sense decodability",
                "cross_position_status": "diagnostic_only",
            })
            logger.info(
                "[H4] %s/%s | P(final adequate | target inadequate)=%s/%s; "
                "P(final inadequate | target adequate)=%s/%s",
                model_name, word,
                transition["n_target_inadequate_final_adequate"],
                transition["n_target_inadequate"],
                transition["n_target_adequate_final_inadequate"],
                transition["n_target_adequate"],
            )

    if aggregate_rows:
        with open(OUTPUT_BASE / "h4_aggregate.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=aggregate_rows[0].keys())
            writer.writeheader()
            writer.writerows(aggregate_rows)
    if all_sentence_rows:
        with open(OUTPUT_BASE / "h4_sentence_level.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_sentence_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_sentence_rows)
