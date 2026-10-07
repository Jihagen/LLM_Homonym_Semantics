"""
H0 — Carrier Norming (prerequisite for H3/H4)
=============================================
Purpose
-------
Computes M_l at three levels for each (model, word):

  1. word_alone_M_l  — bare word ("bank") as its own sentence.
                       Should be near zero; any deviation reveals an intrinsic
                       lexical embedding bias before any sentence context.

  2. carrier_M_l     — word inside the ambiguous carrier sentence
                       ("The bank was unstable."), no disambiguating clause.
                       Shows what the sentence frame alone contributes.

  3. context_gain    — M_l(full sentence) - M_l(carrier)   [computed in H3]
                       Shows what the context clause specifically adds.

Each ambiguous carrier state was historically scored once under each opposing
label, yielding mirrored values M and -M and an apparent count of ten. The
current output collapses those into five independent signed carrier priors:
positive means closer to sense 0 and negative means closer to sense 1.

Output
------
results/study/H0/{safe_model}/h0_{word}.csv
  one row per independent carrier, with signed raw/normalized prior and shift

results/study/H0/h0_summary.csv
  signed direction, magnitude, and cross-carrier directional consistency

Required data
-------------
- data/paired_sentences.json (must contain 'carrier' field on each item)
- results/activations/{word}/{safe_model}/ (centroids)
- Models: all 8 registered study models
"""

import csv
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from experiments.adequacy import (
    compute_carrier_margins,
    compute_word_alone_margins,
    load_centroids,
)
from hypotheses.h3_context_position import H3_MODELS, PAIRED_DATA_PATH, _select_layer
from utils.hpc import configure_hpc_runtime

configure_hpc_runtime()
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
OUTPUT_BASE = Path("results/study/H0")
BIAS_SENSITIVITY_THRESHOLD_NORM = 0.3


def _extract_unique_carriers(paired_data: dict, word: str) -> List[dict]:
    """Extract one record per unique (carrier, sense) pair for a word."""
    seen, records = set(), []
    for item in paired_data.get(word, []):
        if "carrier" not in item:
            logger.warning("[H0] Item %s missing 'carrier' field.", item.get("id"))
            continue
        key = (item["carrier"], item["sense"])
        if key not in seen:
            seen.add(key)
            records.append({"word": word, "carrier": item["carrier"], "sense": item["sense"]})
    return records


def _collapse_mirrored_carriers(
    records: List[dict],
    word_alone_raw: float,
    scale: float,
) -> List[dict]:
    """Collapse M/-M label mirrors into independent signed carrier states."""
    by_carrier = {}
    for record in records:
        by_carrier.setdefault(record["carrier"], {})[int(record["sense"])] = record

    collapsed = []
    for carrier, by_sense in by_carrier.items():
        if 0 in by_sense and 1 in by_sense:
            signed_raw = 0.5 * (
                float(by_sense[0]["M_l_carrier"])
                - float(by_sense[1]["M_l_carrier"])
            )
        elif 0 in by_sense:
            signed_raw = float(by_sense[0]["M_l_carrier"])
        elif 1 in by_sense:
            signed_raw = -float(by_sense[1]["M_l_carrier"])
        else:
            continue

        signed_norm = signed_raw / scale
        if abs(signed_norm) > 1.00001:
            raise ValueError(
                f"H0 normalized carrier prior outside [-1, 1] for {carrier!r}: "
                f"{signed_norm}"
            )
        shift_raw = signed_raw - word_alone_raw
        shift_norm = shift_raw / scale
        direction = "sense_0" if signed_norm > 0 else "sense_1" if signed_norm < 0 else "tie"
        collapsed.append({
            "word": by_sense[next(iter(by_sense))]["word"],
            "carrier": carrier,
            "signed_M_l_word_alone_raw": round(float(word_alone_raw), 4),
            "signed_M_l_word_alone_norm": round(float(word_alone_raw / scale), 4),
            "signed_M_l_carrier_raw": round(float(signed_raw), 4),
            "signed_M_l_carrier_norm": round(float(signed_norm), 4),
            "signed_carrier_shift_raw": round(float(shift_raw), 4),
            "signed_carrier_shift_norm": round(float(shift_norm), 4),
            "prior_direction": direction,
            "abs_norm_gt_0p3": abs(signed_norm) > BIAS_SENSITIVITY_THRESHOLD_NORM,
            "threshold_status": "descriptive_sensitivity_only",
            "layer": by_sense[next(iter(by_sense))]["layer"],
        })
    return collapsed


def run_h0(
    model_names: Optional[List[str]] = None,
    words: Optional[List[str]] = None,
    results_dir: str = RESULTS_DIR,
    paired_data_path: str = PAIRED_DATA_PATH,
) -> None:
    """
    Run H0 carrier norming for all specified models and words.
    Results are saved to results/study/H0/ and used by H3 for context_gain.
    """
    model_names = model_names or H3_MODELS
    words       = words or ["bank", "bark", "bat", "crane", "spring", "match", "pitch"]
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    if not Path(paired_data_path).exists():
        logger.error("[H0] Paired data not found at %s.", paired_data_path)
        return

    with open(paired_data_path) as f:
        paired_data = json.load(f)

    summary_rows = []

    for model_name in model_names:
        safe_model = model_name.replace("/", "_")
        model_out  = OUTPUT_BASE / safe_model
        model_out.mkdir(parents=True, exist_ok=True)

        from models import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(model_name)
        logger.info("[H0] %s", model_name)

        for word in words:
            carrier_items = _extract_unique_carriers(paired_data, word)
            if not carrier_items:
                logger.warning("[H0] No carriers found for word '%s'.", word)
                continue

            try:
                centroids = load_centroids(results_dir, model_name, word)
                layer_idx = _select_layer(model_name, results_dir, word)
            except FileNotFoundError as e:
                logger.warning("[H0] %s", e)
                continue

            # Scale for cross-architecture-comparable margins — see the
            # normalization note in experiments/adequacy.py.
            scale = float(np.linalg.norm(centroids[layer_idx][0] - centroids[layer_idx][1])) + 1e-12

            # ── Level 1: word alone ───────────────────────────────────────
            word_alone = compute_word_alone_margins(
                model, tokenizer, [word], centroids, layer_idx,
            )
            m_word_alone = word_alone.get(word, float("nan"))

            # ── Level 2: carrier sentences ────────────────────────────────
            records = compute_carrier_margins(
                model, tokenizer, carrier_items, centroids, layer_idx,
            )
            if not records:
                continue

            independent = _collapse_mirrored_carriers(records, m_word_alone, scale)
            if not independent:
                continue

            csv_path = model_out / f"h0_{word}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=independent[0].keys())
                writer.writeheader()
                writer.writerows(independent)

            signed_raw = np.asarray([r["signed_M_l_carrier_raw"] for r in independent])
            signed_norm = np.asarray([r["signed_M_l_carrier_norm"] for r in independent])
            shift_raw = np.asarray([r["signed_carrier_shift_raw"] for r in independent])
            nonzero = signed_norm[np.abs(signed_norm) > 1e-8]
            direction_consistency = (
                max(float((nonzero > 0).mean()), float((nonzero < 0).mean()))
                if len(nonzero) else 0.5
            )
            summary_rows.append({
                "model":                 safe_model,
                "word":                  word,
                "n_independent_carriers": len(independent),
                "signed_M_l_word_alone_raw": round(float(m_word_alone), 4),
                "signed_M_l_word_alone_norm": round(float(m_word_alone / scale), 4),
                "mean_signed_M_l_carrier_raw": round(float(signed_raw.mean()), 4),
                "mean_signed_M_l_carrier_norm": round(float(signed_norm.mean()), 4),
                "mean_abs_M_l_carrier_norm": round(float(np.abs(signed_norm).mean()), 4),
                "mean_abs_carrier_shift_raw": round(float(np.abs(shift_raw).mean()), 4),
                "direction_consistency": round(direction_consistency, 3),
                "n_abs_norm_gt_0p3": int(np.sum(np.abs(signed_norm) > BIAS_SENSITIVITY_THRESHOLD_NORM)),
                "threshold_status": "descriptive_sensitivity_only",
                "layer_used":            layer_idx,
            })
            logger.info(
                "[H0] %s / %s | word_prior=%+.3f carrier_prior=%+.3f | n=%d consistency=%.2f",
                model_name, word,
                m_word_alone / scale, signed_norm.mean(),
                len(independent), direction_consistency,
            )

    if summary_rows:
        summary_path = OUTPUT_BASE / "h0_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info("[H0] Summary saved to %s", summary_path)
