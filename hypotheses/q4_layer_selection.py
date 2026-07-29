"""Q4 -- fair decoder layer selection using left-resolved (L-condition)
paired sentences, scored at the homonym position.

Motivation
----------
H1's layer selection is computed at the homonym token position for every
architecture. H3 shows this is only fair for encoders: a causal decoder at
the homonym position can never see a disambiguating clause that comes later
in the sentence (R-condition decoder adequacy is exactly 0.500 there, by
construction of the causal mask). But H3's own L-condition sentences place
the disambiguating clause BEFORE the homonym, so by the time a causal
decoder reaches the homonym token it has already processed that clause --
the same guarantee encoders always have, regardless of clause order (H3
confirms encoder margins barely move between L and R).

Q4 therefore reruns H1's question -- which layer best supports held-out
sense decoding? -- using H3's L-condition paired sentences (guaranteed
disambiguating context available) instead of the plain profiling sentences,
still read out at the homonym position, at *every* layer instead of only
the single H1-selected layer. This keeps the token position H1 always used;
it only changes which stimuli are read out there, removing the
causal-availability confound.

Two stages
----------
1. ``extract_l_condition_activations`` -- one new forward pass per model
   over the L-condition sentences (10 per word), with ``layer_indices=None``
   so every layer's homonym-position state is captured in one pass, cached
   to ``results/activations_paired_left/{word}/{model}/layer_{i}.h5``. This
   is the only step that needs a GPU/model load; a word/model already
   cached is skipped.
2. ``run_q4`` -- pure CPU. Scores every cached layer against the *existing*
   homonym-position profiling centroids (``results/activations/``, already
   computed for H1/H2) using the same ``symmetric_adequacy_margins`` H3/H4
   use, aggregates fraction_adequate and mean normalised margin per layer,
   and identifies *candidate* layers rather than a single winner: every
   layer within ``tolerance`` of that model/word's own peak adequacy, AND
   at or above ``min_adequacy`` (an absolute floor). A fixed adequacy
   threshold alone would be a near-ceiling bar for decoders but a low,
   uninformative one for encoders (which often peak near 0.95-1.0);
   relative-to-peak adapts to each model's own ceiling, and the absolute
   floor keeps a flat/noisy curve (seen in some decoders) from admitting
   most of its layers just because they cluster near a low peak.

No new stimuli and no new metrics: same L-condition sentences H3 already
uses, same margin/adequacy functions, same per-layer scoring style H1 uses.

Output
------
results/study/Q4/q4_layer_curves.csv
    One row per (model, word, layer): model, architecture, homonym, layer,
    relative_depth, adequacy, mean_margin, is_candidate, peak_adequacy,
    best_layer (the single adequacy_best_layer winner, kept for reference/
    labelling only -- the candidate *set* is the primary output).

results/study/Q4/{safe_model}/q4_candidate_layers.csv
    One row per (word, layer) that is a candidate. Use
    q4_candidate_layers(...) as run_h5's layer_candidates_lookup to rerun H5
    once per candidate layer.
"""

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

from experiments.adequacy import (
    adequacy_best_layer,
    load_centroids,
    symmetric_adequacy_margins,
    symmetric_normalized_adequacy_margins,
)
from hypotheses.h3_context_position import H3_MODELS, PAIRED_DATA_PATH, _load_paired_sentences
from models import get_target_activations, load_model_and_tokenizer
from utils.hpc import configure_hpc_runtime

configure_hpc_runtime()
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
OUTPUT_BASE = Path("results/study/Q4")
L_CONDITION_CACHE_SUBDIR = "activations_paired_left"
DEFAULT_WORDS = ["bank", "bark", "bat", "crane", "spring", "match", "pitch"]
DEFAULT_TOLERANCE = 0.05
DEFAULT_MIN_ADEQUACY = 0.6

ENCODERS = {
    "answerdotai/ModernBERT-large",
    "microsoft/deberta-v3-large",
    "FacebookAI/roberta-large",
    "FacebookAI/xlm-roberta-large",
}


def _is_decoder_name(model_name: str) -> bool:
    return model_name not in ENCODERS


def _l_condition_sentences(paired_data_path: str, word: str):
    """Return (sentences, senses) for the L-condition rows of this word,
    reusing H3's exact stimulus-loading path so Q4 sees the identical
    sentence set/parsing H3 does."""
    sentences, conditions, _ids, senses, _carriers = _load_paired_sentences(paired_data_path, word)
    l_sentences, l_senses = [], []
    for sentence, condition, sense in zip(sentences, conditions, senses):
        if condition == "L":
            l_sentences.append(sentence)
            l_senses.append(sense)
    return l_sentences, l_senses


def _l_condition_cache_dir(results_dir: str, word: str, safe_model: str) -> Path:
    return Path(results_dir) / L_CONDITION_CACHE_SUBDIR / word / safe_model


def _has_l_condition_cache(results_dir: str, word: str, safe_model: str) -> bool:
    cache_dir = _l_condition_cache_dir(results_dir, word, safe_model)
    return cache_dir.exists() and any(cache_dir.glob("layer_[1-9]*.h5"))


def _save_l_condition_activations(
    results_dir: str,
    word: str,
    safe_model: str,
    activations: Dict[int, "torch.Tensor"],
    labels: np.ndarray,
    sentences: List[str],
) -> None:
    target_dir = _l_condition_cache_dir(results_dir, word, safe_model)
    target_dir.mkdir(parents=True, exist_ok=True)
    dt = h5py.string_dtype(encoding="utf-8")
    for layer_idx, tensor in activations.items():
        arr = tensor.cpu().numpy()
        with h5py.File(target_dir / f"layer_{layer_idx}.h5", "w") as f:
            f.create_dataset("X", data=arr)
            f.create_dataset("labels", data=labels)
            f.create_dataset(
                "sentences", data=np.array([s.encode("utf-8") for s in sentences]), dtype=dt
            )


def _load_l_condition_layer(results_dir: str, word: str, safe_model: str, layer: int):
    path = _l_condition_cache_dir(results_dir, word, safe_model) / f"layer_{layer}.h5"
    with h5py.File(path, "r") as f:
        return f["X"][:], f["labels"][:]


def extract_l_condition_activations(
    model_names: Optional[List[str]] = None,
    words: Optional[List[str]] = None,
    results_dir: str = RESULTS_DIR,
    paired_data_path: str = PAIRED_DATA_PATH,
    force: bool = False,
) -> None:
    """GPU step: cache every layer's homonym-position state for the
    L-condition sentences. Requires loading each model checkpoint; skipped
    per model/word if already cached, unless force=True."""
    model_names = model_names or H3_MODELS
    words = words or DEFAULT_WORDS

    for model_name in model_names:
        safe_model = model_name.replace("/", "_")
        words_todo = [
            w for w in words
            if force or not _has_l_condition_cache(results_dir, w, safe_model)
        ]
        if not words_todo:
            logger.info("[Q4] %s: all words already cached, skipping model load.", model_name)
            continue

        model, tokenizer = load_model_and_tokenizer(model_name)
        logger.info("[Q4] %s: extracting L-condition activations for %s", model_name, words_todo)
        for word in words_todo:
            try:
                sentences, sense_list = _l_condition_sentences(paired_data_path, word)
            except KeyError as exc:
                logger.warning("[Q4] %s", exc)
                continue
            if not sentences:
                logger.warning("[Q4] No L-condition sentences for '%s'.", word)
                continue

            senses = np.asarray(sense_list)

            # layer_indices=None -> every layer in one forward pass, at the
            # homonym-token position (pooling="target"), same as H3/H1.
            target_acts = get_target_activations(
                model, tokenizer, sentences, [word] * len(sentences),
                batch_size=4, layer_indices=None, pooling="target",
            )
            _save_l_condition_activations(
                results_dir, word, safe_model, target_acts, senses, sentences
            )
            logger.info(
                "[Q4] %s/%s: cached %d layers x %d L-condition sentences",
                model_name, word, len(target_acts), len(sentences),
            )
        del model, tokenizer


def _score_layer(
    results_dir: str, word: str, safe_model: str, layer: int, c0: np.ndarray, c1: np.ndarray,
    epsilon: float = 0.0,
):
    X, labels = _load_l_condition_layer(results_dir, word, safe_model, layer)
    margins_norm = symmetric_normalized_adequacy_margins(X, labels, c0, c1)
    margins_raw = symmetric_adequacy_margins(X, labels, c0, c1)
    return {
        "fraction_adequate": float((margins_raw > epsilon).mean()),
        "mean_norm": float(margins_norm.mean()),
    }


def run_q4(
    model_names: Optional[List[str]] = None,
    words: Optional[List[str]] = None,
    results_dir: str = RESULTS_DIR,
    epsilon: float = 0.0,
    tolerance: float = DEFAULT_TOLERANCE,
    min_adequacy: float = DEFAULT_MIN_ADEQUACY,
) -> None:
    """CPU step: score every cached L-condition layer against the existing
    homonym-position profiling centroids, and mark candidate layers as
    every layer within `tolerance` of that model/word's own peak adequacy
    AND at or above `min_adequacy`."""
    model_names = model_names or H3_MODELS
    words = words or DEFAULT_WORDS
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    curve_rows: List[Dict] = []

    for model_name in model_names:
        safe_model = model_name.replace("/", "_")
        arch_type = "decoder" if _is_decoder_name(model_name) else "encoder"
        candidate_rows: List[Dict] = []

        for word in words:
            if not _has_l_condition_cache(results_dir, word, safe_model):
                logger.warning(
                    "[Q4] No L-condition cache for %s/%s; run extract_l_condition_activations first.",
                    model_name, word,
                )
                continue
            try:
                centroids = load_centroids(results_dir, model_name, word)
            except FileNotFoundError as exc:
                logger.warning("[Q4] %s", exc)
                continue

            cache_dir = _l_condition_cache_dir(results_dir, word, safe_model)
            layers = sorted(
                int(p.stem.removeprefix("layer_")) for p in cache_dir.glob("layer_*.h5")
            )
            if not layers:
                continue
            last_layer = layers[-1]

            profile: Dict[int, Dict] = {}
            for layer in layers:
                if layer not in centroids:
                    continue
                c0, c1 = centroids[layer][0], centroids[layer][1]
                profile[layer] = _score_layer(
                    results_dir, word, safe_model, layer, c0, c1, epsilon
                )
            if not profile:
                logger.warning("[Q4] Empty profile for %s/%s", model_name, word)
                continue

            best_layer = adequacy_best_layer(profile)
            peak_adequacy = max(p["fraction_adequate"] for p in profile.values())
            candidates = [
                layer for layer, p in profile.items()
                if p["fraction_adequate"] >= peak_adequacy - tolerance
                and p["fraction_adequate"] >= min_adequacy
            ]
            candidates.sort()

            for layer in layers:
                if layer not in profile:
                    continue
                curve_rows.append({
                    "model": safe_model,
                    "architecture": arch_type,
                    "homonym": word,
                    "layer": layer,
                    "relative_depth": round(layer / last_layer, 6),
                    "adequacy": round(profile[layer]["fraction_adequate"], 6),
                    "mean_margin": round(profile[layer]["mean_norm"], 6),
                    "is_candidate": layer in candidates,
                    "peak_adequacy": round(peak_adequacy, 6),
                    "best_layer": best_layer,
                })
            for layer in candidates:
                candidate_rows.append({
                    "word": word,
                    "layer": layer,
                    "relative_depth": round(layer / last_layer, 6),
                    "adequacy": round(profile[layer]["fraction_adequate"], 6),
                    "mean_margin": round(profile[layer]["mean_norm"], 6),
                })
            logger.info(
                "[Q4] %s/%s: %d/%d layers are candidates (peak adequacy=%.3f, tolerance=%.2f, floor=%.2f); best=%d",
                model_name, word, len(candidates), len(layers), peak_adequacy, tolerance,
                min_adequacy, best_layer,
            )

        if candidate_rows:
            model_out = OUTPUT_BASE / safe_model
            model_out.mkdir(parents=True, exist_ok=True)
            with open(model_out / "q4_candidate_layers.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=candidate_rows[0].keys())
                writer.writeheader()
                writer.writerows(candidate_rows)

    if curve_rows:
        with open(OUTPUT_BASE / "q4_layer_curves.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=curve_rows[0].keys())
            writer.writeheader()
            writer.writerows(curve_rows)
        logger.info("[Q4] Wrote %d rows to %s", len(curve_rows), OUTPUT_BASE / "q4_layer_curves.csv")


def q4_candidate_layers(model_name: str, results_dir: str, word: str) -> List[int]:
    """layer_candidates_lookup-compatible callable (same signature shape as
    hypotheses.h3_context_position._select_layer, but returns a list) for
    run_h5(layer_candidates_lookup=...). Reads the candidate layers Q4
    selected for this model/word instead of H1's single homonym-position
    layer."""
    safe = model_name.replace("/", "_")
    path = Path(results_dir) / "study" / "Q4" / safe / "q4_candidate_layers.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No Q4 candidate-layer summary at {path}; run run_q4(...) first."
        )
    layers = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["word"] == word:
                layers.append(int(row["layer"]))
    return sorted(layers)
