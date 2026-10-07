"""
H3 — Right-Context Vulnerability
==================================
Hypothesis: Decoder homonym-token representations are inadequate when
decisive context appears after the homonym (R condition); encoders are
largely unaffected by context position.

Method
------
For each (model, word):
  1. Load sense centroids from profiling H5 files.
  2. Run a forward pass on paired sentences (L = left-context biased,
     R = right-context biased) from data/paired_sentences.json.
  3. Extract the representation at the homonym-token position (target pooling).
  4. Compute M_l per sentence. Compare L vs R distributions.

Output
------
results/study/H3/{safe_model}/h3_{word}.csv
  columns: sentence_id, condition (L/R), M_l, adequate (bool)

results/study/H3/h3_aggregate.csv
  mean M_l and fraction adequate per (model, arch_type, condition)

results/study/H3/h3_pair_differences.csv
  paired L-minus-R normalized margins for every rearranged sentence pair.

results/study/H3/h3_paired_summary.csv
  per-model L-R effects with word-cluster bootstrap intervals. Decoder R=0.5
  is treated as a deterministic causal-mask sanity check, not a chance test.

results/study/H3/h3_architecture_interaction.csv
  decoder-minus-encoder difference in the paired L-R effect, using a crossed
  model-by-word bootstrap so repeated carriers are not treated as independent
  evidence for the architecture-level claim.

Required data
-------------
- results/activations/{word}/{safe_model}/ (profiling centroids)
- data/paired_sentences.json (L/R paired sentences — see inspect_paired_sentences.ipynb)
- Models: all 8 (4 encoder + 4 decoder — see model_registry.ALL_MODELS). Full
  coverage strengthens "encoder vs. decoder" claims to 4-vs-4 rather than the
  earlier 2-vs-2 representative subset.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.adequacy import symmetric_adequacy_margins, symmetric_normalized_adequacy_margins, load_centroids
from utils.model_registry import ALL_MODELS
from utils.hpc import configure_hpc_runtime

configure_hpc_runtime()
logger = logging.getLogger(__name__)

RESULTS_DIR       = "results"
PAIRED_DATA_PATH  = "data/paired_sentences.json"
OUTPUT_BASE       = Path("results/study/H3")
# H0/H3/H4/H5 originally ran on a 4-model representative subset (2 enc + 2 dec)
# since they need fresh forward passes over paired/garden-path stimuli, unlike
# H1/H2 which reuse cached activations. Now full 8-model coverage — see
# model_registry.py for the single source of truth on model names/aliases.
H3_MODELS         = ALL_MODELS


def _load_paired_sentences(path: str, word: str) -> Tuple[List[str], List[str], List[str], List[int], List[str]]:
    """
    Load paired sentences for a given word from the JSON dataset.

    Returns (sentences, conditions, sentence_ids, senses, carriers)
    where condition is 'L' or 'R' and sense is each item's own true sense
    (0 or 1) — paired_sentences.json is balanced across both senses per
    word/condition, so callers must score each sentence against its own
    true sense rather than a single fixed sense-0/1 pair.
    """
    with open(path) as f:
        data = json.load(f)
    if word not in data:
        raise KeyError(f"Word '{word}' not found in {path}. Add it via the paired sentences notebook.")
    sentences, conditions, ids, senses, carriers = [], [], [], [], []
    for item in data[word]:
        sentences.append(item["sentence"])
        conditions.append(item["condition"])
        ids.append(item.get("id", f"{word}_{len(ids)}"))
        senses.append(item["sense"])
        carriers.append(item["carrier"])
    return sentences, conditions, ids, senses, carriers


def _select_layer(model_name: str, results_dir: str, word: str) -> int:
    """Pick the adequacy-best layer from the H1 summary if available, else last layer."""
    safe = model_name.replace("/", "_")
    h1_summary = Path(results_dir) / "study" / "H1" / safe / "h1_summary.csv"
    if h1_summary.exists():
        with open(h1_summary) as f:
            for row in csv.DictReader(f):
                if row["word"] == word:
                    return int(row["best_layer_M"])
    # fallback: count layers from H5 files
    h5_dir = Path(results_dir) / "activations" / word / safe
    if h5_dir.exists():
        layers = [int(p.stem.split("_")[1]) for p in h5_dir.glob("layer_*.h5")]
        if layers:
            return max(layers)
    raise FileNotFoundError(f"Cannot determine best layer for {model_name} / {word}")


def run_h3(
    model_names: Optional[List[str]] = None,
    words: Optional[List[str]] = None,
    results_dir: str = RESULTS_DIR,
    paired_data_path: str = PAIRED_DATA_PATH,
    epsilon: float = 0.0,
) -> None:
    """
    Run H3 for specified models and words.
    Requires paired_sentences.json and H5 profiling files.
    """
    model_names = model_names or H3_MODELS
    words = words or ["bank", "bark", "bat", "crane", "spring", "match", "pitch"]
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    if not Path(paired_data_path).exists():
        logger.error(
            "[H3] Paired sentence dataset not found at %s. "
            "Run data/inspect_paired_sentences.ipynb to create it.", paired_data_path
        )
        return

    aggregate_rows = []

    for model_name in model_names:
        safe_model = model_name.replace("/", "_")
        model_out  = OUTPUT_BASE / safe_model
        model_out.mkdir(parents=True, exist_ok=True)

        from models import get_target_activations, is_decoder_only, load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(model_name)
        # Always target-token: we measure the homonym's own representation.
        # For decoders this is causal (left-context only) — which is exactly the
        # effect H3 tests. R-condition sentences will show poor adequacy for
        # decoders because the homonym sits at position ~2 with no useful left context.
        arch_type = "decoder" if is_decoder_only(model) else "encoder"
        logger.info("[H3] %s (%s, pooling=target)", model_name, arch_type)

        for word in words:
            try:
                sentences, conditions, sent_ids, senses, carriers = _load_paired_sentences(paired_data_path, word)
            except KeyError as e:
                logger.warning("[H3] %s", e)
                continue

            try:
                centroids  = load_centroids(results_dir, model_name, word)
                layer_idx  = _select_layer(model_name, results_dir, word)
            except FileNotFoundError as e:
                logger.warning("[H3] %s", e)
                continue

            if layer_idx not in centroids:
                logger.warning("[H3] Layer %d not in centroid cache for %s / %s", layer_idx, model_name, word)
                continue

            c0 = centroids[layer_idx][0]
            c1 = centroids[layer_idx][1]

            # Forward pass — extract homonym-position representations
            targets = [word] * len(sentences)
            acts = get_target_activations(
                model, tokenizer,
                sentences, targets,
                batch_size=4,
                layer_indices=[layer_idx],
                pooling="target",
            )
            H = acts[layer_idx].numpy()
            # paired_sentences.json is balanced across both senses per word/condition,
            # so each sentence must be scored against its own true sense.
            senses_arr  = np.array(senses)
            margins     = symmetric_adequacy_margins(H, senses_arr, c0, c1)
            # Normalized by inter-centroid distance — comparable across
            # architectures/layers, unlike raw M_l (see adequacy.py note).
            margins_norm = symmetric_normalized_adequacy_margins(H, senses_arr, c0, c1)

            # Write per-sentence results
            csv_path = model_out / f"h3_{word}.csv"
            rows = []
            for sid, cond, sense, carrier, margin, margin_norm in zip(
                sent_ids, conditions, senses, carriers, margins, margins_norm
            ):
                rows.append({
                    "sentence_id": sid,
                    "pair_id":     sid.replace("_L_", "_").replace("_R_", "_"),
                    "condition":   cond,
                    "sense":       sense,
                    "carrier":     carrier,
                    "M_l_raw":     round(float(margin), 4),
                    "M_l_norm":    round(float(margin_norm), 4),
                    "adequate":    margin > epsilon,
                    "layer":       layer_idx,
                })
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            # Aggregate per condition
            for cond in ("L", "R"):
                cond_mask = [c == cond for c in conditions]
                cond_margins      = margins[cond_mask]
                cond_margins_norm = margins_norm[cond_mask]
                if len(cond_margins) == 0:
                    continue
                aggregate_rows.append({
                    "model":           safe_model,
                    "arch_type":       arch_type,
                    "word":            word,
                    "condition":       cond,
                    "n":               len(cond_margins),
                    "mean_M_l_raw":    round(float(cond_margins.mean()), 4),
                    "mean_M_l_norm":   round(float(cond_margins_norm.mean()), 4),
                    "frac_adequate":   round(float((cond_margins > epsilon).mean()), 3),
                    "layer_used":      layer_idx,
                })

            logger.info("[H3] %s / %s | layer=%d | L_mean=%.3f R_mean=%.3f",
                        model_name, word, layer_idx,
                        margins[[c == "L" for c in conditions]].mean() if "L" in conditions else float("nan"),
                        margins[[c == "R" for c in conditions]].mean() if "R" in conditions else float("nan"))

    if aggregate_rows:
        agg_path = OUTPUT_BASE / "h3_aggregate.csv"
        with open(agg_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=aggregate_rows[0].keys())
            writer.writeheader()
            writer.writerows(aggregate_rows)
        logger.info("[H3] Aggregate saved to %s", agg_path)

        compute_paired_context_tests(results_dir, paired_data_path)


def _bootstrap_mean_interval(values, rng, n_bootstrap: int = 20000):
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    return np.quantile(draws, [0.025, 0.975])


def compute_paired_context_tests(
    results_dir: str = RESULTS_DIR,
    paired_data_path: str = PAIRED_DATA_PATH,
) -> List[Dict]:
    """Estimate paired L-R effects without treating carriers as independent.

    Sentence pairs are first differenced, then averaged within word. Per-model
    intervals resample the eight word means. The architecture interaction uses
    a crossed model-by-word bootstrap. This is intentionally about L-R change;
    it does not test decoder R against 0.5 because that value is guaranteed by
    the mirrored stimulus/scoring construction.
    """
    output_dir = Path(results_dir) / "study" / "H3"
    aggregate_path = output_dir / "h3_aggregate.csv"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"{aggregate_path} not found — run H3 first")

    with open(paired_data_path) as handle:
        stimulus_data = json.load(handle)
    metadata = {
        item["id"]: {
            "sense": int(item["sense"]),
            "carrier": item["carrier"],
        }
        for items in stimulus_data.values()
        for item in items
    }
    arch_by_model = {}
    with open(aggregate_path) as handle:
        for row in csv.DictReader(handle):
            arch_by_model[row["model"]] = row["arch_type"]

    pair_rows = []
    mirrored_r_values: Dict[Tuple[str, str, str], List[Tuple[int, float]]] = {}
    for model_dir in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        model = model_dir.name
        for csv_path in sorted(model_dir.glob("h3_*.csv")):
            word = csv_path.stem.removeprefix("h3_")
            if word not in stimulus_data:
                logger.warning(
                    "[H3] Skipping stale output %s: %r not in current paired dataset",
                    csv_path, word,
                )
                continue
            by_pair: Dict[str, Dict[str, Dict]] = {}
            with open(csv_path) as handle:
                for row in csv.DictReader(handle):
                    sid = row["sentence_id"]
                    condition = row["condition"]
                    pair_id = row.get("pair_id") or sid.replace("_L_", "_").replace("_R_", "_")
                    meta = metadata[sid]
                    value = float(row["M_l_norm"])
                    adequate = str(row["adequate"]).lower() == "true"
                    by_pair.setdefault(pair_id, {})[condition] = {
                        "sid": sid,
                        "value": value,
                        "adequate": adequate,
                        **meta,
                    }
                    if condition == "R":
                        mirrored_r_values.setdefault(
                            (model, word, meta["carrier"]), []
                        ).append((meta["sense"], value))

            for pair_id, conditions in by_pair.items():
                if set(conditions) != {"L", "R"}:
                    continue
                left, right = conditions["L"], conditions["R"]
                pair_rows.append({
                    "model": model,
                    "arch_type": arch_by_model.get(model, ""),
                    "word": word,
                    "pair_id": pair_id,
                    "sense": left["sense"],
                    "carrier": left["carrier"],
                    "M_norm_L": left["value"],
                    "M_norm_R": right["value"],
                    "delta_M_norm_L_minus_R": left["value"] - right["value"],
                    "adequate_L": left["adequate"],
                    "adequate_R": right["adequate"],
                    "delta_adequate_L_minus_R": int(left["adequate"]) - int(right["adequate"]),
                })

    if not pair_rows:
        raise ValueError("No complete H3 L/R pairs found")
    pair_path = output_dir / "h3_pair_differences.csv"
    with open(pair_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_rows[0].keys())
        writer.writeheader()
        writer.writerows(pair_rows)

    rng = np.random.default_rng(20260714)
    model_summaries = []
    models = sorted({row["model"] for row in pair_rows})
    effect_by_model_word = {}
    for model in models:
        model_rows = [row for row in pair_rows if row["model"] == model]
        words = sorted({row["word"] for row in model_rows})
        word_margin_effects = []
        word_adequacy_effects = []
        for word in words:
            rows = [row for row in model_rows if row["word"] == word]
            margin_effect = float(np.mean([row["delta_M_norm_L_minus_R"] for row in rows]))
            adequacy_effect = float(np.mean([row["delta_adequate_L_minus_R"] for row in rows]))
            effect_by_model_word[(model, word)] = margin_effect
            word_margin_effects.append(margin_effect)
            word_adequacy_effects.append(adequacy_effect)
        margin_ci = _bootstrap_mean_interval(word_margin_effects, rng)
        adequacy_ci = _bootstrap_mean_interval(word_adequacy_effects, rng)

        mirror_errors = []
        mirror_signs_opposite = []
        for (group_model, _, _), values in mirrored_r_values.items():
            if group_model != model or {sense for sense, _ in values} != {0, 1}:
                continue
            by_sense = {sense: value for sense, value in values}
            mirror_errors.append(abs(by_sense[0] + by_sense[1]))
            mirror_signs_opposite.append(by_sense[0] * by_sense[1] <= 0.0)
        max_mirror_error = max(mirror_errors, default=float("nan"))
        is_decoder = arch_by_model.get(model) == "decoder"
        model_summaries.append({
            "model": model,
            "arch_type": arch_by_model.get(model, ""),
            "n_words": len(words),
            "n_sentence_pairs": len(model_rows),
            "mean_delta_M_norm_L_minus_R": round(float(np.mean(word_margin_effects)), 6),
            "ci95_low_delta_M_norm": round(float(margin_ci[0]), 6),
            "ci95_high_delta_M_norm": round(float(margin_ci[1]), 6),
            "mean_delta_fraction_adequate_L_minus_R": round(float(np.mean(word_adequacy_effects)), 6),
            "ci95_low_delta_fraction": round(float(adequacy_ci[0]), 6),
            "ci95_high_delta_fraction": round(float(adequacy_ci[1]), 6),
            "decoder_R_theoretically_deterministic": is_decoder,
            "decoder_R_observed_mirrored_signs_opposite": (
                bool(all(mirror_signs_opposite)) if is_decoder else "not_applicable"
            ),
            "max_abs_mirrored_R_margin_sum": (
                round(float(max_mirror_error), 6) if is_decoder else ""
            ),
        })

    summary_path = output_dir / "h3_paired_summary.csv"
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=model_summaries[0].keys())
        writer.writeheader()
        writer.writerows(model_summaries)

    encoder_models = [model for model in models if arch_by_model.get(model) == "encoder"]
    decoder_models = [model for model in models if arch_by_model.get(model) == "decoder"]
    common_words = sorted(set.intersection(*[
        {word for candidate, word in effect_by_model_word if candidate == model}
        for model in models
    ]))
    encoder_matrix = np.array([
        [effect_by_model_word[(model, word)] for word in common_words]
        for model in encoder_models
    ])
    decoder_matrix = np.array([
        [effect_by_model_word[(model, word)] for word in common_words]
        for model in decoder_models
    ])
    n_bootstrap = 20000
    interaction_draws = np.empty(n_bootstrap)
    for draw in range(n_bootstrap):
        word_idx = rng.integers(0, len(common_words), len(common_words))
        encoder_idx = rng.integers(0, len(encoder_models), len(encoder_models))
        decoder_idx = rng.integers(0, len(decoder_models), len(decoder_models))
        encoder_effect = encoder_matrix[encoder_idx][:, word_idx].mean()
        decoder_effect = decoder_matrix[decoder_idx][:, word_idx].mean()
        interaction_draws[draw] = decoder_effect - encoder_effect
    interaction_ci = np.quantile(interaction_draws, [0.025, 0.975])
    interaction_row = {
        "effect_definition": "(decoder L-R) - (encoder L-R)",
        "n_encoder_models": len(encoder_models),
        "n_decoder_models": len(decoder_models),
        "n_words": len(common_words),
        "encoder_mean_delta_M_norm_L_minus_R": round(float(encoder_matrix.mean()), 6),
        "decoder_mean_delta_M_norm_L_minus_R": round(float(decoder_matrix.mean()), 6),
        "architecture_interaction": round(float(decoder_matrix.mean() - encoder_matrix.mean()), 6),
        "ci95_low_interaction": round(float(interaction_ci[0]), 6),
        "ci95_high_interaction": round(float(interaction_ci[1]), 6),
        "bootstrap_unit": "crossed model-by-word",
    }
    interaction_path = output_dir / "h3_architecture_interaction.csv"
    with open(interaction_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=interaction_row.keys())
        writer.writeheader()
        writer.writerow(interaction_row)

    legacy_path = output_dir / "h3_chance_test.csv"
    if legacy_path.exists():
        legacy_path.unlink()
        logger.info("[H3] Removed invalid legacy chance-test output: %s", legacy_path)
    logger.info("[H3] Paired summaries saved under %s", output_dir)
    return model_summaries
