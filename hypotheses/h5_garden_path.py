"""H5 — Incremental garden-path updating at a fixed sentinel readout.

The old H5 connected the homonym state to the resolution-word state from one
full-sentence pass. That comparison changes token identity and sequence
position, so it cannot demonstrate revision; a resolver such as ``wings`` may
look animal-like simply because it is ``wings``.

The corrected design separates incremental reading time from layer depth and
sequence position. For every item, the model is rerun on three progressively
longer prefixes and the same neutral sentinel word is appended each time:

1. prime context + sentinel;
2. context through the homonym + sentinel;
3. context through the resolution word + sentinel.

The sentinel identity is constant. Its state is scored against sentinel-
position sense centroids learned from independent, ordinary disambiguated
profiling sentences. The pre-homonym stage measures context bias; commitment
is assessed after the homonym. The primary outcome is the paired continuous
resolution-minus-homonym change in correct-sense margin. A conditional sign
transition is reported only among items whose homonym-stage probe first aligns
with the primed sense.

This remains exploratory until the stimulus audit passes: both direction
orders for every homonym, a matched non-conflicting control using the same
resolver, and valid prime→homonym→resolver structure. No external human norms
will be collected, so the stimuli and results are explicitly described as a
model-internal context-conflict design rather than human-validated garden
paths. By default an incomplete design is audited and then blocked; callers
must opt in explicitly to run incomplete exploratory stimuli.
"""

import csv
import json
import logging
import math
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from experiments.adequacy import normalized_adequacy_margin
from hypotheses.h3_context_position import H3_MODELS, _select_layer
from models import (
    find_target_span,
    get_dual_position_activations,
    is_decoder_only,
    load_model_and_tokenizer,
)
from utils.hpc import configure_hpc_runtime

configure_hpc_runtime()
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
GP_DATA_PATH = "data/garden_path_sentences.json"
PROFILING_DATA_PATH = "data/synthetic_data_h2.pkl"
OUTPUT_BASE = Path("results/study/H5")
DEFAULT_WORDS = ["bank", "bark", "bat", "crane", "spring", "match", "pitch"]
H5_EXCLUSIONS = {
    "light": (
        "The study senses are different parts of speech (illumination noun vs "
        "low-weight adjective), so syntactic-category disambiguation cannot be "
        "separated from semantic garden-path resolution."
    )
}
DEFAULT_SENTINEL = "probe"
SENSITIVITY_THRESHOLDS = (0.0, 0.1, 0.3)


def _find_word_span_after(text: str, target: str, start: int) -> Optional[Tuple[int, int]]:
    pattern = r"(?<![a-zA-Z])" + re.escape(target.lower()) + r"(e?s)?(?![a-zA-Z])"
    match = re.search(pattern, text.lower()[start:])
    if match is None:
        return None
    return start + match.start(), start + match.end()


def _append_sentinel(prefix: str, sentinel: str = DEFAULT_SENTINEL) -> str:
    """Put the fixed readout in its own paragraph at every reading stage.

    The paragraph break prevents the sentinel from filling different local
    syntactic slots (for example, turning ``walked to the`` into ``walked to
    the probe`` at the prime stage). The prefix itself is not repaired or
    completed: it remains exactly the context causally available at that
    reading point.
    """
    prefix = prefix.strip()
    if not prefix:
        raise ValueError("Cannot append the sentinel to an empty prefix")
    return f"{prefix}\n\n{sentinel}"


def build_incremental_prefixes(
    item: Dict, word: str, sentinel: str = DEFAULT_SENTINEL
) -> Dict[str, str]:
    """Create prime, homonym, and resolution prefix reruns for one item."""
    sentence = item["sentence"]
    target_span = find_target_span(sentence, word)
    if target_span is None:
        raise ValueError(f"Target '{word}' absent from {item.get('id', sentence)!r}")
    resolution_span = _find_word_span_after(
        sentence, item["resolution_word"], target_span[1]
    )
    if resolution_span is None:
        raise ValueError(
            f"Resolution word '{item['resolution_word']}' is absent after '{word}' "
            f"in {item.get('id', sentence)!r}"
        )

    prime = sentence[: target_span[0]].rstrip(" ,;:-")
    through_homonym = sentence[: target_span[1]]
    through_resolution = sentence[: resolution_span[1]]
    return {
        "prime": _append_sentinel(prime, sentinel),
        "homonym": _append_sentinel(through_homonym, sentinel),
        "resolution": _append_sentinel(through_resolution, sentinel),
    }


def _control_resolution_prefix(
    item: Dict, word: str, sentinel: str = DEFAULT_SENTINEL
) -> Optional[str]:
    control = item.get("matched_control_sentence")
    if not control:
        return None
    target_span = find_target_span(control, word)
    if target_span is None:
        return None
    resolution_span = _find_word_span_after(
        control, item["resolution_word"], target_span[1]
    )
    if resolution_span is None:
        return None
    return _append_sentinel(control[: resolution_span[1]], sentinel)


def audit_h5_design(
    data: Dict[str, List[Dict]], words: Optional[Sequence[str]] = None
) -> Tuple[List[Dict], bool]:
    """Audit stimulus prerequisites without loading a model or using a GPU."""
    words = list(words or DEFAULT_WORDS)
    rows = []
    for word in words:
        items = data.get(word, [])
        if word in H5_EXCLUSIONS:
            rows.append({
                "word": word,
                "eligible_for_h5": False,
                "exclusion_reason": H5_EXCLUSIONS[word],
                "n_items": len(items),
                "n_primed0_to_correct1": 0,
                "n_primed1_to_correct0": 0,
                "both_directions_present": False,
                "directions_balanced": False,
                "n_matched_controls": 0,
                "matched_controls_complete": False,
                "n_human_normed": 0,
                "human_norms_complete": False,
                "human_validation_status": "not_collected_not_required",
                "n_structurally_invalid": 0,
                "model_internal_ready": False,
            })
            continue
        n_01 = sum(
            item.get("primed_sense") == 0 and item.get("correct_sense") == 1
            for item in items
        )
        n_10 = sum(
            item.get("primed_sense") == 1 and item.get("correct_sense") == 0
            for item in items
        )
        n_controls = 0
        n_human_norms = 0
        n_invalid = 0
        for item in items:
            try:
                build_incremental_prefixes(item, word)
            except (KeyError, TypeError, ValueError):
                n_invalid += 1
            if _control_resolution_prefix(item, word) is not None:
                n_controls += 1
            if (
                isinstance(item.get("human_prime_strength"), (int, float))
                and isinstance(item.get("human_resolution_clarity"), (int, float))
            ):
                n_human_norms += 1

        both_directions = n_01 > 0 and n_10 > 0
        balanced_directions = both_directions and n_01 == n_10
        controls_complete = bool(items) and n_controls == len(items)
        norms_complete = bool(items) and n_human_norms == len(items)
        structurally_valid = bool(items) and n_invalid == 0
        # H5 is a model-internal context-conflict experiment. Human norms would
        # be needed for a claim about human garden-path processing, but are not
        # a prerequisite for the model-only estimand used here.
        ready = balanced_directions and controls_complete and structurally_valid
        if norms_complete:
            human_status = "complete_optional"
        elif n_human_norms:
            human_status = "partial_optional"
        else:
            human_status = "not_collected_not_required"
        rows.append({
            "word": word,
            "eligible_for_h5": True,
            "exclusion_reason": "",
            "n_items": len(items),
            "n_primed0_to_correct1": n_01,
            "n_primed1_to_correct0": n_10,
            "both_directions_present": both_directions,
            "directions_balanced": balanced_directions,
            "n_matched_controls": n_controls,
            "matched_controls_complete": controls_complete,
            "n_human_normed": n_human_norms,
            "human_norms_complete": norms_complete,
            "human_validation_status": human_status,
            "n_structurally_invalid": n_invalid,
            "model_internal_ready": ready,
        })
    eligible_rows = [row for row in rows if row["eligible_for_h5"]]
    return rows, bool(eligible_rows) and all(
        row["model_internal_ready"] for row in eligible_rows
    )


def write_h5_design_audit(
    rows: Sequence[Dict], output_base: Path = OUTPUT_BASE
) -> None:
    output_base.mkdir(parents=True, exist_ok=True)
    csv_path = output_base / "h5_design_audit.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_base / "h5_design_audit.md"
    with open(md_path, "w") as f:
        f.write("# H5 stimulus design audit\n\n")
        f.write(
            "H5 is ready for the model-internal context-conflict analysis only when "
            "every requested homonym has both "
            "priming directions, a matched non-garden-path control with the same "
            "resolver for every item, and valid prime→homonym→resolver order. "
            "No external human validation is claimed or required.\n\n"
        )
        f.write("| word | n | 0→1 | 1→0 | balanced | controls | norms | invalid | ready |\n")
        f.write("|---|---:|---:|---:|:---:|---:|---:|---:|:---:|\n")
        for row in rows:
            if not row["eligible_for_h5"]:
                status = "excluded"
            else:
                status = "yes" if row["model_internal_ready"] else "no"
            f.write(
                f"| {row['word']} | {row['n_items']} | "
                f"{row['n_primed0_to_correct1']} | {row['n_primed1_to_correct0']} | "
                f"{'yes' if row['directions_balanced'] else 'no'} | "
                f"{row['n_matched_controls']} | {row['n_human_normed']} | "
                f"{row['n_structurally_invalid']} | "
                f"{status} |\n"
            )
        excluded = [row for row in rows if not row["eligible_for_h5"]]
        if excluded:
            f.write("\n## Documented exclusions\n\n")
            for row in excluded:
                f.write(f"- `{row['word']}`: {row['exclusion_reason']}\n")


def run_design_audit(
    gp_data_path: str = GP_DATA_PATH,
    words: Optional[Sequence[str]] = None,
) -> bool:
    with open(gp_data_path) as f:
        data = json.load(f)
    rows, ready = audit_h5_design(data, words)
    write_h5_design_audit(rows)
    return ready


def _load_profiling_examples(path: str, word: str) -> Tuple[List[str], np.ndarray]:
    """Load ordinary disambiguated examples; pandas is imported only when needed."""
    import pandas as pd

    df = pd.read_pickle(path)
    rows = df[df["word"] == word]
    sentences: List[str] = []
    senses: List[int] = []
    for _, row in rows.iterrows():
        sense = int(row["semantic_group_id"])
        for sentence in row["examples"]:
            sentences.append(str(sentence))
            senses.append(sense)
    if set(senses) != {0, 1}:
        raise ValueError(f"Profiling data for '{word}' must contain senses 0 and 1")
    return sentences, np.asarray(senses)


def _sentinel_activations(
    model,
    tokenizer,
    texts: Sequence[str],
    word: str,
    layer_indices: Sequence[int],
    batch_size: int = 4,
) -> Dict[int, np.ndarray]:
    """Extract the last-token state at every requested layer, in one forward
    pass; every input must end in the same sentinel. Returns {layer: array},
    so scoring several candidate layers for the same word only costs one
    forward pass, not one per candidate."""
    _, final_acts = get_dual_position_activations(
        model,
        tokenizer,
        list(texts),
        [word] * len(texts),
        batch_size=batch_size,
        layer_indices=list(layer_indices),
    )
    return {layer: final_acts[layer].numpy() for layer in layer_indices}


def _raw_margin(h: np.ndarray, c_correct: np.ndarray, c_primed: np.ndarray) -> float:
    return float(np.linalg.norm(h - c_primed) - np.linalg.norm(h - c_correct))


def _score_margin(
    h: np.ndarray, c_correct: np.ndarray, c_primed: np.ndarray
) -> Tuple[float, float]:
    return (
        _raw_margin(h, c_correct, c_primed),
        float(normalized_adequacy_margin(h, c_correct, c_primed)),
    )


def _token_count(tokenizer, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False)
    return len(encoded["input_ids"])


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else math.nan


def _sensitivity_label(margin: float, threshold: float) -> str:
    if margin > threshold:
        return "correct"
    if margin < -threshold:
        return "primed"
    return "boundary"


def run_h5(
    model_names: Optional[List[str]] = None,
    words: Optional[List[str]] = None,
    results_dir: str = RESULTS_DIR,
    gp_data_path: str = GP_DATA_PATH,
    profiling_data_path: str = PROFILING_DATA_PATH,
    epsilon: float = 0.0,
    sentinel: str = DEFAULT_SENTINEL,
    allow_incomplete_design: bool = False,
    output_base: Path = OUTPUT_BASE,
    layer_lookup: Optional[Callable[[str, str, str], int]] = None,
    layer_candidates_lookup: Optional[Callable[[str, str, str], List[int]]] = None,
) -> None:
    """Run fixed-sentinel prefix reruns; block incomplete designs by default.

    layer_lookup, if given, replaces _select_layer (the H1-homonym-position
    best layer) as the (model_name, results_dir, word) -> layer_idx callable
    used for every readout (prime/homonym/resolution/matched-control/
    resolver-only). output_base lets an alternate run write to a separate
    directory so the original H1-based H5 results are never overwritten.

    layer_candidates_lookup, if given, takes priority over layer_lookup: it
    maps (model_name, results_dir, word) -> a *list* of candidate layers, and
    H5 is run once per candidate for that word (all sharing a single forward
    pass per text set, not one pass per candidate). Every output row is
    tagged with the layer actually used ("layer"/"layer_used" columns), and
    in this mode each word's per-item CSV is suffixed with the layer
    (h5_{word}_L{layer}.csv) so candidates never overwrite one another. When
    layer_candidates_lookup is not given (the default), behaviour is
    unchanged: exactly one layer per word, original h5_{word}.csv filename.
    """
    model_names = model_names or H3_MODELS
    words = words or DEFAULT_WORDS
    select_layer = layer_lookup or _select_layer
    multi_layer_mode = layer_candidates_lookup is not None
    output_base.mkdir(parents=True, exist_ok=True)

    with open(gp_data_path) as f:
        gp_data = json.load(f)
    # Documented exclusions (e.g. "light") must appear in the audit for
    # transparency even when the caller's word list omits them; the model
    # loop below still skips any word in H5_EXCLUSIONS regardless.
    audit_words = list(words) + [w for w in H5_EXCLUSIONS if w not in words]
    audit_rows, model_internal_ready = audit_h5_design(gp_data, audit_words)
    write_h5_design_audit(audit_rows, output_base=output_base)
    if not model_internal_ready and not allow_incomplete_design:
        logger.error(
            "[H5] Design audit failed. No model was loaded. See %s. "
            "Add both directions and matched controls, and fix structural errors, or pass "
            "allow_incomplete_design=True for explicitly exploratory output.",
            output_base / "h5_design_audit.md",
        )
        return

    aggregate_rows: List[Dict] = []
    all_sentence_rows: List[Dict] = []

    for model_name in model_names:
        safe_model = model_name.replace("/", "_")
        model_out = output_base / safe_model
        model_out.mkdir(parents=True, exist_ok=True)
        model, tokenizer = load_model_and_tokenizer(model_name)
        arch_type = "decoder" if is_decoder_only(model) else "encoder"

        for word in words:
            if word in H5_EXCLUSIONS:
                logger.info("[H5] Excluding %s: %s", word, H5_EXCLUSIONS[word])
                continue
            items = gp_data.get(word, [])
            if not items:
                logger.warning("[H5] No items for '%s'", word)
                continue
            try:
                if multi_layer_mode:
                    candidate_layers = layer_candidates_lookup(model_name, results_dir, word)
                else:
                    candidate_layers = [select_layer(model_name, results_dir, word)]
                if not candidate_layers:
                    logger.warning("[H5] No candidate layers for %s/%s", model_name, word)
                    continue
                profile_sentences, profile_senses = _load_profiling_examples(
                    profiling_data_path, word
                )
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("[H5] %s", exc)
                continue

            # One forward pass per text set covers every candidate layer for
            # this word (layer_indices=candidate_layers), rather than one
            # forward pass per candidate.
            profile_texts = [_append_sentinel(s, sentinel) for s in profile_sentences]
            profile_h_by_layer = _sentinel_activations(
                model, tokenizer, profile_texts, word, candidate_layers
            )

            extraction_texts: List[str] = []
            extraction_keys: List[Tuple[int, str]] = []
            prefixes_by_item: List[Dict[str, str]] = []
            for i, item in enumerate(items):
                prefixes = build_incremental_prefixes(item, word, sentinel)
                prefixes_by_item.append(prefixes)
                for stage in ("prime", "homonym", "resolution"):
                    extraction_texts.append(prefixes[stage])
                    extraction_keys.append((i, stage))
                isolated = _append_sentinel(item["resolution_word"], sentinel)
                extraction_texts.append(isolated)
                extraction_keys.append((i, "resolver_isolated"))
                control = _control_resolution_prefix(item, word, sentinel)
                if control is not None:
                    extraction_texts.append(control)
                    extraction_keys.append((i, "matched_control"))

            h_all_by_layer = _sentinel_activations(
                model, tokenizer, extraction_texts, word, candidate_layers
            )

            for layer_idx in candidate_layers:
                profile_h = profile_h_by_layer[layer_idx]
                sentinel_centroids = {
                    sense: profile_h[profile_senses == sense].mean(axis=0)
                    for sense in (0, 1)
                }
                h_by_key = {
                    key: h for key, h in zip(extraction_keys, h_all_by_layer[layer_idx])
                }

                csv_rows: List[Dict] = []
                for i, item in enumerate(items):
                    correct = int(item["correct_sense"])
                    primed = int(item["primed_sense"])
                    c_correct = sentinel_centroids[correct]
                    c_primed = sentinel_centroids[primed]
                    scores = {
                        stage: _score_margin(h_by_key[(i, stage)], c_correct, c_primed)
                        for stage in ("prime", "homonym", "resolution", "resolver_isolated")
                    }
                    control_score = (
                        _score_margin(h_by_key[(i, "matched_control")], c_correct, c_primed)
                        if (i, "matched_control") in h_by_key
                        else (math.nan, math.nan)
                    )
                    prime_raw, prime_norm = scores["prime"]
                    homonym_raw, homonym_norm = scores["homonym"]
                    resolution_raw, resolution_norm = scores["resolution"]
                    isolated_raw, isolated_norm = scores["resolver_isolated"]
                    control_raw, control_norm = control_score
                    # Priming context alone is a bias baseline. A garden-path
                    # commitment is defined only after the ambiguous word has
                    # actually been encountered.
                    primed_at_homonym = homonym_raw < epsilon
                    resolved_correct = resolution_raw > epsilon

                    row = {
                        "model": safe_model,
                        "arch_type": arch_type,
                        "word": word,
                        "sentence_id": item.get("id", f"{word}_gp_{i}"),
                        "sentence": item["sentence"],
                        "primed_sense": primed,
                        "correct_sense": correct,
                        "direction": f"{primed}_to_{correct}",
                        "resolution_word": item["resolution_word"],
                        "sentinel": sentinel,
                        "prime_prefix": prefixes_by_item[i]["prime"],
                        "homonym_prefix": prefixes_by_item[i]["homonym"],
                        "resolution_prefix": prefixes_by_item[i]["resolution"],
                        "prime_token_count": _token_count(tokenizer, prefixes_by_item[i]["prime"]),
                        "homonym_token_count": _token_count(tokenizer, prefixes_by_item[i]["homonym"]),
                        "resolution_token_count": _token_count(tokenizer, prefixes_by_item[i]["resolution"]),
                        "prime_correct_margin_raw": round(prime_raw, 4),
                        "homonym_correct_margin_raw": round(homonym_raw, 4),
                        "resolution_correct_margin_raw": round(resolution_raw, 4),
                        "resolver_isolated_correct_margin_raw": round(isolated_raw, 4),
                        "matched_control_correct_margin_raw": (
                            round(control_raw, 4) if math.isfinite(control_raw) else math.nan
                        ),
                        "prime_correct_margin_norm": round(prime_norm, 4),
                        "homonym_correct_margin_norm": round(homonym_norm, 4),
                        "resolution_correct_margin_norm": round(resolution_norm, 4),
                        "resolver_isolated_correct_margin_norm": round(isolated_norm, 4),
                        "matched_control_correct_margin_norm": (
                            round(control_norm, 4) if math.isfinite(control_norm) else math.nan
                        ),
                        "delta_homonym_minus_prime_norm": round(homonym_norm - prime_norm, 4),
                        "delta_resolution_minus_homonym_norm": round(resolution_norm - homonym_norm, 4),
                        "delta_resolution_minus_prime_norm": round(resolution_norm - prime_norm, 4),
                        "delta_resolution_minus_isolated_resolver_norm": round(
                            resolution_norm - isolated_norm, 4
                        ),
                        "garden_path_cost_vs_matched_control_norm": (
                            round(resolution_norm - control_norm, 4)
                            if math.isfinite(control_norm) else math.nan
                        ),
                        "primed_at_homonym": primed_at_homonym,
                        "resolved_correct": resolved_correct,
                        "successful_primed_to_correct_transition": (
                            primed_at_homonym and resolved_correct
                        ),
                        "human_prime_strength": item.get("human_prime_strength", math.nan),
                        "human_resolution_clarity": item.get("human_resolution_clarity", math.nan),
                        "layer": layer_idx,
                        "analysis_status": (
                            "model_internal_ready" if model_internal_ready else "exploratory_incomplete_design"
                        ),
                    }
                    for threshold in SENSITIVITY_THRESHOLDS:
                        suffix = str(threshold).replace(".", "p")
                        row[f"prime_state_thr_{suffix}"] = _sensitivity_label(prime_norm, threshold)
                        row[f"homonym_state_thr_{suffix}"] = _sensitivity_label(
                            homonym_norm, threshold
                        )
                        row[f"resolution_state_thr_{suffix}"] = _sensitivity_label(
                            resolution_norm, threshold
                        )
                    csv_rows.append(row)
                    all_sentence_rows.append(row)

                word_filename = (
                    f"h5_{word}_L{layer_idx}.csv" if multi_layer_mode else f"h5_{word}.csv"
                )
                with open(model_out / word_filename, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(csv_rows)

                primed_homonym_rows = [row for row in csv_rows if row["primed_at_homonym"]]
                transitions = sum(
                    row["successful_primed_to_correct_transition"]
                    for row in primed_homonym_rows
                )
                aggregate_rows.append({
                    "model": safe_model,
                    "arch_type": arch_type,
                    "word": word,
                    "n_items": len(csv_rows),
                    "n_primed_at_homonym": len(primed_homonym_rows),
                    "n_successful_primed_to_correct": transitions,
                    "p_resolved_correct_given_primed_at_homonym": (
                        round(transitions / len(primed_homonym_rows), 3)
                        if primed_homonym_rows else math.nan
                    ),
                    "mean_prime_correct_margin_norm": round(
                        _mean(row["prime_correct_margin_norm"] for row in csv_rows), 4
                    ),
                    "mean_homonym_correct_margin_norm": round(
                        _mean(row["homonym_correct_margin_norm"] for row in csv_rows), 4
                    ),
                    "mean_resolution_correct_margin_norm": round(
                        _mean(row["resolution_correct_margin_norm"] for row in csv_rows), 4
                    ),
                    "mean_delta_resolution_minus_homonym_norm": round(
                        _mean(row["delta_resolution_minus_homonym_norm"] for row in csv_rows), 4
                    ),
                    "mean_delta_resolution_minus_prime_norm": round(
                        _mean(row["delta_resolution_minus_prime_norm"] for row in csv_rows), 4
                    ),
                    "mean_delta_resolution_minus_isolated_resolver_norm": round(
                        _mean(
                            row["delta_resolution_minus_isolated_resolver_norm"]
                            for row in csv_rows
                        ), 4
                    ),
                    "mean_garden_path_cost_vs_matched_control_norm": round(
                        _mean(
                            row["garden_path_cost_vs_matched_control_norm"]
                            for row in csv_rows
                            if math.isfinite(row["garden_path_cost_vs_matched_control_norm"])
                        ), 4
                    ),
                    "layer_used": layer_idx,
                    "sentinel": sentinel,
                    "analysis_status": (
                        "model_internal_ready" if model_internal_ready else "exploratory_incomplete_design"
                    ),
                })
                logger.info(
                    "[H5] %s/%s | layer=%s | resolved after primed homonym=%s/%s | status=%s",
                    model_name, word, layer_idx, transitions, len(primed_homonym_rows),
                    "model-internal-ready" if model_internal_ready else "exploratory-incomplete",
                )

    if aggregate_rows:
        with open(output_base / "h5_aggregate.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=aggregate_rows[0].keys())
            writer.writeheader()
            writer.writerows(aggregate_rows)
    if all_sentence_rows:
        with open(output_base / "h5_sentence_level.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_sentence_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_sentence_rows)
