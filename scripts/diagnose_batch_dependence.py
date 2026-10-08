"""Diagnose why a sentinel state depends on the batch it is read in.

Model step (GPU). For each model it reads the same texts under controlled
changes of one factor at a time and reports how far the states and the
normalised margins move:

  repeat        identical batches, run twice           -> GPU nondeterminism
  composition   same texts, different batch neighbours -> batch composition / padding
  unbatched     one text per forward pass (no padding) -> padding as such
  float32       all of the above with float32 weights and no autocast -> dtype

    python -m scripts.diagnose_batch_dependence --output results/batch_dependence.json

The texts are the coherent-control resolver prefixes of one homonym, read at
the layer H5 uses, exactly as in scripts.extract_h5_matched_states.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
from pathlib import Path

import numpy as np

from experiments.adequacy import normalized_adequacy_margin
from hypotheses.h5_garden_path import (
    DEFAULT_SENTINEL, GP_DATA_PATH, PROFILING_DATA_PATH, _append_sentinel,
    _control_resolution_prefix, _load_profiling_examples, _sentinel_activations,
)
from scripts.extract_h5_matched_states import h5_layers
from utils.model_registry import ALL_MODELS, MODEL_ALIASES

logger = logging.getLogger("diagnose_batch_dependence")


def read(model, tokenizer, texts, word, layer, order=None, batch_size=4):
    order = list(range(len(texts))) if order is None else list(order)
    states = _sentinel_activations(model, tokenizer, [texts[i] for i in order], word, [layer], batch_size=batch_size)[layer]
    out = np.empty_like(states)
    out[order] = states
    return out.astype(np.float64)


def pad_counts(tokenizer, texts, order, batch_size=4):
    counts = np.zeros(len(texts), dtype=int)
    for start in range(0, len(order), batch_size):
        index = order[start:start + batch_size]
        mask = tokenizer([texts[i] for i in index], padding=True, return_tensors="np")["attention_mask"]
        counts[index] = mask.shape[1] - mask.sum(axis=1)
    return counts


def compare(a, b, margins_of):
    relative = np.linalg.norm(a - b, axis=1) / np.linalg.norm(a, axis=1)
    delta = np.abs(margins_of(a) - margins_of(b))
    return {
        "max_relative_state_difference": float(relative.max()),
        "mean_relative_state_difference": float(relative.mean()),
        "max_abs_margin_difference": float(delta.max()),
        "mean_abs_margin_difference": float(delta.mean()),
    }


def run_conditions(model, tokenizer, word, layer, items, label):
    import torch

    profile_sentences, senses = _load_profiling_examples(PROFILING_DATA_PATH, word)
    profile = read(model, tokenizer, [_append_sentinel(s, DEFAULT_SENTINEL) for s in profile_sentences], word, layer)
    centroids = {s: profile[senses == s].mean(axis=0) for s in (0, 1)}
    texts = [_control_resolution_prefix(item, word, DEFAULT_SENTINEL) for item in items]
    correct = [int(item["correct_sense"]) for item in items]

    def margins_of(states):
        return np.array([normalized_adequacy_margin(h, centroids[c], centroids[1 - c]) for h, c in zip(states, correct)])

    base_order = list(range(len(texts)))
    other_order = list(np.random.default_rng(0).permutation(len(texts)))
    base = read(model, tokenizer, texts, word, layer, base_order)
    again = read(model, tokenizer, texts, word, layer, base_order)
    other = read(model, tokenizer, texts, word, layer, other_order)
    single = read(model, tokenizer, texts, word, layer, base_order, batch_size=1)
    pads = pad_counts(tokenizer, texts, base_order)
    unpadded = pads == 0
    result = {
        "precision": label,
        "parameter_dtype": str(next(model.parameters()).dtype),
        "model_in_training_mode": bool(model.training),
        "tokenizer_padding_side": tokenizer.padding_side,
        "attention_mask_passed": True,
        "n_texts": len(texts),
        "n_texts_padded_in_reference_batches": int((~unpadded).sum()),
        "repeat_identical_batches": compare(base, again, margins_of),
        "different_batch_composition": compare(base, other, margins_of),
        "unbatched_vs_batched": compare(base, single, margins_of),
        "unbatched_vs_batched_rows_without_padding": compare(base[unpadded], single[unpadded],
                                                             lambda s: np.zeros(len(s))) if unpadded.any() else None,
        "unbatched_vs_batched_rows_with_padding": compare(base[~unpadded], single[~unpadded],
                                                          lambda s: np.zeros(len(s))) if (~unpadded).any() else None,
        "unbatched_repeat": compare(single, read(model, tokenizer, texts, word, layer, base_order, batch_size=1), margins_of),
    }
    del torch
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="results/batch_dependence.json")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--word", default="bank")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    import torch
    import models.models as model_module
    from models import load_model_and_tokenizer
    from utils.hpc import cleanup_torch

    names = [MODEL_ALIASES.get(m, m) for m in args.models] if args.models else ALL_MODELS
    layers = h5_layers()
    with open(GP_DATA_PATH, encoding="utf-8") as handle:
        items = json.load(handle)[args.word]
    original_context = model_module._inference_context

    @contextlib.contextmanager
    def float32_context(device):
        with torch.inference_mode():
            yield

    report = {"word": args.word, "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "models": {}}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for name in names:
        key = name.replace("/", "_")
        layer = layers[(key, args.word)]
        model, tokenizer = load_model_and_tokenizer(name)
        entry = {"layer": layer, "pipeline_precision": run_conditions(model, tokenizer, args.word, layer, items, "pipeline (bfloat16 weights, bfloat16 autocast)")}
        model.float()
        model_module._inference_context = float32_context
        try:
            entry["float32"] = run_conditions(model, tokenizer, args.word, layer, items, "float32 weights, no autocast")
        finally:
            model_module._inference_context = original_context
        report["models"][key] = entry
        logger.info("%s: %s", key, json.dumps({p: entry[p]["different_batch_composition"]["max_abs_margin_difference"]
                                               for p in ("pipeline_precision", "float32")}))
        out.write_text(json.dumps(report, indent=2) + "\n")
        del model, tokenizer
        cleanup_torch()


if __name__ == "__main__":
    main()
