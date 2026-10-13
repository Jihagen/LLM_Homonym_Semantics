"""Extract the sentinel states needed for matched garden-path trajectories.

Model step (GPU, full requirements). For every model and homonym it reruns the
H5 fixed-sentinel readout at the layer H5 used and additionally reads the
coherent control sentence at all three stages (prime, homonym, resolver). The
released H5 tables only contain the control at the resolver stage.

    python -m scripts.extract_h5_matched_states --output-dir results/h5_matched

Three forward-pass groups per model and word, all read at the sentinel:

  profile   the 40 profiling sentences (defines the two sense centroids)
  h5        exactly the texts and order of hypotheses.h5_garden_path.run_h5
            (prime, homonym, resolution, resolver alone, control resolver)
  control   the control sentence cut at the same three stages

The second group repeats the published H5 inputs unchanged so that the new
states can be checked against data/processed/h5_sentence_level.csv.

Output: one ``<model>_<word>.npz`` per cell (float32 states, no pickles) under
the output directory. These are intermediate hidden states and are not part of
the release; scripts.build_geometry_exports turns them into display coordinates.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from hypotheses.h5_garden_path import (
    DEFAULT_SENTINEL,
    GP_DATA_PATH,
    PROFILING_DATA_PATH,
    _append_sentinel,
    _control_resolution_prefix,
    _load_profiling_examples,
    _sentinel_activations,
    build_incremental_prefixes,
)
from scripts.release_layout import PROCESSED_DIR, WORDS, read_table
from utils.model_registry import ALL_MODELS, MODEL_ALIASES

logger = logging.getLogger("extract_h5_matched_states")
STAGES = ("prime", "homonym", "resolution")


def control_item(item: dict) -> dict:
    """The matched control as an item of its own: same homonym, same
    resolution word, same stage cuts, different opening context."""
    return {
        "id": item["id"],
        "sentence": item["matched_control_sentence"],
        "resolution_word": item["resolution_word"],
    }


def h5_layers() -> dict:
    """Layer used by H5 for each (model key, word), from the released table."""
    table = read_table(PROCESSED_DIR / "h5_aggregate.csv")
    return {(row.model, row.word): int(row.layer_used) for row in table.itertuples()}


def extract_cell(model, tokenizer, word: str, layer: int, items: list, sentinel: str) -> dict:
    profile_sentences, profile_senses = _load_profiling_examples(PROFILING_DATA_PATH, word)
    profile_texts = [_append_sentinel(s, sentinel) for s in profile_sentences]
    profile_h = _sentinel_activations(model, tokenizer, profile_texts, word, [layer])[layer]

    # Group "h5": identical texts and order to run_h5.
    h5_texts, h5_item, h5_stage = [], [], []
    for i, item in enumerate(items):
        prefixes = build_incremental_prefixes(item, word, sentinel)
        for stage in STAGES:
            h5_texts.append(prefixes[stage]); h5_item.append(i); h5_stage.append(stage)
        h5_texts.append(_append_sentinel(item["resolution_word"], sentinel))
        h5_item.append(i); h5_stage.append("resolver_isolated")
        control = _control_resolution_prefix(item, word, sentinel)
        if control is not None:
            h5_texts.append(control); h5_item.append(i); h5_stage.append("matched_control")
    h5_h = _sentinel_activations(model, tokenizer, h5_texts, word, [layer])[layer]

    # Group "control": the control sentence at all three stages. Items whose
    # control cannot be cut at the homonym and resolution word are left out.
    c_texts, c_item, c_stage = [], [], []
    for i, item in enumerate(items):
        if not item.get("matched_control_sentence"):
            continue
        try:
            prefixes = build_incremental_prefixes(control_item(item), word, sentinel)
        except ValueError as exc:
            logger.warning("no three-stage control for %s: %s", item["id"], exc)
            continue
        for stage in STAGES:
            c_texts.append(prefixes[stage]); c_item.append(i); c_stage.append(stage)
    c_h = _sentinel_activations(model, tokenizer, c_texts, word, [layer])[layer]

    return {
        "layer": np.int64(layer),
        "sentinel": np.asarray(sentinel),
        "item_ids": np.asarray([item["id"] for item in items]),
        "profile_H": np.asarray(profile_h, dtype=np.float32),
        "profile_senses": np.asarray(profile_senses, dtype=np.int64),
        "h5_H": np.asarray(h5_h, dtype=np.float32),
        "h5_item_index": np.asarray(h5_item, dtype=np.int64),
        "h5_stage": np.asarray(h5_stage),
        "h5_text": np.asarray(h5_texts),
        "control_H": np.asarray(c_h, dtype=np.float32),
        "control_item_index": np.asarray(c_item, dtype=np.int64),
        "control_stage": np.asarray(c_stage),
        "control_text": np.asarray(c_texts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="results/h5_matched")
    parser.add_argument("--models", nargs="*", default=None, help="names or aliases (default: all 8)")
    parser.add_argument("--words", nargs="*", default=WORDS)
    parser.add_argument("--force", action="store_true", help="recompute cells that already exist")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    from models import load_model_and_tokenizer
    from utils.hpc import cleanup_torch

    models = [MODEL_ALIASES.get(m, m) for m in args.models] if args.models else ALL_MODELS
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    layers = h5_layers()
    with open(GP_DATA_PATH, encoding="utf-8") as handle:
        gp_data = json.load(handle)

    for model_name in models:
        key = model_name.replace("/", "_")
        todo = [w for w in args.words if args.force or not (out / f"{key}_{w}.npz").exists()]
        if not todo:
            logger.info("%s: all cells present", key)
            continue
        model, tokenizer = load_model_and_tokenizer(model_name)
        for word in todo:
            cell = extract_cell(model, tokenizer, word, layers[(key, word)], gp_data[word], DEFAULT_SENTINEL)
            tmp = out / f".{key}_{word}.tmp.npz"
            np.savez_compressed(tmp, **cell)
            tmp.replace(out / f"{key}_{word}.npz")
            logger.info("%s/%s layer %d: profile %s, h5 %s, control %s", key, word, int(cell["layer"]),
                        cell["profile_H"].shape, cell["h5_H"].shape, cell["control_H"].shape)
        del model, tokenizer
        cleanup_torch()


if __name__ == "__main__":
    main()
