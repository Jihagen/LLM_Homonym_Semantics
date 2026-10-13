"""Write the flat, documented stimulus tables in ``data/stimuli/``.

    python -m scripts.build_stimulus_tables

The pipeline reads three hand-curated source files:

    data/profiling_sentences.json     sense-labelled profiling sentences
    data/paired_sentences.json        ambiguous carriers with a resolving clause
                                      before (L) or after (R) the homonym
    data/garden_path_sentences.json   context-conflict items and matched controls

This script re-expresses them as one CSV per stimulus type, adding only
identifiers and the incremental prefixes that the H5 code derives from each
item (via the same functions the pipeline uses). No sentence is edited.
Field definitions are in data/README.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data import load_profiling_dataframe
from hypotheses.h5_garden_path import (
    DEFAULT_SENTINEL,
    _append_sentinel,
    _control_resolution_prefix,
    build_incremental_prefixes,
)
from scripts.release_layout import ROOT, WORDS, write_table

STIMULI_DIR = ROOT / "data" / "stimuli"
SHORT_LABELS = {
    entry["word"]: {sense["sense"]: sense["short_label"] for sense in entry["senses"]}
    for entry in json.loads((ROOT / "configs" / "study.json").read_text(encoding="utf-8"))["words"]
}


def homonyms_and_profiling():
    frame = load_profiling_dataframe(str(ROOT / "data" / "profiling_sentences.json"))
    homonyms, sentences = [], []
    for word in WORDS:
        rows = frame[frame["word"] == word].sort_values("semantic_group_id")
        index = 0
        for row in rows.itertuples():
            sense = int(row.semantic_group_id)
            homonyms.append({
                "word": word,
                "sense": sense,
                "sense_label": row.sense_label,
                "short_label": SHORT_LABELS[word][sense],
                "n_profiling_sentences": len(row.examples),
            })
            for position, sentence in enumerate(row.examples):
                sentences.append({
                    "sentence_id": f"{word}_profile_s{sense}_{position + 1:02d}",
                    "word": word,
                    "sense": sense,
                    "sense_label": row.sense_label,
                    "profile_index": index,
                    "sentence": str(sentence),
                })
                index += 1
    return pd.DataFrame(homonyms), pd.DataFrame(sentences)


def paired_and_carriers():
    with open(ROOT / "data" / "paired_sentences.json", encoding="utf-8") as handle:
        data = json.load(handle)
    paired, carriers = [], []
    for word in WORDS:
        seen = {}
        for item in data[word]:
            carrier = item["carrier"]
            if carrier not in seen:
                seen[carrier] = f"{word}_carrier_{len(seen) + 1:02d}"
                carriers.append({
                    "carrier_id": seen[carrier],
                    "word": word,
                    "condition": "ambiguous_carrier",
                    "text": carrier,
                })
            paired.append({
                "sentence_id": item["id"],
                "word": word,
                "pair_id": item["id"].replace("_L_", "_").replace("_R_", "_"),
                "condition": item["condition"],
                "resolver_position": "before_homonym" if item["condition"] == "L" else "after_homonym",
                "sense": int(item["sense"]),
                "sense_label_short": SHORT_LABELS[word][int(item["sense"])],
                "carrier_id": seen[carrier],
                "carrier": carrier,
                "sentence": item["sentence"],
            })
        carriers.append({
            "carrier_id": f"{word}_bare",
            "word": word,
            "condition": "bare_word",
            "text": word,
        })
    return pd.DataFrame(paired), pd.DataFrame(carriers)


def conflict_items():
    with open(ROOT / "data" / "garden_path_sentences.json", encoding="utf-8") as handle:
        data = json.load(handle)
    rows = []
    for word in WORDS:
        for item in data[word]:
            prefixes = build_incremental_prefixes(item, word, DEFAULT_SENTINEL)
            primed, correct = int(item["primed_sense"]), int(item["correct_sense"])
            rows.append({
                "item_id": item["id"],
                "word": word,
                "primed_sense": primed,
                "correct_sense": correct,
                "direction": f"{primed}_to_{correct}",
                "primed_sense_label_short": SHORT_LABELS[word][primed],
                "correct_sense_label_short": SHORT_LABELS[word][correct],
                "resolution_word": item["resolution_word"],
                "conflicting_sentence": item["sentence"],
                "coherent_control_sentence": item["matched_control_sentence"],
                "sentinel": DEFAULT_SENTINEL,
                "conflicting_prime_prefix": prefixes["prime"],
                "conflicting_homonym_prefix": prefixes["homonym"],
                "conflicting_resolution_prefix": prefixes["resolution"],
                "coherent_control_resolution_prefix": _control_resolution_prefix(item, word, DEFAULT_SENTINEL),
                "resolver_only_text": _append_sentinel(item["resolution_word"], DEFAULT_SENTINEL),
                "design_note": item.get("design_note", ""),
            })
    return pd.DataFrame(rows)


def build() -> dict:
    homonyms, profiling = homonyms_and_profiling()
    paired, carriers = paired_and_carriers()
    return {
        "homonyms": homonyms,
        "profiling_sentences": profiling,
        "carrier_contexts": carriers,
        "paired_context_sentences": paired,
        "conflict_items": conflict_items(),
    }


def main() -> None:
    for name, frame in build().items():
        write_table(frame, STIMULI_DIR / f"{name}.csv")
        print(f"  data/stimuli/{name}.csv  {len(frame)} rows x {len(frame.columns)} columns")


if __name__ == "__main__":
    main()
