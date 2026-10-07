"""Mapping between the study pipeline's ``results/`` tree and the released
tables under ``data/processed/``.

The pipeline (``run_study.py`` and friends) writes one small CSV per model and
per homonym below ``results/``. The public release ships the same values as a
handful of consolidated tables. This module is the single description of that
correspondence, used in both directions:

* ``scripts.build_release_data`` reads ``results/`` and writes ``data/processed/``
  (maintainer step, after a model run);
* ``materialize_results_tree`` rebuilds a ``results/``-shaped directory from
  ``data/processed/`` so the plotting code in ``analysis/`` runs unchanged on
  the released data (``scripts.make_figures``).

Values are carried as text in both directions: no number is reparsed,
rounded, or recomputed here.

Only numpy/pandas/h5py are needed; no torch, no transformers, no model cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.model_registry import ALL_MODELS

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
STATES_DIR = PROCESSED_DIR / "example_states"
MANIFEST_PATH = ROOT / "data" / "release_manifest.json"
DATA_VERSION = "1.0.0"

WORDS = ["bank", "bark", "bat", "crane", "spring", "match", "pitch"]
MODEL_KEYS = [name.replace("/", "_") for name in ALL_MODELS]

# The H2 geometry example in the report (Qwen2.5-7B, "bat", layers 3 and 26).
GEOMETRY_EXAMPLE = {"model": "Qwen_Qwen2.5-7B", "word": "bat", "layers": (3, 26)}
# Single-layer state snapshots behind the two sample-trajectory figures.
TRAJECTORY_EXAMPLES = [("FacebookAI_roberta-large", "bank"), ("Qwen_Qwen2.5-7B", "bank")]


@dataclass(frozen=True)
class SplitTable:
    """A released table that concatenates one pipeline CSV per model (and word).

    ``pattern`` is relative to the results directory and may use ``{model}``
    and ``{word}``. ``added`` lists the key columns that the per-file CSVs do
    not carry themselves and that are therefore prepended on release and
    dropped again on materialisation.
    """

    name: str
    pattern: str
    added: Tuple[str, ...]
    words: Tuple[str, ...] = tuple(WORDS)
    models: Tuple[str, ...] = tuple(MODEL_KEYS)

    def cells(self) -> Iterable[Dict[str, str]]:
        for model in self.models:
            if "{word}" in self.pattern:
                for word in self.words:
                    yield {"model": model, "word": word}
            else:
                yield {"model": model}


@dataclass(frozen=True)
class VerbatimTable:
    """A released table that is a straight copy of one pipeline CSV."""

    name: str
    source: str


SPLIT_TABLES: List[SplitTable] = [
    SplitTable("h0_carrier_lean", "study/H0/{model}/h0_{word}.csv", added=("model",)),
    SplitTable("h1_layer_selection_summary", "study/H1/{model}/h1_summary.csv", added=("model",)),
    SplitTable("h1_nested_loo", "study/H1/{model}/h1_nested_loo_{word}.csv", added=("model", "word")),
    SplitTable("h2_leave_one_word_out", "study/H2/{model}/h2_loo.csv", added=("model",)),
    SplitTable("h3_sentence_level", "study/H3/{model}/h3_{word}.csv", added=("model", "word")),
    SplitTable("q4_candidate_layers", "study/Q4/{model}/q4_candidate_layers.csv", added=("model",)),
    SplitTable(
        "trajectory_h3_context",
        "study/h3_context_trajectory/{model}/{word}_context_trajectory.csv",
        added=("model", "word"),
        words=("bank",),
        models=tuple(m for m, _ in TRAJECTORY_EXAMPLES),
    ),
    SplitTable(
        "trajectory_h5_revision",
        "study/h5_revision_trajectory/{model}/{word}_revision_trajectory.csv",
        added=("model", "word"),
        words=("bank",),
        models=tuple(m for m, _ in TRAJECTORY_EXAMPLES),
    ),
]

VERBATIM_TABLES: List[VerbatimTable] = [
    VerbatimTable("h0_summary", "study/H0/h0_summary.csv"),
    VerbatimTable("h2_strategy_summary", "study/H2/h2_strategy_summary.csv"),
    VerbatimTable("geometry_inference", "study/geometry_inference.csv"),
    VerbatimTable("h3_aggregate", "study/H3/h3_aggregate.csv"),
    VerbatimTable("h3_pair_differences", "study/H3/h3_pair_differences.csv"),
    VerbatimTable("h3_paired_summary", "study/H3/h3_paired_summary.csv"),
    VerbatimTable("h3_architecture_interaction", "study/H3/h3_architecture_interaction.csv"),
    VerbatimTable("h4_sentence_level", "study/H4/h4_sentence_level.csv"),
    VerbatimTable("h4_aggregate", "study/H4/h4_aggregate.csv"),
    VerbatimTable("h5_sentence_level", "study/H5/h5_sentence_level.csv"),
    VerbatimTable("h5_aggregate", "study/H5/h5_aggregate.csv"),
    VerbatimTable("h5_architecture_exploratory", "study/H5/h5_architecture_exploratory.csv"),
    VerbatimTable("h5_design_audit", "study/H5/h5_design_audit.csv"),
    VerbatimTable("cross_hypothesis_associations", "study/cross_hypothesis_associations.csv"),
    VerbatimTable("q4_layer_curves", "study/Q4/q4_layer_curves.csv"),
    VerbatimTable("q4_h1_layer_shift", "study/Q4/q4_h1_layer_shift.csv"),
    VerbatimTable("q4_h5_outcome_comparison", "study/Q4/q4_h5_outcome_comparison.csv"),
    VerbatimTable("h5_by_layer_aggregate", "study/H5_q4/h5_aggregate.csv"),
]

# Tables with a bespoke build step (see scripts.build_release_data):
#   h1_layer_profiles          H1 per-layer profile joined with per-layer GDV
#   gdv_model_layer_summary    across-word mean GDV and mean GDV rank per layer
#   h5_by_layer_sentence_level H5 rerun at every Q4 candidate layer, text columns dropped
#   models                     model registry with architecture and depth
H1_PROFILE_PATTERN = "study/H1/{model}/h1_{word}.csv"
GDV_WORD_PATTERN = "{model}_gdv/gdv_values_{word}.csv"
GDV_MODEL_PATTERN = "{model}_gdv/gdv_values.csv"
GDV_RANK_PATTERN = "{model}_gdv/gdv_rank_aggregated.csv"
H5_BY_LAYER_SOURCE = "study/H5_q4/h5_sentence_level.csv"
# Free-text columns of the by-layer H5 table. They repeat, for every layer, text
# that is already in data/stimuli/conflict_items.csv and h5_sentence_level.csv.
H5_BY_LAYER_DROPPED = ("sentence", "prime_prefix", "homonym_prefix", "resolution_prefix")


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV with every cell kept as the text that is in the file."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def released(name: str) -> pd.DataFrame:
    return read_table(PROCESSED_DIR / f"{name}.csv")


def _write_h5(path: Path, X: np.ndarray, labels: np.ndarray) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=X)
        handle.create_dataset("labels", data=labels)


def geometry_state_path(layer: int) -> Path:
    example = GEOMETRY_EXAMPLE
    return STATES_DIR / f"h2_geometry_{example['model']}_{example['word']}_layer{layer}.npz"


def trajectory_state_path(kind: str, model: str, word: str) -> Path:
    return STATES_DIR / f"{kind}_trajectory_{model}_{word}.npz"


def load_h3_trajectory_states(model: str, word: str) -> Dict:
    """Arrays for analysis.plot_h3_context_trajectory.make_trajectory_figure."""
    with np.load(trajectory_state_path("h3_context", model, word), allow_pickle=False) as npz:
        data = {key: npz[key] for key in npz.files}
    data["layer"] = int(data["layer"])
    data["item_ids"] = [str(x) for x in data["item_ids"]]
    data["item_conditions"] = data["item_conditions"].astype(str)
    return data


def load_h5_trajectory_states(model: str, word: str) -> Dict:
    """Arrays for analysis.plot_h5_revision_trajectory.make_trajectory_figure.

    The pipeline cache stores the stimulus items next to the states; the
    release stores only their ids and rejoins them with the stimulus file.
    """
    with np.load(trajectory_state_path("h5_revision", model, word), allow_pickle=False) as npz:
        data = {key: npz[key] for key in npz.files}
    with open(ROOT / "data" / "garden_path_sentences.json", encoding="utf-8") as handle:
        by_id = {item["id"]: item for item in json.load(handle)[word]}
    item_ids = [str(x) for x in data.pop("item_ids")]
    data["items"] = [by_id[item_id] for item_id in item_ids]
    data["layer"] = int(data["layer"])
    data["stage_keys"] = [
        (int(index), str(stage))
        for index, stage in zip(data.pop("stage_item_index"), data.pop("stage_name"))
    ]
    return data


def materialize_results_tree(target: Path, tables: Sequence[str] | None = None) -> Path:
    """Rebuild a ``results/``-shaped tree under ``target`` from ``data/processed/``.

    Returns ``target`` so it can be passed as ``--results-dir`` to the scripts
    in ``analysis/``. ``tables`` optionally restricts the split/verbatim tables
    that are written.
    """
    target = Path(target)
    wanted = None if tables is None else set(tables)

    for spec in VERBATIM_TABLES:
        if wanted is not None and spec.name not in wanted:
            continue
        write_table(released(spec.name), target / spec.source)

    for spec in SPLIT_TABLES:
        if wanted is not None and spec.name not in wanted:
            continue
        frame = released(spec.name)
        for cell in spec.cells():
            mask = np.ones(len(frame), dtype=bool)
            for column, value in cell.items():
                mask &= (frame[column] == value).to_numpy()
            part = frame.loc[mask].drop(columns=list(spec.added))
            write_table(part, target / spec.pattern.format(**cell))

    if wanted is None or "h1_layer_profiles" in wanted:
        profiles = released("h1_layer_profiles")
        for (model, word), part in profiles.groupby(["model", "word"], sort=False):
            write_table(
                part.drop(columns=["model", "word", "GDV"]),
                target / H1_PROFILE_PATTERN.format(model=model, word=word),
            )
            write_table(
                part[["Layer", "GDV"]],
                target / GDV_WORD_PATTERN.format(model=model, word=word),
            )

    if wanted is None or "gdv_model_layer_summary" in wanted:
        summary = released("gdv_model_layer_summary")
        for model, part in summary.groupby("model", sort=False):
            write_table(part[["Layer", "GDV"]], target / GDV_MODEL_PATTERN.format(model=model))
            write_table(part[["Layer", "MeanRank"]], target / GDV_RANK_PATTERN.format(model=model))

    if wanted is None or "example_states" in wanted:
        example = GEOMETRY_EXAMPLE
        for layer in example["layers"]:
            with np.load(geometry_state_path(layer), allow_pickle=False) as npz:
                _write_h5(
                    target / "activations" / example["word"] / example["model"] / f"layer_{layer}.h5",
                    npz["X"],
                    npz["labels"],
                )
    return target
