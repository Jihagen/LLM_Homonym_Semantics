"""Build ``data/processed/`` from a finished pipeline run under ``results/``.

Maintainer step. It needs the per-model outputs of a full study run (see
"Full rerun" in the README); it does not run any model and recomputes no
statistic. Every released value is the text found in the pipeline CSVs.

    python -m scripts.build_release_data --results-dir results

After writing, the released tables are materialised back into a
``results/``-shaped tree and compared cell by cell with the source files, so a
table that does not round-trip aborts the build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from scripts import release_layout as layout
from scripts.release_layout import (
    DATA_VERSION,
    GDV_MODEL_PATTERN,
    GDV_RANK_PATTERN,
    GDV_WORD_PATTERN,
    GEOMETRY_EXAMPLE,
    H1_PROFILE_PATTERN,
    H5_BY_LAYER_DROPPED,
    H5_BY_LAYER_SOURCE,
    MANIFEST_PATH,
    MODEL_KEYS,
    PROCESSED_DIR,
    ROOT,
    SPLIT_TABLES,
    STATES_DIR,
    TRAJECTORY_EXAMPLES,
    VERBATIM_TABLES,
    WORDS,
    read_table,
    write_table,
)
from utils.model_registry import ALL_MODELS

DISPLAY_NAMES = {
    "answerdotai/ModernBERT-large": "ModernBERT",
    "microsoft/deberta-v3-large": "DeBERTa",
    "FacebookAI/roberta-large": "RoBERTa",
    "FacebookAI/xlm-roberta-large": "XLM-R",
    "Qwen/Qwen2.5-3B": "Qwen-3B",
    "Qwen/Qwen2.5-7B": "Qwen-7B",
    "mistralai/Mistral-Nemo-Base-2407": "Mistral-Nemo",
    "allenai/OLMo-2-1124-7B": "OLMo-7B",
}


def _with_keys(frame: pd.DataFrame, added, cell) -> pd.DataFrame:
    frame = frame.copy()
    for position, column in enumerate(added):
        if column in frame.columns:
            raise ValueError(f"{column!r} already present; cannot add it as a key")
        frame.insert(position, column, cell[column])
    return frame


def build_split_tables(results: Path) -> Dict[str, pd.DataFrame]:
    tables = {}
    for spec in SPLIT_TABLES:
        parts = []
        for cell in spec.cells():
            parts.append(_with_keys(read_table(results / spec.pattern.format(**cell)), spec.added, cell))
        tables[spec.name] = pd.concat(parts, ignore_index=True)
    return tables


def build_verbatim_tables(results: Path) -> Dict[str, pd.DataFrame]:
    return {spec.name: read_table(results / spec.source) for spec in VERBATIM_TABLES}


def build_h1_layer_profiles(results: Path) -> pd.DataFrame:
    parts = []
    for model in MODEL_KEYS:
        for word in WORDS:
            profile = read_table(results / H1_PROFILE_PATTERN.format(model=model, word=word))
            gdv = read_table(results / GDV_WORD_PATTERN.format(model=model, word=word))
            if list(profile["Layer"]) != list(gdv["Layer"]):
                raise ValueError(f"H1 and GDV layer indices differ for {model}/{word}")
            profile = profile.copy()
            profile["GDV"] = gdv["GDV"].to_numpy()
            parts.append(_with_keys(profile, ("model", "word"), {"model": model, "word": word}))
    return pd.concat(parts, ignore_index=True)


def build_gdv_model_summary(results: Path) -> pd.DataFrame:
    parts = []
    for model in MODEL_KEYS:
        values = read_table(results / GDV_MODEL_PATTERN.format(model=model))
        ranks = read_table(results / GDV_RANK_PATTERN.format(model=model))
        if list(values["Layer"]) != list(ranks["Layer"]):
            raise ValueError(f"GDV value and rank layer indices differ for {model}")
        values = values.copy()
        values["MeanRank"] = ranks["MeanRank"].to_numpy()
        parts.append(_with_keys(values, ("model",), {"model": model}))
    return pd.concat(parts, ignore_index=True)


def build_h5_by_layer(results: Path) -> pd.DataFrame:
    frame = read_table(results / H5_BY_LAYER_SOURCE)
    return frame.drop(columns=list(H5_BY_LAYER_DROPPED))


def build_models_table(tables: Dict[str, pd.DataFrame], revisions: Dict[str, str]) -> pd.DataFrame:
    summary = tables["h1_layer_selection_summary"]
    arch = dict(zip(tables["h3_paired_summary"]["model"], tables["h3_paired_summary"]["arch_type"]))
    rows = []
    for name in ALL_MODELS:
        key = name.replace("/", "_")
        n_states = sorted(set(summary.loc[summary["model"] == key, "n_layers"]))
        if len(n_states) != 1:
            raise ValueError(f"inconsistent n_layers for {key}: {n_states}")
        rows.append({
            "model": key,
            "hf_repo_id": name,
            "display_name": DISPLAY_NAMES[name],
            "arch_type": arch[key],
            "n_hidden_states": n_states[0],
            "last_layer_index": str(int(n_states[0]) - 1),
            "hf_revision": revisions.get(name, ""),
        })
    return pd.DataFrame(rows)


def _load_revisions() -> Dict[str, str]:
    with open(ROOT / "configs" / "study.json", encoding="utf-8") as handle:
        config = json.load(handle)
    return {entry["hf_repo_id"]: entry.get("hf_revision", "") for entry in config["models"]}


def build_example_states(results: Path) -> List[Path]:
    """Copy the three small single-layer state snapshots used by figures.

    Stored as float32 ``.npz`` without pickled objects. These are the only
    hidden states in the release; the full per-layer caches are not shipped.
    """
    import h5py

    written = []
    STATES_DIR.mkdir(parents=True, exist_ok=True)
    example = GEOMETRY_EXAMPLE
    for layer in example["layers"]:
        source = results / "activations" / example["word"] / example["model"] / f"layer_{layer}.h5"
        with h5py.File(source, "r") as handle:
            X = handle["X"][:].astype(np.float32)
            labels = handle["labels"][:].astype(np.int64)
            sentences = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in handle["sentences"][:]])
        path = layout.geometry_state_path(layer)
        np.savez_compressed(path, X=X, labels=labels, sentences=sentences, layer=np.int64(layer))
        written.append(path)

    for model, word in TRAJECTORY_EXAMPLES:
        cache = results / "study" / "h3_context_trajectory" / "_extraction_cache" / f"{model}_{word}.pkl"
        with open(cache, "rb") as handle:
            data = pickle.load(handle)
        path = layout.trajectory_state_path("h3_context", model, word)
        np.savez_compressed(
            path,
            layer=np.int64(data["layer"]),
            profile_homonym_H=np.asarray(data["profile_homonym_H"], dtype=np.float32),
            profile_resolved_H=np.asarray(data["profile_resolved_H"], dtype=np.float32),
            profile_senses=np.asarray(data["profile_senses"], dtype=np.int64),
            item_homonym_H=np.asarray(data["item_homonym_H"], dtype=np.float32),
            item_resolved_H=np.asarray(data["item_resolved_H"], dtype=np.float32),
            item_conditions=np.asarray(data["item_conditions"]).astype(str),
            item_senses=np.asarray(data["item_senses"], dtype=np.int64),
            item_ids=np.asarray(data["item_ids"]).astype(str),
        )
        written.append(path)

        cache = results / "study" / "h5_revision_trajectory" / "_extraction_cache" / f"{model}_{word}.pkl"
        with open(cache, "rb") as handle:
            data = pickle.load(handle)
        path = layout.trajectory_state_path("h5_revision", model, word)
        np.savez_compressed(
            path,
            layer=np.int64(data["layer"]),
            profile_H=np.asarray(data["profile_H"], dtype=np.float32),
            profile_senses=np.asarray(data["profile_senses"], dtype=np.int64),
            stage_H=np.asarray(data["stage_H"], dtype=np.float32),
            stage_item_index=np.asarray([key[0] for key in data["stage_keys"]], dtype=np.int64),
            stage_name=np.asarray([key[1] for key in data["stage_keys"]]).astype(str),
            item_ids=np.asarray([item["id"] for item in data["items"]]).astype(str),
        )
        written.append(path)
    return written


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return list(a.columns) == list(b.columns) and a.reset_index(drop=True).equals(b.reset_index(drop=True))


def verify_round_trip(results: Path) -> int:
    """Materialise the release and compare every file with its pipeline source."""
    checked = 0
    with tempfile.TemporaryDirectory() as tmp:
        tree = layout.materialize_results_tree(Path(tmp))
        sources = [spec.source for spec in VERBATIM_TABLES]
        for spec in SPLIT_TABLES:
            sources.extend(spec.pattern.format(**cell) for cell in spec.cells())
        for model in MODEL_KEYS:
            sources.append(GDV_MODEL_PATTERN.format(model=model))
            sources.append(GDV_RANK_PATTERN.format(model=model))
            for word in WORDS:
                sources.append(H1_PROFILE_PATTERN.format(model=model, word=word))
                sources.append(GDV_WORD_PATTERN.format(model=model, word=word))
        for relative in sources:
            if not _frames_equal(read_table(results / relative), read_table(tree / relative)):
                raise AssertionError(f"released data does not reproduce {relative}")
            checked += 1

        import h5py

        example = GEOMETRY_EXAMPLE
        for layer in example["layers"]:
            relative = Path("activations") / example["word"] / example["model"] / f"layer_{layer}.h5"
            with h5py.File(results / relative, "r") as a, h5py.File(tree / relative, "r") as b:
                if not (np.array_equal(a["X"][:], b["X"][:]) and np.array_equal(a["labels"][:], b["labels"][:])):
                    raise AssertionError(f"released states do not reproduce {relative}")
            checked += 1
    return checked


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest() -> None:
    """Record every released data file with its size, checksum, and shape."""
    entries = []
    data_dir = ROOT / "data"
    files = sorted(
        path for path in data_dir.rglob("*")
        if path.is_file()
        and path.suffix in {".csv", ".json", ".npz"}
        and path != MANIFEST_PATH
        and "__pycache__" not in path.parts
    )
    for path in files:
        entry = {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if path.suffix == ".csv":
            frame = read_table(path)
            entry["rows"] = int(len(frame))
            entry["columns"] = list(frame.columns)
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as npz:
                entry["arrays"] = {key: list(npz[key].shape) for key in npz.files}
        entries.append(entry)
    manifest = {
        "name": "LLM_Homonym_Semantics released data",
        "data_version": DATA_VERSION,
        "description": (
            "Stimuli and processed result tables. Checksums are SHA-256 of the "
            "files as committed. See data/README.md for field definitions."
        ),
        "files": entries,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results", help="finished pipeline output directory")
    parser.add_argument(
        "--manifest-only", action="store_true",
        help="only refresh data/release_manifest.json from the files already in data/",
    )
    args = parser.parse_args()

    if args.manifest_only:
        write_manifest()
        print(f"wrote {MANIFEST_PATH.relative_to(ROOT)}")
        return

    results = Path(args.results_dir)
    tables = {}
    tables.update(build_split_tables(results))
    tables.update(build_verbatim_tables(results))
    tables["h1_layer_profiles"] = build_h1_layer_profiles(results)
    tables["gdv_model_layer_summary"] = build_gdv_model_summary(results)
    tables["h5_by_layer_sentence_level"] = build_h5_by_layer(results)
    tables["models"] = build_models_table(tables, _load_revisions())

    for name, frame in sorted(tables.items()):
        write_table(frame, PROCESSED_DIR / f"{name}.csv")
        print(f"  {name}.csv  {len(frame)} rows x {len(frame.columns)} columns")

    for path in build_example_states(results):
        print(f"  {path.relative_to(PROCESSED_DIR)}  {path.stat().st_size / 1e6:.2f} MB")

    checked = verify_round_trip(results)
    print(f"round trip verified against {checked} pipeline files")

    write_manifest()
    print(f"wrote {MANIFEST_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
