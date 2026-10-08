"""Validate the released data: files, schemas, row counts, design balance,
internal consistency, and the headline values of the project report.

    python -m scripts.validate_release        # or: make validate

Reads only ``data/`` and ``configs/``. Needs numpy, pandas, h5py and
scikit-learn; no torch, no transformers, no model cache. Exits non-zero if any
check fails.

The headline values are the numbers printed in the report. Each is
recomputed here from the released tables and compared at the precision the
report prints it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

from scripts import release_layout as layout
from scripts.release_layout import GEOMETRY_EXAMPLE, MANIFEST_PATH, MODEL_KEYS, PROCESSED_DIR, ROOT, TRAJECTORY_EXAMPLES, WORDS
from utils.model_registry import ALL_MODELS

N_MODELS, N_WORDS = 8, 7
# Both the export and the released H5 table round margins to 4 decimals.
MARGIN_TOLERANCE = 0.00011
EXPECTED_ROWS = {
    "data/stimuli/homonyms.csv": 14,
    "data/stimuli/profiling_sentences.csv": 280,          # 7 words x 2 senses x 20
    "data/stimuli/carrier_contexts.csv": 42,              # 7 x (5 carriers + bare word)
    "data/stimuli/paired_context_sentences.csv": 140,     # 7 x 5 carriers x 2 senses x L/R
    "data/stimuli/conflict_items.csv": 98,                # 7 x 14
    "data/processed/models.csv": 8,
    "data/processed/h0_carrier_lean.csv": 280,            # 56 cells x 5 carriers
    "data/processed/h0_summary.csv": 56,
    "data/processed/h1_layer_selection_summary.csv": 56,
    "data/processed/h1_nested_loo.csv": 2240,             # 56 cells x 40 held-out sentences
    "data/processed/h2_leave_one_word_out.csv": 56,
    "data/processed/h2_strategy_summary.csv": 8,
    "data/processed/geometry_inference.csv": 12,
    "data/processed/h3_sentence_level.csv": 1120,         # 56 cells x 20
    "data/processed/h3_aggregate.csv": 112,
    "data/processed/h3_pair_differences.csv": 560,
    "data/processed/h3_paired_summary.csv": 8,
    "data/processed/h3_architecture_interaction.csv": 1,
    "data/processed/h4_sentence_level.csv": 560,          # 56 cells x 10 R sentences
    "data/processed/h4_aggregate.csv": 56,
    "data/processed/h5_sentence_level.csv": 784,          # 8 models x 98 items
    "data/processed/h5_aggregate.csv": 56,
    "data/processed/h5_architecture_exploratory.csv": 3,
    "data/processed/h5_design_audit.csv": 8,
    "data/processed/cross_hypothesis_associations.csv": 5,
    "data/processed/q4_h1_layer_shift.csv": 56,
    "data/processed/q4_h5_outcome_comparison.csv": 56,
    "data/processed/trajectory_h3_context.csv": 40,
    "data/processed/trajectory_h5_revision.csv": 28,
}

_results: List[Tuple[bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    _results.append((bool(condition), label + (f"  [{detail}]" if detail and not condition else "")))


def close(label: str, value: float, reported: float, decimals: int = 3) -> None:
    """Compare with a value as printed in the report (rounded to `decimals`)."""
    ok = abs(float(value) - reported) <= 0.5 * 10 ** (-decimals) + 1e-9
    _results.append((ok, f"{label}: report {reported}, released data {float(value):.{decimals + 2}f}"))


def count(label: str, value, reported) -> None:
    _results.append((value == reported, f"{label}: report {reported}, released data {value}"))


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / f"{name}.csv")


def is_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower() == "true"


# --------------------------------------------------------------------------- files

def check_manifest() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    check("manifest data_version matches code", manifest["data_version"] == layout.DATA_VERSION)
    listed = {entry["path"] for entry in manifest["files"]}
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.exists():
            check(f"file present: {entry['path']}", False)
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        check(f"checksum {entry['path']}", digest == entry["sha256"], "file differs from manifest")
        if path.suffix == ".csv":
            frame = layout.read_table(path)
            check(f"schema {entry['path']}", list(frame.columns) == entry["columns"], "columns differ from manifest")
            check(f"row count {entry['path']}", len(frame) == entry["rows"], f"{len(frame)} != {entry['rows']}")
    for relative, rows in EXPECTED_ROWS.items():
        check(f"listed in manifest: {relative}", relative in listed)
        path = ROOT / relative
        if path.exists():
            n = len(layout.read_table(path))
            check(f"design row count {relative} = {rows}", n == rows, f"found {n}")


def check_config_and_design() -> None:
    config = json.loads((ROOT / "configs" / "study.json").read_text(encoding="utf-8"))
    check("configs/study.json models match utils/model_registry.py",
          [m["hf_repo_id"] for m in config["models"]] == ALL_MODELS)
    check("configs/study.json words match the study set", [w["word"] for w in config["words"]] == WORDS)
    models = table("models")
    check("models.csv keys match registry", list(models["model"]) == MODEL_KEYS)
    check("4 encoders and 4 decoders", models["arch_type"].value_counts().to_dict() == {"encoder": 4, "decoder": 4})
    revisions = {m["hf_repo_id"]: m["hf_revision"] for m in config["models"]}
    check("models.csv revisions match config", all(revisions[r.hf_repo_id] == r.hf_revision for r in models.itertuples()))

    stim = ROOT / "data" / "stimuli"
    profiling = pd.read_csv(stim / "profiling_sentences.csv")
    check("profiling: 20 sentences per word and sense",
          (profiling.groupby(["word", "sense"]).size() == 20).all() and profiling["word"].nunique() == N_WORDS)
    check("profiling: homonym present in every sentence",
          all(_has_word(s, w) for s, w in zip(profiling["sentence"], profiling["word"])))
    paired = pd.read_csv(stim / "paired_context_sentences.csv")
    check("paired: 5 sentences per word, condition and sense",
          (paired.groupby(["word", "condition", "sense"]).size() == 5).all())
    check("paired: every pair has one L and one R sentence with the same carrier and sense",
          (paired.groupby("pair_id").agg(n=("condition", "nunique"), c=("carrier", "nunique"), s=("sense", "nunique"))
           .eq([2, 1, 1]).all(axis=None)))
    check("paired: sentence ids unique", paired["sentence_id"].is_unique)
    conflict = pd.read_csv(stim / "conflict_items.csv", keep_default_na=False)
    check("conflict: 7 items per word and direction",
          (conflict.groupby(["word", "direction"]).size() == 7).all() and len(conflict) == 98)
    check("conflict: every item has a matched coherent control",
          (conflict["coherent_control_sentence"].str.len() > 0).all()
          and (conflict["coherent_control_resolution_prefix"].str.len() > 0).all())
    check("conflict: primed and correct senses differ", (conflict["primed_sense"] != conflict["correct_sense"]).all())

    try:
        from scripts.build_stimulus_tables import build

        rebuilt = build()
        for name, frame in rebuilt.items():
            on_disk = layout.read_table(stim / f"{name}.csv")
            fresh = pd.read_csv(pd.io.common.StringIO(frame.to_csv(index=False)), dtype=str, keep_default_na=False, na_filter=False)
            check(f"data/stimuli/{name}.csv equals the JSON source it is derived from", on_disk.equals(fresh))
    except ImportError as exc:  # pragma: no cover
        check("stimulus tables rebuilt from JSON sources", False, str(exc))

    h5 = pd.read_csv(PROCESSED_DIR / "h5_sentence_level.csv", keep_default_na=False)
    by_id = conflict.set_index("item_id")
    same = all(
        (h5[column] == by_id.loc[h5["sentence_id"], stimulus_column].to_numpy()).all()
        for column, stimulus_column in [
            ("prime_prefix", "conflicting_prime_prefix"),
            ("homonym_prefix", "conflicting_homonym_prefix"),
            ("resolution_prefix", "conflicting_resolution_prefix"),
        ]
    )
    check("H5 result prefixes equal the released stimulus prefixes", same)
    for name in ("h0_summary", "h1_layer_selection_summary", "h3_aggregate", "h4_aggregate", "h5_aggregate"):
        frame = table(name)
        check(f"{name}: all 8 models x 7 words present",
              set(map(tuple, frame[["model", "word"]].drop_duplicates().to_numpy())) ==
              {(m, w) for m in MODEL_KEYS for w in WORDS})
    for name, column in (("h0_carrier_lean", "signed_M_l_carrier_norm"), ("h3_sentence_level", "M_l_norm"),
                         ("h5_sentence_level", "resolution_correct_margin_norm")):
        values = table(name)[column]
        check(f"{name}.{column} within [-1, 1]", values.abs().max() <= 1.00001)


def _has_word(sentence: str, word: str) -> bool:
    from utils.text import find_target_span

    return find_target_span(sentence, word) is not None


# --------------------------------------------------------------------------- headline values

def _arch(frame: pd.DataFrame) -> pd.Series:
    models = table("models").set_index("model")["arch_type"]
    return frame["model"].map(models)


def check_h0() -> None:
    lean = table("h0_carrier_lean")
    summary = table("h0_summary")
    lean["arch"], summary["arch"] = _arch(lean), _arch(summary)
    exceeds = is_true(lean["abs_norm_gt_0p3"])
    close("H0 mean absolute carrier lean", summary["mean_abs_M_l_carrier_norm"].mean(), 0.339)
    close("H0 mean direction consistency", summary["direction_consistency"].mean(), 0.882)
    count("H0 carrier states with |B|>0.3", f"{int(exceeds.sum())}/{len(lean)}", "159/280")
    count("H0 |B|>0.3, encoders", int(exceeds[lean["arch"] == "encoder"].sum()), 65)
    count("H0 |B|>0.3, decoders", int(exceeds[lean["arch"] == "decoder"].sum()), 94)
    close("H0 encoder mean |B|", summary.loc[summary["arch"] == "encoder", "mean_abs_M_l_carrier_norm"].mean(), 0.279)
    close("H0 decoder mean |B|", summary.loc[summary["arch"] == "decoder", "mean_abs_M_l_carrier_norm"].mean(), 0.398)
    agree = np.sign(summary["signed_M_l_word_alone_norm"]) == np.sign(summary["mean_signed_M_l_carrier_norm"])
    count("H0 bare-word and carrier directions agree (cells)", int(agree.sum()), 28)


def check_h1_h2() -> None:
    loo = table("h1_nested_loo")
    loo["arch"] = _arch(loo)
    selected, last = is_true(loo["adequate_selected"]), is_true(loo["adequate_last"])
    count("H1 outer folds", len(loo), 2240)
    close("H1 nested selected-layer adequacy", selected.mean(), 0.917)
    close("H1 final-layer adequacy", last.mean(), 0.893)
    for arch, sel_ref, last_ref in (("encoder", 0.966, 0.960), ("decoder", 0.867, 0.826)):
        mask = loo["arch"] == arch
        close(f"H1 {arch} selected", selected[mask].mean(), sel_ref)
        close(f"H1 {arch} final", last[mask].mean(), last_ref)
    inference = table("geometry_inference").set_index(["analysis", "contrast", "scope"])
    row = inference.loc[("H1", "nested selected - last fraction adequate", "all")]
    close("H1 crossed model-word effect", row["mean_effect"], 0.0237, 4)
    close("H1 effect CI low", row["ci95_low"], -0.0058, 4)
    close("H1 effect CI high", row["ci95_high"], 0.0598, 4)

    summary = table("h1_layer_selection_summary")
    count("H1/H2 adequacy and GDV select the same layer (cells)",
          int((summary["best_layer_M"] == summary["gdv_best_layer"]).sum()), 26)
    summary["arch"] = _arch(summary)
    close("H1 encoder mean relative depth of selected layer",
          summary.loc[summary["arch"] == "encoder", "best_layer_depth"].mean(), 0.762)
    close("H1 decoder mean relative depth of selected layer",
          summary.loc[summary["arch"] == "decoder", "best_layer_depth"].mean(), 0.494)

    h2 = table("h2_leave_one_word_out")
    for strategy, reported in (("gdv", 0.892), ("supervised", 0.914), ("last", 0.893), ("oracle", 0.930)):
        close(f"H2 held-out adequacy, {strategy}", h2[f"fraction_adequate_{strategy}"].mean(), reported)
    close("H2 GDV oracle regret", h2["fraction_regret_to_oracle_gdv"].mean(), 0.038)
    close("H2 GDV layer distance to oracle", h2["layer_distance_to_oracle_gdv"].mean(), 4.62, 2)
    row = inference.loc[("H2", "gdv - last fraction adequate", "all")]
    close("H2 GDV minus last effect", row["mean_effect"], -0.0009, 4)


def check_geometry_example() -> None:
    from experiments.adequacy import leave_one_out_adequacy_margins
    from experiments.gdv_experiments import compute_gdv

    reported = {3: (-0.0505, 0.90), 26: (-0.0634, 0.60)}
    for layer in GEOMETRY_EXAMPLE["layers"]:
        with np.load(layout.geometry_state_path(layer), allow_pickle=False) as npz:
            X, labels = npz["X"], npz["labels"]
        raw, _ = leave_one_out_adequacy_margins(X, labels)
        close(f"H2 example Qwen2.5-7B/bat layer {layer} GDV", compute_gdv(X, labels), reported[layer][0], 4)
        close(f"H2 example layer {layer} leave-one-out adequacy", (raw > 0).mean(), reported[layer][1], 2)


def check_h3_h4() -> None:
    # Sentence-level values, not the 4-decimal cell means of h3_aggregate: the
    # encoder R mean is 0.357501 and a mean of rounded means lands on 0.35750.
    sentences = table("h3_sentence_level")
    sentences["arch"] = _arch(sentences)
    for arch, condition, margin, adequacy in (
        ("encoder", "L", 0.360, 0.843), ("encoder", "R", 0.358, 0.861),
        ("decoder", "L", 0.342, 0.918), ("decoder", "R", -0.0002, 0.500),
    ):
        part = sentences[(sentences["arch"] == arch) & (sentences["condition"] == condition)]
        close(f"H3 {arch} {condition} mean margin", part["M_l_norm"].mean(), margin, 4 if margin < 0 else 3)
        close(f"H3 {arch} {condition} adequacy", is_true(part["adequate"]).mean(), adequacy)
    interaction = table("h3_architecture_interaction").iloc[0]
    close("H3 architecture interaction", interaction["architecture_interaction"], 0.3394, 4)
    close("H3 interaction CI low", interaction["ci95_low_interaction"], 0.2358, 4)
    close("H3 interaction CI high", interaction["ci95_high_interaction"], 0.4355, 4)

    h4 = table("h4_sentence_level")
    target, final = is_true(h4["target_local_adequate"]), is_true(h4["final_local_adequate"])
    for arch, t_ref, f_ref, gain, loss in (
        ("encoder", 0.861, 0.668, "16/39", "70/241"), ("decoder", 0.500, 0.761, "95/140", "22/140"),
    ):
        mask = h4["arch_type"] == arch
        close(f"H4 {arch} adequacy at homonym", target[mask].mean(), t_ref)
        close(f"H4 {arch} adequacy at sentence-final period", final[mask].mean(), f_ref)
        count(f"H4 {arch} gain (inadequate at homonym -> adequate at period)",
              f"{int((~target & final & mask).sum())}/{int((~target & mask).sum())}", gain)
        count(f"H4 {arch} loss (adequate -> inadequate)",
              f"{int((target & ~final & mask).sum())}/{int((target & mask).sum())}", loss)
    close("H4 cross-position diagnostic adequacy", is_true(h4["final_cross_position_adequate"]).mean(), 0.516)


def check_h5() -> None:
    h5 = table("h5_sentence_level")
    count("H5 model-item observations", len(h5), 784)
    close("H5 sentinel mean after prime", h5["prime_correct_margin_norm"].mean(), -0.080)
    close("H5 sentinel mean after homonym", h5["homonym_correct_margin_norm"].mean(), -0.129)
    close("H5 sentinel mean after resolver", h5["resolution_correct_margin_norm"].mean(), -0.001)
    delta = h5["delta_resolution_minus_homonym_norm"]
    close("H5 mean resolver-minus-homonym change", delta.mean(), 0.128)
    count("H5 observations moving toward the resolved sense", f"{int((delta > 0).sum())}/{len(h5)}", "605/784")
    primed = is_true(h5["primed_at_homonym"])
    crossed = is_true(h5["successful_primed_to_correct_transition"])
    count("H5 boundary crossings among initially primed", f"{int(crossed.sum())}/{int(primed.sum())}", "234/580")
    for arch, reference in (("encoder", "111/262"), ("decoder", "123/318")):
        mask = h5["arch_type"] == arch
        count(f"H5 {arch} crossings", f"{int((crossed & mask).sum())}/{int((primed & mask).sum())}", reference)
    for direction, reference in (("0_to_1", "125/256"), ("1_to_0", "109/324")):
        mask = h5["direction"] == direction
        count(f"H5 crossings, direction {direction}", f"{int((crossed & mask).sum())}/{int((primed & mask).sum())}", reference)
    cost = h5["garden_path_cost_vs_matched_control_norm"]
    close("H5 conflict endpoint minus matched control", cost.mean(), -0.140)
    close("H5 conflict minus control, encoders", cost[h5["arch_type"] == "encoder"].mean(), -0.098)
    close("H5 conflict minus control, decoders", cost[h5["arch_type"] == "decoder"].mean(), -0.182)
    close("H5 conflict endpoint minus resolver-only baseline",
          h5["delta_resolution_minus_isolated_resolver_norm"].mean(), -0.056)
    cells = h5.groupby(["model", "word"])
    count("H5 cells with conflict below matched control", int((cells[cost.name].mean() < 0).sum()), 55)
    count("H5 cells with positive mean update", int((cells[delta.name].mean() > 0).sum()), 54)
    count("H5 endpoints on the resolved side at threshold 0",
          f"{int((h5['resolution_correct_margin_norm'] > 0).sum())}/784", "404/784")
    by_word = h5.groupby("word")[delta.name].mean()
    close("H5 largest word-level update (bank)", by_word["bank"], 0.181)
    close("H5 smallest word-level update (spring)", by_word["spring"], 0.068)

    associations = table("cross_hypothesis_associations")
    close("Cross-analysis: H0 lean vs H1 adequacy, Spearman rho", associations.iloc[0]["spearman_rho"], -0.143)

    aggregate = table("h5_aggregate")
    recomputed = cells["resolution_correct_margin_norm"].mean().round(4)
    stored = aggregate.set_index(["model", "word"])["mean_resolution_correct_margin_norm"]
    check("h5_aggregate agrees with h5_sentence_level", np.allclose(recomputed.sort_index(), stored.sort_index(), atol=1.5e-4))


def check_trajectories() -> None:
    from analysis import plot_h3_context_trajectory, plot_h5_revision_trajectory

    h3, h5 = table("trajectory_h3_context"), table("trajectory_h5_revision")
    for model, word in TRAJECTORY_EXAMPLES:
        records = pd.DataFrame(plot_h3_context_trajectory._item_records(layout.load_h3_trajectory_states(model, word)))
        stored = h3[(h3["model"] == model) & (h3["word"] == word)]
        check(f"H3 trajectory margins reproduce from released states ({model})",
              np.allclose(records["resolved_margin_norm"], stored["resolved_margin_norm"], atol=1e-5)
              and np.allclose(records["homonym_margin_norm"], stored["homonym_margin_norm"], atol=1e-5))
        records = pd.DataFrame(plot_h5_revision_trajectory._item_records(layout.load_h5_trajectory_states(model, word)))
        stored = h5[(h5["model"] == model) & (h5["word"] == word)]
        check(f"H5 trajectory margins reproduce from released states ({model})",
              all(np.allclose(records[c], stored[c], atol=1e-5)
                  for c in ("prime_margin_norm", "homonym_margin_norm", "resolution_margin_norm")))


# --------------------------------------------------------------------------- web exports

def _load_web(name: str):
    return json.loads((ROOT / "web_export" / name).read_text(encoding="utf-8"))


def _finite(obj) -> bool:
    if isinstance(obj, float):
        return np.isfinite(obj)
    if isinstance(obj, dict):
        return all(_finite(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_finite(v) for v in obj)
    return True


def check_web_existing() -> None:
    """The four table-derived exports must equal a fresh export of the released tables."""
    from scripts import export_web_data as web

    for name in ("layer_separation.json", "revision_comparison.json", "prior_heatmap.json",
                 "revision_trajectories.json", "meta.json", "trajectory_examples.json"):
        fresh = json.dumps(web.EXPORTS[name](), ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        on_disk = (ROOT / "web_export" / name).read_text(encoding="utf-8")
        check(f"web_export/{name} equals a fresh export from the released tables", fresh == on_disk)
    prior = _load_web("prior_heatmap.json")
    meta = _load_web("meta.json")
    check("prior heatmap: 8 x 7 matrices for bare word and carriers, with model and word order",
          prior["models"] == MODEL_KEYS and prior["words"] == WORDS
          and all(np.asarray(prior[k]).shape == (8, 7) for k in ("bare_word_lean", "carrier_mean_lean",
                                                                 "carrier_mean_abs_lean", "carrier_direction_consistency")))
    check("prior heatmap: five labelled carriers per cell; sense labels available in meta.json",
          all(len(prior["carriers"][m][w]) == 5 for m in MODEL_KEYS for w in WORDS)
          and all(len(entry["senses"]) == 2 for entry in meta["words"]))


def check_web_geometry() -> None:
    from scripts.build_geometry_exports import N_COMPONENTS, stable_hash

    geometry = _load_web("geometry_by_layer.json")
    profiles = table("h1_layer_profiles")
    separation = _load_web("layer_separation.json")
    stimuli = pd.read_csv(ROOT / "data" / "stimuli" / "profiling_sentences.csv").sort_values(["word", "profile_index"])
    cells = geometry["cells"]
    check("geometry: status preliminary, projection metadata present",
          geometry["status"] == "preliminary" and geometry["projection"]["n_components"] == N_COMPONENTS == 3
          and geometry["projection"]["coordinate_kind"] == "within-layer standardised display coordinates in one shared PCA basis"
          and geometry["projection"]["pca_solver"] == "full" and "random_state" in geometry["projection"])
    check("geometry: all 56 model-homonym combinations present",
          sorted(cells) == sorted(MODEL_KEYS) and all(list(cells[m]) == WORDS for m in MODEL_KEYS))
    n_layers = 0
    problems = {"layers": [], "points": [], "labels": [], "metrics": [], "hash": [], "basis": [], "centroid": []}
    for model in MODEL_KEYS:
        for word in WORDS:
            cell = cells[model][word]
            rows = profiles[(profiles["model"] == model) & (profiles["word"] == word)].sort_values("Layer")
            layers = [entry["layer"] for entry in cell["layers"]]
            n_layers += len(layers)
            name = f"{model}/{word}"
            if layers != list(rows["Layer"]) or layers != list(range(cell["last_layer"] + 1)) \
                    or layers != separation["cells"][model][word]["layers"]:
                problems["layers"].append(name)
            senses = list(stimuli.loc[stimuli["word"] == word, "sense"])
            for entry, row in zip(cell["layers"], rows.itertuples()):
                points = np.asarray(entry["profile_points"])
                if points.shape != (40, 3) or entry["n_profile_points"] != 40:
                    problems["points"].append(name)
                if entry["profile_senses"] != senses:
                    problems["labels"].append(name)
                if entry["gdv_full_space"] != float(row.GDV) or entry["fraction_adequate_full_space"] != float(row.FractionAdequate):
                    problems["metrics"].append(name)
                labels = np.asarray(entry["profile_senses"])
                centroids = np.array([points[labels == s].mean(axis=0) for s in (0, 1)])
                if not np.allclose(centroids, entry["sense_centroids"], atol=1e-4):
                    problems["centroid"].append(name)
            if stable_hash([entry["profile_points"] for entry in cell["layers"]]) != cell["coordinates_sha256"] \
                    or len(cell["source_activations_sha256"]) != 64 or len(cell["explained_variance_ratio"]) != 3:
                problems["hash"].append(name)
            # A shared basis fitted on per-layer centred data leaves every layer centred on the origin
            # (up to rounding), and gives more than two distinct layer clouds.
            means = np.array([np.mean(entry["profile_points"], axis=0) for entry in cell["layers"]])
            distinct = {json.dumps(entry["profile_points"]) for entry in cell["layers"]}
            if np.abs(means).max() > 5e-3 or len(distinct) < len(layers) - 1:
                problems["basis"].append(name)
    count("geometry: total model-homonym-layer entries equal the released layer table", n_layers, len(profiles))
    check("geometry: every cell has every layer from 0 to the last, matching layer_separation.json", not problems["layers"], str(problems["layers"][:3]))
    check("geometry: every layer has exactly 40 three-dimensional points", not problems["points"], str(problems["points"][:3]))
    check("geometry: sense labels equal the released profiling stimuli at every layer", not problems["labels"], str(problems["labels"][:3]))
    check("geometry: full-space GDV and adequacy equal h1_layer_profiles.csv exactly", not problems["metrics"], str(problems["metrics"][:3]))
    check("geometry: sense centroids are the means of the published points", not problems["centroid"], str(problems["centroid"][:3]))
    check("geometry: coordinate hashes reproduce; source and variance metadata present", not problems["hash"], str(problems["hash"][:3]))
    check("geometry: one shared basis per cell and distinct coordinates per layer (not two snapshots)", not problems["basis"], str(problems["basis"][:3]))
    check("geometry: all values finite", _finite(cells))


def check_web_landscapes() -> None:
    from scripts.build_geometry_exports import (CONTOUR_FRACTIONS, GRID_SIZE, gaussian_kde_grid, scott_bandwidth,
                                                significant, stable_hash)

    data = _load_web("garden_path_landscapes.json")
    cells = data["cells"]
    stimuli = pd.read_csv(ROOT / "data" / "stimuli" / "conflict_items.csv", keep_default_na=False).set_index("item_id")
    profile_stimuli = pd.read_csv(ROOT / "data" / "stimuli" / "profiling_sentences.csv").sort_values(["word", "profile_index"])
    h5 = table("h5_sentence_level").set_index(["model", "sentence_id"])
    layers = table("h5_aggregate").set_index(["model", "word"])["layer_used"]
    check("landscapes: status preliminary; stages are prime, homonym, resolver",
          data["status"] == "preliminary" and data["stage_names"] == ["prime", "homonym", "resolver"])
    check("landscapes: all 56 model-homonym combinations present",
          sorted(cells) == sorted(MODEL_KEYS) and all(list(cells[m]) == WORDS for m in MODEL_KEYS))
    bad = {k: [] for k in ("layer", "profile", "items", "text", "senses", "shape", "kde", "hash", "side", "flag")}
    margin_diff, control_diff, n_items, n_unavailable = [], [], 0, 0
    for model in MODEL_KEYS:
        for word in WORDS:
            cell, name = cells[model][word], f"{model}/{word}"
            if cell["analysis_layer"] != int(layers.loc[(model, word)]):
                bad["layer"].append(name)
            points, senses = np.asarray(cell["profile"]["points"]), np.asarray(cell["profile"]["senses"])
            if points.shape != (40, 3) or senses.tolist() != list(profile_stimuli.loc[profile_stimuli["word"] == word, "sense"]):
                bad["profile"].append(name)
            n_items += len(cell["items"]); n_unavailable += len(cell["unavailable_items"])
            ids = [item["item_id"] for item in cell["items"]] + [u["item_id"] for u in cell["unavailable_items"]]
            if sorted(ids) != sorted(stimuli.index[stimuli["word"] == word]) or cell["n_items"] != len(cell["items"]):
                bad["items"].append(name)
            extent = [points]
            for item in cell["items"]:
                stim = stimuli.loc[item["item_id"]]
                if item["target_sense"] != int(stim["correct_sense"]) or item["primed_sense"] != int(stim["primed_sense"]):
                    bad["senses"].append(item["item_id"])
                for path in (item["conflicting"], item["coherent_control"]):
                    if path["stage_names"] != ["prime", "homonym", "resolver"] or np.asarray(path["points"]).shape != (3, 3) \
                            or len(path["full_space_correct_margins"]) != 3 or len(path["stage_text"]) != 3:
                        bad["shape"].append(item["item_id"])
                    extent.append(np.asarray(path["points"]))
                expected = [stim[f"conflicting_{s}_prefix"].rsplit("\n\n", 1)[0] for s in ("prime", "homonym", "resolution")]
                control = item["coherent_control"]["stage_text"]
                sentence = stim["coherent_control_sentence"]
                if item["conflicting"]["stage_text"] != expected \
                        or control[2] != stim["coherent_control_resolution_prefix"].rsplit("\n\n", 1)[0] \
                        or not (sentence.startswith(control[0]) and sentence.startswith(control[1]) and sentence.startswith(control[2])) \
                        or not (len(control[0]) < len(control[1]) < len(control[2])) \
                        or item["coherent_control_sentence"] != sentence or item["conflicting_sentence"] != stim["conflicting_sentence"]:
                    bad["text"].append(item["item_id"])
                row = h5.loc[(model, item["item_id"])]
                released = [row["prime_correct_margin_norm"], row["homonym_correct_margin_norm"], row["resolution_correct_margin_norm"]]
                margin_diff += list(np.abs(np.asarray(item["conflicting"]["full_space_correct_margins"]) - released))
                coherent = item["coherent_control"]
                control_diff.append(abs(coherent["full_space_correct_margins"][2] - row["matched_control_correct_margin_norm"]))
                if abs(coherent["released_table_resolver_margin"] - row["matched_control_correct_margin_norm"]) > 1e-12 \
                        or coherent["resolver_side_differs_from_released_table"] != (
                            (coherent["full_space_correct_margins"][2] > 0) != (row["matched_control_correct_margin_norm"] > 0)):
                    bad["flag"].append(item["item_id"])
                for value, reference in zip(item["conflicting"]["full_space_correct_margins"], released):
                    if abs(reference) > MARGIN_TOLERANCE and (value > 0) != (reference > 0):
                        bad["side"].append(item["item_id"])
            land = cell["landscape"]
            x, y = np.asarray(land["x"]), np.asarray(land["y"])
            d0 = gaussian_kde_grid(points[senses == 0, :2], x, y, land["bandwidth"])
            d1 = gaussian_kde_grid(points[senses == 1, :2], x, y, land["bandwidth"])
            all_xy = np.vstack(extent)[:, :2]
            if land["grid_size"] != [GRID_SIZE, GRID_SIZE] or len(x) != GRID_SIZE or len(y) != GRID_SIZE \
                    or abs(land["bandwidth"] - scott_bandwidth(points[:, :2], senses)) > 1e-4 \
                    or significant(d0) != land["sense_0_density"] or significant(d1) != land["sense_1_density"] \
                    or significant(0.5 * (d0 + d1)) != land["total_density"] \
                    or significant(np.asarray(CONTOUR_FRACTIONS) * (0.5 * (d0 + d1)).max()) != land["contour_levels"] \
                    or all_xy[:, 0].min() < x[0] or all_xy[:, 0].max() > x[-1] or all_xy[:, 1].min() < y[0] or all_xy[:, 1].max() > y[-1]:
                bad["kde"].append(name)
            coords = [cell["profile"]["points"]] + [[i["conflicting"]["points"], i["coherent_control"]["points"]] for i in cell["items"]]
            if stable_hash(coords) != cell["projection"]["coordinates_sha256"] or len(cell["projection"]["source_states_sha256"]) != 64 \
                    or len(cell["projection"]["explained_variance_ratio"]) != 3:
                bad["hash"].append(name)
    count("landscapes: published item pairs plus unavailable items", n_items + n_unavailable, 784)
    print(f"      landscapes: {n_items} item pairs published, {n_unavailable} marked unavailable")
    check("landscapes: analysis layer equals the layer used by H5", not bad["layer"], str(bad["layer"][:3]))
    check("landscapes: 40 labelled profile points per cell", not bad["profile"], str(bad["profile"][:3]))
    check("landscapes: every stimulus item is either published or listed as unavailable", not bad["items"], str(bad["items"][:3]))
    check("landscapes: target and primed senses equal the released stimuli", not bad["senses"], str(bad["senses"][:3]))
    check("landscapes: each item has one conflicting and one coherent path with exactly three stages", not bad["shape"], str(bad["shape"][:3]))
    check("landscapes: stage texts are the released prefixes; control stages are prefixes of the matched control sentence", not bad["text"], str(bad["text"][:3]))
    check("landscapes: density grids recompute exactly from the published points, bandwidth and grid", not bad["kde"], str(bad["kde"][:3]))
    check("landscapes: coordinate hashes reproduce; source and variance metadata present", not bad["hash"], str(bad["hash"][:3]))
    check(f"landscapes: conflicting-path margins reproduce h5_sentence_level.csv within {MARGIN_TOLERANCE} "
          f"(max difference {max(margin_diff):.5f})", max(margin_diff) <= MARGIN_TOLERANCE)
    # The coherent control is a single-pass path and is NOT expected to reproduce the
    # released control margin exactly (documented reproducibility discrepancy). The
    # checks are that the discrepancy is recorded truthfully and stays small.
    note = data["numerical_note"]
    control_diff = np.asarray(control_diff)
    flips = sum(item["coherent_control"]["resolver_side_differs_from_released_table"]
                for by_word in cells.values() for cell in by_word.values() for item in cell["items"])
    print(f"      landscapes: control resolver vs released table: mean |difference| {control_diff.mean():.5f}, "
          f"max {control_diff.max():.5f}, {flips} of {len(control_diff)} on the other side of the boundary")
    check("landscapes: every control path carries the released-table resolver margin and a correct side flag", not bad["flag"], str(bad["flag"][:3]))
    defaults_ok = all(
        any(item["item_id"] == cell["default_item_id"] and not item["coherent_control"]["resolver_side_differs_from_released_table"]
            for item in cell["items"])
        for by_word in cells.values() for cell in by_word.values())
    check("landscapes: every cell has a default item, and it is never one flagged for a changed control side", defaults_ok)
    check("landscapes: recorded control discrepancy statistics match the published values",
          abs(note["max_abs_difference_from_released_table"] - control_diff.max()) < 1.5e-4
          and abs(note["mean_abs_difference_from_released_table"] - control_diff.mean()) < 1.5e-4
          and note["n_resolver_states_on_other_side_than_released_table"] == flips
          and note["n_control_resolver_readings"] == len(control_diff))
    check("landscapes: control discrepancy from the released table stays below 0.05 for every item", control_diff.max() < 0.05)
    check("landscapes: no conflicting-path state changes side of the boundary relative to the released table", not bad["side"], str(bad["side"][:3]))
    check("landscapes: all values finite", _finite(cells))


def check_web_hygiene() -> None:
    import re

    web = ROOT / "web_export"
    files = sorted(p for p in web.rglob("*") if p.is_file())
    check("web_export contains only JSON files and its README",
          all(p.suffix == ".json" or p.name == "README.md" for p in files), str([p.name for p in files if p.suffix != ".json"]))
    text = "".join(p.read_text(encoding="utf-8") for p in files if p.suffix == ".json") + MANIFEST_PATH.read_text(encoding="utf-8")
    hits = sorted(set(re.findall(r"/anvme[^\s\"]*|/home/[^\s\"]*|\biwi\d[a-z0-9]{3,}\b|hf_[A-Za-z0-9]{20,}|\.npy", text)))
    check("web exports and manifest contain no cluster paths, usernames, tokens or .npy references", not hits, str(hits[:3]))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    listed = {entry["path"]: entry for entry in manifest["files"]}
    for p in files:
        if p.suffix == ".json":
            check(f"manifest lists web_export/{p.name}", f"web_export/{p.name}" in listed)
    for name, key in (("geometry_by_layer.json", "layers"), ("garden_path_landscapes.json", "item_pairs")):
        entry = listed.get(f"web_export/{name}", {})
        check(f"manifest records counts and source hashes for {name}",
              entry.get("counts", {}).get("cells") == 56 and key in entry.get("counts", {}) and len(entry.get("source_hashes", {})) == 56)

CHECKS: List[Callable[[], None]] = [
    check_manifest, check_config_and_design, check_h0, check_h1_h2, check_geometry_example,
    check_h3_h4, check_h5, check_trajectories,
    check_web_existing, check_web_geometry, check_web_landscapes, check_web_hygiene,
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true", help="print passing checks too")
    args = parser.parse_args()
    for function in CHECKS:
        try:
            function()
        except Exception as exc:  # a crashed group is a failed check, not a silent skip
            _results.append((False, f"{function.__name__} raised {type(exc).__name__}: {exc}"))
    failed = [label for ok, label in _results if not ok]
    for ok, label in _results:
        if args.verbose or not ok:
            print(("PASS  " if ok else "FAIL  ") + label)
    print(f"{len(_results) - len(failed)}/{len(_results)} checks passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
