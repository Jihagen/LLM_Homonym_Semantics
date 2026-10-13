"""Build the two display-geometry web exports from saved hidden states.

    python -m scripts.build_geometry_exports \
        --results-dir results --h5-matched-dir results/h5_matched

Maintainer step. Unlike ``scripts.export_web_data`` it cannot run from the
released tables alone, because it projects hidden states that are too large to
release:

  web_export/geometry_by_layer.json
      needs results/activations/<word>/<model>/layer_<n>.h5 (written by run_h2.py)
  web_export/garden_path_landscapes.json
      needs the output of scripts.extract_h5_matched_states

Only three-dimensional display coordinates, density grids and metadata are
written. The coordinates are a visualisation: every GDV, adequacy and margin
value in the two files is computed in the full hidden space, and is either
copied from the released tables (geometry) or checked against them
(garden path). Re-running on the same inputs rewrites identical files.

Needs numpy, pandas, h5py and scikit-learn; no model and no GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from analysis.plot_centroid_separation import center_scale_params
from experiments.adequacy import normalized_adequacy_margin
from scripts.release_layout import DATA_VERSION, MODEL_KEYS, PROCESSED_DIR, ROOT, WORDS

WEB_DIR = ROOT / "web_export"
GEOMETRY_FILE = "geometry_by_layer.json"
LANDSCAPE_FILE = "garden_path_landscapes.json"
STATUS = "preliminary"

N_COMPONENTS = 3
PCA_SOLVER = "full"          # exact LAPACK SVD: no random state is involved
COORD_DECIMALS = 4
PIPELINE_STAGES = ("prime", "homonym", "resolution")   # names used by the pipeline
STAGE_NAMES = ["prime", "homonym", "resolver"]         # names used in the export
GRID_SIZE = 33
GRID_PADDING = 0.10
CONTOUR_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 0.9]
DENSITY_SIGNIFICANT = 5


# --------------------------------------------------------------------------- helpers

def round_list(values, decimals: int = COORD_DECIMALS):
    array = np.round(np.asarray(values, dtype=np.float64), decimals) + 0.0   # +0.0 turns -0.0 into 0.0
    return array.tolist()


def significant(values, digits: int = DENSITY_SIGNIFICANT):
    array = np.asarray(values, dtype=np.float64)
    return np.vectorize(lambda v: float(f"{v:.{digits}g}"), otypes=[float])(array).tolist()


def stable_hash(obj) -> str:
    """SHA-256 of the canonical JSON form of ``obj``."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode() + str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def fit_pca(reference: np.ndarray) -> PCA:
    return PCA(n_components=N_COMPONENTS, svd_solver=PCA_SOLVER).fit(reference)


def write_json(name: str, payload: Dict) -> Path:
    path = WEB_DIR / name
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    path.write_text(text + "\n", encoding="utf-8")
    return path


def table(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / f"{name}.csv", **kwargs)


# --------------------------------------------------------------------------- density landscape

def scott_bandwidth(points_2d: np.ndarray, senses: np.ndarray) -> float:
    """One isotropic Gaussian bandwidth per cell: Scott's factor n^(-1/(d+4))
    with d = 2 and n = points per sense, times the pooled within-sense standard
    deviation of the two plotted coordinates."""
    residuals = np.vstack([points_2d[senses == s] - points_2d[senses == s].mean(axis=0) for s in (0, 1)])
    sigma = float(np.sqrt((residuals ** 2).mean()))
    n = int(min((senses == 0).sum(), (senses == 1).sum()))
    return max(sigma * n ** (-1.0 / 6.0), 1e-6)


def gaussian_kde_grid(points_2d: np.ndarray, x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    """Mean of isotropic Gaussian kernels; rows index y, columns index x.
    Integrates to 1 over the plane."""
    gx, gy = np.meshgrid(x, y)
    d2 = (gx[..., None] - points_2d[:, 0]) ** 2 + (gy[..., None] - points_2d[:, 1]) ** 2
    return np.exp(-d2 / (2.0 * bandwidth ** 2)).mean(axis=-1) / (2.0 * np.pi * bandwidth ** 2)


def landscape(profile_xyz: Sequence, senses: Sequence[int], extent_points: Sequence, bandwidth: float) -> Dict:
    """Density landscape from the *published* (rounded) coordinates, so that it
    can be recomputed exactly from the JSON file."""
    profile = np.asarray(profile_xyz, dtype=np.float64)[:, :2]
    senses = np.asarray(senses)
    extent = np.asarray(extent_points, dtype=np.float64)[:, :2]
    lo, hi = extent.min(axis=0), extent.max(axis=0)
    pad = GRID_PADDING * np.maximum(hi - lo, 1e-9) + bandwidth
    x = np.round(np.linspace(lo[0] - pad[0], hi[0] + pad[0], GRID_SIZE), COORD_DECIMALS)
    y = np.round(np.linspace(lo[1] - pad[1], hi[1] + pad[1], GRID_SIZE), COORD_DECIMALS)
    d0 = gaussian_kde_grid(profile[senses == 0], x, y, bandwidth)
    d1 = gaussian_kde_grid(profile[senses == 1], x, y, bandwidth)
    total = 0.5 * (d0 + d1)
    return {
        "coordinate_system": "x = PC1, y = PC2 of the profile projection; height is KDE density of the "
                             "profile points, not PC3 and not a model quantity",
        "kernel": "isotropic Gaussian",
        "bandwidth": bandwidth,
        "bandwidth_rule": "Scott factor n^(-1/6) (n = 20 points per sense) times the pooled within-sense "
                          "standard deviation of PC1 and PC2",
        "grid_size": [GRID_SIZE, GRID_SIZE],
        "array_layout": "density[iy][ix] is the value at (x[ix], y[iy])",
        "x": x.tolist(),
        "y": y.tolist(),
        "sense_0_density": significant(d0),
        "sense_1_density": significant(d1),
        "total_density": significant(total),
        "contour_levels": significant(np.asarray(CONTOUR_FRACTIONS) * float(total.max())),
        "contour_level_rule": f"{CONTOUR_FRACTIONS} times the maximum of total_density",
    }


# --------------------------------------------------------------------------- geometry by layer

def load_layer(results: Path, word: str, model: str, layer: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    import h5py

    with h5py.File(results / "activations" / word / model / f"layer_{layer}.h5", "r") as handle:
        X = handle["X"][:].astype(np.float32)
        labels = handle["labels"][:].astype(np.int64)
        sentences = [s.decode() if isinstance(s, bytes) else str(s) for s in handle["sentences"][:]]
    return X, labels, sentences


def geometry_cell(results: Path, model: str, word: str, profile_rows: pd.DataFrame, stimuli: pd.DataFrame) -> Tuple[Dict, Dict]:
    layers = [int(v) for v in profile_rows["Layer"]]
    last = max(layers)
    expected_sentences = list(stimuli["sentence"])
    expected_senses = [int(v) for v in stimuli["sense"]]

    raw, normalised, labels = [], [], None
    for layer in layers:
        X, layer_labels, sentences = load_layer(results, word, model, layer)
        if sentences != expected_sentences or layer_labels.tolist() != expected_senses:
            raise ValueError(f"{model}/{word} layer {layer}: cache order differs from data/stimuli/profiling_sentences.csv")
        labels = layer_labels
        center, scale = center_scale_params(X.astype(np.float64), layer_labels)
        raw.append(X)
        normalised.append((X.astype(np.float64) - center) / scale)

    pca = fit_pca(np.vstack(normalised))
    entries = []
    for layer, norm_X, row in zip(layers, normalised, profile_rows.itertuples()):
        points = np.asarray(round_list(pca.transform(norm_X)))
        entries.append({
            "layer": layer,
            "relative_depth": round(layer / last, 4),
            "profile_points": points.tolist(),
            "profile_senses": labels.tolist(),
            "sense_centroids": [round_list(points[labels == s].mean(axis=0)) for s in (0, 1)],
            "n_profile_points": int(len(points)),
            "gdv_full_space": float(row.GDV),
            "fraction_adequate_full_space": float(row.FractionAdequate),
        })
    cell = {
        "n_layers": len(layers),
        "last_layer": last,
        "explained_variance_ratio": round_list(pca.explained_variance_ratio_, 6),
        "source_activations_sha256": array_hash(np.stack(raw), labels),
        "coordinates_sha256": stable_hash([entry["profile_points"] for entry in entries]),
        "layers": entries,
    }
    # Recompute the full-space metrics from the same activations, for the build report.
    from experiments.adequacy import leave_one_out_adequacy_margins
    from experiments.gdv_experiments import compute_gdv

    gdv = np.array([compute_gdv(X, labels) for X in raw])
    adequate = np.array([(leave_one_out_adequacy_margins(X, labels)[0] > 0).mean() for X in raw])
    report = {
        "max_abs_gdv_difference": float(np.abs(gdv - profile_rows["GDV"].to_numpy(float)).max()),
        "max_abs_adequacy_difference": float(np.abs(adequate - profile_rows["FractionAdequate"].to_numpy(float)).max()),
    }
    return cell, report


def build_geometry(results: Path) -> Tuple[Dict, Dict]:
    profiles = table("h1_layer_profiles")
    stimuli = pd.read_csv(ROOT / "data" / "stimuli" / "profiling_sentences.csv")
    cells, reports = {}, {}
    for model in MODEL_KEYS:
        cells[model] = {}
        for word in WORDS:
            rows = profiles[(profiles["model"] == model) & (profiles["word"] == word)].sort_values("Layer")
            stim = stimuli[stimuli["word"] == word].sort_values("profile_index")
            cells[model][word], reports[f"{model}/{word}"] = geometry_cell(results, model, word, rows, stim)
    payload = {
        "data_version": DATA_VERSION,
        "status": STATUS,
        "description": "Display coordinates of the 40 sense-labelled profiling sentences at the homonym "
                       "token, for every model, homonym and hidden-state layer, in one projection per "
                       "model and homonym that is shared by all of its layers.",
        "kind": "within-layer standardised display coordinates in one shared PCA basis; not an analysis result",
        "comparability": "The coordinates support inspecting how well the two sense clusters separate at each "
                         "depth. Because every layer is standardised on its own, absolute cloud size is not "
                         "comparable between layers. Full-space GDV, adequacy and margins remain the analysis "
                         "metrics.",
        "source_tables": ["data/processed/h1_layer_profiles.csv", "data/stimuli/profiling_sentences.csv"],
        "source_states": "per-layer homonym-token hidden states of the profiling sentences "
                         "(results/activations/, not released; identified by source_activations_sha256)",
        "projection": {
            "coordinate_kind": "within-layer standardised display coordinates in one shared PCA basis",
            "method": "PCA fitted once per model and homonym on the profile states of all layers pooled, "
                      "after per-layer standardisation; every layer is then transformed with that one basis",
            "per_layer_standardisation": "subtract the midpoint of the two sense centroids of that layer "
                                       "(equal to the layer mean, the senses being balanced) and divide by "
                                       "the mean distance of the points to their own sense centroid; a "
                                       "layer whose 40 states are identical is left unscaled",
            "unit": "one unit is the typical within-sense spread of the layer in the full hidden space",
            "n_components": N_COMPONENTS,
            "pca_solver": PCA_SOLVER,
            "random_state": None,
            "sign_convention": "scikit-learn svd_flip (deterministic)",
            "coordinate_decimals": COORD_DECIMALS,
            "library": "scikit-learn PCA",
        },
        "full_space_metrics": "gdv_full_space and fraction_adequate_full_space are copied from "
                              "data/processed/h1_layer_profiles.csv; they are computed in the full hidden "
                              "space and are not derived from the coordinates",
        "point_order": "profile_points[i] is the sentence with profile_index i in "
                       "data/stimuli/profiling_sentences.csv, at every layer",
        "cells": cells,
    }
    return payload, reports


# --------------------------------------------------------------------------- garden path

def load_matched(directory: Path, model: str, word: str) -> Dict:
    with np.load(directory / f"{model}_{word}.npz", allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def strip_sentinel(text: str) -> str:
    return text.rsplit("\n\n", 1)[0]


def landscape_cell(data: Dict, items: pd.DataFrame, released: pd.DataFrame) -> Tuple[Dict, Dict]:
    profile_H = data["profile_H"].astype(np.float64)
    senses = data["profile_senses"]
    centroids = {s: profile_H[senses == s].mean(axis=0) for s in (0, 1)}
    center, scale = center_scale_params(profile_H, senses)
    pca = fit_pca((profile_H - center) / scale)

    def project(H):
        return np.asarray(round_list(pca.transform((np.asarray(H, dtype=np.float64) - center) / scale)))

    profile_points = project(profile_H)
    item_ids = [str(v) for v in data["item_ids"]]

    def group(prefix: str) -> Dict:
        return {
            (int(i), str(stage)): (H, str(text))
            for i, stage, H, text in zip(data[f"{prefix}_item_index"], data[f"{prefix}_stage"],
                                         data[f"{prefix}_H"], data[f"{prefix}_text"])
        }

    h5, control = group("h5"), group("control")
    by_id = items.set_index("item_id")
    released = released.set_index("sentence_id")

    exported, unavailable, extent = [], [], [profile_points]
    differences = {"conflicting": [], "control_resolver": [], "control_two_passes": [], "reference_reading_vs_table": []}
    for i, item_id in enumerate(item_ids):
        stimulus = by_id.loc[item_id]
        correct, primed = int(stimulus["correct_sense"]), int(stimulus["primed_sense"])

        def path(states: Dict, expected_texts: List[str]):
            if any((i, stage) not in states for stage in PIPELINE_STAGES):
                return None
            H = np.stack([states[(i, stage)][0] for stage in PIPELINE_STAGES]).astype(np.float64)
            texts = [states[(i, stage)][1] for stage in PIPELINE_STAGES]
            if expected_texts is not None and texts != expected_texts:
                raise ValueError(f"{item_id}: extracted prefixes differ from data/stimuli/conflict_items.csv")
            margins = [float(normalized_adequacy_margin(h, centroids[correct], centroids[primed])) for h in H]
            return {
                "stage_names": STAGE_NAMES,
                "stage_text": [strip_sentinel(t) for t in texts],
                "points": project(H).tolist(),
                "full_space_correct_margins": round_list(margins),
            }, margins

        conflicting = path(h5, [stimulus[f"conflicting_{stage}_prefix"] for stage in PIPELINE_STAGES])
        # Each path comes from one forward-pass group and is never assembled from
        # two: the conflicting path from the group that repeats the published H5
        # inputs, the coherent control from the group that reads the control
        # sentence at its three stages.
        coherent = path(control, None)
        if conflicting is None or coherent is None:
            unavailable.append({"item_id": item_id, "reason": "no matched coherent control at all three stages"})
            continue
        if coherent[0]["stage_text"][2] != strip_sentinel(stimulus["coherent_control_resolution_prefix"]):
            raise ValueError(f"{item_id}: control resolver prefix differs from the released stimulus")

        row = released.loc[item_id]
        differences["conflicting"] += [
            abs(m - float(row[f"{stage}_correct_margin_norm"])) for m, stage in zip(conflicting[1], PIPELINE_STAGES)
        ]
        differences["control_resolver"].append(abs(coherent[1][2] - float(row["matched_control_correct_margin_norm"])))
        # The control resolver text was also read in the H5-repeating group (the
        # reading behind the released control margin). It is compared, not used.
        other = h5[(i, "matched_control")]
        if other[1] != control[(i, "resolution")][1]:
            raise ValueError(f"{item_id}: the two control resolver readings are of different texts")
        repeat = float(normalized_adequacy_margin(other[0].astype(np.float64), centroids[correct], centroids[primed]))
        differences["control_two_passes"].append(abs(coherent[1][2] - repeat))
        released_control = float(row["matched_control_correct_margin_norm"])
        differences["reference_reading_vs_table"].append(abs(repeat - released_control))
        coherent[0]["released_table_resolver_margin"] = released_control
        coherent[0]["resolver_side_differs_from_released_table"] = bool((coherent[1][2] > 0) != (released_control > 0))

        exported.append({
            "item_id": item_id,
            "target_sense": correct,
            "primed_sense": primed,
            "resolution_word": str(stimulus["resolution_word"]),
            "conflicting_sentence": str(stimulus["conflicting_sentence"]),
            "coherent_control_sentence": str(stimulus["coherent_control_sentence"]),
            "conflicting": conflicting[0],
            "coherent_control": coherent[0],
        })
        extent += [np.asarray(conflicting[0]["points"]), np.asarray(coherent[0]["points"])]

    bandwidth = round(scott_bandwidth(profile_points[:, :2], senses), COORD_DECIMALS)
    cell = {
        "analysis_layer": int(data["layer"]),
        "projection": {
            "method": "PCA fitted on the 40 sense-profile sentinel states of this model, homonym and layer; "
                      "trajectory states are transformed with the same basis",
            "dimensions": N_COMPONENTS,
            "explained_variance_ratio": round_list(pca.explained_variance_ratio_, 6),
            "explained_variance_scope": "share of the variance of the 40 profile states only",
            "source_states_sha256": array_hash(data["profile_H"], data["h5_H"], data["control_H"]),
            "coordinates_sha256": None,
        },
        "profile": {
            "points": profile_points.tolist(),
            "senses": senses.tolist(),
            "centroids": [round_list(profile_points[senses == s].mean(axis=0)) for s in (0, 1)],
        },
        "n_items": len(exported),
        "default_item_id": next(
            (item["item_id"] for item in exported
             if not item["coherent_control"]["resolver_side_differs_from_released_table"]), None),
        "control_resolver_max_abs_difference_from_released_table":
            round(max(differences["control_resolver"]), COORD_DECIMALS) if differences["control_resolver"] else None,
        "items": exported,
        "unavailable_items": unavailable,
        "landscape": landscape(profile_points, senses, np.vstack(extent), bandwidth),
    }
    cell["projection"]["coordinates_sha256"] = stable_hash(
        [cell["profile"]["points"]] + [[it["conflicting"]["points"], it["coherent_control"]["points"]] for it in exported]
    )
    report = {name: (float(max(values)) if values else None) for name, values in differences.items()}
    report["two_pass_values"] = differences["control_two_passes"]
    return cell, report


def build_landscapes(matched_dir: Path) -> Tuple[Dict, Dict]:
    items = pd.read_csv(ROOT / "data" / "stimuli" / "conflict_items.csv", keep_default_na=False)
    h5 = table("h5_sentence_level", keep_default_na=False)
    layers = table("h5_aggregate").set_index(["model", "word"])["layer_used"]
    cells, reports = {}, {}
    for model in MODEL_KEYS:
        cells[model] = {}
        for word in WORDS:
            data = load_matched(matched_dir, model, word)
            if int(data["layer"]) != int(layers.loc[(model, word)]):
                raise ValueError(f"{model}/{word}: states are not from the H5 analysis layer")
            cells[model][word], reports[f"{model}/{word}"] = landscape_cell(
                data, items[items["word"] == word], h5[(h5["model"] == model) & (h5["word"] == word)]
            )
    payload = {
        "data_version": DATA_VERSION,
        "status": STATUS,
        "description": "For every model and homonym, at the layer used by the H5 analysis: each "
                       "context-conflict item as a three-stage trajectory together with the trajectory of "
                       "its matched coherent control, in one three-dimensional projection, with the "
                       "sense-profile point cloud and a two-dimensional density landscape of that cloud.",
        "kind": "visualisation coordinates with full-space margins; the density landscape is a display aid",
        "source_tables": ["data/stimuli/conflict_items.csv", "data/processed/h5_sentence_level.csv",
                          "data/processed/h5_aggregate.csv"],
        "source_states": "sentinel hidden states from scripts/extract_h5_matched_states.py (not released; "
                         "identified by source_states_sha256). The conflicting path repeats the published "
                         "H5 inputs; the coherent control at the prime and homonym stages is new.",
        "stage_names": STAGE_NAMES,
        "stage_definition": "prime: text before the homonym; homonym: text through the homonym; resolver: "
                            "text through the resolution word. Each prefix is run on its own with the "
                            "sentinel appended and read at the sentinel. 'resolver' is the stage called "
                            "'resolution' in the other files.",
        "sentinel": "probe",
        "projection": {
            "normalisation": "subtract the midpoint of the two profile sense centroids and divide by the mean "
                             "distance of the profile states to their own sense centroid",
            "unit": "one unit is the typical within-sense spread of the profile states in the full hidden space",
            "n_components": N_COMPONENTS,
            "pca_solver": PCA_SOLVER,
            "random_state": None,
            "sign_convention": "scikit-learn svd_flip (deterministic)",
            "coordinate_decimals": COORD_DECIMALS,
            "library": "scikit-learn PCA",
        },
        "default_item_rule": "default_item_id is the first item of the cell, in stimulus order, whose coherent "
                             "control lies on the same side of the boundary as in the released table. Items with "
                             "resolver_side_differs_from_released_table = true must not be used as a default or "
                             "showcase example.",
        "margin_definition": "full_space_correct_margins are normalised nearest-centroid margins in the full "
                             "hidden space, positive toward target_sense, against the profile sense centroids "
                             "at the sentinel. They are not computed from the coordinates.",
        "numerical_note": None,
        "cells": cells,
    }
    repeats = np.concatenate([r["two_pass_values"] for r in reports.values()])
    flips = sum(item["coherent_control"]["resolver_side_differs_from_released_table"]
                for by_word in cells.values() for cell in by_word.values() for item in cell["items"])
    payload["numerical_note"] = {
        "what": "Reproducibility discrepancy. Each published path comes from one forward-pass group. The "
                "conflicting path repeats the published H5 inputs and reproduces the released table at all "
                "three stages. The coherent control is read in a separate group, because the published H5 "
                "run read the control at the resolver stage only. Its resolver margin therefore differs "
                "slightly from matched_control_correct_margin_norm in h5_sentence_level.csv: a state depends "
                "on which texts share its batch under the pipeline's bfloat16 arithmetic (see STATUS.md).",
        "n_control_resolver_readings": int(len(repeats)),
        "mean_abs_difference_from_released_table": round(float(repeats.mean()), 5),
        "max_abs_difference_from_released_table": round(float(repeats.max()), 5),
        "share_above_0.002": round(float((repeats > 0.002).mean()), 4),
        "n_resolver_states_on_other_side_than_released_table": int(flips),
        "per_item_fields": ["coherent_control.released_table_resolver_margin",
                            "coherent_control.resolver_side_differs_from_released_table"],
        "per_cell_maximum": "control_resolver_max_abs_difference_from_released_table",
    }
    return payload, reports


# --------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results", help="pipeline results with activations/")
    parser.add_argument("--h5-matched-dir", default="results/h5_matched")
    parser.add_argument("--only", choices=["geometry", "landscapes"], default=None)
    args = parser.parse_args()

    if args.only in (None, "geometry"):
        payload, reports = build_geometry(Path(args.results_dir))
        path = write_json(GEOMETRY_FILE, payload)
        n_layers = sum(cell["n_layers"] for by_word in payload["cells"].values() for cell in by_word.values())
        print(f"  web_export/{GEOMETRY_FILE}  {path.stat().st_size / 1e6:.2f} MB, 56 cells, {n_layers} layers")
        print(f"    recomputed from activations vs released table: max |GDV difference| "
              f"{max(r['max_abs_gdv_difference'] for r in reports.values()):.2e}, max |adequacy difference| "
              f"{max(r['max_abs_adequacy_difference'] for r in reports.values()):.2e}")

    if args.only in (None, "landscapes"):
        payload, reports = build_landscapes(Path(args.h5_matched_dir))
        path = write_json(LANDSCAPE_FILE, payload)
        n_items = sum(cell["n_items"] for by_word in payload["cells"].values() for cell in by_word.values())
        n_missing = sum(len(cell["unavailable_items"]) for by_word in payload["cells"].values() for cell in by_word.values())
        print(f"  web_export/{LANDSCAPE_FILE}  {path.stat().st_size / 1e6:.2f} MB, 56 cells, "
              f"{n_items} item pairs, {n_missing} unavailable")
        labels = {"conflicting": "conflicting path (single pass) vs released H5 table",
                  "reference_reading_vs_table": "control resolver, H5-repeating pass vs released H5 table (not published)",
                  "control_resolver": "control resolver, published single-pass path vs released H5 table"}
        for name, label in labels.items():
            values = [r[name] for r in reports.values() if r[name] is not None]
            print(f"    max |margin difference|, {label}: {max(values):.2e}")


if __name__ == "__main__":
    main()
