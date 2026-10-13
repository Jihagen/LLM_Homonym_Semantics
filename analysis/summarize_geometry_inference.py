"""Reproduce crossed model-by-word intervals for the corrected H1/H2 estimands."""

import csv
from pathlib import Path

import numpy as np

from utils.model_registry import ALL_MODELS


WORDS = ["bank", "bark", "bat", "crane", "spring", "match", "pitch"]
ENCODERS = {
    "answerdotai/ModernBERT-large",
    "microsoft/deberta-v3-large",
    "FacebookAI/roberta-large",
    "FacebookAI/xlm-roberta-large",
}


def _bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _crossed_interval(matrix: np.ndarray, seed: int, n_bootstrap: int = 20000):
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap)
    for draw in range(n_bootstrap):
        model_idx = rng.integers(0, matrix.shape[0], matrix.shape[0])
        word_idx = rng.integers(0, matrix.shape[1], matrix.shape[1])
        draws[draw] = matrix[model_idx][:, word_idx].mean()
    return np.quantile(draws, [0.025, 0.975])


def _effect_row(analysis, contrast, scope, matrix, n_models, seed):
    interval = _crossed_interval(matrix, seed)
    return {
        "analysis": analysis,
        "contrast": contrast,
        "scope": scope,
        "n_models": n_models,
        "n_words": matrix.shape[1],
        "mean_effect": round(float(matrix.mean()), 8),
        "ci95_low": round(float(interval[0]), 8),
        "ci95_high": round(float(interval[1]), 8),
        "bootstrap_unit": "crossed model-by-word",
        "n_bootstrap": 20000,
        "seed": seed,
    }


def _h1_matrix(results: Path, models):
    matrix = np.empty((len(models), len(WORDS)))
    for model_index, model in enumerate(models):
        safe = model.replace("/", "_")
        for word_index, word in enumerate(WORDS):
            path = results / "study" / "H1" / safe / f"h1_nested_loo_{word}.csv"
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                raise ValueError(f"empty H1 nested output: {path}")
            matrix[model_index, word_index] = np.mean([
                int(_bool(row["adequate_selected"])) - int(_bool(row["adequate_last"]))
                for row in rows
            ])
    return matrix


def _h2_matrix(results: Path, models, strategy):
    matrix = np.empty((len(models), len(WORDS)))
    for model_index, model in enumerate(models):
        safe = model.replace("/", "_")
        path = results / "study" / "H2" / safe / "h2_loo.csv"
        with path.open(newline="") as handle:
            by_word = {row["held_out_word"]: row for row in csv.DictReader(handle)}
        if set(by_word) != set(WORDS):
            raise ValueError(f"incomplete H2 held-out words: {path}")
        for word_index, word in enumerate(WORDS):
            row = by_word[word]
            matrix[model_index, word_index] = (
                float(row[f"fraction_adequate_{strategy}"])
                - float(row["fraction_adequate_last"])
            )
    return matrix


def summarize_geometry_inference(results_dir="results"):
    results = Path(results_dir)
    rows = []
    groups = {
        "all": ALL_MODELS,
        "encoder": [model for model in ALL_MODELS if model in ENCODERS],
        "decoder": [model for model in ALL_MODELS if model not in ENCODERS],
    }
    seed = 20260714
    for scope, models in groups.items():
        rows.append(_effect_row(
            "H1", "nested selected - last fraction adequate", scope,
            _h1_matrix(results, models), len(models), seed,
        ))
        seed += 1
    for strategy in ("gdv", "supervised", "oracle"):
        for scope, models in groups.items():
            rows.append(_effect_row(
                "H2", f"{strategy} - last fraction adequate", scope,
                _h2_matrix(results, models, strategy), len(models), seed,
            ))
            seed += 1

    path = results / "study" / "geometry_inference.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    summarize_geometry_inference()
