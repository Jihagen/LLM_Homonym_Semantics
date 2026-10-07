"""Exploratory links between H0, H1, and H5 report outcomes."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

RESULTS = Path("results/study")
OUT_ASSOC = RESULTS / "cross_hypothesis_associations.csv"
OUT_ARCH = RESULTS / "H5" / "h5_architecture_exploratory.csv"
SEED = 20260716
N_BOOT = 20000


def load_h1(models):
    frames = []
    for model in models:
        frame = pd.read_csv(RESULTS / "H1" / model / "h1_summary.csv")
        frame["model"] = model
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def association_row(analysis, predictor, outcome, frame):
    data = frame[[predictor, outcome]].dropna()
    pearson = pearsonr(data[predictor], data[outcome])
    spearman = spearmanr(data[predictor], data[outcome])
    return {
        "analysis": analysis,
        "predictor": predictor,
        "outcome": outcome,
        "n_cells": len(data),
        "pearson_r": pearson.statistic,
        "pearson_p_uncorrected": pearson.pvalue,
        "spearman_rho": spearman.statistic,
        "spearman_p_uncorrected": spearman.pvalue,
        "status": "post_hoc_descriptive",
    }


def crossed_difference(frame, value, rng):
    words = frame["word"].unique()
    encoder = frame[frame["arch_type"] == "encoder"].pivot(
        index="model", columns="word", values=value
    ).loc[:, words].to_numpy()
    decoder = frame[frame["arch_type"] == "decoder"].pivot(
        index="model", columns="word", values=value
    ).loc[:, words].to_numpy()
    draws = np.empty(N_BOOT)
    for index in range(N_BOOT):
        word_index = rng.integers(0, len(words), len(words))
        encoder_index = rng.integers(0, encoder.shape[0], encoder.shape[0])
        decoder_index = rng.integers(0, decoder.shape[0], decoder.shape[0])
        draws[index] = (
            decoder[np.ix_(decoder_index, word_index)].mean()
            - encoder[np.ix_(encoder_index, word_index)].mean()
        )
    return {
        "outcome": value,
        "decoder_mean": decoder.mean(),
        "encoder_mean": encoder.mean(),
        "decoder_minus_encoder": decoder.mean() - encoder.mean(),
        "ci_low": np.quantile(draws, 0.025),
        "ci_high": np.quantile(draws, 0.975),
        "bootstrap_unit": "crossed model-by-word",
        "n_bootstrap": N_BOOT,
        "status": "post_hoc_exploratory",
    }


def main():
    h0 = pd.read_csv(RESULTS / "H0" / "h0_summary.csv")
    h5 = pd.read_csv(RESULTS / "H5" / "h5_sentence_level.csv")
    h1 = load_h1(h0["model"].unique())

    eligible_h0 = h0[h0["word"] != "light"]
    h1_cells = eligible_h0.merge(
        h1[["model", "word", "nested_frac_selected"]],
        on=["model", "word"],
    )
    rows = [
        association_row(
            "H0_absolute_lean_to_H1",
            "mean_abs_M_l_carrier_norm",
            "nested_frac_selected",
            h1_cells,
        )
    ]

    direction = h5.groupby(
        ["model", "arch_type", "word", "primed_sense", "correct_sense"],
        as_index=False,
    ).agg(
        homonym_correct_margin_norm=("homonym_correct_margin_norm", "mean"),
        resolution_correct_margin_norm=("resolution_correct_margin_norm", "mean"),
        delta_resolution_minus_homonym_norm=(
            "delta_resolution_minus_homonym_norm", "mean"
        ),
        n_primed=("primed_at_homonym", "sum"),
        n_transition=("successful_primed_to_correct_transition", "sum"),
    )
    direction["transition_rate"] = direction["n_transition"] / direction["n_primed"]
    direction = direction.merge(
        eligible_h0[["model", "word", "mean_signed_M_l_carrier_norm"]],
        on=["model", "word"],
    )
    direction["h0_lean_aligned_with_primed_sense"] = np.where(
        direction["primed_sense"] == 0,
        direction["mean_signed_M_l_carrier_norm"],
        -direction["mean_signed_M_l_carrier_norm"],
    )
    for outcome in (
        "homonym_correct_margin_norm",
        "resolution_correct_margin_norm",
        "delta_resolution_minus_homonym_norm",
        "transition_rate",
    ):
        rows.append(
            association_row(
                "H0_direction_aligned_lean_to_H5",
                "h0_lean_aligned_with_primed_sense",
                outcome,
                direction,
            )
        )
    pd.DataFrame(rows).to_csv(OUT_ASSOC, index=False)

    architecture_cells = h5.groupby(
        ["model", "arch_type", "word"], as_index=False
    ).agg(
        update=("delta_resolution_minus_homonym_norm", "mean"),
        resolver_endpoint=("resolution_correct_margin_norm", "mean"),
        conflict_minus_matched_control=(
            "garden_path_cost_vs_matched_control_norm", "mean"
        ),
    )
    rng = np.random.default_rng(SEED)
    contrasts = [
        crossed_difference(architecture_cells, outcome, rng)
        for outcome in (
            "update",
            "resolver_endpoint",
            "conflict_minus_matched_control",
        )
    ]
    pd.DataFrame(contrasts).to_csv(OUT_ARCH, index=False)
    print(OUT_ASSOC)
    print(OUT_ARCH)


if __name__ == "__main__":
    main()
