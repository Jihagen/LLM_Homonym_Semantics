"""Did Q4 change the selected layer relative to H1, and did that change the
outcome of H4 (fair decoder adequacy) or H5 (garden-path revision)?

Two questions, CPU-only, from cached CSVs -- no GPU needed:

1. Layer selection: for each model/word, is H1's homonym-position best layer
   itself a Q4 candidate (fair, L-condition, homonym-position adequacy
   within tolerance of that model/word's own peak, above the absolute
   floor)? If not, how far (relative depth) is the nearest candidate, and in
   which direction? Summarised by architecture, since the whole point of Q4
   is to check whether decoders specifically were being mis-served by H1's
   token-position choice.

   NB H4 itself was not rerun under Q4 (only H5 was, per the agreed design)
   -- H4 still reports R-condition adequacy at H1's original layer. What Q4
   *does* let us check directly is the flip side of H4's claim: at that same
   H1 layer, is a decoder's adequacy on the guaranteed-context L-condition
   sentences meaningfully higher than what H1 measured on plain profiling
   sentences? If so, that supports H4/H3's claim that the low apparent
   decoder adequacy is a token-position confound rather than a genuine
   representational limit. If L-condition adequacy at that same layer is
   *also* low, the position confound is not the (whole) explanation.

   Caveat: the L-condition set is only 10 sentences per word, so
   fraction_adequate is quantised in steps of 0.1 and single-item noise
   swings it by that much -- small differences between architectures should
   not be over-read, and this is reported explicitly.

2. H5 outcome: h5_aggregate.csv (original, one row per model/word, at H1's
   single layer) vs h5_q4/h5_aggregate.csv (one row per model/word per Q4
   candidate layer). For each model/word, compares the original
   p_resolved_correct_given_primed_at_homonym (the direct operationalisation
   of "did the garden-path prime get revised toward the correct sense") to
   the best value found across that word's Q4 candidate layers, and reports
   the layer at which that best value occurs.

Writes:
  results/study/Q4/q4_h1_layer_shift.csv
  results/study/Q4/q4_h5_outcome_comparison.csv
and prints a short console summary.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("results/study")
Q4_DIR = RESULTS / "Q4"
H1_DIR = RESULTS / "H1"
H5_DIR = RESULTS / "H5"
H5_Q4_DIR = RESULTS / "H5_q4"

ENCODERS = {
    "answerdotai_ModernBERT-large",
    "microsoft_deberta-v3-large",
    "FacebookAI_roberta-large",
    "FacebookAI_xlm-roberta-large",
}


def _arch(safe_model: str) -> str:
    return "encoder" if safe_model in ENCODERS else "decoder"


def load_h1_summaries(models):
    frames = []
    for model in models:
        path = H1_DIR / model / "h1_summary.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["model"] = model
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_q4_curves():
    return pd.read_csv(Q4_DIR / "q4_layer_curves.csv")


def load_q4_candidates(models):
    frames = []
    for model in models:
        path = Q4_DIR / model / "q4_candidate_layers.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["model"] = model
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["model", "word", "layer", "relative_depth", "adequacy", "mean_margin"]
    )


def layer_shift_table(h1, curves, candidates):
    rows = []
    for (model, word), h1_row in h1.set_index(["model", "word"]).groupby(level=[0, 1]):
        h1_row = h1_row.iloc[0]
        h1_layer = int(h1_row["best_layer_M"])
        last_layer = int(h1_row["n_layers"]) - 1
        cand = candidates[(candidates["model"] == model) & (candidates["word"] == word)]
        curve = curves[(curves["model"] == model) & (curves["homonym"] == word)]
        if curve.empty:
            continue

        cand_layers = sorted(cand["layer"].astype(int).tolist())
        h1_in_candidates = h1_layer in cand_layers
        if cand_layers:
            nearest = min(cand_layers, key=lambda c: abs(c - h1_layer))
            shift = (nearest - h1_layer) / last_layer if last_layer else 0.0
        else:
            nearest, shift = None, np.nan

        h1_layer_curve_row = curve[curve["layer"] == h1_layer]
        q4_adequacy_at_h1_layer = (
            float(h1_layer_curve_row["adequacy"].iloc[0]) if not h1_layer_curve_row.empty else np.nan
        )
        peak_row = curve.loc[curve["adequacy"].idxmax()]

        rows.append({
            "model": model,
            "arch_type": _arch(model),
            "word": word,
            "h1_layer": h1_layer,
            "h1_relative_depth": round(h1_layer / last_layer, 4) if last_layer else 0.0,
            "h1_frac_adeq_best": h1_row["frac_adeq_best"],
            "q4_adequacy_at_h1_layer": q4_adequacy_at_h1_layer,
            "q4_adequacy_minus_h1": (
                q4_adequacy_at_h1_layer - h1_row["frac_adeq_best"]
                if not np.isnan(q4_adequacy_at_h1_layer) else np.nan
            ),
            "h1_layer_is_q4_candidate": h1_in_candidates,
            "n_q4_candidates": len(cand_layers),
            "nearest_q4_candidate": nearest,
            "relative_depth_shift_to_nearest_candidate": round(shift, 4) if shift == shift else np.nan,
            "q4_peak_layer": int(peak_row["layer"]),
            "q4_peak_adequacy": float(peak_row["adequacy"]),
        })
    return pd.DataFrame(rows)


def h5_outcome_comparison(shift_table):
    orig = pd.read_csv(H5_DIR / "h5_aggregate.csv")
    q4 = pd.read_csv(H5_Q4_DIR / "h5_aggregate.csv")

    rows = []
    for _, orig_row in orig.iterrows():
        model, word = orig_row["model"], orig_row["word"]
        q4_word = q4[(q4["model"] == model) & (q4["word"] == word)]
        if q4_word.empty:
            continue
        best = q4_word.loc[q4_word["p_resolved_correct_given_primed_at_homonym"].idxmax()]
        shift_row = shift_table[(shift_table["model"] == model) & (shift_table["word"] == word)]
        h1_in_candidates = (
            bool(shift_row["h1_layer_is_q4_candidate"].iloc[0]) if not shift_row.empty else None
        )
        rows.append({
            "model": model,
            "arch_type": orig_row["arch_type"],
            "word": word,
            "h1_layer_used": int(orig_row["layer_used"]),
            "h1_layer_is_q4_candidate": h1_in_candidates,
            "orig_p_resolved": orig_row["p_resolved_correct_given_primed_at_homonym"],
            "orig_delta_resolution_minus_homonym": orig_row["mean_delta_resolution_minus_homonym_norm"],
            "best_q4_layer": int(best["layer_used"]),
            "best_q4_p_resolved": best["p_resolved_correct_given_primed_at_homonym"],
            "best_q4_delta_resolution_minus_homonym": best["mean_delta_resolution_minus_homonym_norm"],
            "delta_p_resolved_best_q4_minus_orig": (
                best["p_resolved_correct_given_primed_at_homonym"]
                - orig_row["p_resolved_correct_given_primed_at_homonym"]
            ),
            "n_q4_layers_tested": len(q4_word),
        })
    return pd.DataFrame(rows)


def main():
    models = sorted(p.name for p in H1_DIR.iterdir() if p.is_dir())
    h1 = load_h1_summaries(models)
    curves = load_q4_curves()
    candidates = load_q4_candidates(models)

    shift_table = layer_shift_table(h1, curves, candidates)
    shift_out = Q4_DIR / "q4_h1_layer_shift.csv"
    shift_table.to_csv(shift_out, index=False)

    outcome_table = h5_outcome_comparison(shift_table)
    outcome_out = Q4_DIR / "q4_h5_outcome_comparison.csv"
    outcome_table.to_csv(outcome_out, index=False)

    print(f"Wrote {shift_out} ({len(shift_table)} rows)")
    print(f"Wrote {outcome_out} ({len(outcome_table)} rows)")

    print("\n=== Layer-selection shift, by architecture ===")
    summary = shift_table.groupby("arch_type").agg(
        n_cells=("word", "count"),
        pct_h1_layer_validated=("h1_layer_is_q4_candidate", "mean"),
        mean_n_candidates=("n_q4_candidates", "mean"),
        mean_adequacy_gain_at_h1_layer=("q4_adequacy_minus_h1", "mean"),
        mean_abs_relative_shift=(
            "relative_depth_shift_to_nearest_candidate", lambda s: s.abs().mean()
        ),
    ).round(3)
    print(summary.to_string())

    print("\n=== H5 outcome, by architecture: best-of-Q4-candidates vs H1's original layer ===")
    outcome_summary = outcome_table.groupby("arch_type").agg(
        n_cells=("word", "count"),
        pct_h1_layer_was_candidate=("h1_layer_is_q4_candidate", "mean"),
        mean_orig_p_resolved=("orig_p_resolved", "mean"),
        mean_best_q4_p_resolved=("best_q4_p_resolved", "mean"),
        mean_delta_p_resolved=("delta_p_resolved_best_q4_minus_orig", "mean"),
    ).round(3)
    print(outcome_summary.to_string())


if __name__ == "__main__":
    main()
