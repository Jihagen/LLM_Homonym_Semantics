"""Root-level runner for the Q4 fair-decoder layer reanalysis.

Three stages, run in order:

1. Extract every layer's homonym-position state for H3's L-condition paired
   sentences (GPU: loads each model checkpoint; skipped per model/word if
   already cached under results/activations_paired_left/). L-condition
   sentences present the disambiguating clause before the homonym, so a
   causal decoder has legitimate access to it by the time it reaches the
   homonym token -- unlike H1's plain profiling sentences, which give no
   such guarantee.
2. Score every cached layer against the existing homonym-position profiling
   centroids (CPU-only; writes results/study/Q4/) and select, per
   model/word, the set of *candidate* layers: every layer within
   `--tolerance` of that model/word's own peak adequacy AND at or above
   `--min-adequacy` (relative-to-peak with an absolute floor, so encoders
   and decoders are held to their own ceiling rather than one shared bar).
3. Rerun H5 across every candidate layer per word (not just one), writing
   to a separate output directory (results/study/H5_q4/ by default) so the
   original H1-based H5 results are never touched.

Mirrors run_h2.py's standalone-script convention; the H0-H5 dispatch in
run_study.py is intentionally left untouched, since Q4 is a new reanalysis
rather than one of the six original hypotheses.
"""

import argparse
import gc
import logging
from functools import partial
from pathlib import Path

from utils.hpc import configure_hpc_runtime

configure_hpc_runtime()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_WORDS = ["bank", "bark", "bat", "crane", "spring", "match", "pitch"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--words", nargs="*", default=DEFAULT_WORDS)
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Restrict to specific models (safe or slash-form names); default all 8.",
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip stage 1 (assume results/activations_paired_left/ is already complete).",
    )
    parser.add_argument(
        "--skip-h5-rerun", action="store_true",
        help="Skip stage 3 (only produce the Q4 layer curves/candidate selection, no H5 rerun).",
    )
    parser.add_argument("--force-extraction", action="store_true")
    parser.add_argument(
        "--h5-output-base", default="results/study/H5_q4",
        help="Where the Q4-candidate-layer H5 rerun is written (never the original results/study/H5/).",
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.05,
        help="Candidate layers must be within this much of the model/word's peak adequacy.",
    )
    parser.add_argument(
        "--min-adequacy", type=float, default=0.6,
        help="Absolute adequacy floor a layer must also clear to be a candidate.",
    )
    parser.add_argument("--allow-incomplete-h5-design", action="store_true")
    args = parser.parse_args()

    from utils.model_registry import ALL_MODELS, MODEL_ALIASES

    if args.models:
        models = [MODEL_ALIASES.get(m, m) for m in args.models]
    else:
        models = ALL_MODELS

    if not args.skip_extraction:
        from hypotheses.q4_layer_selection import extract_l_condition_activations
        logger.info("=== Q4 stage 1: extracting L-condition activations | %d models ===", len(models))
        extract_l_condition_activations(
            model_names=models, words=args.words, force=args.force_extraction
        )
        gc.collect()
    else:
        logger.info("=== Q4 stage 1 skipped (--skip-extraction) ===")

    from hypotheses.q4_layer_selection import run_q4
    logger.info("=== Q4 stage 2: scoring and candidate-layer selection ===")
    run_q4(
        model_names=models, words=args.words,
        tolerance=args.tolerance, min_adequacy=args.min_adequacy,
    )

    if not args.skip_h5_rerun:
        from hypotheses.h5_garden_path import run_h5
        from hypotheses.q4_layer_selection import q4_candidate_layers
        logger.info(
            "=== Q4 stage 3: rerunning H5 across Q4 candidate layers -> %s ===",
            args.h5_output_base,
        )
        run_h5(
            model_names=models,
            words=args.words,
            output_base=Path(args.h5_output_base),
            layer_candidates_lookup=q4_candidate_layers,
            allow_incomplete_design=args.allow_incomplete_h5_design,
        )
    else:
        logger.info("=== Q4 stage 3 skipped (--skip-h5-rerun) ===")

    logger.info("Q4 fair-decoder reanalysis complete.")


if __name__ == "__main__":
    main()
