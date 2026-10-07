"""
Master runner for H1–H5.

Usage examples
--------------
  python run_study.py                          # run H1+H2 (no forward passes needed)
  python run_study.py --hypotheses H1 H2 H3   # specific subset
  python run_study.py --hypotheses H3 H4 H5   # forward-pass experiments only
  python run_study.py --models deberta roberta # restrict to named model subsets

Per-hypothesis data requirements
---------------------------------
H0  Carrier norming (prerequisite for H3)
    Data   : data/paired_sentences.json  +  H5 profiling files (homonym-position)
    Models : all 8 (see model_registry.ALL_MODELS)

H1  Layer adequacy profile
    Data   : results/activations/{word}/{model}/  (H5 files, homonym-position)
    Models : all 8 (encoder + decoder)

H2  GDV generalisation (leave-one-out)
    Data   : H5 files + gdv_values_{word}.csv for all words
    Models : all 8

H3  Right-context vulnerability
    Data   : data/paired_sentences.json  +  H5 profiling files
    Models : all 8

H4  Token-position sense decodability
    Data   : data/paired_sentences.json (R-condition only)  +  H5 files
             (homonym-position) + activations_final/ files (final-position,
             for encoder final-token scoring — see hypotheses/h4_dissociation.py)
    Models : all 8

H5  Incremental garden-path updating at a fixed sentinel (exploratory)
    Data   : data/garden_path_sentences.json + data/profiling_sentences.json
    Guard  : blocked until direction/control/structure audit passes unless the
             explicitly exploratory override is supplied
    Models : all 8
"""

import argparse
import gc
import logging
import sys

from utils.model_registry import ALL_MODELS, MODEL_ALIASES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("run_study")

# ── Model sets ────────────────────────────────────────────────────────────────
# H0/H3/H4/H5 now also run on the full 8-model set (previously a 4-model
# representative subset) — see model_registry.py, the single source of truth.
H3_MODELS = ALL_MODELS

WORDS_DEFAULT = ["bank", "bark", "bat", "crane", "spring", "match", "pitch"]


def _resolve_models(names):
    resolved = []
    for n in names:
        if n in MODEL_ALIASES:
            resolved.append(MODEL_ALIASES[n])
        elif n in ALL_MODELS:
            resolved.append(n)
        else:
            logger.warning("Unknown model alias '%s' - skipping.", n)
    return resolved or ALL_MODELS


def _run_h0(args):
    from hypotheses.h0_carrier_norming import run_h0
    models = _resolve_models(args.models) if args.models else H3_MODELS
    logger.info("=== H0: Carrier Norming | %d models ===", len(models))
    run_h0(model_names=models, words=args.words)
    gc.collect()


def _run_h1(args):
    from hypotheses.h1_layer_adequacy import run_h1
    models = _resolve_models(args.models) if args.models else ALL_MODELS
    logger.info("=== H1: Layer Adequacy | %d models | words=%s ===", len(models), args.words)
    run_h1(model_names=models, words=args.words)


def _run_h2(args):
    from hypotheses.h2_gdv_generalization import run_h2
    models = _resolve_models(args.models) if args.models else ALL_MODELS
    logger.info("=== H2: GDV Generalisation | %d models ===", len(models))
    run_h2(model_names=models, words=args.words)


def _run_h3(args):
    from hypotheses.h3_context_position import run_h3
    models = _resolve_models(args.models) if args.models else H3_MODELS
    logger.info("=== H3: Context Position | %d models ===", len(models))
    run_h3(model_names=models, words=args.words)
    gc.collect()


def _run_h4(args):
    from hypotheses.h4_dissociation import run_h4
    models = _resolve_models(args.models) if args.models else H3_MODELS
    logger.info("=== H4: Token-position decodability | %d models ===", len(models))
    run_h4(model_names=models, words=args.words, epsilon=args.epsilon)
    gc.collect()


def _run_h5(args):
    from hypotheses.h5_garden_path import run_h5
    models = _resolve_models(args.models) if args.models else H3_MODELS
    logger.info("=== H5: Incremental fixed-sentinel updating | %d models ===", len(models))
    run_h5(
        model_names=models,
        words=args.words,
        epsilon=args.epsilon,
        allow_incomplete_design=args.allow_incomplete_h5,
    )
    gc.collect()


RUNNERS = {
    "H0": _run_h0,
    "H1": _run_h1,
    "H2": _run_h2,
    "H3": _run_h3,
    "H4": _run_h4,
    "H5": _run_h5,
}


def main():
    parser = argparse.ArgumentParser(description="Run H1–H5 study experiments.")
    parser.add_argument(
        "--hypotheses", nargs="+",
        choices=["H0", "H1", "H2", "H3", "H4", "H5"],
        default=["H1", "H2"],
        help="Which hypotheses to run (default: H1 H2).",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=(
            "Model names or aliases to run. "
            "Aliases: deberta, roberta, xlm, modernbert, qwen3b, qwen7b, mistral, olmo. "
            "Default: all 8, for every hypothesis."
        ),
    )
    parser.add_argument(
        "--words", nargs="+",
        default=WORDS_DEFAULT,
        help=f"Homonyms to include (default: {WORDS_DEFAULT}).",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.0,
        help="Adequacy threshold epsilon (default: 0.0).",
    )
    parser.add_argument(
        "--allow-incomplete-h5",
        action="store_true",
        help=(
            "Run H5 despite missing reverse directions, matched controls, or "
            "structurally valid resolver order. Outputs are labelled "
            "exploratory_incomplete_design."
        ),
    )
    args = parser.parse_args()

    logger.info("Study run: hypotheses=%s, words=%s", args.hypotheses, args.words)

    failures = []
    for hyp in args.hypotheses:
        try:
            RUNNERS[hyp](args)
        except Exception as exc:
            failures.append((hyp, str(exc)))
            logger.error("=== %s FAILED: %s ===", hyp, exc, exc_info=True)

    if failures:
        failed_names = ", ".join(hyp for hyp, _ in failures)
        logger.error("Study failed for: %s", failed_names)
        raise SystemExit(1)
    logger.info("Study complete.")


if __name__ == "__main__":
    main()
