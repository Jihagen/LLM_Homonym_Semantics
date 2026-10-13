"""Recompute corrected GDV, H1, and H2 outputs from cached activations.

No transformer models are loaded and no forward passes are performed.
"""

import argparse
import logging

from experiments.gdv_experiments import recompute_gdv_from_cache
from utils.model_registry import ALL_MODELS, MODEL_ALIASES

DEFAULT_WORDS = ["bank", "bark", "bat", "crane", "spring", "match", "pitch"]


def _resolve_models(names):
    if not names:
        return ALL_MODELS
    resolved = []
    for name in names:
        if name in MODEL_ALIASES:
            resolved.append(MODEL_ALIASES[name])
        elif name in ALL_MODELS:
            resolved.append(name)
        else:
            raise ValueError(f"Unknown model or alias: {name}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute geometry metrics from cached H5 activations only."
    )
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--words", nargs="*", default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--gdv-only",
        action="store_true",
        help="Only regenerate corrected GDV CSVs; skip H1/H2.",
    )
    args = parser.parse_args()
    words = args.words or DEFAULT_WORDS

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    models = _resolve_models(args.models)
    for model_name in models:
        recompute_gdv_from_cache(args.results_dir, model_name, words)

    if not args.gdv_only:
        from hypotheses.h1_layer_adequacy import run_h1
        from hypotheses.h2_gdv_generalization import run_h2

        run_h1(models, words=words, results_dir=args.results_dir)
        run_h2(models, words=words, results_dir=args.results_dir)

        if models == ALL_MODELS and words == DEFAULT_WORDS:
            from analysis.summarize_geometry_inference import summarize_geometry_inference

            summarize_geometry_inference(args.results_dir)


if __name__ == "__main__":
    main()
