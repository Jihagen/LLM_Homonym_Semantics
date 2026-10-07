import argparse
import gc
import logging
import os
import sys
import traceback
from pathlib import Path

import pandas as pd

from data import PROFILING_DATA_PATH, load_profiling_dataframe
from experiments import run_gdv_experiment
from utils.file_manager import FileManager
from utils.hpc import configure_hpc_runtime


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
configure_hpc_runtime()


def _words_needing_processing(df: pd.DataFrame, model_name: str) -> list:
    """
    Return words that do not yet have a complete activation cache.

    Checks BOTH activations/ (homonym-position) and activations_final/
    (final-content-token position, used by H4's final-position centroids) —
    a word only counts as cached if both exist, otherwise re-running would
    silently skip words that have stale homonym-only caches from before
    activations_final/ existed.
    """
    safe = model_name.replace("/", "_")
    all_words = sorted(df["word"].unique())
    missing = []
    for word in all_words:
        h5_dir       = Path("results/activations") / word / safe
        h5_final_dir = Path("results/activations_final") / word / safe
        has_files = (
            h5_dir.exists() and any(h5_dir.glob("layer_[1-9]*.h5"))
            and h5_final_dir.exists() and any(h5_final_dir.glob("layer_[1-9]*.h5"))
        )
        if not has_files:
            missing.append(word)
        else:
            logging.info("Skipping %s / %s - activation cache exists.", model_name, word)
    return missing


def main():
    parser = argparse.ArgumentParser(description="Run GDV profiling for H2 dataset.")
    parser.add_argument(
        "--words", nargs="*", default=None,
        help="Restrict to specific homonyms (default: all in data file).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process all words even if activation cache exists.",
    )
    args = parser.parse_args()

    sys.path.append(os.path.abspath(".."))
    print("PYTORCH_CUDA_ALLOC_CONF:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    print("HF_HOME:", os.environ.get("HF_HOME"))
    print("OFFLOAD_DIR:", os.environ.get("OFFLOAD_DIR"))

    file_manager = FileManager()
    del file_manager

    data_file = PROFILING_DATA_PATH
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"File {data_file} not found.")
    df = load_profiling_dataframe(data_file)
    logging.info(
        "Loaded %s: %d word-sense pairs across %d unique words",
        data_file, len(df), df['word'].nunique(),
    )

    # Filter to requested words if specified
    if args.words:
        df = df[df["word"].isin(args.words)]
        logging.info("Restricted to words: %s", sorted(df["word"].unique()))

    model_configs = {
        "answerdotai/ModernBERT-large":         {"model_type": "default"},
        "microsoft/deberta-v3-large":           {"model_type": "default"},
        "FacebookAI/roberta-large":             {"model_type": "default"},
        "FacebookAI/xlm-roberta-large":         {"model_type": "default"},
        "Qwen/Qwen2.5-3B":                      {"model_type": "default"},
        "Qwen/Qwen2.5-7B":                      {"model_type": "default"},
        "mistralai/Mistral-Nemo-Base-2407":     {"model_type": "default"},
        "allenai/OLMo-2-1124-7B":              {"model_type": "default"},
    }

    for model_name, cfg in model_configs.items():
        # Determine which words still need processing for this model
        if args.force:
            words_todo = sorted(df["word"].unique())
        else:
            words_todo = _words_needing_processing(df, model_name)

        if not words_todo:
            logging.info("All words already cached for %s - skipping.", model_name)
            continue

        df_todo = df[df["word"].isin(words_todo)]
        logging.info("Running GDV for %s | words: %s", model_name, words_todo)
        try:
            run_gdv_experiment(df_todo, model_name, model_type=cfg["model_type"])
        except Exception as exc:
            logging.error("Failed to run %s: %s", model_name, exc)
            logging.debug(traceback.format_exc())
        finally:
            gc.collect()


if __name__ == "__main__":
    main()
