import json
from pathlib import Path

import pandas as pd

PROFILING_DATA_PATH = "data/profiling_sentences.json"


def load_profiling_dataframe(path: str = PROFILING_DATA_PATH) -> pd.DataFrame:
    """
    Load the sense-labelled profiling sentences as one row per (word, sense)
    with columns "word", "semantic_group_id", "sense_label", "examples" (a list).

    The canonical file is JSON. A legacy pandas pickle with the same columns
    (the format used before v1.0.0) is still accepted by file extension.
    """
    if Path(path).suffix == ".pkl":
        return pd.read_pickle(path)
    with open(path, encoding="utf-8") as handle:
        return pd.DataFrame(json.load(handle))


def flatten_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe with columns "examples" (a list), "word", and "semantic_group_id",
    flatten the "examples" list so that each generated sentence becomes its own row.
    """
    rows = []
    for _, row in df.iterrows():
        word = row["word"]
        group_id = row["semantic_group_id"]
        for sent in row["examples"]:
            rows.append({"sentence": sent, "word": word, "semantic_group_id": group_id})
    return pd.DataFrame(rows)
