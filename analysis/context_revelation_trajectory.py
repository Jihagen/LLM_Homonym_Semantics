"""Track sense geometry as right context is progressively revealed.

For each neutral carrier, the two sense continuations share an identical
prefix.  We then reveal a partial and complete sense-specific continuation.
One PCA is fitted across every stage at the selected layer, so visible motion
is in a fixed coordinate system.  Encoder plots track the homonym token;
decoder plots additionally track the newest content-token readout.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from experiments.gdv_experiments import compute_gdv
from hypotheses.h3_context_position import _select_layer
from utils.model_registry import ALL_MODELS, MODEL_ALIASES
from models import (
    get_dual_position_activations,
    is_decoder_only,
    load_model_and_tokenizer,
)
from utils.visual_style import INK, METHOD, MUTED, SENSE, apply_report_style


STAGES = (0.0, 0.5, 1.0)
STAGE_LABELS = ("shared", "partial", "complete")


def _common_prefix_length(left, right):
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def build_prefix_stages(paired_path: Path, word: str):
    """Return aligned staged texts for every carrier's two R-condition senses."""
    with open(paired_path) as handle:
        items = json.load(handle)[word]
    by_carrier = defaultdict(dict)
    for item in items:
        if item["condition"] == "R":
            by_carrier[item["carrier"]][int(item["sense"])] = item

    records = []
    for pair_index, (carrier, senses) in enumerate(by_carrier.items()):
        if set(senses) != {0, 1}:
            continue
        tokens = {sense: senses[sense]["sentence"].split() for sense in (0, 1)}
        common_length = _common_prefix_length(tokens[0], tokens[1])
        common = tokens[0][:common_length]
        tails = {sense: tokens[sense][common_length:] for sense in (0, 1)}
        if not common or word.lower() not in " ".join(common).lower():
            raise ValueError(f"Shared prefix does not contain '{word}': {carrier}")

        for stage_index, fraction in enumerate(STAGES):
            for sense in (0, 1):
                n_tail = 0 if fraction == 0 else max(1, math.ceil(fraction * len(tails[sense])))
                staged_tokens = common + tails[sense][:n_tail]
                records.append(
                    {
                        "pair_id": pair_index,
                        "carrier": carrier,
                        "sense": sense,
                        "stage": stage_index,
                        "stage_fraction": fraction,
                        "text": " ".join(staged_tokens),
                    }
                )
    return records


def _pair_heldout_scores(X, labels, pair_ids, standardized=False):
    normalized_margins = []
    correctness = []
    centroid_distances = []
    for heldout_pair in np.unique(pair_ids):
        train = pair_ids != heldout_pair
        test = ~train
        X_train = X[train]
        X_test = X[test]
        if standardized:
            mean = X_train.mean(axis=0)
            scale = X_train.std(axis=0)
            scale[scale < 1e-12] = 1.0
            X_train = (X_train - mean) / scale
            X_test = (X_test - mean) / scale
        c0 = X_train[labels[train] == 0].mean(axis=0)
        c1 = X_train[labels[train] == 1].mean(axis=0)
        denominator = float(np.linalg.norm(c1 - c0))
        centroid_distances.append(denominator)
        for vector, sense in zip(X_test, labels[test]):
            correct = c0 if sense == 0 else c1
            wrong = c1 if sense == 0 else c0
            margin = float(np.linalg.norm(vector - wrong) - np.linalg.norm(vector - correct))
            if denominator < 1e-10:
                correctness.append(0.5)
            else:
                normalized_margins.append(margin / denominator)
                correctness.append(1.0 if margin > 0 else 0.5 if abs(margin) < 1e-12 else 0.0)
    return {
        "mean_margin_norm": float(np.mean(normalized_margins)) if normalized_margins else float("nan"),
        "tie_aware_accuracy": float(np.mean(correctness)),
        "mean_centroid_distance": float(np.mean(centroid_distances)),
    }


def _compute_metrics(activations, records, representation):
    stages = np.array([record["stage"] for record in records])
    labels = np.array([record["sense"] for record in records])
    pair_ids = np.array([record["pair_id"] for record in records])
    rows = []
    for layer, tensor in activations.items():
        X_all = tensor.numpy()
        for stage in range(len(STAGES)):
            mask = stages == stage
            X = X_all[mask]
            raw = _pair_heldout_scores(X, labels[mask], pair_ids[mask], standardized=False)
            standardized = _pair_heldout_scores(
                X, labels[mask], pair_ids[mask], standardized=True
            )
            rows.append(
                {
                    "representation": representation,
                    "layer": layer,
                    "stage": stage,
                    "stage_label": STAGE_LABELS[stage],
                    "gdv": compute_gdv(X, labels[mask]),
                    "raw_mean_margin_norm": raw["mean_margin_norm"],
                    "raw_tie_aware_accuracy": raw["tie_aware_accuracy"],
                    "standardized_mean_margin_norm": standardized["mean_margin_norm"],
                    "standardized_tie_aware_accuracy": standardized["tie_aware_accuracy"],
                    "raw_centroid_distance": raw["mean_centroid_distance"],
                    "standardized_centroid_distance": standardized["mean_centroid_distance"],
                }
            )
    return rows


def _save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=190, bbox_inches="tight")
    plt.close(fig)


def _trajectory_panel(ax, X, records, title):
    coords = PCA(n_components=2).fit_transform(X)
    stages = np.array([record["stage"] for record in records])
    senses = np.array([record["sense"] for record in records])
    pairs = np.array([record["pair_id"] for record in records])
    for pair in np.unique(pairs):
        for sense in (0, 1):
            mask = (pairs == pair) & (senses == sense)
            order = np.argsort(stages[mask])
            points = coords[mask][order]
            ax.plot(points[:, 0], points[:, 1], color=SENSE[sense], alpha=0.16, linewidth=1)
    for sense in (0, 1):
        centroids = []
        for stage in range(len(STAGES)):
            mask = (stages == stage) & (senses == sense)
            centroid = coords[mask].mean(axis=0)
            centroids.append(centroid)
            ax.scatter(
                centroid[0],
                centroid[1],
                s=48 + 14 * stage,
                facecolor=SENSE[sense],
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
            ax.annotate(
                STAGE_LABELS[stage],
                centroid,
                xytext=(4, 4 if sense == 0 else -11),
                textcoords="offset points",
                fontsize=8,
                color=SENSE[sense],
            )
        centroids = np.asarray(centroids)
        ax.plot(centroids[:, 0], centroids[:, 1], color=SENSE[sense], linewidth=2.4)
    ax.set_xlabel("joint-trajectory PC1")
    ax.set_ylabel("joint-trajectory PC2")
    ax.set_title(title, loc="left", fontweight="bold")


def _make_trajectory_figure(target, readout, records, layer, model_label, word, decoder, output):
    n_panels = 2 if decoder else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7.1 * n_panels, 5.3), constrained_layout=True)
    axes = np.atleast_1d(axes)
    _trajectory_panel(axes[0], target[layer].numpy(), records, "Homonym token")
    if decoder:
        _trajectory_panel(axes[1], readout[layer].numpy(), records, "Newest-token context readout")
    fig.suptitle(
        f"Context revelation · {model_label} · {word} · layer {layer}",
        fontsize=14,
        fontweight="bold",
    )
    _save_figure(fig, output)


def _make_metric_figure(metric_rows, layer, model_label, word, output):
    selected = [row for row in metric_rows if row["layer"] == layer]
    representations = list(dict.fromkeys(row["representation"] for row in selected))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3), constrained_layout=True)
    styles = {
        "homonym": {"marker": "o", "fillstyle": "none", "linestyle": "--"},
        "context_readout": {"marker": "o", "fillstyle": "full", "linestyle": "-"},
    }
    for representation in representations:
        rows = sorted(
            (row for row in selected if row["representation"] == representation),
            key=lambda row: row["stage"],
        )
        x = [row["stage"] for row in rows]
        label = representation.replace("_", " ")
        axes[0].plot(
            x,
            [row["gdv"] for row in rows],
            color=METHOD["gdv"],
            label=label,
            **styles[representation],
        )
        axes[1].plot(
            x,
            [row["standardized_tie_aware_accuracy"] for row in rows],
            color=METHOD["adequacy"],
            label=label,
            **styles[representation],
        )
    axes[0].set_ylabel("GDV · more negative is stronger")
    axes[0].set_title("Global geometry", loc="left", fontweight="bold")
    axes[1].set_ylabel("Pair-held-out accuracy · ties count as 0.5")
    axes[1].set_ylim(0.35, 1.03)
    axes[1].set_title("Standardized centroid decision", loc="left", fontweight="bold")
    for ax in axes:
        ax.set_xticks(range(len(STAGES)), STAGE_LABELS)
        ax.set_xlabel("Sense-specific context revealed")
        ax.legend()
    fig.suptitle(
        f"Separation as context arrives · {model_label} · {word} · layer {layer}",
        fontsize=14,
        fontweight="bold",
    )
    _save_figure(fig, output)


def _resolve_models(names):
    if not names:
        return [MODEL_ALIASES["roberta"], MODEL_ALIASES["qwen7b"]]
    return [MODEL_ALIASES.get(name, name) for name in names]


def run(models, words, paired_path, results_dir, output_base, dry_run=False):
    records_by_word = {
        word: build_prefix_stages(paired_path, word)
        for word in words
    }
    if dry_run:
        for word, records in records_by_word.items():
            print(f"\n{word}: {len(records)} staged sentences")
            for record in records[:10]:
                print(record["stage_label"] if "stage_label" in record else STAGE_LABELS[record["stage"]], record["sense"], record["text"])
        return

    for model_name in models:
        if model_name not in ALL_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        model, tokenizer = load_model_and_tokenizer(model_name)
        decoder = is_decoder_only(model)
        safe_model = model_name.replace("/", "_")
        out_dir = output_base / safe_model
        for word, records in records_by_word.items():
            texts = [record["text"] for record in records]
            targets = [word] * len(records)
            target, readout = get_dual_position_activations(
                model, tokenizer, texts, targets, batch_size=4, layer_indices=None
            )
            layer = _select_layer(model_name, str(results_dir), word)
            metric_rows = _compute_metrics(target, records, "homonym")
            if decoder:
                metric_rows.extend(_compute_metrics(readout, records, "context_readout"))
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"{word}_metrics.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=metric_rows[0].keys())
                writer.writeheader()
                writer.writerows(metric_rows)
            _make_trajectory_figure(
                target,
                readout,
                records,
                layer,
                model_name,
                word,
                decoder,
                out_dir / f"{word}_trajectory.svg",
            )
            _make_metric_figure(
                metric_rows,
                layer,
                model_name,
                word,
                out_dir / f"{word}_separation.svg",
            )
        del model, tokenizer


def main():
    apply_report_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--words", nargs="+", default=["bank"])
    parser.add_argument("--paired-path", default="data/paired_sentences.json")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-base", default="results/study/context_trajectory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(
        _resolve_models(args.models),
        args.words,
        Path(args.paired_path),
        Path(args.results_dir),
        Path(args.output_base),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
