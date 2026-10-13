"""Render the per-model centroid-separation summary (produced alongside
plot_centroid_separation.py) as a readable table image.

Columns:
  Peak sep.        max(||c1-c0|| / mean within-class distance to centroid)
                   over all layers, and the relative depth it occurs at.
  H1-best layer    the layer H1's full-profile nested selection actually
                   picked, its relative depth, and the separation ratio
                   and raw centroid distance there.
  Max raw dist.    largest raw (unnormalised) inter-centroid distance seen
                   at any layer -- included to make the scale-artifact
                   contrast with the normalised ratio visible directly
                   (e.g. Qwen2.5-7B / Mistral-Nemo have huge raw distances
                   but modest-to-low normalised separation).
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.model_registry import ALL_MODELS
from utils.visual_style import ARCHITECTURE, INK, apply_report_style
from analysis.plot_centroid_separation import ENCODERS, _read_h1_best_layer, _separation_curve


def _build_rows(results_dir: Path, word: str):
    rows = []
    for model in ALL_MODELS:
        safe_model = model.replace("/", "_")
        arch = "encoder" if model in ENCODERS else "decoder"
        depth, raw_dist, norm_ratio, layers, last_layer = _separation_curve(
            results_dir, word, safe_model
        )
        best_layer = _read_h1_best_layer(results_dir, safe_model, word)
        peak_idx = int(np.nanargmax(norm_ratio))
        best_idx = layers.index(best_layer) if best_layer in layers else None
        rows.append(
            {
                "model": model.split("/")[-1],
                "arch": arch,
                "n_layers": last_layer,
                "peak_norm_sep": norm_ratio[peak_idx],
                "peak_rel_depth": depth[peak_idx],
                "h1_best_layer": best_layer,
                "h1_best_rel_depth": depth[best_idx] if best_idx is not None else np.nan,
                "norm_sep_at_h1_best": norm_ratio[best_idx] if best_idx is not None else np.nan,
                "raw_dist_at_h1_best": raw_dist[best_idx] if best_idx is not None else np.nan,
                "raw_dist_max": np.nanmax(raw_dist),
            }
        )
    return rows


def make_table(results_dir: Path, word: str, output_path: Path, csv_path: Path = None) -> None:
    apply_report_style()
    rows = _build_rows(results_dir, word)

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    columns = [
        "Model", "Arch.", "Layers",
        "Peak sep.\n(rel. depth)",
        "H1-best layer\n(rel. depth)",
        "Sep. @\nH1-best",
        "Raw dist. @\nH1-best",
        "Max raw\ndist.",
    ]
    cell_text = []
    for r in rows:
        cell_text.append([
            r["model"],
            r["arch"],
            str(r["n_layers"]),
            f'{r["peak_norm_sep"]:.2f} ({r["peak_rel_depth"]:.2f})',
            f'L{r["h1_best_layer"]} ({r["h1_best_rel_depth"]:.2f})',
            f'{r["norm_sep_at_h1_best"]:.2f}',
            f'{r["raw_dist_at_h1_best"]:.1f}',
            f'{r["raw_dist_max"]:.1f}',
        ])

    col_widths = [0.20, 0.09, 0.07, 0.135, 0.145, 0.10, 0.125, 0.115]

    fig, ax = plt.subplots(figsize=(13.0, 0.62 * len(rows) + 1.3))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 2.05)
    for (row_idx, col_idx), cell in table.get_celld().items():
        if col_idx == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.03

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#D9D6CF")
        if row_idx == 0:
            cell.set_text_props(fontweight="bold", color=INK)
            cell.set_facecolor("#EFECE4")
            continue
        arch = rows[row_idx - 1]["arch"]
        cell.set_facecolor(ARCHITECTURE[arch] + "22")
        if col_idx == 0:
            cell.set_text_props(fontweight="bold", color=INK)

    ax.set_title(
        f"Centroid-separation summary across depth · {word}\n"
        "green = encoder, mauve = decoder",
        fontsize=12.5,
        fontweight="bold",
        pad=14,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--word", default="bank")
    parser.add_argument(
        "--output", default="results/study/figures/h1_centroid_separation_bank_table.svg"
    )
    parser.add_argument(
        "--csv", default="results/study/figures/h1_centroid_separation_bank_summary.csv"
    )
    args = parser.parse_args()
    make_table(
        Path(args.results_dir), args.word, Path(args.output),
        Path(args.csv) if args.csv else None,
    )


if __name__ == "__main__":
    main()
