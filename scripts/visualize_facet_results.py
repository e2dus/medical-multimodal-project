# -*- coding: utf-8 -*-
"""
Visualize FACET experiment outputs.

Inputs:
  - reconstruction_results.csv
  - temporal_gap_results.csv
  - classification_lead_selection_results.csv

Example:
  python visualize_facet_results.py --results_dir ./outputs/facet_full

Or with explicit paths:
    # Windows cmd (single line)
    python visualize_facet_results.py --reconstruction_csv .\reconstruction_results.csv --temporal_csv .\temporal_gap_results.csv --classification_csv .\classification_lead_selection_results.csv --out_dir .\figures

    # PowerShell (multi-line)
    python visualize_facet_results.py `
        --reconstruction_csv .\reconstruction_results.csv `
        --temporal_csv .\temporal_gap_results.csv `
        --classification_csv .\classification_lead_selection_results.csv `
        --out_dir .\figures

    # bash/zsh (multi-line)
    python visualize_facet_results.py \
        --reconstruction_csv ./reconstruction_results.csv \
        --temporal_csv ./temporal_gap_results.csv \
        --classification_csv ./classification_lead_selection_results.csv \
        --out_dir ./figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
GAP_ORDER = ["0-1d", "1-7d", "7-30d", "1-6m", "6-12m", "1y+"]
METHOD_ORDER = [
    "lead_i",
    "lead_i_meta",
    "concat",
    "past_as_output",
    "no_physics",
    "facet",
    "facet_meta",
]
SOURCE_ORDER = [
    "lead_i",
    "past12",
    "recon_random_selected",
    "recon_all",
    "recon_selected",
    "real12",
]


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def savefig(out_dir: Path, name: str) -> None:
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    plt.tight_layout()
    plt.savefig(png, dpi=240, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {png}")
    print(f"[SAVE] {pdf}")


def ordered_categories(values: Iterable[str], preferred: Sequence[str]) -> List[str]:
    values = [str(v) for v in values if pd.notna(v)]
    seen = set(values)
    ordered = [x for x in preferred if x in seen]
    ordered += sorted([x for x in seen if x not in ordered])
    return ordered


def pretty_method(x: str) -> str:
    mapping = {
        "lead_i": "Lead I",
        "lead_i_meta": "Lead I + meta",
        "concat": "Concat",
        "past_as_output": "Past copy",
        "facet": "FACET",
        "facet_meta": "FACET + meta",
        "no_physics": "No physics",
    }
    return mapping.get(str(x), str(x))


def pretty_source(x: str) -> str:
    mapping = {
        "real12": "Real 12-lead",
        "lead_i": "Lead I",
        "past12": "Past 12-lead",
        "recon_all": "Recon all",
        "recon_selected": "Recon selected",
        "recon_random_selected": "Random selected",
    }
    return mapping.get(str(x), str(x))


def plot_grouped_bars(df: pd.DataFrame, x_col: str, metrics: Sequence[str], title: str, out_dir: Path, name: str) -> None:
    available = [m for m in metrics if m in df.columns]
    if not available or x_col not in df.columns or len(df) == 0:
        print(f"[SKIP] {name}: required columns missing")
        return

    x = np.arange(len(df))
    width = 0.8 / len(available)
    plt.figure(figsize=(max(8, len(df) * 0.8), 4.8))
    for i, metric in enumerate(available):
        offset = (i - (len(available) - 1) / 2) * width
        plt.bar(x + offset, df[metric].astype(float), width=width, label=metric)
    plt.xticks(x, df[x_col].astype(str), rotation=35, ha="right")
    plt.title(title)
    plt.ylabel("Score")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False)
    savefig(out_dir, name)


def plot_heatmap(matrix: np.ndarray, row_labels: Sequence[str], col_labels: Sequence[str], title: str, out_dir: Path, name: str, cmap: str = "viridis") -> None:
    if matrix.size == 0:
        print(f"[SKIP] {name}: empty matrix")
        return
    plt.figure(figsize=(max(8, len(col_labels) * 0.65), max(3.5, len(row_labels) * 0.45)))
    im = plt.imshow(matrix, aspect="auto", cmap=cmap)
    plt.colorbar(im, fraction=0.025, pad=0.02)
    plt.xticks(np.arange(len(col_labels)), col_labels, rotation=35, ha="right")
    plt.yticks(np.arange(len(row_labels)), row_labels)
    plt.title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if val > np.nanmean(matrix) else "black")
    savefig(out_dir, name)


def visualize_reconstruction(path: Path, out_dir: Path) -> None:
    if not path.exists():
        print(f"[SKIP] reconstruction file not found: {path}")
        return

    df = pd.read_csv(path)
    if "method" not in df.columns:
        print("[SKIP] reconstruction: no method column")
        return

    main = df.copy()
    if "ablation" in main.columns:
        main = main[main["ablation"].fillna("none").eq("none")]
    if "past_mode" in main.columns:
        main = main[main["past_mode"].fillna("self").eq("self")]
    order = ordered_categories(main["method"].tolist(), METHOD_ORDER)
    main["method"] = pd.Categorical(main["method"], categories=order, ordered=True)
    main = main.sort_values("method")
    main["label"] = main["method"].astype(str).map(pretty_method)

    plot_grouped_bars(
        main,
        x_col="label",
        metrics=["mae_all", "mae_v1_v4", "mae_chest"],
        title="Reconstruction Error by Method",
        out_dir=out_dir,
        name="reconstruction_mae_by_method",
    )
    plot_grouped_bars(
        main,
        x_col="label",
        metrics=["pearson_all", "pearson_v1_v4", "pearson_chest"],
        title="Reconstruction Correlation by Method",
        out_dir=out_dir,
        name="reconstruction_pearson_by_method",
    )

    lead_cols = [f"mae_{lead}" for lead in LEADS if f"mae_{lead}" in main.columns]
    if lead_cols:
        matrix = main[lead_cols].astype(float).to_numpy()
        plot_heatmap(
            matrix,
            row_labels=main["label"].tolist(),
            col_labels=[c.replace("mae_", "") for c in lead_cols],
            title="Per-Lead MAE",
            out_dir=out_dir,
            name="reconstruction_per_lead_mae_heatmap",
            cmap="magma_r",
        )

    pearson_cols = [f"pearson_{lead}" for lead in LEADS if f"pearson_{lead}" in main.columns]
    if pearson_cols:
        matrix = main[pearson_cols].astype(float).to_numpy()
        plot_heatmap(
            matrix,
            row_labels=main["label"].tolist(),
            col_labels=[c.replace("pearson_", "") for c in pearson_cols],
            title="Per-Lead Pearson Correlation",
            out_dir=out_dir,
            name="reconstruction_per_lead_pearson_heatmap",
            cmap="viridis",
        )

    if "ablation" in df.columns:
        ab = df[df["method"].astype(str).str.contains("facet", case=False, na=False)].copy()
        if len(ab):
            ab["condition"] = ab["ablation"].fillna("none").astype(str)
            if "past_mode" in ab.columns:
                ab["condition"] = ab["condition"] + " / " + ab["past_mode"].fillna("").astype(str)
            ab = ab.drop_duplicates(subset=["condition"], keep="first")
            plot_grouped_bars(
                ab,
                x_col="condition",
                metrics=["mae_all", "mae_v1_v4", "pearson_all"],
                title="FACET Ablations",
                out_dir=out_dir,
                name="facet_ablation_summary",
            )


def visualize_temporal(path: Path, out_dir: Path) -> None:
    if not path.exists():
        print(f"[SKIP] temporal file not found: {path}")
        return
    df = pd.read_csv(path)
    required = {"method", "gap_bin"}
    if not required.issubset(df.columns):
        print("[SKIP] temporal: required columns missing")
        return

    gap_order = [g for g in GAP_ORDER if g in set(df["gap_bin"].astype(str))]
    extra = sorted(set(df["gap_bin"].astype(str)) - set(gap_order))
    gap_order += extra
    method_order = ordered_categories(df["method"].tolist(), METHOD_ORDER)

    for metric, ylabel, name in [
        ("mae_all", "MAE", "temporal_mae_all"),
        ("mae_v1_v4", "MAE V1-V4", "temporal_mae_v1_v4"),
        ("pearson_all", "Pearson", "temporal_pearson_all"),
    ]:
        if metric not in df.columns:
            continue
        plt.figure(figsize=(8, 4.8))
        for method in method_order:
            sub = df[df["method"].astype(str).eq(method)].copy()
            sub["gap_bin"] = pd.Categorical(sub["gap_bin"].astype(str), categories=gap_order, ordered=True)
            sub = sub.sort_values("gap_bin")
            plt.plot(sub["gap_bin"].astype(str), sub[metric].astype(float), marker="o", linewidth=2, label=pretty_method(method))
        plt.title(f"Temporal Gap Robustness: {ylabel}")
        plt.xlabel("Past-current time gap")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.25)
        plt.legend(frameon=False)
        savefig(out_dir, name)

    if "n_pairs" in df.columns:
        counts = df.drop_duplicates("gap_bin").copy()
        counts["gap_bin"] = pd.Categorical(counts["gap_bin"].astype(str), categories=gap_order, ordered=True)
        counts = counts.sort_values("gap_bin")
        plt.figure(figsize=(7.5, 3.8))
        plt.bar(counts["gap_bin"].astype(str), counts["n_pairs"].astype(float), color="#5B8DEF")
        plt.title("Number of Test Pairs by Temporal Gap")
        plt.xlabel("Past-current time gap")
        plt.ylabel("Pairs")
        plt.grid(axis="y", alpha=0.25)
        savefig(out_dir, "temporal_gap_pair_counts")


def disease_names_from_columns(df: pd.DataFrame) -> List[str]:
    names = []
    for col in df.columns:
        if col.startswith("auroc_"):
            label = col.replace("auroc_", "")
            if label not in names:
                names.append(label)
    return names


def visualize_classification(path: Path, out_dir: Path) -> None:
    if not path.exists():
        print(f"[SKIP] classification file not found: {path}")
        return
    df = pd.read_csv(path)
    if "source" not in df.columns:
        print("[SKIP] classification: no source column")
        return
    order = ordered_categories(df["source"].tolist(), SOURCE_ORDER)
    df["source"] = pd.Categorical(df["source"], categories=order, ordered=True)
    df = df.sort_values("source")
    df["label"] = df["source"].astype(str).map(pretty_source)

    plot_grouped_bars(
        df,
        x_col="label",
        metrics=["macro_auroc", "macro_auprc", "macro_f1"],
        title="Downstream Classification by Input Source",
        out_dir=out_dir,
        name="classification_macro_metrics",
    )

    diseases = disease_names_from_columns(df)
    for prefix, title, name in [
        ("auroc", "Per-Disease AUROC", "classification_per_disease_auroc"),
        ("auprc", "Per-Disease AUPRC", "classification_per_disease_auprc"),
        ("f1", "Per-Disease F1", "classification_per_disease_f1"),
    ]:
        cols = [f"{prefix}_{d}" for d in diseases if f"{prefix}_{d}" in df.columns]
        if not cols:
            continue
        matrix = df[cols].astype(float).to_numpy()
        plot_heatmap(
            matrix,
            row_labels=df["label"].tolist(),
            col_labels=[c.replace(f"{prefix}_", "") for c in cols],
            title=title,
            out_dir=out_dir,
            name=name,
            cmap="viridis",
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="./outputs/facet_full", help="Folder containing the three result CSV files.")
    p.add_argument("--reconstruction_csv", default="", help="Optional explicit reconstruction_results.csv path.")
    p.add_argument("--temporal_csv", default="", help="Optional explicit temporal_gap_results.csv path.")
    p.add_argument("--classification_csv", default="", help="Optional explicit classification_lead_selection_results.csv path.")
    p.add_argument("--out_dir", default="", help="Output directory for figures. Default: <results_dir>/figures.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "figures"
    ensure_out_dir(out_dir)

    reconstruction_csv = Path(args.reconstruction_csv) if args.reconstruction_csv else results_dir / "reconstruction_results.csv"
    temporal_csv = Path(args.temporal_csv) if args.temporal_csv else results_dir / "temporal_gap_results.csv"
    classification_csv = Path(args.classification_csv) if args.classification_csv else results_dir / "classification_lead_selection_results.csv"

    visualize_reconstruction(reconstruction_csv, out_dir)
    visualize_temporal(temporal_csv, out_dir)
    visualize_classification(classification_csv, out_dir)
    print(f"[DONE] figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
