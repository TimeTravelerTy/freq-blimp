#!/usr/bin/env python3
"""Make compact dataset-only regime diagnostics figures.

This script intentionally uses the cached dataset diagnostics points table
instead of reparsing generated JSONL files. It produces the minimum diagnostics
needed for the paper narrative: realized content-word Zipf distributions and
sentence-length controls by regime.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "results/regime_diagnostics_minimal/.mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REGIME_ORDER = ["original", "head", "tail", "xtail"]
REGIME_LABELS = {
    "original": "Original",
    "head": "Head",
    "tail": "Tail",
    "xtail": "XTail",
}
REGIME_COLORS = {
    "original": "#6B7280",
    "head": "#2563A6",
    "tail": "#D97706",
    "xtail": "#C2410C",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["regime"].isin(REGIME_ORDER)].copy()
    out["regime"] = pd.Categorical(out["regime"], categories=REGIME_ORDER, ordered=True)
    return out.sort_values("regime")


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, part in df.groupby("regime", observed=True):
        rows.append(
            {
                "regime": str(regime),
                "n": int(len(part)),
                "zipf_median_mean": float(part["zipf_median"].mean()),
                "zipf_median_p10": float(part["zipf_median"].quantile(0.10)),
                "zipf_median_p50": float(part["zipf_median"].quantile(0.50)),
                "zipf_median_p90": float(part["zipf_median"].quantile(0.90)),
                "char_count_mean": float(part["char_count"].mean()),
                "char_count_p25": float(part["char_count"].quantile(0.25)),
                "char_count_p75": float(part["char_count"].quantile(0.75)),
                "content_word_count_mean": float(part["content_word_count"].mean()),
                "content_word_count_p25": float(part["content_word_count"].quantile(0.25)),
                "content_word_count_p75": float(part["content_word_count"].quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def _plot_zipf(df: pd.DataFrame, out_dir: Path) -> None:
    data = [df.loc[df["regime"].eq(regime), "zipf_median"].dropna().to_numpy() for regime in REGIME_ORDER]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    positions = np.arange(len(REGIME_ORDER))
    vp = ax.violinplot(data, positions=positions, vert=False, widths=0.78, showmeans=False, showmedians=False, showextrema=False)
    for body, regime in zip(vp["bodies"], REGIME_ORDER):
        body.set_facecolor(REGIME_COLORS[regime])
        body.set_edgecolor(REGIME_COLORS[regime])
        body.set_alpha(0.32)
        body.set_linewidth(1.0)

    bp = ax.boxplot(
        data,
        positions=positions,
        vert=False,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 1.8},
        boxprops={"facecolor": "white", "edgecolor": "#111827", "linewidth": 1.0},
        whiskerprops={"color": "#111827", "linewidth": 1.0},
        capprops={"color": "#111827", "linewidth": 1.0},
    )
    for _patch in bp["boxes"]:
        _patch.set_alpha(0.88)

    target_windows = {
        "head": (3.5, 5.5),
        "tail": (2.4, 3.2),
        "xtail": (1.2, 2.2),
    }
    for regime, (low, high) in target_windows.items():
        idx = REGIME_ORDER.index(regime)
        ax.hlines(idx, low, high, color=REGIME_COLORS[regime], linewidth=5.0, alpha=0.35)

    ax.set_yticks(positions)
    ax.set_yticklabels([REGIME_LABELS[r] for r in REGIME_ORDER])
    ax.set_xlabel("Realized content-word median Zipf")
    ax.set_title("Realized Frequency Separates the Dataset Regimes")
    ax.set_xlim(5.8, 1.0)
    ax.grid(axis="y", visible=False)
    ax.text(
        0.99,
        0.02,
        "Rarer words to the right",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#4B5563",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "regime_diagnostics_zipf_distribution.png")
    fig.savefig(out_dir / "regime_diagnostics_zipf_distribution.pdf")
    plt.close(fig)


def _plot_lengths(summary: pd.DataFrame, out_dir: Path) -> None:
    summary = summary.set_index("regime").reindex(REGIME_ORDER).reset_index()
    labels = [REGIME_LABELS[r] for r in REGIME_ORDER]
    x = np.arange(len(REGIME_ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.45), sharex=True)
    specs = [
        ("char_count", "Characters per good sentence"),
        ("content_word_count", "Content words per good sentence"),
    ]
    for ax, (prefix, ylabel) in zip(axes, specs):
        mean = summary[f"{prefix}_mean"].to_numpy()
        low = summary[f"{prefix}_p25"].to_numpy()
        high = summary[f"{prefix}_p75"].to_numpy()
        colors = [REGIME_COLORS[r] for r in REGIME_ORDER]
        ax.vlines(x, low, high, color=colors, linewidth=4.0, alpha=0.35)
        ax.scatter(x, mean, s=58, color=colors, edgecolor="#111827", linewidth=0.6, zorder=3)
        ax.plot(x, mean, color="#9CA3AF", linewidth=1.0, zorder=2)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.grid(axis="x", visible=False)
    axes[0].set_title("Sentence Length")
    axes[1].set_title("Content-Word Count")
    fig.suptitle("Length Diagnostics Are Modest Relative to Frequency Shift", y=1.03, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out_dir / "regime_diagnostics_length_controls.png")
    fig.savefig(out_dir / "regime_diagnostics_length_controls.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--points",
        type=Path,
        default=Path(
            "/Users/tyronewhite/masters_research_code/freq-blimp/outputs/"
            "advisor_meeting_20260429_dropdiv_full/dataset_regime_diagnostics_points.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/Users/tyronewhite/masters_research_code/freq-blimp/outputs/"
            "advisor_meeting_20260429_dropdiv_full/regime_diagnostics_minimal"
        ),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / ".mplconfig").mkdir(exist_ok=True)
    _style()

    df = _ordered(pd.read_csv(args.points))
    summary = _summary(df)
    summary.to_csv(args.out_dir / "regime_diagnostics_minimal_summary.csv", index=False)
    _plot_zipf(df, args.out_dir)
    _plot_lengths(summary, args.out_dir)
    print(f"Wrote minimal regime diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
