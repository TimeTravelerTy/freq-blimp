#!/usr/bin/env python3
"""Analyze good-vs-bad asymmetry in COCA local support."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "results/coca_collocation_support/good_bad_asymmetry/.mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ITEM_SUPPORT = Path("/Users/tyronewhite/masters_research_code/blimp-rare/results/coca_collocation_support/item_local_support.csv")
DEFAULT_SCORE_SUPPORT = Path("/Users/tyronewhite/masters_research_code/blimp-rare/results/coca_collocation_support/model_item_support_scores.csv")
DEFAULT_OUT_DIR = Path("/Users/tyronewhite/masters_research_code/blimp-rare/results/coca_collocation_support/good_bad_asymmetry")

REGIME_ORDER = ["original", "head", "tail", "xtail"]
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
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def _standardize(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        return pd.Series(np.nan, index=values.index)
    return (values - float(values.mean())) / std


def _demean(df: pd.DataFrame, cols: Sequence[str], groups: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    out[list(cols)] = out[list(cols)].astype(float) - out.groupby(list(groups), observed=True)[list(cols)].transform("mean")
    return out


def _ols(df: pd.DataFrame, outcome: str, features: Sequence[str]) -> dict:
    work = df[[outcome, *features]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(work) < 50:
        return {"n": len(work), "r2": np.nan, **{f"beta_{f}": np.nan for f in features}}
    y = _standardize(work[outcome])
    xs = []
    valid = []
    for feature in features:
        x = _standardize(work[feature])
        if x.notna().all():
            xs.append(x.to_numpy())
            valid.append(feature)
    if not xs:
        return {"n": len(work), "r2": np.nan, **{f"beta_{f}": np.nan for f in features}}
    xmat = np.column_stack(xs)
    coef, *_ = np.linalg.lstsq(xmat, y.to_numpy(), rcond=None)
    pred = xmat @ coef
    sst = float(np.sum((y.to_numpy() - y.mean()) ** 2))
    sse = float(np.sum((y.to_numpy() - pred) ** 2))
    out = {"n": int(len(work)), "r2": float(1.0 - sse / sst) if sst else np.nan}
    for feature in features:
        out[f"beta_{feature}"] = np.nan
    for feature, beta in zip(valid, coef):
        out[f"beta_{feature}"] = float(beta)
    return out


def _asymmetry_summary(item_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    delta = "local_support_mean_delta_good_minus_bad"
    for regime, part in item_df.groupby("regime", observed=True):
        vals = part[delta].dropna()
        rows.append(
            {
                "regime": regime,
                "n": int(len(vals)),
                "mean_delta_good_minus_bad": float(vals.mean()),
                "median_delta_good_minus_bad": float(vals.median()),
                "p10": float(vals.quantile(0.10)),
                "p90": float(vals.quantile(0.90)),
                "share_good_higher": float((vals > 0).mean()),
                "share_near_tie_abs_lt_0_25": float((vals.abs() < 0.25).mean()),
                "mean_good_support": float(part["local_support_mean_good"].mean()),
                "mean_bad_support": float(part["local_support_mean_bad"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("regime", key=lambda s: s.map({r: i for i, r in enumerate(REGIME_ORDER)}))


def _behavior_by_delta_bin(score_df: pd.DataFrame) -> pd.DataFrame:
    df = score_df.dropna(subset=["local_support_mean_delta_good_minus_bad", "correctness", "margin_logprob"]).copy()
    df["delta_bin"] = pd.cut(
        df["local_support_mean_delta_good_minus_bad"],
        bins=[-np.inf, -1.0, -0.25, 0.25, 1.0, np.inf],
        labels=["bad much higher", "bad slightly higher", "near tie", "good slightly higher", "good much higher"],
    )
    return (
        df.groupby(["model_slug", "regime", "delta_bin"], observed=True)
        .agg(
            accuracy=("correctness", "mean"),
            margin=("margin_logprob", "mean"),
            n=("correctness", "size"),
        )
        .reset_index()
    )


def _controlled_models(score_df: pd.DataFrame) -> pd.DataFrame:
    df = score_df.copy()
    df["zipf_delta_good_minus_bad"] = df["good_zipf_mean"] - df["bad_zipf_mean"]
    df["char_delta_good_minus_bad"] = df["good_char_count"] - df["bad_char_count"]
    df["token_delta_good_minus_bad"] = df["good_token_count"] - df["bad_token_count"]
    features = [
        "local_support_mean_delta_good_minus_bad",
        "zipf_delta_good_minus_bad",
        "char_delta_good_minus_bad",
        "token_delta_good_minus_bad",
    ]
    rows = []
    for model, part in df.groupby("model_slug", observed=True):
        for outcome in ["correctness", "margin_logprob"]:
            needed = [outcome, *features]
            raw = part.dropna(subset=needed).copy()
            fe = _demean(raw, needed, ["uid", "regime"])
            full = _ols(fe, outcome, features)
            controls = _ols(fe, outcome, features[1:])
            rows.append(
                {
                    "model": model,
                    "outcome": outcome,
                    "spec": "uid_regime_fixed_effects",
                    "n": full["n"],
                    "beta_support_delta": full["beta_local_support_mean_delta_good_minus_bad"],
                    "beta_zipf_delta": full["beta_zipf_delta_good_minus_bad"],
                    "beta_char_delta": full["beta_char_delta_good_minus_bad"],
                    "beta_token_delta": full["beta_token_delta_good_minus_bad"],
                    "r2_full": full["r2"],
                    "delta_r2_support_over_controls": full["r2"] - controls["r2"],
                }
            )
    return pd.DataFrame(rows)


def _plot_delta_distribution(item_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    data = [
        item_df.loc[item_df["regime"].eq(regime), "local_support_mean_delta_good_minus_bad"].dropna().clip(-5, 5).to_numpy()
        for regime in REGIME_ORDER
    ]
    positions = np.arange(len(REGIME_ORDER))
    vp = ax.violinplot(data, positions=positions, widths=0.78, showmeans=False, showmedians=False, showextrema=False)
    for body, regime in zip(vp["bodies"], REGIME_ORDER):
        body.set_facecolor(REGIME_COLORS[regime])
        body.set_edgecolor(REGIME_COLORS[regime])
        body.set_alpha(0.35)
    ax.boxplot(
        data,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 1.7},
        boxprops={"facecolor": "white", "edgecolor": "#111827", "linewidth": 1.0},
        whiskerprops={"color": "#111827", "linewidth": 1.0},
        capprops={"color": "#111827", "linewidth": 1.0},
    )
    ax.axhline(0, color="#111827", linewidth=1.0, alpha=0.75)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Original", "Head", "Tail", "XTail"])
    ax.set_ylabel("Good - bad local support")
    ax.set_title("Good-Bad COCA Support Asymmetry")
    ax.text(0.02, 0.97, "Positive = grammatical item has higher support", transform=ax.transAxes, va="top", color="#4B5563")
    fig.tight_layout()
    fig.savefig(out_dir / "good_bad_support_delta_distribution.png")
    fig.savefig(out_dir / "good_bad_support_delta_distribution.pdf")
    plt.close(fig)


def _plot_behavior_bins(bin_df: pd.DataFrame, out_dir: Path) -> None:
    pooled = (
        bin_df.groupby("delta_bin", observed=True)
        .agg(accuracy=("accuracy", "mean"), margin=("margin", "mean"), n=("n", "sum"))
        .reset_index()
    )
    labels = [str(x) for x in pooled["delta_bin"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    axes[0].plot(x, pooled["accuracy"] * 100, marker="o", color="#2563A6")
    axes[0].set_ylabel("Accuracy (%)")
    axes[1].plot(x, pooled["margin"], marker="o", color="#D97706")
    axes[1].set_ylabel("Mean logprob margin")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_xlabel("Good-bad support delta bin")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Model Behavior by Good-Bad Support Asymmetry")
    fig.tight_layout()
    fig.savefig(out_dir / "behavior_by_good_bad_support_delta_bin.png")
    fig.savefig(out_dir / "behavior_by_good_bad_support_delta_bin.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item-support", type=Path, default=DEFAULT_ITEM_SUPPORT)
    parser.add_argument("--score-support", type=Path, default=DEFAULT_SCORE_SUPPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / ".mplconfig").mkdir(exist_ok=True)
    _style()

    item_cols = [
        "regime",
        "uid",
        "pair_id",
        "local_support_mean_good",
        "local_support_mean_bad",
        "local_support_mean_delta_good_minus_bad",
    ]
    score_cols = [
        "model_slug",
        "regime",
        "uid",
        "pair_id",
        "correctness",
        "margin_logprob",
        "local_support_mean_delta_good_minus_bad",
        "good_zipf_mean",
        "bad_zipf_mean",
        "good_char_count",
        "bad_char_count",
        "good_token_count",
        "bad_token_count",
    ]
    item_df = pd.read_csv(args.item_support, usecols=item_cols)
    score_df = pd.read_csv(args.score_support, usecols=score_cols)

    summary = _asymmetry_summary(item_df)
    summary.to_csv(args.out_dir / "good_bad_support_asymmetry_summary.csv", index=False)
    bins = _behavior_by_delta_bin(score_df)
    bins.to_csv(args.out_dir / "behavior_by_good_bad_support_delta_bin.csv", index=False)
    controlled = _controlled_models(score_df)
    controlled.to_csv(args.out_dir / "controlled_good_bad_support_delta_models.csv", index=False)
    _plot_delta_distribution(item_df, args.out_dir)
    _plot_behavior_bins(bins, args.out_dir)
    print(f"Wrote good-bad support asymmetry analyses to {args.out_dir}")


if __name__ == "__main__":
    main()
