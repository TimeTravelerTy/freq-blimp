#!/usr/bin/env python3
"""Legible item-level COCA-support analyses.

These plots avoid item-level scatter clouds. They residualize item behavior and
support within paradigm-by-regime, control for Zipf/length confounds, then plot
binned trends over residual local support.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "results/coca_collocation_support/item_level_effects/.mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("/Users/tyronewhite/masters_research_code/blimp-rare/results/coca_collocation_support/model_item_support_scores.csv")
DEFAULT_OUT_DIR = Path("/Users/tyronewhite/masters_research_code/blimp-rare/results/coca_collocation_support/item_level_effects")
MODEL_ORDER = ["Llama-3_1-8B", "Mistral-7B-v0_1", "gemma-4-E4B"]
MODEL_COLORS = {
    "Llama-3_1-8B": "#2563A6",
    "Mistral-7B-v0_1": "#D97706",
    "gemma-4-E4B": "#7C3AED",
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


def _rank_corr(left: pd.Series, right: pd.Series) -> float:
    ranks = pd.DataFrame({"left": left, "right": right}).rank(method="average")
    return float(ranks["left"].corr(ranks["right"]))


def _residualize_against_controls(df: pd.DataFrame, target: str, controls: Sequence[str], groups: Sequence[str]) -> pd.Series:
    needed = [target, *controls]
    work = df[needed + list(groups)].copy()
    group_means = work.groupby(list(groups), observed=True)[needed].transform("mean")
    demeaned = work[needed].astype(float) - group_means.astype(float)
    y = demeaned[target].to_numpy(dtype=float)
    x = demeaned[list(controls)].to_numpy(dtype=float)
    if x.shape[1] == 0:
        resid = y
    else:
        valid_cols = np.nanstd(x, axis=0) > 0
        x = x[:, valid_cols]
        if x.shape[1] == 0:
            resid = y
        else:
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            resid = y - x @ coef
    return pd.Series(resid, index=df.index)


def _standardize(values: pd.Series) -> pd.Series:
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        return pd.Series(np.nan, index=values.index)
    return (values - float(values.mean())) / std


def _residual_dataset(df: pd.DataFrame, feature: str, controls: Sequence[str], outcome: str) -> pd.DataFrame:
    needed = ["model_slug", "uid", "regime", outcome, feature, *controls]
    sub = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    rows = []
    for model, part in sub.groupby("model_slug", observed=True):
        part = part.copy()
        part["x_resid"] = _residualize_against_controls(part, feature, controls, ["uid", "regime"])
        part["y_resid"] = _residualize_against_controls(part, outcome, controls, ["uid", "regime"])
        part["x_resid_z"] = _standardize(part["x_resid"])
        part["model_slug"] = model
        part["outcome"] = outcome
        part["feature"] = feature
        rows.append(part[["model_slug", "outcome", "feature", "x_resid", "x_resid_z", "y_resid"]])
    return pd.concat(rows, ignore_index=True)


def _binned(df: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    rows = []
    for (model, outcome, feature), part in df.groupby(["model_slug", "outcome", "feature"], observed=True):
        part = part.dropna(subset=["x_resid_z", "y_resid"]).copy()
        if len(part) < bins or part["x_resid_z"].nunique() < 4:
            continue
        part["bin"] = pd.qcut(part["x_resid_z"], q=bins, labels=False, duplicates="drop") + 1
        for bin_i, bpart in part.groupby("bin", observed=True):
            y = bpart["y_resid"].astype(float)
            rows.append(
                {
                    "model_slug": model,
                    "outcome": outcome,
                    "feature": feature,
                    "bin": int(bin_i),
                    "x_mean": float(bpart["x_resid_z"].mean()),
                    "y_mean": float(y.mean()),
                    "y_se": float(y.std(ddof=1) / np.sqrt(len(y))) if len(y) > 1 else 0.0,
                    "n": int(len(y)),
                }
            )
    return pd.DataFrame(rows)


def _correlations(resid_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, outcome, feature), part in resid_df.groupby(["model_slug", "outcome", "feature"], observed=True):
        sub = part[["x_resid", "y_resid"]].dropna()
        if len(sub) < 10 or sub["x_resid"].nunique() < 2:
            continue
        xz = _standardize(sub["x_resid"])
        yz = _standardize(sub["y_resid"])
        beta = float(xz.corr(yz))
        rows.append(
            {
                "model_slug": model,
                "outcome": outcome,
                "feature": feature,
                "n": int(len(sub)),
                "residual_pearson_r": float(sub["x_resid"].corr(sub["y_resid"])),
                "residual_spearman_rho": _rank_corr(sub["x_resid"], sub["y_resid"]),
                "single_predictor_standardized_beta": beta,
            }
        )
    return pd.DataFrame(rows)


def _plot_binned(binned: pd.DataFrame, feature: str, out_dir: Path, stem: str, title: str, x_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.75), sharex=True)
    outcomes = [("correctness", "Accuracy residual (percentage points)"), ("margin_logprob", "Logprob-margin residual")]
    for ax, (outcome, ylabel) in zip(axes, outcomes):
        part = binned[(binned["feature"].eq(feature)) & (binned["outcome"].eq(outcome))]
        for model in MODEL_ORDER:
            sub = part[part["model_slug"].eq(model)].sort_values("x_mean")
            if sub.empty:
                continue
            y = sub["y_mean"].to_numpy()
            se = sub["y_se"].to_numpy()
            if outcome == "correctness":
                y = y * 100.0
                se = se * 100.0
            color = MODEL_COLORS.get(model, "#374151")
            ax.plot(sub["x_mean"], y, marker="o", ms=3.2, lw=1.7, color=color, label=model)
            ax.fill_between(sub["x_mean"], y - 1.96 * se, y + 1.96 * se, color=color, alpha=0.10, linewidth=0)
        ax.axhline(0, color="#111827", lw=0.8, alpha=0.7)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", alpha=0.18)
    axes[1].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def _plot_deciles(df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    feature = "local_support_mean_good"
    for (model, regime), part in df.dropna(subset=[feature, "correctness", "margin_logprob"]).groupby(["model_slug", "regime"], observed=True):
        if part[feature].nunique() < 5:
            continue
        part = part.copy()
        part["decile"] = pd.qcut(part[feature], q=10, labels=False, duplicates="drop") + 1
        agg = (
            part.groupby("decile", observed=True)
            .agg(accuracy=("correctness", "mean"), margin=("margin_logprob", "mean"), n=("correctness", "size"))
            .reset_index()
        )
        agg["model_slug"] = model
        agg["regime"] = regime
        rows.append(agg)
    dec = pd.concat(rows, ignore_index=True)
    dec.to_csv(out_dir / "item_level_accuracy_by_support_decile.csv", index=False)

    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(10.2, 3.4), sharey=True)
    regime_colors = {"original": "#6B7280", "head": "#2563A6", "tail": "#D97706", "xtail": "#C2410C"}
    for ax, model in zip(axes, MODEL_ORDER):
        part = dec[dec["model_slug"].eq(model)]
        for regime, sub in part.groupby("regime", observed=True):
            ax.plot(sub["decile"], sub["accuracy"] * 100, marker="o", ms=3, lw=1.5, color=regime_colors.get(regime), label=regime)
        ax.set_title(model)
        ax.set_xlabel("Local-support decile")
        ax.grid(axis="x", alpha=0.18)
    axes[0].set_ylabel("Accuracy (%)")
    axes[-1].legend(frameon=False, title="Regime")
    fig.suptitle("Item-Level Accuracy by Local-Support Decile")
    fig.tight_layout()
    fig.savefig(out_dir / "item_level_accuracy_by_support_decile.png")
    fig.savefig(out_dir / "item_level_accuracy_by_support_decile.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / ".mplconfig").mkdir(exist_ok=True)
    _style()

    usecols = [
        "model_slug",
        "regime",
        "uid",
        "correctness",
        "margin_logprob",
        "local_support_mean_good",
        "local_support_mean_delta_good_minus_bad",
        "good_zipf_mean",
        "bad_zipf_mean",
        "good_char_count",
        "bad_char_count",
        "good_token_count",
        "bad_token_count",
    ]
    df = pd.read_csv(args.input, usecols=usecols)
    df["zipf_delta_good_minus_bad"] = df["good_zipf_mean"] - df["bad_zipf_mean"]
    df["char_delta_good_minus_bad"] = df["good_char_count"] - df["bad_char_count"]
    df["token_delta_good_minus_bad"] = df["good_token_count"] - df["bad_token_count"]

    residual_parts = []
    for outcome in ["correctness", "margin_logprob"]:
        residual_parts.append(
            _residual_dataset(
                df,
                "local_support_mean_good",
                ["good_zipf_mean", "good_char_count", "good_token_count"],
                outcome,
            )
        )
        residual_parts.append(
            _residual_dataset(
                df,
                "local_support_mean_delta_good_minus_bad",
                ["zipf_delta_good_minus_bad", "char_delta_good_minus_bad", "token_delta_good_minus_bad"],
                outcome,
            )
        )
    resid = pd.concat(residual_parts, ignore_index=True)
    corr = _correlations(resid)
    bins = _binned(resid, bins=20)
    corr.to_csv(args.out_dir / "item_level_residual_correlations.csv", index=False)
    bins.to_csv(args.out_dir / "item_level_residual_binned_trends.csv", index=False)

    _plot_binned(
        bins,
        "local_support_mean_good",
        args.out_dir,
        "item_level_residual_local_support_trend",
        "Item-Level Effect of Local COCA Support",
        "Residual local support (z)",
    )
    _plot_binned(
        bins,
        "local_support_mean_delta_good_minus_bad",
        args.out_dir,
        "item_level_residual_good_bad_support_delta_trend",
        "Item-Level Effect of Good-Bad Support Asymmetry",
        "Residual good-bad support delta (z)",
    )
    _plot_deciles(df, args.out_dir)
    print(f"Wrote item-level collocation effects to {args.out_dir}")


if __name__ == "__main__":
    main()
