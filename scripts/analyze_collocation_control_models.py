#!/usr/bin/env python3
"""Controlled analyses for COCA local-collocation support.

The goal is deliberately modest: quantify whether local support still predicts
model behavior after accounting for unigram Zipf and length diagnostics already
saved in the collocation-support table.  We use dependency-light linear models
instead of statsmodels so this runs in the local environment.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "results/coca_collocation_support/controlled_models/.mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("/Users/tyronewhite/masters_research_code/blimp-rare/results/coca_collocation_support/model_item_support_scores.csv")
DEFAULT_OUT_DIR = Path("/Users/tyronewhite/masters_research_code/blimp-rare/results/coca_collocation_support/controlled_models")

OUTCOMES = ["correctness", "margin_logprob"]
MAIN_FEATURE = "local_support_mean_good"
CONTROLS = ["good_zipf_mean", "good_char_count", "good_token_count"]
FEATURES = [MAIN_FEATURE] + CONTROLS


def _standardize(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        return pd.Series(np.nan, index=values.index)
    return (values - float(values.mean())) / std


def _demean(df: pd.DataFrame, cols: Sequence[str], group_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    group_means = out.groupby(list(group_cols), observed=True)[list(cols)].transform("mean")
    out[list(cols)] = out[list(cols)].astype(float) - group_means.astype(float)
    return out


def _ols_standardized(df: pd.DataFrame, outcome: str, features: Sequence[str]) -> dict:
    work = df[[outcome, *features]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(work) < 50:
        return {"n": len(work), "r2": np.nan, **{f"beta_{f}": np.nan for f in features}}

    y = _standardize(work[outcome])
    x_cols: List[np.ndarray] = []
    valid_features: List[str] = []
    for feature in features:
        x = _standardize(work[feature])
        if x.notna().all():
            x_cols.append(x.to_numpy())
            valid_features.append(feature)
    if not x_cols:
        return {"n": len(work), "r2": np.nan, **{f"beta_{f}": np.nan for f in features}}

    xmat = np.column_stack(x_cols)
    yvec = y.to_numpy()
    coef, *_ = np.linalg.lstsq(xmat, yvec, rcond=None)
    pred = xmat @ coef
    sse = float(np.sum((yvec - pred) ** 2))
    sst = float(np.sum((yvec - yvec.mean()) ** 2))
    out = {"n": int(len(work)), "r2": float(1.0 - sse / sst) if sst else np.nan}
    for feature in features:
        out[f"beta_{feature}"] = np.nan
    for feature, beta in zip(valid_features, coef):
        out[f"beta_{feature}"] = float(beta)
    return out


def _fit_specs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    specs = [
        ("raw", []),
        ("uid_regime_fixed_effects", ["uid", "regime"]),
    ]

    for model, model_df in df.groupby("model_slug", observed=True):
        for outcome in OUTCOMES:
            for spec_name, group_cols in specs:
                needed = [outcome, *FEATURES]
                part = model_df.dropna(subset=needed).copy()
                if group_cols:
                    part = _demean(part, needed, group_cols)

                full = _ols_standardized(part, outcome, FEATURES)
                controls_only = _ols_standardized(part, outcome, CONTROLS)
                local_only = _ols_standardized(part, outcome, [MAIN_FEATURE])

                rows.append(
                    {
                        "model": model,
                        "outcome": outcome,
                        "spec": spec_name,
                        "n": full["n"],
                        "beta_local_support": full[f"beta_{MAIN_FEATURE}"],
                        "beta_zipf": full["beta_good_zipf_mean"],
                        "beta_char_count": full["beta_good_char_count"],
                        "beta_token_count": full["beta_good_token_count"],
                        "r2_full": full["r2"],
                        "r2_controls_only": controls_only["r2"],
                        "delta_r2_local_over_controls": full["r2"] - controls_only["r2"],
                        "r2_local_only": local_only["r2"],
                    }
                )

    pooled = df.copy()
    pooled["_model_uid_regime"] = pooled["model_slug"].astype(str) + "::" + pooled["uid"].astype(str) + "::" + pooled["regime"].astype(str)
    for outcome in OUTCOMES:
        needed = [outcome, *FEATURES]
        part = pooled.dropna(subset=needed).copy()
        part = _demean(part, needed, ["_model_uid_regime"])
        full = _ols_standardized(part, outcome, FEATURES)
        controls_only = _ols_standardized(part, outcome, CONTROLS)
        local_only = _ols_standardized(part, outcome, [MAIN_FEATURE])
        rows.append(
            {
                "model": "pooled",
                "outcome": outcome,
                "spec": "model_uid_regime_fixed_effects",
                "n": full["n"],
                "beta_local_support": full[f"beta_{MAIN_FEATURE}"],
                "beta_zipf": full["beta_good_zipf_mean"],
                "beta_char_count": full["beta_good_char_count"],
                "beta_token_count": full["beta_good_token_count"],
                "r2_full": full["r2"],
                "r2_controls_only": controls_only["r2"],
                "delta_r2_local_over_controls": full["r2"] - controls_only["r2"],
                "r2_local_only": local_only["r2"],
            }
        )
    return pd.DataFrame(rows)


def _uid_gap_controls(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "model_slug",
        "uid",
        "regime",
        "correctness",
        "margin_logprob",
        MAIN_FEATURE,
        "good_zipf_mean",
        "good_char_count",
        "good_token_count",
    ]
    agg = (
        df[cols]
        .dropna()
        .groupby(["model_slug", "uid", "regime"], observed=True, as_index=False)
        .agg(
            accuracy=("correctness", "mean"),
            margin=("margin_logprob", "mean"),
            local_support=(MAIN_FEATURE, "mean"),
            zipf=("good_zipf_mean", "mean"),
            char_count=("good_char_count", "mean"),
            token_count=("good_token_count", "mean"),
            n=("correctness", "size"),
        )
    )
    original = agg[agg["regime"].eq("original")].rename(
        columns={
            "accuracy": "accuracy_original",
            "margin": "margin_original",
            "local_support": "local_support_original",
            "zipf": "zipf_original",
            "char_count": "char_count_original",
            "token_count": "token_count_original",
        }
    )
    freq = agg[~agg["regime"].eq("original")].rename(
        columns={
            "accuracy": "accuracy_freq",
            "margin": "margin_freq",
            "local_support": "local_support_freq",
            "zipf": "zipf_freq",
            "char_count": "char_count_freq",
            "token_count": "token_count_freq",
        }
    )
    merged = freq.merge(original, on=["model_slug", "uid"], how="inner", suffixes=("", "_origrow"))
    merged["accuracy_gap"] = merged["accuracy_original"] - merged["accuracy_freq"]
    merged["margin_gap"] = merged["margin_original"] - merged["margin_freq"]
    merged["local_support_gap"] = merged["local_support_original"] - merged["local_support_freq"]
    merged["zipf_gap"] = merged["zipf_original"] - merged["zipf_freq"]
    merged["char_count_gap"] = merged["char_count_original"] - merged["char_count_freq"]
    merged["token_count_gap"] = merged["token_count_original"] - merged["token_count_freq"]

    rows = []
    for model, model_df in merged.groupby("model_slug", observed=True):
        for regime, part in model_df.groupby("regime", observed=True):
            for outcome in ["accuracy_gap", "margin_gap"]:
                full = _ols_standardized(part, outcome, ["local_support_gap", "zipf_gap", "char_count_gap", "token_count_gap"])
                controls = _ols_standardized(part, outcome, ["zipf_gap", "char_count_gap", "token_count_gap"])
                rows.append(
                    {
                        "model": model,
                        "regime": regime,
                        "outcome": outcome,
                        "n_uid": full["n"],
                        "beta_local_support_gap": full["beta_local_support_gap"],
                        "beta_zipf_gap": full["beta_zipf_gap"],
                        "beta_char_count_gap": full["beta_char_count_gap"],
                        "beta_token_count_gap": full["beta_token_count_gap"],
                        "r2_full": full["r2"],
                        "delta_r2_local_over_controls": full["r2"] - controls["r2"],
                    }
                )
    return pd.DataFrame(rows), merged


def _plot_coefficients(coefs: pd.DataFrame, out_dir: Path) -> None:
    plot_df = coefs[coefs["spec"].eq("uid_regime_fixed_effects")].copy()
    plot_df = plot_df[plot_df["outcome"].isin(OUTCOMES)]
    if plot_df.empty:
        return
    models = list(plot_df["model"].unique())
    outcomes = OUTCOMES
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    if len(outcomes) == 1:
        axes = [axes]
    x = np.arange(len(models))
    for ax, outcome in zip(axes, outcomes):
        part = plot_df[plot_df["outcome"].eq(outcome)].set_index("model").reindex(models)
        ax.axhline(0, color="#374151", lw=0.8, alpha=0.7)
        ax.bar(x, part["beta_local_support"], color="#2563A6", alpha=0.82)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.set_title("Accuracy" if outcome == "correctness" else "Logprob margin")
        ax.set_ylabel("Standardized beta for local support")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Local Support Predicts Behavior After Zipf/Length Controls")
    fig.tight_layout()
    fig.savefig(out_dir / "controlled_local_support_betas.png", dpi=300)
    fig.savefig(out_dir / "controlled_local_support_betas.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / ".mplconfig").mkdir(exist_ok=True)

    usecols = [
        "model_slug",
        "uid",
        "regime",
        "correctness",
        "margin_logprob",
        "local_support_mean_good",
        "good_zipf_mean",
        "good_char_count",
        "good_token_count",
    ]
    df = pd.read_csv(args.input, usecols=usecols)

    coefs = _fit_specs(df)
    coefs.to_csv(args.out_dir / "controlled_item_level_models.csv", index=False)
    gap_coefs, gap_points = _uid_gap_controls(df)
    gap_coefs.to_csv(args.out_dir / "controlled_uid_gap_models.csv", index=False)
    gap_points.to_csv(args.out_dir / "controlled_uid_gap_points.csv", index=False)
    _plot_coefficients(coefs, args.out_dir)

    print(f"Wrote controlled collocation analyses to {args.out_dir}")


if __name__ == "__main__":
    main()
