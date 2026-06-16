#!/usr/bin/env python3
"""Paradigm-level decomposition of original-to-FreqBLiMP model gaps.

This treats each UID/paradigm as the unit of analysis and asks:

1. Which external/support/length predictors explain the original-Freq gap?
2. Is the gap broadly distributed, or concentrated in a small set of paradigms?
"""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "results/coca_collocation_support/paradigm_gap_decomposition/.mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("/Users/tyronewhite/masters_research_code/blimp-rare/results/coca_collocation_support")
DEFAULT_GAP_POINTS = BASE / "controlled_models/controlled_uid_gap_points.csv"
DEFAULT_ITEM_SUPPORT = BASE / "item_local_support.csv"
DEFAULT_OUT_DIR = BASE / "paradigm_gap_decomposition"

PREDICTORS = ["local_support_gap", "zipf_gap", "char_count_gap", "token_count_gap"]
OUTCOMES = ["accuracy_gap", "margin_gap"]
REGIME_ORDER = ["head", "tail", "xtail"]
MODEL_ORDER = ["Llama-3_1-8B", "Mistral-7B-v0_1", "gemma-4-E4B"]
REGIME_COLORS = {"head": "#2563A6", "tail": "#D97706", "xtail": "#C2410C"}


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


def _standardize(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        return pd.Series(np.nan, index=series.index)
    return (series.astype(float) - float(series.mean())) / std


def _ols(df: pd.DataFrame, outcome: str, predictors: Sequence[str]) -> dict:
    work = df[[outcome, *predictors]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(work) < len(predictors) + 4:
        return {"n": len(work), "r2": np.nan, **{f"beta_{p}": np.nan for p in predictors}}
    y = _standardize(work[outcome]).to_numpy()
    xs = []
    valid = []
    for pred in predictors:
        x = _standardize(work[pred])
        if x.notna().all():
            xs.append(x.to_numpy())
            valid.append(pred)
    if not xs:
        return {"n": len(work), "r2": np.nan, **{f"beta_{p}": np.nan for p in predictors}}
    xmat = np.column_stack(xs)
    coef, *_ = np.linalg.lstsq(xmat, y, rcond=None)
    fitted = xmat @ coef
    sse = float(np.sum((y - fitted) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    out = {"n": int(len(work)), "r2": float(1.0 - sse / sst) if sst else np.nan}
    for pred in predictors:
        out[f"beta_{pred}"] = np.nan
    for pred, beta in zip(valid, coef):
        out[f"beta_{pred}"] = float(beta)
    return out


def _avg_incremental_r2(df: pd.DataFrame, outcome: str, predictors: Sequence[str]) -> Dict[str, float]:
    """Average incremental R2 over all insertion orders (Shapley-style)."""
    increments = {pred: [] for pred in predictors}
    for order in itertools.permutations(predictors):
        current: List[str] = []
        current_r2 = 0.0
        for pred in order:
            next_preds = current + [pred]
            next_r2 = _ols(df, outcome, next_preds)["r2"]
            if not np.isfinite(next_r2):
                next_r2 = current_r2
            increments[pred].append(float(next_r2 - current_r2))
            current = next_preds
            current_r2 = next_r2
    return {pred: float(np.mean(vals)) for pred, vals in increments.items()}


def _fit_decomposition(points: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model_rows = []
    shapley_rows = []
    for (model, regime), part in points.groupby(["model_slug", "regime"], observed=True):
        for outcome in OUTCOMES:
            full = _ols(part, outcome, PREDICTORS)
            local_only = _ols(part, outcome, ["local_support_gap"])
            no_local = _ols(part, outcome, [p for p in PREDICTORS if p != "local_support_gap"])
            model_rows.append(
                {
                    "model": model,
                    "regime": regime,
                    "outcome": outcome,
                    "n_uid": full["n"],
                    "r2_full": full["r2"],
                    "r2_local_only": local_only["r2"],
                    "delta_r2_local_over_controls": full["r2"] - no_local["r2"],
                    **{f"beta_{p}": full[f"beta_{p}"] for p in PREDICTORS},
                }
            )
            inc = _avg_incremental_r2(part, outcome, PREDICTORS)
            total = sum(max(v, 0.0) for v in inc.values())
            for pred, value in inc.items():
                shapley_rows.append(
                    {
                        "model": model,
                        "regime": regime,
                        "outcome": outcome,
                        "predictor": pred,
                        "avg_incremental_r2": value,
                        "share_positive_incremental_r2": value / total if total > 0 else np.nan,
                    }
                )
    return pd.DataFrame(model_rows), pd.DataFrame(shapley_rows)


def _concentration(points: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    top_rows = []
    for (model, regime, outcome), part in points.groupby(["model_slug", "regime", "outcome_name"], observed=True):
        part = part.copy()
        gap_col = outcome
        part["positive_gap"] = part[gap_col].clip(lower=0)
        part["abs_gap"] = part[gap_col].abs()
        for basis in ["positive_gap", "abs_gap"]:
            ranked = part.sort_values(basis, ascending=False).reset_index(drop=True)
            total = float(ranked[basis].sum())
            if total <= 0:
                continue
            ranked["cum_share"] = ranked[basis].cumsum() / total
            ranked["rank"] = np.arange(1, len(ranked) + 1)
            for k in [5, 10, 15, 20]:
                k_eff = min(k, len(ranked))
                rows.append(
                    {
                        "model": model,
                        "regime": regime,
                        "outcome": gap_col,
                        "basis": basis,
                        "top_k": k,
                        "share_of_gap": float(ranked.loc[k_eff - 1, "cum_share"]),
                    }
                )
            half_idx = int(np.searchsorted(ranked["cum_share"].to_numpy(), 0.5, side="left"))
            rows.append(
                {
                    "model": model,
                    "regime": regime,
                    "outcome": gap_col,
                    "basis": basis,
                    "top_k": "k_for_50pct",
                    "share_of_gap": half_idx + 1,
                }
            )
            for _, row in ranked.head(15).iterrows():
                top_rows.append(
                    {
                        "model": model,
                        "regime": regime,
                        "outcome": gap_col,
                        "basis": basis,
                        "rank": int(row["rank"]),
                        "uid": row["uid"],
                        "field": row.get("field", ""),
                        "phenomenon": row.get("phenomenon", ""),
                        "gap": float(row[gap_col]),
                        "local_support_gap": float(row["local_support_gap"]),
                        "zipf_gap": float(row["zipf_gap"]),
                        "char_count_gap": float(row["char_count_gap"]),
                        "token_count_gap": float(row["token_count_gap"]),
                        "cum_share": float(row["cum_share"]),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(top_rows)


def _plot_concentration(conc: pd.DataFrame, out_dir: Path) -> None:
    sub = conc[
        conc["basis"].eq("positive_gap")
        & conc["outcome"].eq("accuracy_gap")
        & conc["top_k"].isin([5, 10, 15, 20])
    ].copy()
    if sub.empty:
        return
    sub["top_k"] = sub["top_k"].astype(int)
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(10.2, 3.35), sharey=True)
    for ax, model in zip(axes, MODEL_ORDER):
        part = sub[sub["model"].eq(model)]
        for regime in REGIME_ORDER:
            rpart = part[part["regime"].eq(regime)].sort_values("top_k")
            if rpart.empty:
                continue
            ax.plot(rpart["top_k"], rpart["share_of_gap"] * 100, marker="o", lw=1.8, color=REGIME_COLORS[regime], label=regime)
        ax.set_title(model)
        ax.set_xlabel("Top K paradigms")
        ax.set_xticks([5, 10, 15, 20])
        ax.grid(axis="x", alpha=0.15)
    axes[0].set_ylabel("Share of positive accuracy gap (%)")
    axes[-1].legend(title="Regime")
    fig.suptitle("How Concentrated Is the Original-Freq Accuracy Gap?")
    fig.tight_layout()
    fig.savefig(out_dir / "paradigm_gap_concentration_topk.png", dpi=300)
    fig.savefig(out_dir / "paradigm_gap_concentration_topk.pdf")
    plt.close(fig)


def _plot_decomposition(shapley: pd.DataFrame, out_dir: Path) -> None:
    sub = shapley[shapley["outcome"].eq("accuracy_gap")].copy()
    if sub.empty:
        return
    labels = {
        "local_support_gap": "Local support",
        "zipf_gap": "Zipf",
        "char_count_gap": "Chars",
        "token_count_gap": "Tokens",
    }
    sub["predictor_label"] = sub["predictor"].map(labels)
    summary = (
        sub.groupby("predictor_label", as_index=False)["share_positive_incremental_r2"]
        .mean()
        .sort_values("share_positive_incremental_r2", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    ax.barh(summary["predictor_label"], summary["share_positive_incremental_r2"] * 100, color="#2563A6", alpha=0.82)
    ax.invert_yaxis()
    ax.set_xlabel("Mean share of explained variance (%)")
    ax.set_title("Predictor Contributions to Paradigm-Level Accuracy Gap")
    fig.tight_layout()
    fig.savefig(out_dir / "paradigm_gap_decomposition_accuracy.png", dpi=300)
    fig.savefig(out_dir / "paradigm_gap_decomposition_accuracy.pdf")
    plt.close(fig)


def _metadata(item_support: Path) -> pd.DataFrame:
    meta = pd.read_csv(item_support, usecols=["uid", "field", "phenomenon"]).drop_duplicates("uid")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-points", type=Path, default=DEFAULT_GAP_POINTS)
    parser.add_argument("--item-support", type=Path, default=DEFAULT_ITEM_SUPPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / ".mplconfig").mkdir(exist_ok=True)
    _style()

    points = pd.read_csv(args.gap_points)
    points = points.merge(_metadata(args.item_support), on="uid", how="left")
    model_table, shapley = _fit_decomposition(points)

    long_points = []
    for outcome in OUTCOMES:
        tmp = points.copy()
        tmp["outcome_name"] = outcome
        long_points.append(tmp)
    long_points_df = pd.concat(long_points, ignore_index=True)
    conc, top = _concentration(long_points_df)

    model_table.to_csv(args.out_dir / "paradigm_gap_multivariate_models.csv", index=False)
    shapley.to_csv(args.out_dir / "paradigm_gap_predictor_contributions.csv", index=False)
    conc.to_csv(args.out_dir / "paradigm_gap_concentration.csv", index=False)
    top.to_csv(args.out_dir / "top_gap_paradigms.csv", index=False)
    points.to_csv(args.out_dir / "paradigm_gap_points_with_metadata.csv", index=False)
    _plot_concentration(conc, args.out_dir)
    _plot_decomposition(shapley, args.out_dir)
    print(f"Wrote paradigm gap decomposition to {args.out_dir}")


if __name__ == "__main__":
    main()
