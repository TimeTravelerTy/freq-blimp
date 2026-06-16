#!/usr/bin/env python3
"""Relation-specific COCA local-collocation support analysis.

This reuses the completed probe-support CSV from the main COCA scan.  It does
not rescan COCA and does not reload text-heavy score columns.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "results/coca_collocation_support/relation_specific/.mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("/Users/tyronewhite/masters_research_code/blimp-rare/results/coca_collocation_support")
DEFAULT_PROBE_SUPPORT = BASE / "local_collocation_probe_support.csv"
DEFAULT_SCORE_SUPPORT = BASE / "model_item_support_scores.csv"
DEFAULT_GAP_POINTS = BASE / "controlled_models/controlled_uid_gap_points.csv"
DEFAULT_TOP_GAPS = BASE / "paradigm_gap_decomposition/top_gap_paradigms.csv"
DEFAULT_OUT_DIR = BASE / "relation_specific"

REGIME_ORDER = ["original", "head", "tail", "xtail"]
RELATION_ORDER = [
    "verb_subject",
    "verb_object",
    "adjective_noun",
    "compound_noun",
    "prep_object",
    "verb_adverb",
    "generic_dependency",
    "adjacent_content",
]
REGIME_COLORS = {
    "original": "#6B7280",
    "head": "#2563A6",
    "tail": "#D97706",
    "xtail": "#C2410C",
}
RELATION_LABELS = {
    "verb_subject": "Verb-subject",
    "verb_object": "Verb-object",
    "adjective_noun": "Adj-noun",
    "compound_noun": "Compound",
    "prep_object": "Prep-object",
    "verb_adverb": "Verb-adverb",
    "generic_dependency": "Generic dep.",
    "adjacent_content": "Adjacent",
}
SUPPORT_FEATURES = [
    "relation_support_mean_good",
    "relation_support_min_good",
    "relation_support_zero_rate_good",
    "relation_support_pmi_mean_good",
    "relation_support_mean_delta_good_minus_bad",
]
CONTROL_FEATURES = ["good_zipf_mean", "good_char_count", "good_token_count"]
OUTCOMES = ["correctness", "margin_logprob"]


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


def _relation_sort_key(values: pd.Series) -> pd.Series:
    order = {relation: i for i, relation in enumerate(RELATION_ORDER)}
    return values.astype(str).map(order).fillna(len(order)).astype(int)


def _pearson(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3:
        return np.nan
    return float(x.corr(y, method="pearson"))


def _spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3:
        return np.nan
    return float(x.rank(method="average").corr(y.rank(method="average"), method="pearson"))


def _demean(df: pd.DataFrame, cols: Sequence[str], groups: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    out[list(cols)] = out[list(cols)].astype(float) - out.groupby(list(groups), observed=True)[list(cols)].transform("mean")
    return out


def _residualize(df: pd.DataFrame, target: str, controls: Sequence[str]) -> pd.Series:
    work = df[[target, *controls]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    residuals = pd.Series(np.nan, index=df.index, dtype=float)
    if len(work) < len(controls) + 5:
        return residuals
    y = work[target].astype(float).to_numpy()
    xmat = np.column_stack([np.ones(len(work)), *[work[col].astype(float).to_numpy() for col in controls]])
    coef, *_ = np.linalg.lstsq(xmat, y, rcond=None)
    residuals.loc[work.index] = y - xmat @ coef
    return residuals


def _load_relation_probe_support(path: Path) -> pd.DataFrame:
    usecols = [
        "regime",
        "uid",
        "pair_id",
        "side",
        "relation",
        "local_log_doc_count",
        "local_pair_attested",
        "local_pmi_doc",
    ]
    dtype = {
        "regime": "category",
        "uid": "category",
        "pair_id": "int32",
        "side": "category",
        "relation": "category",
        "local_pair_attested": "float32",
    }
    return pd.read_csv(path, usecols=usecols, dtype=dtype)


def _side_relation_support(probes: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        probes.groupby(["regime", "uid", "pair_id", "side", "relation"], observed=True)
        .agg(
            relation_probe_count=("relation", "size"),
            relation_support_mean=("local_log_doc_count", "mean"),
            relation_support_min=("local_log_doc_count", "min"),
            relation_support_zero_rate=("local_pair_attested", lambda s: 1.0 - float(np.mean(s))),
            relation_support_pmi_mean=("local_pmi_doc", "mean"),
        )
        .reset_index()
    )
    return grouped


def _item_relation_support(side_support: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["regime", "uid", "pair_id", "relation"]
    metric_cols = [
        "relation_probe_count",
        "relation_support_mean",
        "relation_support_min",
        "relation_support_zero_rate",
        "relation_support_pmi_mean",
    ]
    wide = side_support.pivot_table(index=id_cols, columns="side", values=metric_cols, aggfunc="first", observed=True)
    wide.columns = [f"{metric}_{side}" for metric, side in wide.columns]
    wide = wide.reset_index()
    for metric in [
        "relation_support_mean",
        "relation_support_min",
        "relation_support_zero_rate",
        "relation_support_pmi_mean",
    ]:
        good = f"{metric}_good"
        bad = f"{metric}_bad"
        if good in wide.columns and bad in wide.columns:
            wide[f"{metric}_delta_good_minus_bad"] = wide[good] - wide[bad]
    return wide


def _load_score_support(path: Path) -> pd.DataFrame:
    usecols = [
        "regime",
        "uid",
        "pair_id",
        "model_slug",
        "correctness",
        "margin_logprob",
        "field",
        "phenomenon",
        "good_zipf_mean",
        "good_char_count",
        "good_token_count",
    ]
    dtype = {
        "regime": "category",
        "uid": "category",
        "pair_id": "int32",
        "model_slug": "category",
        "correctness": "float32",
        "good_token_count": "float32",
    }
    return pd.read_csv(path, usecols=usecols, dtype=dtype)


def _relation_support_by_regime(side_support: pd.DataFrame) -> pd.DataFrame:
    return (
        side_support.groupby(["relation", "regime", "side"], observed=True)
        .agg(
            n_item_sides=("relation_support_mean", "size"),
            mean_support=("relation_support_mean", "mean"),
            median_support=("relation_support_mean", "median"),
            mean_min_support=("relation_support_min", "mean"),
            zero_rate=("relation_support_zero_rate", "mean"),
            pmi_mean=("relation_support_pmi_mean", "mean"),
            mean_probe_count=("relation_probe_count", "mean"),
        )
        .reset_index()
        .sort_values(["relation", "regime", "side"], key=lambda s: _relation_sort_key(s) if s.name == "relation" else s)
    )


def _relation_support_gap_by_uid(item_rel: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    uid_rel = (
        item_rel.groupby(["uid", "regime", "relation"], observed=True)
        .agg(
            n_items=("pair_id", "nunique"),
            relation_support_good=("relation_support_mean_good", "mean"),
            relation_support_bad=("relation_support_mean_bad", "mean"),
            relation_zero_rate_good=("relation_support_zero_rate_good", "mean"),
            relation_zero_rate_bad=("relation_support_zero_rate_bad", "mean"),
            relation_support_delta_good_minus_bad=("relation_support_mean_delta_good_minus_bad", "mean"),
        )
        .reset_index()
    )
    original = uid_rel[uid_rel["regime"].eq("original")].rename(
        columns={
            "n_items": "n_items_original",
            "relation_support_good": "relation_support_good_original",
            "relation_support_bad": "relation_support_bad_original",
            "relation_zero_rate_good": "relation_zero_rate_good_original",
            "relation_zero_rate_bad": "relation_zero_rate_bad_original",
            "relation_support_delta_good_minus_bad": "relation_support_delta_good_minus_bad_original",
        }
    )
    freq = uid_rel[~uid_rel["regime"].eq("original")].rename(
        columns={
            "n_items": "n_items_freq",
            "relation_support_good": "relation_support_good_freq",
            "relation_support_bad": "relation_support_bad_freq",
            "relation_zero_rate_good": "relation_zero_rate_good_freq",
            "relation_zero_rate_bad": "relation_zero_rate_bad_freq",
            "relation_support_delta_good_minus_bad": "relation_support_delta_good_minus_bad_freq",
        }
    )
    merged = freq.merge(original.drop(columns=["regime"]), on=["uid", "relation"], how="inner")
    merged["relation_support_gap_original_minus_freq"] = (
        merged["relation_support_good_original"] - merged["relation_support_good_freq"]
    )
    merged["relation_zero_rate_gap_freq_minus_original"] = (
        merged["relation_zero_rate_good_freq"] - merged["relation_zero_rate_good_original"]
    )
    merged["relation_delta_gap_original_minus_freq"] = (
        merged["relation_support_delta_good_minus_bad_original"]
        - merged["relation_support_delta_good_minus_bad_freq"]
    )
    merged = merged.merge(metadata, on="uid", how="left")
    return merged.sort_values(["uid", "regime", "relation"], key=lambda s: _relation_sort_key(s) if s.name == "relation" else s)


def _relation_behavior_correlations(score_rel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, relation), part in score_rel.groupby(["model_slug", "relation"], observed=True):
        for outcome in OUTCOMES:
            for feature in SUPPORT_FEATURES:
                work = part[[outcome, feature]].replace([np.inf, -np.inf], np.nan).dropna()
                rows.append(
                    {
                        "model": model,
                        "relation": relation,
                        "outcome": outcome,
                        "feature": feature,
                        "n": int(len(work)),
                        "pearson": _pearson(work[feature], work[outcome]),
                        "spearman": _spearman(work[feature], work[outcome]),
                    }
                )
    return pd.DataFrame(rows)


def _relation_behavior_controlled_correlations(score_rel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, relation), part in score_rel.groupby(["model_slug", "relation"], observed=True):
        for outcome in OUTCOMES:
            for feature in SUPPORT_FEATURES:
                needed = [outcome, feature, *CONTROL_FEATURES]
                work = part[["uid", "regime", *needed]].replace([np.inf, -np.inf], np.nan).dropna().copy()
                if len(work) < 100:
                    rows.append(
                        {
                            "model": model,
                            "relation": relation,
                            "outcome": outcome,
                            "feature": feature,
                            "n": int(len(work)),
                            "partial_pearson": np.nan,
                            "partial_spearman": np.nan,
                        }
                    )
                    continue
                fe = _demean(work, needed, ["uid", "regime"])
                y_resid = _residualize(fe, outcome, CONTROL_FEATURES)
                x_resid = _residualize(fe, feature, CONTROL_FEATURES)
                resid = pd.DataFrame({"x": x_resid, "y": y_resid}).dropna()
                rows.append(
                    {
                        "model": model,
                        "relation": relation,
                        "outcome": outcome,
                        "feature": feature,
                        "n": int(len(resid)),
                        "partial_pearson": _pearson(resid["x"], resid["y"]),
                        "partial_spearman": _spearman(resid["x"], resid["y"]),
                    }
                )
    return pd.DataFrame(rows)


def _relation_gap_decomposition(relation_gaps: pd.DataFrame, gap_points: pd.DataFrame) -> pd.DataFrame:
    merged = gap_points.merge(
        relation_gaps[
            [
                "uid",
                "regime",
                "relation",
                "relation_support_gap_original_minus_freq",
                "relation_zero_rate_gap_freq_minus_original",
                "relation_delta_gap_original_minus_freq",
            ]
        ],
        on=["uid", "regime"],
        how="inner",
    )
    rows = []
    for (model, regime, relation), part in merged.groupby(["model_slug", "regime", "relation"], observed=True):
        for gap_outcome in ["accuracy_gap", "margin_gap"]:
            for feature in [
                "relation_support_gap_original_minus_freq",
                "relation_zero_rate_gap_freq_minus_original",
                "relation_delta_gap_original_minus_freq",
            ]:
                work = part[[gap_outcome, feature]].replace([np.inf, -np.inf], np.nan).dropna()
                r = _pearson(work[feature], work[gap_outcome])
                rows.append(
                    {
                        "model": model,
                        "regime": regime,
                        "outcome": gap_outcome,
                        "relation": relation,
                        "feature": feature,
                        "n_uid": int(len(work)),
                        "pearson": r,
                        "spearman": _spearman(work[feature], work[gap_outcome]),
                        "univariate_r2": float(r * r) if np.isfinite(r) else np.nan,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["positive_univariate_r2"] = out["univariate_r2"].clip(lower=0)
    denom = out.groupby(["model", "regime", "outcome", "feature"], observed=True)["positive_univariate_r2"].transform("sum")
    out["share_positive_univariate_r2"] = np.where(denom > 0, out["positive_univariate_r2"] / denom, np.nan)
    return out


def _top_gap_relation_profile(relation_gaps: pd.DataFrame, top_gaps: pd.DataFrame) -> pd.DataFrame:
    top = top_gaps[
        top_gaps["basis"].eq("positive_gap")
        & top_gaps["outcome"].eq("accuracy_gap")
        & top_gaps["rank"].le(15)
    ].copy()
    if top.empty:
        return pd.DataFrame()
    relation_cols = relation_gaps.drop(columns=["field", "phenomenon"], errors="ignore")
    return top.merge(relation_cols, on=["uid", "regime"], how="left").sort_values(
        ["model", "regime", "rank", "relation"],
        key=lambda s: _relation_sort_key(s) if s.name == "relation" else s,
    )


def _plot_relation_support_by_regime(summary: pd.DataFrame, out_dir: Path) -> None:
    sub = summary[summary["side"].eq("good")].copy()
    if sub.empty:
        return
    relations = [rel for rel in RELATION_ORDER if rel in set(sub["relation"])]
    x = np.arange(len(relations))
    width = 0.18
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 6.6), sharex=True)
    for idx, regime in enumerate(REGIME_ORDER):
        part = sub[sub["regime"].eq(regime)].set_index("relation")
        offsets = x + (idx - 1.5) * width
        axes[0].bar(
            offsets,
            [part.loc[rel, "mean_support"] if rel in part.index else np.nan for rel in relations],
            width=width,
            color=REGIME_COLORS[regime],
            label=regime,
            alpha=0.86,
        )
        axes[1].bar(
            offsets,
            [part.loc[rel, "zero_rate"] if rel in part.index else np.nan for rel in relations],
            width=width,
            color=REGIME_COLORS[regime],
            label=regime,
            alpha=0.86,
        )
    axes[0].set_ylabel("Mean log doc count")
    axes[0].set_title("Relation-Specific COCA Support by Regime")
    axes[1].set_ylabel("Mean zero-rate")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([RELATION_LABELS.get(rel, rel) for rel in relations], rotation=25, ha="right")
    axes[1].legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.33))
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out_dir / "relation_support_by_regime.png")
    fig.savefig(out_dir / "relation_support_by_regime.pdf")
    plt.close(fig)


def _plot_relation_gap_decomposition(decomp: pd.DataFrame, out_dir: Path) -> None:
    sub = decomp[
        decomp["outcome"].eq("accuracy_gap")
        & decomp["feature"].eq("relation_support_gap_original_minus_freq")
    ].copy()
    if sub.empty:
        return
    summary = (
        sub.groupby("relation", observed=True)
        .agg(mean_share=("share_positive_univariate_r2", "mean"), mean_r=("pearson", "mean"))
        .reset_index()
        .sort_values("relation", key=_relation_sort_key)
    )
    summary["label"] = summary["relation"].map(RELATION_LABELS).fillna(summary["relation"])
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.barh(summary["label"], summary["mean_share"] * 100, color="#2563A6", alpha=0.86)
    ax.invert_yaxis()
    ax.set_xlabel("Mean share of relation-gap univariate R2 (%)")
    ax.set_title("Which Relation Gaps Track the Accuracy Gap?")
    fig.tight_layout()
    fig.savefig(out_dir / "relation_gap_decomposition.png")
    fig.savefig(out_dir / "relation_gap_decomposition.pdf")
    plt.close(fig)


def _write_summary(out_dir: Path, probes: pd.DataFrame, item_rel: pd.DataFrame, score_rel: pd.DataFrame) -> None:
    summary = {
        "probe_rows": int(len(probes)),
        "item_relation_rows": int(len(item_rel)),
        "score_relation_rows": int(len(score_rel)),
        "relations": sorted(str(x) for x in item_rel["relation"].dropna().unique()),
        "regimes": sorted(str(x) for x in item_rel["regime"].dropna().unique()),
        "models": sorted(str(x) for x in score_rel["model_slug"].dropna().unique()),
    }
    (out_dir / "relation_analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-support", type=Path, default=DEFAULT_PROBE_SUPPORT)
    parser.add_argument("--score-support", type=Path, default=DEFAULT_SCORE_SUPPORT)
    parser.add_argument("--gap-points", type=Path, default=DEFAULT_GAP_POINTS)
    parser.add_argument("--top-gaps", type=Path, default=DEFAULT_TOP_GAPS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / ".mplconfig").mkdir(exist_ok=True)
    _style()

    probes = _load_relation_probe_support(args.probe_support)
    side_support = _side_relation_support(probes)
    item_rel = _item_relation_support(side_support)
    scores = _load_score_support(args.score_support)
    metadata = scores[["uid", "field", "phenomenon"]].drop_duplicates("uid")

    item_rel = item_rel.merge(metadata, on="uid", how="left")
    score_rel = scores.merge(item_rel, on=["regime", "uid", "pair_id"], how="inner")

    support_summary = _relation_support_by_regime(side_support)
    relation_gaps = _relation_support_gap_by_uid(item_rel, metadata)
    behavior_corr = _relation_behavior_correlations(score_rel)
    behavior_controlled = _relation_behavior_controlled_correlations(score_rel)

    gap_points = pd.read_csv(args.gap_points)
    gap_decomp = _relation_gap_decomposition(relation_gaps, gap_points)
    top_gaps = pd.read_csv(args.top_gaps)
    top_profile = _top_gap_relation_profile(relation_gaps, top_gaps)

    side_support.to_csv(args.out_dir / "relation_side_support.csv", index=False)
    item_rel.to_csv(args.out_dir / "relation_item_support.csv", index=False)
    support_summary.to_csv(args.out_dir / "relation_support_by_regime.csv", index=False)
    relation_gaps.to_csv(args.out_dir / "relation_support_gap_by_uid.csv", index=False)
    behavior_corr.to_csv(args.out_dir / "relation_behavior_correlations.csv", index=False)
    behavior_controlled.to_csv(args.out_dir / "relation_behavior_controlled_correlations.csv", index=False)
    gap_decomp.to_csv(args.out_dir / "relation_gap_decomposition.csv", index=False)
    top_profile.to_csv(args.out_dir / "top_gap_paradigm_relation_profile.csv", index=False)

    _plot_relation_support_by_regime(support_summary, args.out_dir)
    _plot_relation_gap_decomposition(gap_decomp, args.out_dir)
    _write_summary(args.out_dir, probes, item_rel, score_rel)
    print(f"Wrote relation-specific COCA analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
