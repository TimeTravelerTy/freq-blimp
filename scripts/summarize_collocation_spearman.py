#!/usr/bin/env python3
"""Summarize Spearman correlations for COCA-support analyses."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "results/coca_collocation_support/spearman_summary/.mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("/Users/tyronewhite/masters_research_code/blimp-rare/results/coca_collocation_support")
DEFAULT_OUT = BASE / "spearman_summary"


def _spearman(left: pd.Series, right: pd.Series) -> float:
    ranks = pd.DataFrame({"left": left, "right": right}).rank(method="average")
    return float(ranks["left"].corr(ranks["right"]))


def _residualize(y: pd.Series, controls: pd.DataFrame) -> pd.Series:
    data = pd.concat([y.rename("target"), controls], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return pd.Series(dtype=float)
    yv = data["target"].astype(float).to_numpy()
    x = data.drop(columns=["target"]).astype(float).to_numpy()
    valid = np.nanstd(x, axis=0) > 0
    x = x[:, valid]
    if x.shape[1] == 0:
        resid = yv - yv.mean()
    else:
        x = np.column_stack([np.ones(len(x)), x])
        coef, *_ = np.linalg.lstsq(x, yv, rcond=None)
        resid = yv - x @ coef
    return pd.Series(resid, index=data.index)


def _paradigm_gap_spearman(base: Path) -> pd.DataFrame:
    points = pd.read_csv(base / "controlled_models/controlled_uid_gap_points.csv")
    controls = ["zipf_gap", "char_count_gap", "token_count_gap"]
    rows = []
    for (model, regime), part in points.groupby(["model_slug", "regime"], observed=True):
        for outcome in ["accuracy_gap", "margin_gap"]:
            sub = part[[outcome, "local_support_gap", *controls]].dropna()
            if len(sub) < 5 or sub["local_support_gap"].nunique() < 2:
                continue
            y_resid = _residualize(sub[outcome], sub[controls])
            x_resid = _residualize(sub["local_support_gap"], sub[controls])
            aligned = pd.concat([y_resid.rename("y"), x_resid.rename("x")], axis=1).dropna()
            rows.append(
                {
                    "level": "paradigm_gap",
                    "model": model,
                    "regime": regime,
                    "outcome": outcome,
                    "feature": "local_support_gap",
                    "n": int(len(sub)),
                    "spearman_raw": _spearman(sub[outcome], sub["local_support_gap"]),
                    "spearman_partial_controls": _spearman(aligned["y"], aligned["x"]) if len(aligned) >= 5 else np.nan,
                    "controls": "+".join(controls),
                }
            )
    return pd.DataFrame(rows)


def _item_spearman(base: Path) -> pd.DataFrame:
    item = pd.read_csv(base / "item_level_effects/item_level_residual_correlations.csv")
    out = item.rename(
        columns={
            "model_slug": "model",
            "residual_spearman_rho": "spearman_partial_controls",
            "residual_pearson_r": "pearson_partial_controls",
        }
    )
    out["level"] = "item_residual"
    out["regime"] = "within_paradigm_regime"
    out["spearman_raw"] = np.nan
    out["controls"] = np.where(
        out["feature"].eq("local_support_mean_good"),
        "good_zipf_mean+good_char_count+good_token_count+paradigm_regime_FE",
        "zipf_delta+char_delta+token_delta+paradigm_regime_FE",
    )
    return out[
        [
            "level",
            "model",
            "regime",
            "outcome",
            "feature",
            "n",
            "spearman_raw",
            "spearman_partial_controls",
            "controls",
        ]
    ]


def _plot_paradigm_heatmap(paradigm: pd.DataFrame, out_dir: Path) -> None:
    if paradigm.empty:
        return
    plot = paradigm.copy()
    plot["row"] = plot["model"] + " / " + plot["regime"]
    plot["col"] = plot["outcome"].map({"accuracy_gap": "Accuracy", "margin_gap": "Margin"})
    for value_col, stem, title in [
        ("spearman_raw", "paradigm_gap_spearman_raw", "Paradigm Gap Spearman: Local-Support Gap vs Behavior Gap"),
        (
            "spearman_partial_controls",
            "paradigm_gap_spearman_partial_controls",
            "Paradigm Gap Partial Spearman, Controlling Zipf/Length Gaps",
        ),
    ]:
        piv = plot.pivot(index="row", columns="col", values=value_col)
        piv = piv.reindex(columns=["Accuracy", "Margin"])
        fig, ax = plt.subplots(figsize=(5.4, max(3.6, 0.36 * len(piv) + 1.0)))
        im = ax.imshow(piv.values, aspect="auto", vmin=-0.4, vmax=0.4, cmap="RdBu_r")
        ax.set_xticks(np.arange(len(piv.columns)))
        ax.set_xticklabels(piv.columns)
        ax.set_yticks(np.arange(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_title(title)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                val = piv.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.5)
        fig.colorbar(im, ax=ax, label="Spearman rho")
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}.png", dpi=300)
        fig.savefig(out_dir / f"{stem}.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / ".mplconfig").mkdir(exist_ok=True)

    paradigm = _paradigm_gap_spearman(args.base)
    item = _item_spearman(args.base)
    paradigm.to_csv(args.out_dir / "paradigm_gap_spearman.csv", index=False)
    item.to_csv(args.out_dir / "item_level_spearman.csv", index=False)
    pd.concat([paradigm, item], ignore_index=True).to_csv(args.out_dir / "collocation_spearman_summary.csv", index=False)
    _plot_paradigm_heatmap(paradigm, args.out_dir)
    print(f"Wrote Spearman summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
