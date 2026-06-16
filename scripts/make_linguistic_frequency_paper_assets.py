#!/usr/bin/env python3
"""Create paper-facing tables and plots for linguistic frequency effects."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "results/linguistic_frequency_effects_20260511_main_suite/paper_assets/.mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "llama31",
    "llama3170b",
    "mistral7b",
    "qwen25_7b",
    "qwen25_72b",
    "gemma4e4b",
    "gemma4_31b",
]

INSTRUCT_MODEL_ORDER = [
    "llama31i70b",
    "qwen25_72bi",
    "gemma4_31bit",
]

MODEL_SHORT = {
    "llama31": "L3.1-8B",
    "llama3170b": "L3.1-70B",
    "llama31i70b": "L3.1-70B-I",
    "mistral7b": "Mistral-7B",
    "qwen25_7b": "Q2.5-7B",
    "qwen25_72b": "Q2.5-72B",
    "qwen25_72bi": "Q2.5-72B-I",
    "gemma4e4b": "G4-E4B",
    "gemma4_31b": "G4-31B",
    "gemma4_31bit": "G4-31B-it",
}

METHOD_LABEL = {
    "in_template_lp": "LP",
    "nll": "NLL",
}

FIELD_ORDER = ["syntax", "syntax/semantics", "semantics", "morphology"]
FIELD_LABEL = {
    "syntax": "Syntax",
    "syntax/semantics": "Syntax/Semantics",
    "semantics": "Semantics",
    "morphology": "Morphology",
}

PHENOMENON_SHORT = {
    "anaphor_agreement": "Anaphor agr.",
    "argument_structure": "Arg. structure",
    "binding": "Binding",
    "control_raising": "Control/raising",
    "determiner_noun_agreement": "Det-noun agr.",
    "ellipsis": "Ellipsis",
    "filler_gap_dependency": "Filler-gap",
    "irregular_forms": "Irreg. forms",
    "island_effects": "Islands",
    "npi_licensing": "NPI licensing",
    "quantifiers": "Quantifiers",
    "s-selection": "S-selection",
    "subject_verb_agreement": "SVA",
}


def to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = []
    for i, header in enumerate(headers):
        widths.append(max([len(header)] + [len(row[i]) for row in rows]))
    lines = []
    lines.append("| " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)) + " |")
    lines.append("| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    return "\n".join(lines)


def prettify(value: str) -> str:
    return value.replace("_", " ")


def short_phenomenon(value: str) -> str:
    return PHENOMENON_SHORT.get(value, prettify(value))


def fmt_pp(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):+.1f}"


def method_consensus(df: pd.DataFrame, method: str, key: str) -> pd.DataFrame:
    sub = df[df["method"].eq(method)].copy()
    if "model_slug" in sub.columns:
        sub = sub[sub["model_slug"].isin(MODEL_ORDER)].copy()
    rows = []
    for name, group in sub.groupby(key, dropna=False):
        vals = pd.to_numeric(group["drop_head_xtail_pp"], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append(
            {
                key: name,
                f"{method}_median_drop_pp": float(vals.median()),
                f"{method}_n_drop": int((vals > 0).sum()),
                f"{method}_n_models": int(vals.shape[0]),
                f"{method}_min_drop_pp": float(vals.min()),
                f"{method}_max_drop_pp": float(vals.max()),
            }
        )
    return pd.DataFrame(rows)


def make_heatmap(input_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(input_dir / "phenomenon_effects_by_model_method.csv")
    df = df[df["method"].isin(["in_template_lp", "nll"])].copy()
    df = df[df["model_slug"].isin(MODEL_ORDER)].copy()

    nll_consensus = method_consensus(df, "nll", "phenomenon")
    ordered = (
        nll_consensus.sort_values("nll_median_drop_pp", ascending=False)["phenomenon"]
        .tolist()
    )

    vmax = max(20.0, float(np.nanmax(np.abs(df["drop_head_xtail_pp"]))))
    vlim = min(math.ceil(vmax / 5.0) * 5.0, 55.0)

    fig, axes = plt.subplots(
        ncols=2,
        figsize=(15.2, 8.4),
        gridspec_kw={"wspace": 0.36},
        constrained_layout=False,
    )

    image = None
    for ax, method in zip(axes, ["in_template_lp", "nll"]):
        sub = df[df["method"].eq(method)]
        matrix = (
            sub.pivot_table(
                index="phenomenon",
                columns="model_slug",
                values="drop_head_xtail_pp",
                aggfunc="first",
            )
            .reindex(index=ordered, columns=MODEL_ORDER)
        )

        image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
        ax.set_title(METHOD_LABEL[method], fontsize=13, pad=10)
        ax.set_xticks(np.arange(len(MODEL_ORDER)))
        ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER], fontsize=8)
        ax.set_yticks(np.arange(len(ordered)))
        if method == "in_template_lp":
            ax.set_yticklabels([prettify(x) for x in ordered], fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", length=0)
        ax.set_xticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ordered), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)

        consensus = method_consensus(sub, method, "phenomenon").set_index("phenomenon")
        for y, phenomenon in enumerate(ordered):
            med = consensus.loc[phenomenon, f"{method}_median_drop_pp"]
            n_drop = int(consensus.loc[phenomenon, f"{method}_n_drop"])
            n_models = int(consensus.loc[phenomenon, f"{method}_n_models"])
            ax.text(
                len(MODEL_ORDER) + 0.18,
                y,
                f"{med:+.1f}  {n_drop}/{n_models}",
                va="center",
                ha="left",
                fontsize=8.5,
            )
        ax.text(
            len(MODEL_ORDER) + 0.18,
            -0.8,
            "median  n",
            va="bottom",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )
        ax.set_xlim(-0.5, len(MODEL_ORDER) + 2.7)

    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.025)
    cbar.set_label("Head - xtail accuracy (percentage points)")
    fig.suptitle(
        "Head-to-xtail accuracy drop by BLiMP phenomenon and model",
        fontsize=15,
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "Rows sorted by NLL median drop. Positive values indicate worse accuracy on xtail than head.",
        ha="center",
        fontsize=9,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_phenomenon_model_head_xtail_drop_lp_nll.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def make_nll_lp_readout_heatmap(input_dir: Path, out_dir: Path) -> None:
    effects = pd.read_csv(input_dir / "phenomenon_effects_by_model_method.csv")
    effects = effects[effects["method"].eq("nll") & effects["model_slug"].isin(MODEL_ORDER)].copy()

    meta = pd.read_csv(input_dir / "accuracy_by_uid_normalized.csv")
    phen_fields = (
        meta[["phenomenon", "field"]]
        .dropna()
        .drop_duplicates()
        .groupby("phenomenon")["field"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    effects["field"] = effects["phenomenon"].map(phen_fields)
    effects["signed_drop_pp"] = -effects["drop_head_xtail_pp"]

    consensus = []
    for (field, phenomenon), group in effects.groupby(["field", "phenomenon"], dropna=False):
        vals = group.set_index("model_slug").reindex(MODEL_ORDER)["signed_drop_pp"].dropna()
        if vals.empty:
            continue
        consensus.append(
            {
                "field": field,
                "phenomenon": phenomenon,
                "median_signed_drop_pp": float(vals.median()),
                "n_drop": int((vals < 0).sum()),
                "n_models": int(vals.shape[0]),
            }
        )
    consensus_df = pd.DataFrame(consensus)
    consensus_df["field_order"] = consensus_df["field"].map(
        {field: i for i, field in enumerate(FIELD_ORDER)}
    ).fillna(99)
    consensus_df = consensus_df.sort_values(
        ["field_order", "median_signed_drop_pp", "phenomenon"],
        ascending=[True, True, True],
    )
    ordered = consensus_df["phenomenon"].tolist()

    matrix = (
        effects.pivot_table(
            index="phenomenon",
            columns="model_slug",
            values="signed_drop_pp",
            aggfunc="first",
        )
        .reindex(index=ordered, columns=MODEL_ORDER)
    )

    vlim = 25.0
    fig = plt.figure(figsize=(12.8, 8.2))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=4,
        width_ratios=[0.12, 1.0, 0.018, 0.2],
        wspace=0.03,
    )
    field_ax = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])
    sep_ax = fig.add_subplot(gs[0, 2])
    stat_ax = fig.add_subplot(gs[0, 3])

    image = ax.imshow(matrix.to_numpy(), cmap="RdBu", vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_title("LP Readout", fontsize=13, pad=10)
    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels([prettify(x) for x in ordered], fontsize=9)
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ordered), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Field group separators and labels.
    row_fields = consensus_df["field"].tolist()
    boundaries = []
    start = 0
    for i in range(1, len(row_fields) + 1):
        if i == len(row_fields) or row_fields[i] != row_fields[start]:
            boundaries.append((row_fields[start], start, i - 1))
            start = i
    for _, _, end in boundaries[:-1]:
        ax.axhline(end + 0.5, color="black", linewidth=1.0)
        stat_ax.axhline(end + 0.5, color="black", linewidth=1.0)
        field_ax.axhline(end + 0.5, color="black", linewidth=1.0)

    field_ax.set_xlim(0, 1)
    field_ax.set_ylim(len(ordered) - 0.5, -0.5)
    field_ax.axis("off")
    field_colors = {
        "syntax": "#d7e8f5",
        "syntax/semantics": "#e6ddf2",
        "semantics": "#e3edd7",
        "morphology": "#f3e1d2",
    }
    for field, first, last in boundaries:
        y0 = first - 0.5
        height = last - first + 1
        field_ax.add_patch(
            plt.Rectangle((0.18, y0), 0.64, height, facecolor=field_colors.get(field, "#eeeeee"), edgecolor="none")
        )
        field_ax.text(
            0.5,
            (first + last) / 2,
            FIELD_LABEL.get(field, prettify(str(field))),
            rotation=90,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )

    sep_ax.set_ylim(len(ordered) - 0.5, -0.5)
    sep_ax.set_xlim(0, 1)
    sep_ax.axis("off")
    sep_ax.axvline(0.5, color="black", linewidth=1.1)

    stat_ax.set_xlim(0, 1)
    stat_ax.set_ylim(len(ordered) - 0.5, -0.5)
    stat_ax.axis("off")
    stat_ax.text(0.08, -0.85, "median", ha="left", va="bottom", fontsize=8.5, fontweight="bold")
    stat_ax.text(0.66, -0.85, f"n/{len(MODEL_ORDER)}", ha="left", va="bottom", fontsize=8.5, fontweight="bold")
    stats = consensus_df.set_index("phenomenon")
    for y, phenomenon in enumerate(ordered):
        med = stats.loc[phenomenon, "median_signed_drop_pp"]
        n_drop = int(stats.loc[phenomenon, "n_drop"])
        stat_ax.text(0.08, y, f"{med:+.1f}", ha="left", va="center", fontsize=8.7)
        stat_ax.text(0.66, y, f"{n_drop}/10", ha="left", va="center", fontsize=8.7)

    cbar = fig.colorbar(image, ax=[ax, stat_ax], fraction=0.026, pad=0.03)
    cbar.set_label("Xtail - head accuracy (percentage points)")
    fig.suptitle(
        "Head-to-xtail frequency effect by BLiMP phenomenon and model",
        fontsize=14,
        y=0.985,
    )
    fig.text(
        0.53,
        0.012,
        "Negative values indicate an accuracy drop from head to xtail. Rows grouped by linguistic field and sorted by median within field.",
        ha="center",
        fontsize=9,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_lp_readout_phenomenon_model_xtail_minus_head_nll.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def make_base_lp_readout_faceted_heatmap(input_dir: Path, out_dir: Path) -> None:
    effects = pd.read_csv(input_dir / "phenomenon_effects_by_model_method.csv")
    effects = effects[effects["method"].eq("nll")].copy()
    present_models = [m for m in MODEL_ORDER if m in set(effects["model_slug"])]
    effects = effects[effects["model_slug"].isin(present_models)].copy()

    meta = pd.read_csv(input_dir / "accuracy_by_uid_normalized.csv")
    phen_fields = (
        meta[["phenomenon", "field"]]
        .dropna()
        .drop_duplicates()
        .groupby("phenomenon")["field"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    effects["field"] = effects["phenomenon"].map(phen_fields)
    effects["signed_drop_pp"] = -effects["drop_head_xtail_pp"]

    stats = []
    for (field, phenomenon), group in effects.groupby(["field", "phenomenon"], dropna=False):
        vals = group["signed_drop_pp"].dropna()
        if vals.empty:
            continue
        stats.append(
            {
                "field": field,
                "phenomenon": phenomenon,
                "median_signed_drop_pp": float(vals.median()),
                "n_drop": int((vals < 0).sum()),
                "n_models": int(vals.shape[0]),
            }
        )
    stats_df = pd.DataFrame(stats)

    field_to_rows = {}
    for field in FIELD_ORDER:
        rows = (
            stats_df[stats_df["field"].eq(field)]
            .sort_values(["median_signed_drop_pp", "phenomenon"])
            ["phenomenon"]
            .tolist()
        )
        if rows:
            field_to_rows[field] = rows

    total_rows = sum(len(rows) for rows in field_to_rows.values())
    height_ratios = [len(rows) for rows in field_to_rows.values()]
    fig = plt.figure(figsize=(7.6, max(7.2, 0.56 * total_rows + 1.7)))
    gs = fig.add_gridspec(
        nrows=len(field_to_rows),
        ncols=1,
        height_ratios=height_ratios,
        hspace=0.18,
    )

    image = None
    for i, (field, rows) in enumerate(field_to_rows.items()):
        ax = fig.add_subplot(gs[i, 0])
        matrix = (
            effects[effects["field"].eq(field)]
            .pivot_table(index="phenomenon", columns="model_slug", values="signed_drop_pp", aggfunc="first")
            .reindex(index=rows, columns=present_models)
        )
        values = matrix.to_numpy()
        image = ax.imshow(values, cmap="RdBu", vmin=-25, vmax=25, aspect="auto")
        ax.set_title(FIELD_LABEL.get(field, prettify(field)), loc="left", fontsize=14.0, fontweight="bold", pad=4)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels([short_phenomenon(r) for r in rows], fontsize=13.0)
        ax.set_xticks(np.arange(len(present_models)))
        if i == len(field_to_rows) - 1:
            ax.set_xticklabels([MODEL_SHORT[m] for m in present_models], rotation=40, ha="right", fontsize=13.0)
        else:
            ax.set_xticklabels([])
        for y in range(values.shape[0]):
            for x in range(values.shape[1]):
                value = values[y, x]
                if np.isnan(value):
                    continue
                text_color = "white" if abs(value) >= 12 else "#222222"
                ax.text(
                    x,
                    y,
                    f"{value:+.0f}",
                    ha="center",
                    va="center",
                    fontsize=11.4,
                    fontweight="bold",
                    color=text_color,
                )
        ax.tick_params(axis="both", length=0)
        ax.set_xticks(np.arange(-0.5, len(present_models), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(image, ax=fig.axes, fraction=0.025, pad=0.025)
    cbar.set_label("Xtail - head accuracy (percentage points)", fontsize=12.5)
    cbar.ax.tick_params(labelsize=11.5)
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_lp_readout_base_phenomenon_field_facets.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)

    stats_out = stats_df.sort_values(["field", "median_signed_drop_pp", "phenomenon"]).copy()
    stats_out["phenomenon"] = stats_out["phenomenon"].map(prettify)
    stats_out["field"] = stats_out["field"].map(prettify)
    stats_out["median_pp"] = stats_out["median_signed_drop_pp"].map(lambda x: f"{x:+.1f}")
    stats_out["n_drop"] = stats_out.apply(lambda r: f"{int(r['n_drop'])}/{int(r['n_models'])}", axis=1)
    stats_out[["field", "phenomenon", "median_pp", "n_drop"]].to_csv(
        out_dir / "phenomenon_lp_readout_base_summary.csv", index=False
    )


def make_instruct_yes_no_heatmap(input_dir: Path, out_dir: Path) -> None:
    effects_path = input_dir / "phenomenon_effects_by_model_method.csv"
    if not effects_path.exists():
        return
    effects = pd.read_csv(effects_path)
    effects = effects[effects["method"].eq("yes_no")].copy()
    if effects.empty:
        return
    present_models = [m for m in INSTRUCT_MODEL_ORDER if m in set(effects["model_slug"])]
    effects = effects[effects["model_slug"].isin(present_models)].copy()
    if effects.empty:
        return

    meta = pd.read_csv(input_dir / "accuracy_by_uid_normalized.csv")
    phen_fields = (
        meta[["phenomenon", "field"]]
        .dropna()
        .drop_duplicates()
        .groupby("phenomenon")["field"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    effects["field"] = effects["phenomenon"].map(phen_fields)
    effects["signed_drop_pp"] = -effects["drop_head_xtail_pp"]

    stats = []
    for (field, phenomenon), group in effects.groupby(["field", "phenomenon"], dropna=False):
        vals = group["signed_drop_pp"].dropna()
        if vals.empty:
            continue
        stats.append(
            {
                "field": field,
                "phenomenon": phenomenon,
                "median_signed_drop_pp": float(vals.median()),
                "n_drop": int((vals < 0).sum()),
                "n_models": int(vals.shape[0]),
            }
        )
    stats_df = pd.DataFrame(stats)

    field_to_rows = {}
    for field in FIELD_ORDER:
        rows = (
            stats_df[stats_df["field"].eq(field)]
            .sort_values(["median_signed_drop_pp", "phenomenon"])
            ["phenomenon"]
            .tolist()
        )
        if rows:
            field_to_rows[field] = rows
    if not field_to_rows:
        return

    total_rows = sum(len(rows) for rows in field_to_rows.values())
    height_ratios = [len(rows) for rows in field_to_rows.values()]
    fig = plt.figure(figsize=(5.0, max(6.8, 0.48 * total_rows + 1.45)))
    gs = fig.add_gridspec(
        nrows=len(field_to_rows),
        ncols=1,
        height_ratios=height_ratios,
        hspace=0.18,
    )

    values_all = effects["signed_drop_pp"].abs().dropna()
    vlim = max(15.0, min(35.0, math.ceil(float(values_all.max()) / 5.0) * 5.0))
    image = None
    for i, (field, rows) in enumerate(field_to_rows.items()):
        ax = fig.add_subplot(gs[i, 0])
        matrix = (
            effects[effects["field"].eq(field)]
            .pivot_table(index="phenomenon", columns="model_slug", values="signed_drop_pp", aggfunc="first")
            .reindex(index=rows, columns=present_models)
        )
        values = matrix.to_numpy()
        image = ax.imshow(values, cmap="RdBu", vmin=-vlim, vmax=vlim, aspect="auto")
        ax.set_title(FIELD_LABEL.get(field, prettify(field)), loc="left", fontsize=12.4, fontweight="bold", pad=3)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels([short_phenomenon(r) for r in rows], fontsize=12.0)
        ax.set_xticks(np.arange(len(present_models)))
        if i == len(field_to_rows) - 1:
            ax.set_xticklabels([MODEL_SHORT[m] for m in present_models], rotation=32, ha="right", fontsize=10.6)
        else:
            ax.set_xticklabels([])
        for y in range(values.shape[0]):
            for x in range(values.shape[1]):
                value = values[y, x]
                if np.isnan(value):
                    continue
                text_color = "white" if abs(value) >= 0.48 * vlim else "#222222"
                ax.text(
                    x,
                    y,
                    f"{value:+.0f}",
                    ha="center",
                    va="center",
                    fontsize=8.8,
                    color=text_color,
                )
        ax.tick_params(axis="both", length=0)
        ax.set_xticks(np.arange(-0.5, len(present_models), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(image, ax=fig.axes, fraction=0.035, pad=0.03)
    cbar.set_label("Xtail - head accuracy (percentage points)", fontsize=11.0)
    cbar.ax.tick_params(labelsize=10.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_yes_no_instruct_phenomenon_field_facets.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)

    stats_out = stats_df.sort_values(["field", "median_signed_drop_pp", "phenomenon"]).copy()
    stats_out["phenomenon"] = stats_out["phenomenon"].map(prettify)
    stats_out["field"] = stats_out["field"].map(prettify)
    stats_out["median_pp"] = stats_out["median_signed_drop_pp"].map(lambda x: f"{x:+.1f}")
    stats_out["n_drop"] = stats_out.apply(lambda r: f"{int(r['n_drop'])}/{int(r['n_models'])}", axis=1)
    stats_out[["field", "phenomenon", "median_pp", "n_drop"]].to_csv(
        out_dir / "phenomenon_yes_no_instruct_summary.csv", index=False
    )


def _mean_score_margin(path: Path) -> float:
    margins = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            margin = obj.get("score_margin")
            if isinstance(margin, (int, float)):
                margins.append(float(margin))
            else:
                good = obj.get("score_good")
                bad = obj.get("score_bad")
                if isinstance(good, (int, float)) and isinstance(bad, (int, float)):
                    margins.append(float(good) - float(bad))
    if not margins:
        return float("nan")
    return float(np.mean(margins))


def _load_in_template_lp_margins(input_dir: Path, model_slugs: list[str]) -> pd.DataFrame:
    acc_path = input_dir / "accuracy_by_uid_normalized.csv"
    if not acc_path.exists():
        return pd.DataFrame()
    acc = pd.read_csv(acc_path)
    acc = acc[
        acc["model_slug"].isin(model_slugs)
        & acc["method"].eq("in_template_lp")
        & acc["regime"].isin(["head", "xtail"])
    ].copy()
    if acc.empty:
        return pd.DataFrame()

    rows = []
    for _, row in acc.iterrows():
        score_path = Path(row["repo_relative_path"])
        if not score_path.is_absolute():
            score_path = Path.cwd() / score_path
        rows.append(
            {
                "model_slug": row["model_slug"],
                "paradigm": row["paradigm"],
                "regime": row["regime"],
                "lp_margin": _mean_score_margin(score_path) if score_path.exists() else float("nan"),
            }
        )
    margins = pd.DataFrame(rows)
    if margins.empty:
        return margins
    margins = (
        margins.pivot_table(
            index=["model_slug", "paradigm"],
            columns="regime",
            values="lp_margin",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"head": "lp_margin_head", "xtail": "lp_margin_xtail"})
    )
    margins["lp_margin_delta"] = margins["lp_margin_xtail"] - margins["lp_margin_head"]
    return margins


def make_instruct_yes_no_paradigm_extremes(input_dir: Path, out_dir: Path, top_n: int = 10) -> None:
    effects_path = input_dir / "paradigm_effects_by_model_method.csv"
    if not effects_path.exists():
        return
    effects = pd.read_csv(effects_path)
    effects = effects[effects["method"].eq("yes_no")].copy()
    if effects.empty:
        return
    present_models = [m for m in INSTRUCT_MODEL_ORDER if m in set(effects["model_slug"])]
    effects = effects[effects["model_slug"].isin(present_models)].copy()
    if effects.empty:
        return
    lp_margins = _load_in_template_lp_margins(input_dir, present_models)
    if not lp_margins.empty:
        effects = effects.merge(lp_margins, on=["model_slug", "paradigm"], how="left")
    else:
        effects["lp_margin_head"] = np.nan
        effects["lp_margin_xtail"] = np.nan
        effects["lp_margin_delta"] = np.nan
    effects["xtail_minus_head_pp"] = -effects["drop_head_xtail_pp"]
    effects["abs_delta_pp"] = effects["xtail_minus_head_pp"].abs()
    effects["model"] = effects["model_slug"].map(MODEL_SHORT).fillna(effects["model_slug"])

    rows = []
    for model_slug in present_models:
        group = effects[effects["model_slug"].eq(model_slug)].copy()
        group = group.sort_values(["abs_delta_pp", "paradigm"], ascending=[False, True]).head(top_n)
        for rank, (_, row) in enumerate(group.iterrows(), start=1):
            rows.append(
                {
                    "model": row["model"],
                    "rank": rank,
                    "paradigm": prettify(row["paradigm"]),
                    "phenomenon": prettify(row["phenomenon"]),
                    "field": prettify(row["field"]),
                    "head": f"{100 * row['accuracy_head']:.1f}",
                    "xtail": f"{100 * row['accuracy_xtail']:.1f}",
                    "delta": f"{row['xtail_minus_head_pp']:+.1f}",
                    "lp margin head": fmt_pp(row["lp_margin_head"]),
                    "lp margin xtail": fmt_pp(row["lp_margin_xtail"]),
                    "lp margin delta": fmt_pp(row["lp_margin_delta"]),
                }
            )

    display = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    display.to_csv(out_dir / "top_instruct_yes_no_paradigm_effects_by_model.csv", index=False)
    (out_dir / "top_instruct_yes_no_paradigm_effects_by_model.md").write_text(
        to_markdown(display),
        encoding="utf-8",
    )

    signed_rows = []
    for model_slug in present_models:
        group = effects[effects["model_slug"].eq(model_slug)].copy()
        selections = [
            ("drop", group.sort_values(["xtail_minus_head_pp", "paradigm"], ascending=[True, True]).head(5)),
            ("gain", group.sort_values(["xtail_minus_head_pp", "paradigm"], ascending=[False, True]).head(5)),
        ]
        for direction, selected in selections:
            for rank, (_, row) in enumerate(selected.iterrows(), start=1):
                signed_rows.append(
                    {
                        "model": row["model"],
                        "direction": direction,
                        "rank": rank,
                        "paradigm": prettify(row["paradigm"]),
                        "phenomenon": prettify(row["phenomenon"]),
                        "field": prettify(row["field"]),
                        "head": f"{100 * row['accuracy_head']:.1f}",
                        "xtail": f"{100 * row['accuracy_xtail']:.1f}",
                        "delta": f"{row['xtail_minus_head_pp']:+.1f}",
                        "lp margin head": fmt_pp(row["lp_margin_head"]),
                        "lp margin xtail": fmt_pp(row["lp_margin_xtail"]),
                        "lp margin delta": fmt_pp(row["lp_margin_delta"]),
                    }
                )

    signed_display = pd.DataFrame(signed_rows)
    signed_display.to_csv(out_dir / "top_instruct_yes_no_paradigm_drops_and_gains_by_model.csv", index=False)
    (out_dir / "top_instruct_yes_no_paradigm_drops_and_gains_by_model.md").write_text(
        to_markdown(signed_display),
        encoding="utf-8",
    )


def make_top_paradigm_tables(input_dir: Path, out_dir: Path) -> None:
    effects = pd.read_csv(input_dir / "paradigm_effects_by_model_method.csv")
    effects = effects[effects["method"].isin(["in_template_lp", "nll"])].copy()
    meta = (
        effects[["paradigm", "phenomenon", "field"]]
        .dropna(subset=["paradigm"])
        .drop_duplicates("paradigm")
    )

    nll = method_consensus(effects, "nll", "paradigm")
    lp = method_consensus(effects, "in_template_lp", "paradigm")
    table = meta.merge(nll, on="paradigm", how="left").merge(lp, on="paradigm", how="left")
    table = table.sort_values("nll_median_drop_pp", ascending=False)

    cols = [
        "paradigm",
        "phenomenon",
        "field",
        "nll_median_drop_pp",
        "nll_n_drop",
        "nll_n_models",
        "nll_min_drop_pp",
        "nll_max_drop_pp",
        "in_template_lp_median_drop_pp",
        "in_template_lp_n_drop",
        "in_template_lp_n_models",
        "in_template_lp_min_drop_pp",
        "in_template_lp_max_drop_pp",
    ]
    out = table[cols].copy()
    out.to_csv(out_dir / "top_paradigm_drop_table_lp_nll_full.csv", index=False)
    out.head(20).to_csv(out_dir / "top_20_paradigm_drops_lp_nll.csv", index=False)
    out.tail(15).sort_values("nll_median_drop_pp").to_csv(
        out_dir / "top_15_paradigm_reversals_lp_nll.csv", index=False
    )

    display = out.head(20).copy()
    display["paradigm"] = display["paradigm"].map(prettify)
    display["phenomenon"] = display["phenomenon"].map(prettify)
    display["field"] = display["field"].map(prettify)
    display["NLL median pp"] = display["nll_median_drop_pp"].map(fmt_pp)
    display[f"NLL n/{len(MODEL_ORDER)}"] = display.apply(
        lambda r: f"{int(r['nll_n_drop'])}/{int(r['nll_n_models'])}",
        axis=1,
    )
    display["LP median pp"] = display["in_template_lp_median_drop_pp"].map(fmt_pp)
    display[f"LP n/{len(MODEL_ORDER)}"] = display.apply(
        lambda r: f"{int(r['in_template_lp_n_drop'])}/{int(r['in_template_lp_n_models'])}",
        axis=1,
    )
    md = to_markdown(
        display[
            [
                "paradigm",
                "phenomenon",
                "field",
                "NLL median pp",
                f"NLL n/{len(MODEL_ORDER)}",
                "LP median pp",
                f"LP n/{len(MODEL_ORDER)}",
            ]
        ]
    )
    (out_dir / "top_20_paradigm_drops_lp_nll.md").write_text(md + "\n", encoding="utf-8")

    signed = out.head(20).copy()
    signed["LP Readout median pp"] = signed["nll_median_drop_pp"].map(lambda x: f"{-x:+.1f}")
    signed["LP Readout n/models"] = signed.apply(
        lambda r: f"{int(r['nll_n_drop'])}/{int(r['nll_n_models'])}",
        axis=1,
    )
    signed["worst model pp"] = signed["nll_max_drop_pp"].map(lambda x: f"{-x:+.1f}")
    signed["least affected pp"] = signed["nll_min_drop_pp"].map(lambda x: f"{-x:+.1f}")
    signed_display = signed[[
        "paradigm",
        "phenomenon",
        "field",
        "LP Readout median pp",
        "LP Readout n/models",
        "worst model pp",
        "least affected pp",
    ]].copy()
    signed_display["paradigm"] = signed_display["paradigm"].map(prettify)
    signed_display["phenomenon"] = signed_display["phenomenon"].map(prettify)
    signed_display["field"] = signed_display["field"].map(prettify)
    signed_display.to_csv(out_dir / "top_20_paradigm_drops_lp_readout_signed.csv", index=False)
    (out_dir / "top_20_paradigm_drops_lp_readout_signed.md").write_text(
        to_markdown(signed_display) + "\n",
        encoding="utf-8",
    )

    drops = out.head(10).copy()
    drops["pattern"] = "drop"
    reversals = out.tail(10).sort_values("nll_median_drop_pp", ascending=True).copy()
    reversals["pattern"] = "reversal"
    extremes = pd.concat([drops, reversals], ignore_index=True)
    extremes["LP Readout median pp"] = extremes["nll_median_drop_pp"].map(lambda x: f"{-x:+.1f}")
    extremes["LP Readout n/models"] = extremes.apply(
        lambda r: f"{int(r['nll_n_drop'])}/{int(r['nll_n_models'])}",
        axis=1,
    )
    extremes["worst drop pp"] = extremes["nll_max_drop_pp"].map(lambda x: f"{-x:+.1f}")
    extremes["strongest reversal pp"] = extremes["nll_min_drop_pp"].map(lambda x: f"{-x:+.1f}")
    extremes_display = extremes[
        [
            "pattern",
            "paradigm",
            "phenomenon",
            "field",
            "LP Readout median pp",
            "LP Readout n/models",
            "worst drop pp",
            "strongest reversal pp",
        ]
    ].copy()
    extremes_display["paradigm"] = extremes_display["paradigm"].map(prettify)
    extremes_display["phenomenon"] = extremes_display["phenomenon"].map(prettify)
    extremes_display["field"] = extremes_display["field"].map(prettify)
    extremes_display.to_csv(out_dir / "paradigm_drop_and_reversal_examples_lp_readout_signed.csv", index=False)
    (out_dir / "paradigm_drop_and_reversal_examples_lp_readout_signed.md").write_text(
        to_markdown(extremes_display) + "\n",
        encoding="utf-8",
    )


def make_context_summary(input_dir: Path, out_dir: Path) -> None:
    overall = pd.read_csv(input_dir / "overall_effects_by_model_method.csv")
    field_effects = pd.read_csv(input_dir / "field_effects_by_model_method.csv")
    phen_effects = pd.read_csv(input_dir / "phenomenon_effects_by_model_method.csv")

    overall = overall[overall["model_slug"].isin(MODEL_ORDER)].copy()
    overall_nll = overall[overall["method"].eq("nll")].copy()
    overall_lp = overall[overall["method"].eq("in_template_lp")].copy()
    keep = [
        "model_label",
        "accuracy_original",
        "accuracy_head",
        "accuracy_tail",
        "accuracy_xtail",
        "drop_head_xtail_pp",
        "drop_original_xtail_pp",
    ]
    nll_table = overall_nll[keep].sort_values("drop_head_xtail_pp", ascending=False)
    lp_table = overall_lp[keep].sort_values("drop_head_xtail_pp", ascending=False)

    lines = ["# Main-suite linguistic frequency effect context", ""]
    lines.append("## Overall NLL by model")
    nll_display = nll_table.copy()
    for col in nll_display.columns:
        if col != "model_label":
            nll_display[col] = nll_display[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    lines.append(to_markdown(nll_display))
    lines.append("")
    lines.append("## Overall LP by model")
    lp_display = lp_table.copy()
    for col in lp_display.columns:
        if col != "model_label":
            lp_display[col] = lp_display[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    lines.append(to_markdown(lp_display))
    lines.append("")
    lines.append("## Field consensus")
    field_rows = []
    for method in ["in_template_lp", "nll"]:
        consensus = method_consensus(field_effects, method, "field")
        for row in consensus.itertuples(index=False):
            field_rows.append(
                {
                    "method": method,
                    "field": row.field,
                    "n_models": getattr(row, f"{method}_n_models"),
                    "n_positive_head_xtail": getattr(row, f"{method}_n_drop"),
                    "median_drop_head_xtail_pp": getattr(row, f"{method}_median_drop_pp"),
                    "min_drop_head_xtail_pp": getattr(row, f"{method}_min_drop_pp"),
                    "max_drop_head_xtail_pp": getattr(row, f"{method}_max_drop_pp"),
                }
            )
    field_display = pd.DataFrame(field_rows)
    for col in ["median_drop_head_xtail_pp", "min_drop_head_xtail_pp", "max_drop_head_xtail_pp"]:
        field_display[col] = field_display[col].map(lambda x: f"{x:.2f}")
    lines.append(to_markdown(field_display))
    lines.append("")
    lines.append("## NLL phenomenon consensus")
    phen_consensus = method_consensus(phen_effects, "nll", "phenomenon")
    phen_display = phen_consensus.rename(
        columns={
            "nll_n_models": "n_models",
            "nll_n_drop": "n_positive_head_xtail",
            "nll_median_drop_pp": "median_drop_head_xtail_pp",
            "nll_min_drop_pp": "min_drop_head_xtail_pp",
            "nll_max_drop_pp": "max_drop_head_xtail_pp",
        }
    )[
        [
            "phenomenon",
            "n_models",
            "n_positive_head_xtail",
            "median_drop_head_xtail_pp",
            "min_drop_head_xtail_pp",
            "max_drop_head_xtail_pp",
        ]
    ].copy()
    for col in ["median_drop_head_xtail_pp", "min_drop_head_xtail_pp", "max_drop_head_xtail_pp"]:
        phen_display[col] = phen_display[col].map(lambda x: f"{x:.2f}")
    lines.append(to_markdown(phen_display))
    lines.append("")
    (out_dir / "paper_context_numbers.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default="results/linguistic_frequency_effects_20260511_main_suite")
    ap.add_argument("--output-dir", default="results/linguistic_frequency_effects_20260511_main_suite/paper_assets")
    args = ap.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", str(Path(args.output_dir) / ".mplconfig"))
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    make_heatmap(input_dir, out_dir)
    make_nll_lp_readout_heatmap(input_dir, out_dir)
    make_base_lp_readout_faceted_heatmap(input_dir, out_dir)
    make_instruct_yes_no_heatmap(input_dir, out_dir)
    make_instruct_yes_no_paradigm_extremes(input_dir, out_dir)
    make_top_paradigm_tables(input_dir, out_dir)
    make_context_summary(input_dir, out_dir)
    print(f"Wrote paper assets to {out_dir}")


if __name__ == "__main__":
    main()
