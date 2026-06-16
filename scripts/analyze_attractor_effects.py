#!/usr/bin/env python3
"""Analyze BLiMP/FreqBLiMP distractor and short-distance agreement effects.

This script consumes the normalized UID-level tables produced by
scripts/analyze_linguistic_frequency_effects.py and collapses the relevant
agreement paradigms into linguistically motivated conditions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SUBJECT_VERB_GROUPS = {
    "sv_no_distractor": [
        "regular_plural_subject_verb_agreement_1",
        "regular_plural_subject_verb_agreement_2",
        "irregular_plural_subject_verb_agreement_1",
        "irregular_plural_subject_verb_agreement_2",
    ],
    "sv_relational_noun_attractor": ["distractor_agreement_relational_noun"],
    "sv_relative_clause_attractor": ["distractor_agreement_relative_clause"],
}

DET_NOUN_GROUPS = {
    "det_noun_local": [
        "determiner_noun_agreement_1",
        "determiner_noun_agreement_2",
        "determiner_noun_agreement_irregular_1",
        "determiner_noun_agreement_irregular_2",
    ],
    "det_noun_adjective_intervened": [
        "determiner_noun_agreement_with_adjective_1",
        "determiner_noun_agreement_with_adj_2",
        "determiner_noun_agreement_with_adj_irregular_1",
        "determiner_noun_agreement_with_adj_irregular_2",
    ],
}

CONDITION_NOTES = {
    "sv_no_distractor": (
        "Subject-verb agreement controls without an intervening opposite-number noun. "
        "Includes regular and irregular plural controls."
    ),
    "sv_relational_noun_attractor": (
        "Subject followed by a relational-noun complement containing an opposite-number noun, "
        "then the agreeing verb."
    ),
    "sv_relative_clause_attractor": (
        "Subject followed by a relative clause whose embedded object has the opposite number, "
        "then the agreeing matrix verb."
    ),
    "det_noun_local": (
        "Demonstrative-noun agreement with no adjective between demonstrative and noun."
    ),
    "det_noun_adjective_intervened": (
        "Demonstrative-noun agreement with one adjective between demonstrative and noun."
    ),
}

SOURCE_HINTS = {
    "sv_relational_noun_attractor": (
        "/Users/tyronewhite/masters_research_code/freq-blimp/"
        "generation_projects/blimp/distractor_agreement_relational_noun.py"
    ),
    "sv_relative_clause_attractor": (
        "/Users/tyronewhite/masters_research_code/freq-blimp/"
        "generation_projects/blimp/distractor_agreement_rc.py"
    ),
    "det_noun_local": (
        "/Users/tyronewhite/masters_research_code/freq-blimp/"
        "generation_projects/blimp/determiner_noun_agreement_1.py"
    ),
    "det_noun_adjective_intervened": (
        "/Users/tyronewhite/masters_research_code/freq-blimp/"
        "generation_projects/blimp/determiner_noun_agreement_with_adj_1.py"
    ),
}

REGIME_ORDER = ["original", "head", "tail", "xtail"]


def load_accuracy(path: Path, methods: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if methods:
        df = df[df["method"].isin(methods)].copy()
    keep = set().union(*SUBJECT_VERB_GROUPS.values(), *DET_NOUN_GROUPS.values())
    df = df[df["uid"].isin(keep)].copy()
    if df.empty:
        raise SystemExit(f"No target agreement rows found in {path}")
    return df


def condition_inventory() -> pd.DataFrame:
    rows = []
    for domain, groups in [
        ("subject_verb_agreement", SUBJECT_VERB_GROUPS),
        ("determiner_noun_agreement", DET_NOUN_GROUPS),
    ]:
        for condition, paradigms in groups.items():
            for uid in paradigms:
                rows.append(
                    {
                        "domain": domain,
                        "condition": condition,
                        "uid": uid,
                        "condition_note": CONDITION_NOTES[condition],
                        "source_hint": SOURCE_HINTS.get(condition, ""),
                    }
                )
    return pd.DataFrame(rows)


def add_conditions(df: pd.DataFrame) -> pd.DataFrame:
    inventory = condition_inventory()[["domain", "condition", "uid"]]
    out = df.merge(inventory, on="uid", how="inner")
    if out.empty:
        raise SystemExit("Condition mapping removed every row.")
    return out


def summarize_conditions(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["domain", "condition", "model_slug", "model_label", "model", "method", "regime"]
    # Paradigm-balanced accuracy: every UID contributes one mean, regardless of
    # generated item count or future partial reruns.
    out = (
        df.groupby(group_cols, observed=True)
        .agg(
            accuracy=("accuracy", "mean"),
            n_paradigms=("uid", "nunique"),
            total_items=("total", "sum"),
            paradigms=("uid", lambda s: ";".join(sorted(set(s)))),
        )
        .reset_index()
    )
    out["regime"] = pd.Categorical(out["regime"], categories=REGIME_ORDER, ordered=True)
    return out.sort_values(["domain", "condition", "method", "model_slug", "regime"]).reset_index(drop=True)


def frequency_effects(summary: pd.DataFrame) -> pd.DataFrame:
    wide = summary.pivot_table(
        index=["domain", "condition", "model_slug", "model_label", "model", "method"],
        columns="regime",
        values="accuracy",
        aggfunc="first",
        observed=False,
    ).reset_index()
    for regime in REGIME_ORDER:
        if regime not in wide:
            wide[regime] = pd.NA
    wide["drop_original_head_pp"] = (wide["original"] - wide["head"]) * 100
    wide["drop_original_tail_pp"] = (wide["original"] - wide["tail"]) * 100
    wide["drop_original_xtail_pp"] = (wide["original"] - wide["xtail"]) * 100
    wide["drop_head_tail_pp"] = (wide["head"] - wide["tail"]) * 100
    wide["drop_tail_xtail_pp"] = (wide["tail"] - wide["xtail"]) * 100
    wide["drop_head_xtail_pp"] = (wide["head"] - wide["xtail"]) * 100
    wide["monotonic_head_tail_xtail"] = (
        wide["head"].notna()
        & wide["tail"].notna()
        & wide["xtail"].notna()
        & (wide["head"] >= wide["tail"])
        & (wide["tail"] >= wide["xtail"])
    )
    return wide.sort_values(["domain", "condition", "method", "model_slug"]).reset_index(drop=True)


def condition_penalties(summary: pd.DataFrame) -> pd.DataFrame:
    idx = ["model_slug", "model_label", "model", "method", "regime"]
    wide = summary.pivot_table(
        index=idx,
        columns="condition",
        values="accuracy",
        aggfunc="first",
        observed=False,
    ).reset_index()
    rows = []
    comparisons = [
        ("subject_verb_agreement", "relational_noun_attractor_penalty_pp", "sv_no_distractor", "sv_relational_noun_attractor"),
        ("subject_verb_agreement", "relative_clause_attractor_penalty_pp", "sv_no_distractor", "sv_relative_clause_attractor"),
        ("determiner_noun_agreement", "adjective_distance_penalty_pp", "det_noun_local", "det_noun_adjective_intervened"),
    ]
    for rec in wide.to_dict("records"):
        for domain, metric, baseline, target in comparisons:
            if pd.isna(rec.get(baseline)) or pd.isna(rec.get(target)):
                continue
            rows.append(
                {
                    "domain": domain,
                    "metric": metric,
                    "model_slug": rec["model_slug"],
                    "model_label": rec["model_label"],
                    "model": rec["model"],
                    "method": rec["method"],
                    "regime": rec["regime"],
                    "baseline_condition": baseline,
                    "target_condition": target,
                    "baseline_accuracy": rec[baseline],
                    "target_accuracy": rec[target],
                    "penalty_pp": (rec[baseline] - rec[target]) * 100,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["regime"] = pd.Categorical(out["regime"], categories=REGIME_ORDER, ordered=True)
    return out.sort_values(["domain", "metric", "method", "model_slug", "regime"]).reset_index(drop=True)


def consensus(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in df.groupby(group_cols, observed=True, dropna=False):
        vals = pd.to_numeric(part[value_col], errors="coerce").dropna()
        if vals.empty:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_models": int(part["model_slug"].nunique()),
                "mean": float(vals.mean()),
                "median": float(vals.median()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "n_positive": int((vals > 0).sum()),
                "n_negative": int((vals < 0).sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_readme(out_dir: Path, manifest: dict) -> None:
    text = f"""# Attractor and Distance Effects

Inputs: `{manifest["input"]}`

This analysis keeps the base-model rows from the normalized linguistic-frequency table and groups the relevant BLiMP paradigms into:

- subject-verb controls with no distractor
- subject-verb relational-noun attractors
- subject-verb relative-clause attractors
- determiner-noun local agreement
- determiner-noun adjective-intervened agreement

Positive `penalty_pp` means the distractor/distance condition is harder than its local/no-distractor baseline. Positive frequency drops mean accuracy is higher in the first named regime than the second.

Primary files:

- `condition_inventory.csv`: UID-to-condition mapping and generator-source hints.
- `condition_accuracy_by_model_method_regime.csv`: paradigm-balanced accuracy by condition.
- `condition_frequency_effects.csv`: original/head/tail/xtail pivots and regime drops by condition.
- `condition_penalties.csv`: attractor/distance penalties by model, method, and regime.
- `condition_penalty_consensus.csv`: cross-model summaries of the penalties.
- `condition_frequency_consensus.csv`: cross-model summaries of head-to-xtail drops by condition.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        default="results/linguistic_frequency_effects_20260512_base/accuracy_by_uid_normalized.csv",
        help="Normalized UID accuracy CSV from analyze_linguistic_frequency_effects.py.",
    )
    ap.add_argument("--output-dir", default="results/attractor_effects_20260512_base")
    ap.add_argument(
        "--method",
        action="append",
        default=[],
        help="Restrict to method. Repeatable. Default: all methods present.",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = set(args.method)
    raw = load_accuracy(Path(args.input), methods)
    conditioned = add_conditions(raw)
    summary = summarize_conditions(conditioned)
    effects = frequency_effects(summary)
    penalties = condition_penalties(summary)
    penalty_consensus = consensus(
        penalties,
        "penalty_pp",
        ["domain", "metric", "method", "regime"],
    )
    frequency_consensus = consensus(
        effects,
        "drop_head_xtail_pp",
        ["domain", "condition", "method"],
    )

    inventory = condition_inventory()
    inventory.to_csv(out_dir / "condition_inventory.csv", index=False)
    conditioned.to_csv(out_dir / "target_paradigm_accuracy_rows.csv", index=False)
    summary.to_csv(out_dir / "condition_accuracy_by_model_method_regime.csv", index=False)
    effects.to_csv(out_dir / "condition_frequency_effects.csv", index=False)
    penalties.to_csv(out_dir / "condition_penalties.csv", index=False)
    penalty_consensus.to_csv(out_dir / "condition_penalty_consensus.csv", index=False)
    frequency_consensus.to_csv(out_dir / "condition_frequency_consensus.csv", index=False)

    manifest = {
        "input": args.input,
        "rows": int(conditioned.shape[0]),
        "models": sorted(conditioned["model_slug"].dropna().unique().tolist()),
        "methods": sorted(conditioned["method"].dropna().unique().tolist()),
        "regimes": sorted(conditioned["regime"].dropna().unique().tolist()),
        "conditions": sorted(conditioned["condition"].dropna().unique().tolist()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_readme(out_dir, manifest)
    print(f"Wrote attractor analysis to {out_dir}")


if __name__ == "__main__":
    main()
