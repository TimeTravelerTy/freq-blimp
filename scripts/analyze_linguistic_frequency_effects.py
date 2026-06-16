#!/usr/bin/env python3
"""Summarize FreqBLiMP frequency effects by linguistic category.

The paper eval bundle stores frequency runs one file per paradigm, but original
BLiMP runs as aggregate files. This script normalizes both into per-UID accuracy
rows, then writes model-specific field/phenomenon/paradigm effect tables.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable

import pandas as pd


REGIMES = ("original", "head", "tail", "xtail")
FREQ_REGIMES = ("head", "tail", "xtail")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc


def load_uid_metadata(path: Path) -> pd.DataFrame:
    rows = {}
    for rec in iter_jsonl(path):
        uid = rec.get("subtask")
        if not isinstance(uid, str) or not uid:
            continue
        rows.setdefault(
            uid,
            {
                "uid": uid,
                "phenomenon": rec.get("phenomenon") or "",
                "field": rec.get("field") or "",
            },
        )
    return pd.DataFrame(rows.values())


def load_selected_files(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def frequency_rows(selected: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    freq = selected[selected["variant"].eq("freq")].copy()
    freq = freq.rename(columns={"rows": "total"})
    keep = [
        "model_slug",
        "model_label",
        "model",
        "variant",
        "regime",
        "uid",
        "method",
        "correct",
        "total",
        "accuracy",
        "repo_relative_path",
    ]
    freq = freq[keep]
    freq["correct"] = freq["correct"].astype(int)
    freq["total"] = freq["total"].astype(int)
    freq["accuracy"] = freq["accuracy"].astype(float)
    out = freq.merge(meta, on="uid", how="left")
    missing = out[out["field"].isna() | out["phenomenon"].isna()]["uid"].unique()
    if len(missing):
        raise ValueError(f"Missing metadata for frequency UIDs: {sorted(missing)}")
    return out


def original_rows(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    originals = selected[selected["variant"].eq("original")].copy()
    for file_row in originals.to_dict("records"):
        path = Path(file_row["repo_relative_path"])
        buckets: DefaultDict[str, dict] = defaultdict(
            lambda: {
                "model_slug": file_row["model_slug"],
                "model_label": file_row["model_label"],
                "model": file_row["model"],
                "variant": "original",
                "regime": "original",
                "uid": "",
                "method": file_row["method"],
                "correct": 0,
                "total": 0,
                "repo_relative_path": file_row["repo_relative_path"],
                "phenomenon": "",
                "field": "",
            }
        )
        for rec in iter_jsonl(path):
            uid = rec.get("subtask")
            correct = rec.get("correctness")
            if not isinstance(uid, str) or not isinstance(correct, int):
                continue
            bucket = buckets[uid]
            bucket["uid"] = uid
            bucket["correct"] += correct
            bucket["total"] += 1
            if not bucket["phenomenon"]:
                bucket["phenomenon"] = rec.get("phenomenon") or ""
            if not bucket["field"]:
                bucket["field"] = rec.get("field") or ""
        for bucket in buckets.values():
            if bucket["total"]:
                bucket["accuracy"] = bucket["correct"] / bucket["total"]
                rows.append(bucket)
    return pd.DataFrame(rows)


def add_group_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["overall"] = "all"
    out["paradigm"] = out["uid"]
    return out


def aggregate_level(df: pd.DataFrame, level: str) -> pd.DataFrame:
    group_cols = ["model_slug", "model_label", "model", "method", "regime", level]
    agg = (
        df.groupby(group_cols, dropna=False)[["correct", "total"]]
        .sum()
        .reset_index()
    )
    agg["accuracy"] = agg["correct"] / agg["total"]
    return agg


def effect_table(df: pd.DataFrame, level: str) -> pd.DataFrame:
    agg = aggregate_level(df, level)
    wide = agg.pivot_table(
        index=["model_slug", "model_label", "model", "method", level],
        columns="regime",
        values=["accuracy", "correct", "total"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{regime}" for metric, regime in wide.columns]
    wide = wide.reset_index()
    for regime in REGIMES:
        acc_col = f"accuracy_{regime}"
        if acc_col not in wide:
            wide[acc_col] = pd.NA
    wide["drop_head_tail_pp"] = (wide["accuracy_head"] - wide["accuracy_tail"]) * 100
    wide["drop_tail_xtail_pp"] = (wide["accuracy_tail"] - wide["accuracy_xtail"]) * 100
    wide["drop_head_xtail_pp"] = (wide["accuracy_head"] - wide["accuracy_xtail"]) * 100
    wide["drop_original_head_pp"] = (wide["accuracy_original"] - wide["accuracy_head"]) * 100
    wide["drop_original_xtail_pp"] = (wide["accuracy_original"] - wide["accuracy_xtail"]) * 100
    wide["monotonic_head_tail_xtail"] = (
        wide["accuracy_head"].notna()
        & wide["accuracy_tail"].notna()
        & wide["accuracy_xtail"].notna()
        & (wide["accuracy_head"] >= wide["accuracy_tail"])
        & (wide["accuracy_tail"] >= wide["accuracy_xtail"])
    )
    if level == "paradigm":
        meta = (
            df[["uid", "phenomenon", "field"]]
            .drop_duplicates()
            .rename(columns={"uid": "paradigm"})
        )
        wide = wide.merge(meta, on="paradigm", how="left")
    return wide.sort_values(["method", "model_slug", level]).reset_index(drop=True)


def consensus_table(effect: pd.DataFrame, level: str) -> pd.DataFrame:
    rows = []
    for (method, group), sub in effect.groupby(["method", level], dropna=False):
        vals = pd.to_numeric(sub["drop_head_xtail_pp"], errors="coerce").dropna()
        monotonic = sub["monotonic_head_tail_xtail"].fillna(False)
        if vals.empty:
            continue
        rows.append(
            {
                "method": method,
                level: group,
                "n_models": int(vals.shape[0]),
                "n_positive_head_xtail": int((vals > 0).sum()),
                "n_negative_head_xtail": int((vals < 0).sum()),
                "n_monotonic": int(monotonic.sum()),
                "median_drop_head_xtail_pp": float(vals.median()),
                "mean_drop_head_xtail_pp": float(vals.mean()),
                "min_drop_head_xtail_pp": float(vals.min()),
                "max_drop_head_xtail_pp": float(vals.max()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["method", "median_drop_head_xtail_pp"], ascending=[True, False]
    )


def top_paradigms(paradigm_effect: pd.DataFrame, method: str, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = paradigm_effect[paradigm_effect["method"].eq(method)].copy()
    sub["drop_head_xtail_pp"] = pd.to_numeric(sub["drop_head_xtail_pp"], errors="coerce")
    sub = sub.dropna(subset=["drop_head_xtail_pp"])
    cols = [
        "model_slug",
        "model_label",
        "method",
        "paradigm",
        "phenomenon",
        "field",
        "accuracy_head",
        "accuracy_tail",
        "accuracy_xtail",
        "drop_head_xtail_pp",
        "drop_original_xtail_pp",
    ]
    largest = (
        sub.sort_values(["model_slug", "drop_head_xtail_pp"], ascending=[True, False])
        .groupby("model_slug", group_keys=False)
        .head(n)[cols]
    )
    smallest = (
        sub.sort_values(["model_slug", "drop_head_xtail_pp"], ascending=[True, True])
        .groupby("model_slug", group_keys=False)
        .head(n)[cols]
    )
    return largest, smallest


def write_outputs(df: pd.DataFrame, out_dir: Path, top_n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "accuracy_by_uid_normalized.csv", index=False)

    effects = {}
    for level in ("overall", "field", "phenomenon", "paradigm"):
        table = effect_table(df, level)
        table.to_csv(out_dir / f"{level}_effects_by_model_method.csv", index=False)
        consensus = consensus_table(table, level)
        consensus.to_csv(out_dir / f"{level}_consensus_by_method.csv", index=False)
        effects[level] = table

    for method in ("nll", "in_template_lp", "yes_no"):
        largest, smallest = top_paradigms(effects["paradigm"], method, top_n)
        largest.to_csv(out_dir / f"top_{top_n}_paradigm_drops_{method}.csv", index=False)
        smallest.to_csv(out_dir / f"top_{top_n}_paradigm_reversals_{method}.csv", index=False)

    manifest = {
        "rows": int(df.shape[0]),
        "models": sorted(df["model_slug"].dropna().unique().tolist()),
        "methods": sorted(df["method"].dropna().unique().tolist()),
        "regimes": sorted(df["regime"].dropna().unique().tolist()),
        "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default="results/paper_eval_bundle_final_20260509")
    ap.add_argument("--metadata", default="data/processed/blimp_original.jsonl")
    ap.add_argument("--output-dir", default="results/linguistic_frequency_effects_20260511")
    ap.add_argument(
        "--exclude-model",
        action="append",
        default=[],
        help="Model slug to exclude. Repeatable.",
    )
    ap.add_argument(
        "--include-model",
        action="append",
        default=[],
        help="If set, keep only these model slugs. Repeatable.",
    )
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()

    bundle = Path(args.bundle)
    selected = load_selected_files(bundle / "catalog" / "selected_files.csv")
    if args.include_model:
        selected = selected[selected["model_slug"].isin(args.include_model)].copy()
    if args.exclude_model:
        selected = selected[~selected["model_slug"].isin(args.exclude_model)].copy()
    meta = load_uid_metadata(Path(args.metadata))
    df = pd.concat([frequency_rows(selected, meta), original_rows(selected)], ignore_index=True)
    df = add_group_labels(df)
    write_outputs(df, Path(args.output_dir), args.top_n)
    print(f"Wrote linguistic frequency effect tables to {args.output_dir}")


if __name__ == "__main__":
    main()
