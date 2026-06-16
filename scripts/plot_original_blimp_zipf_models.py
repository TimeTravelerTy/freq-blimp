#!/usr/bin/env python3
"""Plot original BLiMP base-model accuracy, margin, and LP/word by Zipf.

Rows are binned by realized content-word Zipf in the original grammatical
sentence. Within each model/bin, item scores are averaged per BLiMP paradigm
first, then paradigm means are averaged so paradigms contribute equally.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "results/original_blimp_zipf_models/.mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordfreq import zipf_frequency


TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "were", "with", "not", "nt", "s", "d", "ll", "re", "ve", "m",
}
METHODS = ("nll", "in_template_lp", "in_template_meanlp", "in_template_penlp", "yes_no", "ensemble")
DEFAULT_OUT_DIR = Path("results/original_blimp_zipf_models")


@dataclass(frozen=True)
class ScoreFile:
    path: Path
    model: str
    method: str
    paradigm: str
    source_rank: str
    mtime: float


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc


def _iter_score_paths(values: Sequence[str]) -> List[Path]:
    paths = set()
    for value in values:
        p = Path(value)
        if any(ch in value for ch in "*?[]"):
            paths.update(Path().glob(value))
        elif p.is_dir():
            paths.update(p.rglob("*blimp_original_original*_acceptability.jsonl"))
        elif p.is_file():
            paths.add(p)
    return sorted(p for p in paths if p.is_file() and p.suffix == ".jsonl")


def _first_record(path: Path) -> Optional[dict]:
    for rec in _iter_jsonl(path):
        return rec
    return None


def _source_rank(path: Path) -> str:
    match = re.match(r"(?P<rank>\d{8}(?:-\d{6})?(?:-[A-Za-z0-9]+)?)", path.name)
    return match.group("rank") if match else path.name


def _method(rec: dict, path: Path) -> str:
    value = rec.get("method")
    if isinstance(value, str) and value:
        return value
    match = re.search(rf"_original_({'|'.join(METHODS)})_acceptability\.jsonl$", path.name)
    return match.group(1) if match else "unknown"


def _paradigm(rec: dict, path: Path) -> str:
    for key in ("subtask", "UID"):
        value = rec.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("dataset_name", "dataset_path"):
        value = rec.get(key)
        if isinstance(value, str) and value:
            return Path(value).stem
    return path.stem


def _score_file(path: Path) -> Optional[ScoreFile]:
    rec = _first_record(path)
    if rec is None:
        return None
    model = rec.get("model")
    if not isinstance(model, str) or not model:
        model = "unknown"
    return ScoreFile(
        path=path,
        model=model,
        method=_method(rec, path),
        paradigm=_paradigm(rec, path),
        source_rank=_source_rank(path),
        mtime=path.stat().st_mtime,
    )


def _model_label(model: str) -> str:
    return model.split("/")[-1].replace("_", ".")


def _model_slug(model: str) -> str:
    return model.split("/")[-1].replace(".", "_").replace("/", "_")


def _model_matches_exclude(model: str, patterns: Sequence[str]) -> bool:
    model_low = model.lower()
    label_low = _model_label(model).lower()
    slug_low = _model_slug(model).lower()
    return any(pattern.lower() in model_low or pattern.lower() in label_low or pattern.lower() in slug_low for pattern in patterns)


def _latest_files(paths: Sequence[Path], method: str, exclude_model: Sequence[str]) -> List[ScoreFile]:
    latest: Dict[Tuple[str, str, str], ScoreFile] = {}
    for path in paths:
        meta = _score_file(path)
        if meta is None:
            continue
        if meta.method != method:
            continue
        if "blimp_original" not in path.name:
            continue
        if _model_matches_exclude(meta.model, exclude_model):
            continue
        key = (meta.model, meta.method, meta.paradigm)
        previous = latest.get(key)
        if previous is None or (meta.mtime, meta.source_rank, meta.path.name) > (
            previous.mtime,
            previous.source_rank,
            previous.path.name,
        ):
            latest[key] = meta
    return sorted(latest.values(), key=lambda f: (f.model, f.method, f.paradigm, f.path.name))


def _text(rec: dict, default_field: str, fallback_key: str) -> Optional[str]:
    value = rec.get(fallback_key)
    if isinstance(value, str):
        return value
    field = rec.get(default_field)
    if isinstance(field, str):
        value = rec.get(field)
        if isinstance(value, str):
            return value
    return None


def _logprob(rec: dict, side: str) -> Optional[float]:
    lp = rec.get(f"{side}_total_logprob")
    if isinstance(lp, (int, float)):
        return float(lp)
    nll = rec.get(f"{side}_total_nll")
    if isinstance(nll, (int, float)):
        return -float(nll)
    score = rec.get(f"score_{side}")
    if isinstance(score, (int, float)):
        return float(score)
    return None


def _content_words(text: str) -> List[str]:
    words = []
    for raw in TOKEN_RE.findall(text):
        token = raw.lower()
        if token.endswith("'s"):
            token = token[:-2]
        if token in STOPWORDS or len(token) <= 1:
            continue
        words.append(token)
    return words


def _zipf_stat(text: str, stat: str, unique_words: bool) -> Optional[float]:
    words = _content_words(text)
    if unique_words:
        words = sorted(set(words))
    values = [float(zipf_frequency(word, "en")) for word in words]
    values = [value for value in values if value > 0.0]
    if not values:
        return None
    if stat == "mean":
        return float(sum(values) / len(values))
    if stat == "median":
        return float(statistics.median(values))
    raise ValueError(f"Unknown Zipf stat: {stat}")


def _word_count(text: str) -> int:
    return max(1, len(TOKEN_RE.findall(text)))


def _collect_points(files: Sequence[ScoreFile], args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    zipf_cache: Dict[Tuple[str, str, bool], Optional[float]] = {}
    for meta in files:
        for row_i, rec in enumerate(_iter_jsonl(meta.path)):
            if rec.get("variant") not in (None, "original"):
                continue
            good_text = _text(rec, "good_field", "good_text")
            bad_text = _text(rec, "bad_field", "bad_text")
            good_lp = _logprob(rec, "good")
            bad_lp = _logprob(rec, "bad")
            correctness = rec.get("correctness")
            if not isinstance(good_text, str) or not isinstance(bad_text, str):
                continue
            if good_lp is None or bad_lp is None or not isinstance(correctness, (int, bool)):
                continue
            cache_key = (good_text, args.zipf_stat, bool(args.unique_words))
            if cache_key not in zipf_cache:
                zipf_cache[cache_key] = _zipf_stat(good_text, args.zipf_stat, args.unique_words)
            zipf_value = zipf_cache[cache_key]
            if zipf_value is None:
                continue
            rows.append(
                {
                    "model": meta.model,
                    "model_slug": _model_slug(meta.model),
                    "model_label": _model_label(meta.model),
                    "method": meta.method,
                    "paradigm": _paradigm(rec, meta.path),
                    "source_file": str(meta.path),
                    "source_rank": meta.source_rank,
                    "row": row_i,
                    "correct": int(correctness),
                    "margin_lp": float(good_lp) - float(bad_lp),
                    "good_lp": float(good_lp),
                    "good_lp_per_word": float(good_lp) / _word_count(good_text),
                    "zipf": float(zipf_value),
                    "good_text": good_text,
                    "bad_text": bad_text,
                }
            )
    if not rows:
        raise SystemExit("No usable original BLiMP score rows found after filtering.")
    return pd.DataFrame(rows)


def _mean_ci(values: np.ndarray, z: float = 1.96) -> Tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1:
        mean = float(values[0])
        return mean, mean
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(values.size))
    return mean - z * se, mean + z * se


def _make_bins(points: pd.DataFrame, bins: int, strategy: str) -> pd.DataFrame:
    work = points.dropna(subset=["zipf"]).copy()
    if strategy == "quantile":
        work["zipf_bin"] = pd.qcut(work["zipf"], q=bins, duplicates="drop")
    elif strategy == "equal":
        work["zipf_bin"] = pd.cut(work["zipf"], bins=bins)
    else:
        raise ValueError(f"Unknown bin strategy: {strategy}")
    return work


def _balanced_binned(points: pd.DataFrame, args: argparse.Namespace, metrics: Sequence[str]) -> pd.DataFrame:
    binned = _make_bins(points, args.bins, args.bin_strategy)
    if "accuracy" in metrics and "accuracy" not in binned.columns:
        binned["accuracy"] = binned["correct"]
    aggregations = {metric: (metric, "mean") for metric in metrics}
    aggregations.update(n_items=("correct", "size"), zipf_mid=("zipf", "mean"))
    paradigm = (
        binned.groupby(["model", "model_slug", "model_label", "zipf_bin", "paradigm"], observed=True)
        .agg(**aggregations)
        .reset_index()
    )
    if args.min_items_per_paradigm_bin > 1:
        paradigm = paradigm[paradigm["n_items"] >= args.min_items_per_paradigm_bin].copy()

    rows = []
    for keys, part in paradigm.groupby(["model", "model_slug", "model_label", "zipf_bin"], observed=True):
        model, model_slug, model_label, bin_label = keys
        if len(part) < args.min_paradigms:
            continue
        row = {
            "model": model,
            "model_slug": model_slug,
            "model_label": model_label,
            "zipf_bin": str(bin_label),
            "zipf_mid": float(bin_label.mid),
            "zipf_left": float(bin_label.left),
            "zipf_right": float(bin_label.right),
            "n_paradigms": int(len(part)),
            "n_items": int(part["n_items"].sum()),
        }
        for metric in metrics:
            values = part[metric].to_numpy(dtype=float)
            lo, hi = _mean_ci(values)
            row[metric] = float(np.mean(values))
            row[f"{metric}_lo"] = lo
            row[f"{metric}_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model_label", "zipf_mid"])


def _average_line(summary: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows = []
    for zipf_mid, part in summary.groupby("zipf_mid", observed=True):
        row = {"zipf_mid": float(zipf_mid)}
        for metric in metrics:
            values = part[metric].to_numpy(dtype=float)
            lo, hi = _mean_ci(values)
            row[metric] = float(np.mean(values))
            row[f"{metric}_lo"] = lo
            row[f"{metric}_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values("zipf_mid")


def _sensitivity_models(summary: pd.DataFrame, metric: str) -> Tuple[str, str]:
    sensitivities = {}
    for model, part in summary.groupby("model_label", observed=True):
        part = part.sort_values("zipf_mid")
        sensitivities[str(model)] = abs(float(part.iloc[0][metric] - part.iloc[-1][metric]))
    ordered = sorted(sensitivities.items(), key=lambda item: item[1])
    return ordered[0][0], ordered[-1][0]


def _plot_acc_margin(summary: pd.DataFrame, out_base: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.4), sharex=True)
    ax_acc, ax_margin = axes
    model_means = summary.groupby("model_label", observed=True)["accuracy"].mean().sort_values(ascending=False)
    less_model = str(model_means.index[0])
    more_model = str(model_means.index[-1])
    avg = _average_line(summary, ["accuracy", "margin_lp"])
    specs = [
        ("Average", avg, "#111827", 2.6, "--"),
        (f"Less Sensitive: {less_model}", summary[summary["model_label"].eq(less_model)], "#2563A6", 2.1, "-"),
        (f"More Sensitive: {more_model}", summary[summary["model_label"].eq(more_model)], "#C2410C", 2.1, "-"),
    ]
    for label, part, color, linewidth, linestyle in specs:
        part = part.sort_values("zipf_mid", ascending=False)
        x = part["zipf_mid"].to_numpy(float)
        ax_acc.plot(x, part["accuracy"], marker="o", linewidth=linewidth, linestyle=linestyle, color=color, label=label)
        ax_acc.fill_between(x, part["accuracy_lo"], part["accuracy_hi"], color=color, alpha=0.14, linewidth=0)
        ax_margin.plot(x, part["margin_lp"], marker="o", linewidth=linewidth, linestyle=linestyle, color=color, label=label)
        ax_margin.fill_between(x, part["margin_lp_lo"], part["margin_lp_hi"], color=color, alpha=0.14, linewidth=0)

    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(loc="best")
    ax_acc.set_title("Original BLiMP Accuracy by Realized Zipf")
    ax_margin.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
    ax_margin.set_ylabel("Margin: LP(good) - LP(bad)")
    ax_margin.set_xlabel("Realized content-word Zipf")
    ax_margin.set_title("Original BLiMP LP Margin")
    _set_descending_zipf_axis(ax_margin, summary["zipf_mid"])
    _savefig(fig, out_base)


def _plot_lp(summary: pd.DataFrame, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    model_means = summary.groupby("model_label", observed=True)["good_lp_per_word"].mean().sort_values(ascending=False)
    less_model = str(model_means.index[0])
    more_model = str(model_means.index[-1])
    avg = _average_line(summary, ["good_lp_per_word"])
    specs = [
        ("Average", avg, "#111827", 2.6, "--"),
        (f"Less Sensitive: {less_model}", summary[summary["model_label"].eq(less_model)], "#2563A6", 2.1, "-"),
        (f"More Sensitive: {more_model}", summary[summary["model_label"].eq(more_model)], "#C2410C", 2.1, "-"),
    ]
    for label, part, color, linewidth, linestyle in specs:
        part = part.sort_values("zipf_mid", ascending=False)
        x = part["zipf_mid"].to_numpy(float)
        ax.plot(x, part["good_lp_per_word"], marker="o", linewidth=linewidth, linestyle=linestyle, color=color, label=label)
        if not label.startswith("Average"):
            ax.fill_between(x, part["good_lp_per_word_lo"], part["good_lp_per_word_hi"], color=color, alpha=0.16, linewidth=0)
    ax.set_ylabel("LP/word, paradigm-balanced")
    ax.set_xlabel("Realized content-word Zipf")
    ax.set_title("Original BLiMP LP/word by Realized Zipf")
    ax.legend(loc="best")
    _set_descending_zipf_axis(ax, summary["zipf_mid"])
    _savefig(fig, out_base)


def _plot_lp_accuracy(lp_summary: pd.DataFrame, acc_summary: pd.DataFrame, out_base: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.4), sharex=True)
    ax_lp, ax_acc = axes

    lp_means = lp_summary.groupby("model_label", observed=True)["good_lp_per_word"].mean().sort_values(ascending=False)
    acc_means = acc_summary.groupby("model_label", observed=True)["accuracy"].mean().sort_values(ascending=False)
    lp_less = str(lp_means.index[0])
    lp_more = str(lp_means.index[-1])
    acc_less = str(acc_means.index[0])
    acc_more = str(acc_means.index[-1])

    lp_avg = _average_line(lp_summary, ["good_lp_per_word"])
    acc_avg = _average_line(acc_summary, ["accuracy"])
    lp_specs = [
        ("Average", lp_avg, "#111827", 2.6, "--"),
        (f"Less Sensitive: {lp_less}", lp_summary[lp_summary["model_label"].eq(lp_less)], "#2563A6", 2.1, "-"),
        (f"More Sensitive: {lp_more}", lp_summary[lp_summary["model_label"].eq(lp_more)], "#C2410C", 2.1, "-"),
    ]
    acc_specs = [
        ("Average", acc_avg, "#111827", 2.6, "--"),
        (f"Less Sensitive: {acc_less}", acc_summary[acc_summary["model_label"].eq(acc_less)], "#2563A6", 2.1, "-"),
        (f"More Sensitive: {acc_more}", acc_summary[acc_summary["model_label"].eq(acc_more)], "#C2410C", 2.1, "-"),
    ]

    for label, part, color, linewidth, linestyle in lp_specs:
        part = part.sort_values("zipf_mid", ascending=False)
        x = part["zipf_mid"].to_numpy(float)
        ax_lp.plot(x, part["good_lp_per_word"], marker="o", linewidth=linewidth, linestyle=linestyle, color=color, label=label)
        if not label.startswith("Average"):
            ax_lp.fill_between(x, part["good_lp_per_word_lo"], part["good_lp_per_word_hi"], color=color, alpha=0.16, linewidth=0)
    for label, part, color, linewidth, linestyle in acc_specs:
        part = part.sort_values("zipf_mid", ascending=False)
        x = part["zipf_mid"].to_numpy(float)
        ax_acc.plot(x, part["accuracy"], marker="o", linewidth=linewidth, linestyle=linestyle, color=color, label=label)
        if not label.startswith("Average"):
            ax_acc.fill_between(x, part["accuracy_lo"], part["accuracy_hi"], color=color, alpha=0.14, linewidth=0)

    ax_lp.set_ylabel("LP(good) / word")
    ax_lp.legend(loc="best")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_xlabel("Realized content-word median Zipf")
    ax_acc.legend(loc="lower left")
    _set_descending_zipf_axis(ax_acc, acc_summary["zipf_mid"])
    _savefig(fig, out_base)


def _set_descending_zipf_axis(ax, values: pd.Series) -> None:
    x_min = float(values.min())
    x_max = float(values.max())
    left_lim = x_max + 0.25
    right_lim = x_min - 0.25
    tick_min = math.ceil(right_lim * 2.0) / 2.0
    tick_max = math.floor(left_lim * 2.0) / 2.0
    ticks = np.arange(tick_max, tick_min - 0.001, -0.5)
    ax.set_xlim(left_lim, right_lim)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:.1f}" for tick in ticks])


def _savefig(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def _spearman_rho(x: pd.Series, y: pd.Series) -> float:
    return float(x.rank(method="average").corr(y.rank(method="average"), method="pearson"))


def _write_spearman(summary: pd.DataFrame, metrics: Sequence[str], out_path: Path) -> None:
    rows = []
    for metric in metrics:
        for model, part in summary.groupby("model", observed=True):
            rows.append(
                {
                    "model": model,
                    "model_label": str(part["model_label"].iloc[0]),
                    "metric": metric,
                    "spearman_rho": _spearman_rho(part["zipf_mid"], part[metric]),
                    "n_bins": int(len(part)),
                    "n_items": int(part["n_items"].sum()),
                    "n_paradigms_mean": float(part["n_paradigms"].mean()),
                }
            )
        avg = _average_line(summary, [metric])
        rows.append(
            {
                "model": "AVERAGE",
                "model_label": "AVERAGE",
                "metric": metric,
                "spearman_rho": _spearman_rho(avg["zipf_mid"], avg[metric]),
                "n_bins": int(len(avg)),
                "n_items": int(summary["n_items"].sum()),
                "n_paradigms_mean": float(summary["n_paradigms"].mean()),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _write_effects(summary: pd.DataFrame, metrics: Sequence[str], out_path: Path) -> None:
    rows = []
    for metric in metrics:
        for model, part in summary.groupby("model", observed=True):
            rows.append(_effect_row(part.sort_values("zipf_mid"), metric, str(part["model_label"].iloc[0]), model))
        avg = _average_line(summary, [metric]).sort_values("zipf_mid")
        rows.append(_effect_row(avg, metric, "AVERAGE", "AVERAGE"))
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _effect_row(part: pd.DataFrame, metric: str, model_label: str, model: str) -> dict:
    low = part.iloc[0]
    high = part.iloc[-1]
    slope_per_zipf_increase = float(np.polyfit(part["zipf_mid"].to_numpy(float), part[metric].to_numpy(float), 1)[0])
    row = {
        "model": model,
        "model_label": model_label,
        "metric": metric,
        "lowest_zipf_mid": float(low["zipf_mid"]),
        "highest_zipf_mid": float(high["zipf_mid"]),
        "lowest_bin_value": float(low[metric]),
        "highest_bin_value": float(high[metric]),
        "change_highest_to_lowest": float(low[metric] - high[metric]),
        "slope_per_1_zipf_decrease": -slope_per_zipf_increase,
        "n_bins": int(len(part)),
    }
    if metric == "accuracy":
        row["change_highest_to_lowest_points"] = 100.0 * row["change_highest_to_lowest"]
        row["slope_points_per_1_zipf_decrease"] = 100.0 * row["slope_per_1_zipf_decrease"]
    return row


def _write_manifest(files: Sequence[ScoreFile], path: Path) -> None:
    rows = [
        {
            "path": str(meta.path),
            "model": meta.model,
            "method": meta.method,
            "paradigm": meta.paradigm,
            "source_rank": meta.source_rank,
            "mtime": meta.mtime,
        }
        for meta in files
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scores",
        nargs="*",
        default=["results/acceptability_pair_scores"],
        help="Original BLiMP acceptability JSONL files, directories, or globs.",
    )
    parser.add_argument("--accuracy-method", default="nll")
    parser.add_argument("--lp-method", default="in_template_lp")
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--bin-strategy", choices=["quantile", "equal"], default="quantile")
    parser.add_argument("--zipf-stat", choices=["median", "mean"], default="median")
    parser.add_argument("--unique-words", action="store_true")
    parser.add_argument("--min-paradigms", type=int, default=2)
    parser.add_argument("--min-items-per-paradigm-bin", type=int, default=1)
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=["Instruct", "-it"],
        help="Exclude models whose full name, label, or slug contains this token. Repeatable.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / ".mplconfig").mkdir(exist_ok=True)
    _style()

    paths = _iter_score_paths(args.scores)
    if not paths:
        raise SystemExit("No original BLiMP acceptability JSONL files found.")

    acc_files = _latest_files(paths, args.accuracy_method, args.exclude_model)
    lp_files = _latest_files(paths, args.lp_method, args.exclude_model)
    acc_points = _collect_points(acc_files, args)
    lp_points = _collect_points(lp_files, args)

    acc_summary = _balanced_binned(acc_points, args, ["accuracy", "margin_lp"])
    lp_summary = _balanced_binned(lp_points, args, ["good_lp_per_word"])

    suffix = f"{args.zipf_stat}_{args.bin_strategy}{args.bins}"
    acc_points_path = args.out_dir / f"original_blimp_accuracy_margin_points_{args.accuracy_method}_{suffix}.csv"
    acc_summary_path = args.out_dir / f"original_blimp_accuracy_margin_binned_{args.accuracy_method}_{suffix}.csv"
    acc_spearman_path = args.out_dir / f"original_blimp_accuracy_margin_spearman_{args.accuracy_method}_{suffix}.csv"
    acc_effects_path = args.out_dir / f"original_blimp_accuracy_margin_effects_{args.accuracy_method}_{suffix}.csv"
    lp_points_path = args.out_dir / f"original_blimp_lp_points_{args.lp_method}_{suffix}.csv"
    lp_summary_path = args.out_dir / f"original_blimp_lp_binned_{args.lp_method}_{suffix}.csv"
    lp_spearman_path = args.out_dir / f"original_blimp_lp_spearman_{args.lp_method}_{suffix}.csv"
    lp_effects_path = args.out_dir / f"original_blimp_lp_effects_{args.lp_method}_{suffix}.csv"

    acc_points.to_csv(acc_points_path, index=False)
    acc_summary.to_csv(acc_summary_path, index=False)
    _write_spearman(acc_summary, ["accuracy", "margin_lp"], acc_spearman_path)
    _write_effects(acc_summary, ["accuracy", "margin_lp"], acc_effects_path)
    lp_points.to_csv(lp_points_path, index=False)
    lp_summary.to_csv(lp_summary_path, index=False)
    _write_spearman(lp_summary, ["good_lp_per_word"], lp_spearman_path)
    _write_effects(lp_summary, ["good_lp_per_word"], lp_effects_path)
    _write_manifest(acc_files, args.out_dir / f"original_blimp_accuracy_margin_manifest_{args.accuracy_method}.csv")
    _write_manifest(lp_files, args.out_dir / f"original_blimp_lp_manifest_{args.lp_method}.csv")

    _plot_acc_margin(acc_summary, args.out_dir / f"original_blimp_accuracy_margin_{args.accuracy_method}_{suffix}")
    _plot_lp(lp_summary, args.out_dir / f"original_blimp_lp_{args.lp_method}_{suffix}")
    _plot_lp_accuracy(
        lp_summary,
        acc_summary,
        args.out_dir / f"original_blimp_lp_accuracy_{args.lp_method}_{args.accuracy_method}_{suffix}",
    )

    print(f"Loaded {len(acc_points):,} accuracy/margin rows from {len(acc_files):,} file(s).")
    print(f"Loaded {len(lp_points):,} LP rows from {len(lp_files):,} file(s).")
    print(f"Models: {', '.join(sorted(acc_points['model_label'].unique()))}")
    print(f"Saved accuracy/margin plot to {args.out_dir / f'original_blimp_accuracy_margin_{args.accuracy_method}_{suffix}.pdf'}")
    print(f"Saved LP plot to {args.out_dir / f'original_blimp_lp_{args.lp_method}_{suffix}.pdf'}")
    print(
        "Saved LP/accuracy plot to "
        f"{args.out_dir / f'original_blimp_lp_accuracy_{args.lp_method}_{args.accuracy_method}_{suffix}.pdf'}"
    )


if __name__ == "__main__":
    main()
