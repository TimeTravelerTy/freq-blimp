#!/usr/bin/env python3
"""Plot recent generated-regime accuracy and LP margin by realized Zipf.

Panel A: accuracy by realized content-word Zipf bin.
Panel B: margin = LP(good) - LP(bad) by the same bins.

Within each model/bin, rows are first averaged per paradigm and then those
paradigm means are averaged, so paradigms contribute equally regardless of item
count. By default, the script uses generated full-3-regime FreqBLiMP results
and keeps only the latest file per model/method/regime/paradigm.
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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "results/frequency_effects/latest_zipf_accuracy_margin/.mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordfreq import zipf_frequency

try:
    from analyze_frequency_effects import STOPWORDS, TOKEN_RE
except ImportError:
    TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
    STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }


REGIMES = ("head", "tail", "xtail")
METHODS = ("nll", "in_template_lp", "in_template_meanlp", "in_template_penlp", "yes_no")
DEFAULT_OUT_DIR = Path("results/frequency_effects/latest_zipf_accuracy_margin")


@dataclass(frozen=True)
class ScoreFile:
    path: Path
    model: str
    method: str
    regime: str
    paradigm: str
    variant: str
    dataset_path: str
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
            paths.update(p.rglob("*_acceptability.jsonl"))
        elif p.is_file():
            paths.add(p)
    return sorted(p for p in paths if p.is_file() and p.suffix == ".jsonl")


def _source_rank(path: Path) -> str:
    match = re.match(r"(?P<rank>\d{8}(?:-\d{6})?(?:-[A-Za-z0-9]+)?)", path.name)
    return match.group("rank") if match else path.name


def _method(rec: dict, path: Path) -> str:
    method = rec.get("method")
    if isinstance(method, str) and method:
        return method
    m = re.search(rf"_({'|'.join(METHODS)})_acceptability\.jsonl$", path.name)
    return m.group(1) if m else "unknown"


def _variant(rec: dict, path: Path) -> str:
    variant = rec.get("variant")
    if isinstance(variant, str) and variant:
        return variant
    if "_original_" in path.name:
        return "original"
    if "_freq_" in path.name:
        return "freq"
    return "unknown"


def _regime(rec: dict, path: Path) -> str:
    text = " ".join(
        str(part)
        for part in (
            rec.get("dataset_path", ""),
            rec.get("dataset_name", ""),
            path.name,
        )
    )
    for regime in REGIMES:
        if re.search(rf"(^|[/_-]){regime}($|[/_.-])", text):
            return regime
        if f"freq_blimp_{regime}_" in text:
            return regime
    return "unknown"


def _paradigm(rec: dict, path: Path) -> str:
    for key in ("subtask", "UID"):
        value = rec.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("dataset_name", "dataset_path"):
        value = rec.get(key)
        if isinstance(value, str) and value:
            stem = Path(value).stem
            if stem:
                return stem
    stem = path.name
    for method in METHODS:
        suffix = f"_freq_{method}_acceptability.jsonl"
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.rsplit("_", 1)[-1] if stem else "unknown"


def _first_record(path: Path) -> Optional[dict]:
    for rec in _iter_jsonl(path):
        return rec
    return None


def _score_file(path: Path) -> Optional[ScoreFile]:
    rec = _first_record(path)
    if rec is None:
        return None
    model = rec.get("model")
    if not isinstance(model, str) or not model:
        model = "unknown"
    dataset_path = rec.get("dataset_path")
    if not isinstance(dataset_path, str):
        dataset_path = ""
    return ScoreFile(
        path=path,
        model=model,
        method=_method(rec, path),
        regime=_regime(rec, path),
        paradigm=_paradigm(rec, path),
        variant=_variant(rec, path),
        dataset_path=dataset_path,
        source_rank=_source_rank(path),
        mtime=path.stat().st_mtime,
    )


def _latest_score_files(paths: Sequence[Path], args: argparse.Namespace) -> List[ScoreFile]:
    cutoff = None
    if args.latest_days > 0:
        cutoff = (datetime.now() - timedelta(days=args.latest_days)).timestamp()

    files: List[ScoreFile] = []
    for path in paths:
        if cutoff is not None and path.stat().st_mtime < cutoff:
            continue
        meta = _score_file(path)
        if meta is None:
            continue
        if meta.variant != "freq":
            continue
        if meta.regime not in REGIMES:
            continue
        if args.method != "any" and meta.method != args.method:
            continue
        if not args.no_dataset_filter and args.dataset_contains:
            haystack = f"{meta.dataset_path} {meta.path}"
            if not any(token in haystack for token in args.dataset_contains):
                continue
        files.append(meta)

    if not args.latest_per_key:
        return sorted(files, key=lambda f: (f.model, f.regime, f.paradigm, f.method, f.path.name))

    latest: Dict[Tuple[str, str, str, str], ScoreFile] = {}
    for meta in files:
        key = (meta.model, meta.method, meta.regime, meta.paradigm)
        previous = latest.get(key)
        if previous is None or (meta.mtime, meta.source_rank, meta.path.name) > (
            previous.mtime,
            previous.source_rank,
            previous.path.name,
        ):
            latest[key] = meta
    return sorted(latest.values(), key=lambda f: (f.model, f.regime, f.paradigm, f.method, f.path.name))


def _latest_original_files(paths: Sequence[Path], method: str) -> List[ScoreFile]:
    candidates: List[ScoreFile] = []
    for path in paths:
        meta = _score_file(path)
        if meta is None:
            continue
        if meta.variant != "original":
            continue
        if method != "any" and meta.method != method:
            continue
        haystack = f"{meta.dataset_path} {meta.path}"
        if "blimp_original" not in haystack:
            continue
        candidates.append(meta)

    latest: Dict[Tuple[str, str], ScoreFile] = {}
    for meta in candidates:
        key = (meta.model, meta.method)
        previous = latest.get(key)
        if previous is None or (meta.mtime, meta.source_rank, meta.path.name) > (
            previous.mtime,
            previous.source_rank,
            previous.path.name,
        ):
            latest[key] = meta
    return sorted(latest.values(), key=lambda f: (f.model, f.method, f.path.name))


def _model_label(model: str) -> str:
    return model.split("/")[-1].replace("_", ".")


def _model_slug(model: str) -> str:
    return model.split("/")[-1].replace(".", "_").replace("/", "_")


def _model_matches_exclude(model: str, patterns: Sequence[str]) -> bool:
    model_low = model.lower()
    label_low = _model_label(model).lower()
    slug_low = _model_slug(model).lower()
    return any(
        pattern.lower() in model_low or pattern.lower() in label_low or pattern.lower() in slug_low
        for pattern in patterns
    )


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


def _words(text: str) -> List[str]:
    words: List[str] = []
    for raw in TOKEN_RE.findall(text):
        token = raw.lower()
        if token.endswith("'s"):
            token = token[:-2]
        if token in STOPWORDS or len(token) <= 1:
            continue
        words.append(token)
    return words


def _zipf_stat(texts: Sequence[str], stat: str, unique_words: bool) -> Optional[float]:
    words: List[str] = []
    for text in texts:
        words.extend(_words(text))
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


def _spearman_rho(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    xr = pd.Series(x, dtype="float64").rank(method="average")
    yr = pd.Series(y, dtype="float64").rank(method="average")
    return float(xr.corr(yr, method="pearson"))


def _write_binned_spearman(summary: pd.DataFrame, out_path: Path) -> None:
    rows: List[dict] = []
    metrics = (("accuracy", "accuracy"), ("margin_lp", "margin_lp"))
    for value_col, metric in metrics:
        for model, part in summary.groupby("model", observed=True):
            rows.append(
                {
                    "model": model,
                    "model_label": str(part["model_label"].iloc[0]),
                    "metric": metric,
                    "spearman_rho": _spearman_rho(part["zipf_mid"], part[value_col]),
                    "n_bins": int(len(part)),
                    "n_items": int(part["n_items"].sum()),
                    "n_paradigms_mean": float(part["n_paradigms"].mean()),
                }
            )
        rows.append(
            {
                "model": "ALL_BINS_POOLED",
                "model_label": "ALL_BINS_POOLED",
                "metric": metric,
                "spearman_rho": _spearman_rho(summary["zipf_mid"], summary[value_col]),
                "n_bins": int(len(summary)),
                "n_items": int(summary["n_items"].sum()),
                "n_paradigms_mean": float(summary["n_paradigms"].mean()),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _write_binned_effects(summary: pd.DataFrame, out_path: Path) -> None:
    rows: List[dict] = []
    metrics = (("accuracy", "accuracy"), ("margin_lp", "margin_lp"))
    for value_col, metric in metrics:
        for model, part in summary.groupby("model", observed=True):
            part = part.sort_values("zipf_mid")
            low = part.iloc[0]
            high = part.iloc[-1]
            slope_per_zipf_increase = float(np.polyfit(part["zipf_mid"].to_numpy(float), part[value_col].to_numpy(float), 1)[0])
            row = {
                "model": model,
                "model_label": str(part["model_label"].iloc[0]),
                "metric": metric,
                "lowest_zipf_mid": float(low["zipf_mid"]),
                "highest_zipf_mid": float(high["zipf_mid"]),
                "lowest_bin_value": float(low[value_col]),
                "highest_bin_value": float(high[value_col]),
                "change_highest_to_lowest": float(low[value_col] - high[value_col]),
                "slope_per_1_zipf_decrease": -slope_per_zipf_increase,
                "n_bins": int(len(part)),
                "n_items": int(part["n_items"].sum()),
                "n_paradigms_mean": float(part["n_paradigms"].mean()),
            }
            if metric == "accuracy":
                row["change_highest_to_lowest_points"] = 100.0 * row["change_highest_to_lowest"]
                row["slope_points_per_1_zipf_decrease"] = 100.0 * row["slope_per_1_zipf_decrease"]
            rows.append(row)

        avg = (
            summary.groupby("zipf_mid", observed=True)
            .agg(value=(value_col, "mean"), n_items=("n_items", "sum"), n_paradigms=("n_paradigms", "mean"))
            .reset_index()
            .sort_values("zipf_mid")
        )
        low = avg.iloc[0]
        high = avg.iloc[-1]
        slope_per_zipf_increase = float(np.polyfit(avg["zipf_mid"].to_numpy(float), avg["value"].to_numpy(float), 1)[0])
        row = {
            "model": "AVERAGE",
            "model_label": "AVERAGE",
            "metric": metric,
            "lowest_zipf_mid": float(low["zipf_mid"]),
            "highest_zipf_mid": float(high["zipf_mid"]),
            "lowest_bin_value": float(low["value"]),
            "highest_bin_value": float(high["value"]),
            "change_highest_to_lowest": float(low["value"] - high["value"]),
            "slope_per_1_zipf_decrease": -slope_per_zipf_increase,
            "n_bins": int(len(avg)),
            "n_items": int(avg["n_items"].sum()),
            "n_paradigms_mean": float(avg["n_paradigms"].mean()),
        }
        if metric == "accuracy":
            row["change_highest_to_lowest_points"] = 100.0 * row["change_highest_to_lowest"]
            row["slope_points_per_1_zipf_decrease"] = 100.0 * row["slope_per_1_zipf_decrease"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _collect_points(files: Sequence[ScoreFile], args: argparse.Namespace) -> pd.DataFrame:
    rows: List[dict] = []
    zipf_cache: Dict[Tuple[Tuple[str, ...], str, bool], Optional[float]] = {}
    for meta in files:
        for row_i, rec in enumerate(_iter_jsonl(meta.path)):
            if rec.get("variant") not in (None, "freq"):
                continue
            good_text = _text(rec, "good_field", "good_text")
            bad_text = _text(rec, "bad_field", "bad_text")
            good_lp = _logprob(rec, "good")
            bad_lp = _logprob(rec, "bad")
            correctness = rec.get("correctness")
            if (
                not isinstance(good_text, str)
                or not isinstance(bad_text, str)
                or good_lp is None
                or bad_lp is None
                or not isinstance(correctness, (int, bool))
            ):
                continue
            if args.zipf_source == "good_content":
                zipf_texts = (good_text,)
            elif args.zipf_source == "pair_content":
                zipf_texts = (good_text, bad_text)
            else:
                raise ValueError(f"Unknown Zipf source: {args.zipf_source}")
            cache_key = (zipf_texts, args.zipf_stat, bool(args.unique_words))
            if cache_key not in zipf_cache:
                zipf_cache[cache_key] = _zipf_stat(zipf_texts, args.zipf_stat, args.unique_words)
            zipf_value = zipf_cache[cache_key]
            if zipf_value is None:
                continue
            rows.append(
                {
                    "model": meta.model,
                    "model_slug": _model_slug(meta.model),
                    "method": meta.method,
                    "regime": meta.regime,
                    "paradigm": meta.paradigm,
                    "source_file": str(meta.path),
                    "source_rank": meta.source_rank,
                    "row": row_i,
                    "correct": int(correctness),
                    "margin_lp": float(good_lp) - float(bad_lp),
                    "zipf": float(zipf_value),
                    "good_text": good_text,
                    "bad_text": bad_text,
                }
            )
    if not rows:
        raise SystemExit("No usable generated-regime score rows found after filtering.")
    return pd.DataFrame(rows)


def _collect_behavior_points(files: Sequence[ScoreFile], allowed_variants: Sequence[str]) -> pd.DataFrame:
    allowed = set(allowed_variants)
    rows: List[dict] = []
    for meta in files:
        for row_i, rec in enumerate(_iter_jsonl(meta.path)):
            if rec.get("variant") not in (None, meta.variant) or meta.variant not in allowed:
                continue
            good_lp = _logprob(rec, "good")
            bad_lp = _logprob(rec, "bad")
            correctness = rec.get("correctness")
            if good_lp is None or bad_lp is None or not isinstance(correctness, (int, bool)):
                continue
            rows.append(
                {
                    "model": meta.model,
                    "model_slug": _model_slug(meta.model),
                    "model_label": _model_label(meta.model),
                    "method": meta.method,
                    "variant": meta.variant,
                    "regime": meta.regime,
                    "paradigm": _paradigm(rec, meta.path),
                    "source_file": str(meta.path),
                    "row": row_i,
                    "correct": int(correctness),
                    "margin_lp": float(good_lp) - float(bad_lp),
                }
            )
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


def _make_bins(points: pd.DataFrame, bins: int, strategy: str) -> Tuple[pd.DataFrame, List[str]]:
    work = points.dropna(subset=["zipf"]).copy()
    if strategy == "quantile":
        work["zipf_bin"] = pd.qcut(work["zipf"], q=bins, duplicates="drop")
    elif strategy == "equal":
        work["zipf_bin"] = pd.cut(work["zipf"], bins=bins)
    else:
        raise ValueError(f"Unknown bin strategy: {strategy}")
    labels = [str(interval) for interval in work["zipf_bin"].cat.categories]
    return work, labels


def _balanced_binned(points: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    binned, _ = _make_bins(points, args.bins, args.bin_strategy)
    paradigm = (
        binned.groupby(["model", "model_slug", "zipf_bin", "paradigm"], observed=True)
        .agg(
            accuracy=("correct", "mean"),
            margin_lp=("margin_lp", "mean"),
            n_items=("correct", "size"),
            zipf_mid=("zipf", "mean"),
        )
        .reset_index()
    )
    if args.min_items_per_paradigm_bin > 1:
        paradigm = paradigm[paradigm["n_items"] >= args.min_items_per_paradigm_bin].copy()

    rows: List[dict] = []
    for (model, model_slug, bin_label), part in paradigm.groupby(["model", "model_slug", "zipf_bin"], observed=True):
        if len(part) < args.min_paradigms:
            continue
        acc = part["accuracy"].to_numpy(dtype=float)
        margin = part["margin_lp"].to_numpy(dtype=float)
        acc_lo, acc_hi = _mean_ci(acc)
        margin_lo, margin_hi = _mean_ci(margin)
        rows.append(
            {
                "model": model,
                "model_slug": model_slug,
                "model_label": _model_label(model),
                "zipf_bin": str(bin_label),
                "zipf_mid": float(bin_label.mid),
                "zipf_left": float(bin_label.left),
                "zipf_right": float(bin_label.right),
                "n_paradigms": int(len(part)),
                "n_items": int(part["n_items"].sum()),
                "accuracy": float(np.mean(acc)),
                "accuracy_lo": acc_lo,
                "accuracy_hi": acc_hi,
                "margin_lp": float(np.mean(margin)),
                "margin_lp_lo": margin_lo,
                "margin_lp_hi": margin_hi,
            }
        )
    return pd.DataFrame(rows).sort_values(["model_label", "zipf_mid"])


def _plot(summary: pd.DataFrame, out_png: Path, out_pdf: Optional[Path], title: str) -> None:
    if summary.empty:
        raise SystemExit("No binned rows to plot; reduce --bins, --min-paradigms, or --min-items-per-paradigm-bin.")

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.4), sharex=True)
    ax_acc, ax_margin = axes
    model_means = summary.groupby("model_label", observed=True)["accuracy"].mean().sort_values(ascending=False)
    less_model = str(model_means.index[0])
    more_model = str(model_means.index[-1])

    avg_rows: List[dict] = []
    for zipf_mid, part in summary.groupby("zipf_mid", observed=True):
        acc = part["accuracy"].to_numpy(dtype=float)
        margin = part["margin_lp"].to_numpy(dtype=float)
        acc_lo, acc_hi = _mean_ci(acc)
        margin_lo, margin_hi = _mean_ci(margin)
        avg_rows.append(
            {
                "zipf_mid": float(zipf_mid),
                "accuracy": float(np.mean(acc)),
                "accuracy_lo": acc_lo,
                "accuracy_hi": acc_hi,
                "margin_lp": float(np.mean(margin)),
                "margin_lp_lo": margin_lo,
                "margin_lp_hi": margin_hi,
            }
        )
    avg = pd.DataFrame(avg_rows).sort_values("zipf_mid", ascending=False)

    plot_specs = [
        ("Average", avg, "#111827", 2.6, "--"),
        (f"Less Sensitive: {less_model}", summary[summary["model_label"].eq(less_model)], "#2563A6", 2.1, "-"),
        (f"More Sensitive: {more_model}", summary[summary["model_label"].eq(more_model)], "#C2410C", 2.1, "-"),
    ]

    for label, part, color, linewidth, linestyle in plot_specs:
        part = part.sort_values("zipf_mid", ascending=False)
        x = part["zipf_mid"].to_numpy(dtype=float)
        ax_acc.plot(x, part["accuracy"], marker="o", linewidth=linewidth, linestyle=linestyle, color=color, label=label)
        ax_acc.fill_between(x, part["accuracy_lo"], part["accuracy_hi"], color=color, alpha=0.14, linewidth=0)
        ax_margin.plot(x, part["margin_lp"], marker="o", linewidth=linewidth, linestyle=linestyle, color=color, label=label)
        ax_margin.fill_between(x, part["margin_lp_lo"], part["margin_lp_hi"], color=color, alpha=0.14, linewidth=0)

    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0.6, 0.85)
    ax_acc.set_title(title)
    ax_acc.legend(loc="best")

    ax_margin.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
    ax_margin.set_ylabel("Margin: LP(good) - LP(bad)")
    ax_margin.set_xlabel("Realized content-word Zipf")
    ax_margin.set_title("Balanced LP Margin")

    x_min = float(summary["zipf_mid"].min())
    x_max = float(summary["zipf_mid"].max())
    left_lim = x_max + 0.25
    right_lim = x_min - 0.25
    tick_min = math.ceil(right_lim * 2.0) / 2.0
    tick_max = math.floor(left_lim * 2.0) / 2.0
    ticks = np.arange(tick_max, tick_min - 0.001, -0.5)
    ax_margin.set_xlim(left_lim, right_lim)
    ax_margin.set_xticks(ticks)
    ax_margin.set_xticklabels([f"{tick:.1f}" for tick in ticks])

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def _balanced_metric(values: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    per_paradigm = (
        values.groupby([*group_cols, "paradigm"], observed=True)
        .agg(
            accuracy=("correct", "mean"),
            margin_lp=("margin_lp", "mean"),
            n_items=("correct", "size"),
        )
        .reset_index()
    )
    rows: List[dict] = []
    for key, part in per_paradigm.groupby(list(group_cols), observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        acc = part["accuracy"].to_numpy(dtype=float)
        margin = part["margin_lp"].to_numpy(dtype=float)
        acc_lo, acc_hi = _mean_ci(acc)
        margin_lo, margin_hi = _mean_ci(margin)
        row = {col: value for col, value in zip(group_cols, key)}
        row.update(
            {
                "accuracy": float(np.mean(acc)),
                "accuracy_lo": acc_lo,
                "accuracy_hi": acc_hi,
                "margin_lp": float(np.mean(margin)),
                "margin_lp_lo": margin_lo,
                "margin_lp_hi": margin_hi,
                "n_paradigms": int(len(part)),
                "n_items": int(part["n_items"].sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _make_regime_table(points: pd.DataFrame, original: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    generated_summary = _balanced_metric(points, ["model", "model_label", "regime"])
    original_summary = _balanced_metric(original, ["model", "model_label"]) if not original.empty else pd.DataFrame()

    rows: List[dict] = []
    for model, model_label in sorted(points[["model", "model_label"]].drop_duplicates().itertuples(index=False)):
        row = {"model": model_label}
        orig = original_summary[original_summary["model"].eq(model)] if not original_summary.empty else pd.DataFrame()
        if orig.empty:
            for col in ("original_acc", "original_acc_lo", "original_acc_hi", "original_margin", "original_n_paradigms", "original_n_items"):
                row[col] = math.nan
        else:
            o = orig.iloc[0]
            row.update(
                {
                    "original_acc": float(o["accuracy"]),
                    "original_acc_lo": float(o["accuracy_lo"]),
                    "original_acc_hi": float(o["accuracy_hi"]),
                    "original_margin": float(o["margin_lp"]),
                    "original_n_paradigms": int(o["n_paradigms"]),
                    "original_n_items": int(o["n_items"]),
                }
            )
        by_regime = {}
        for regime in REGIMES:
            part = generated_summary[generated_summary["model"].eq(model) & generated_summary["regime"].eq(regime)]
            if part.empty:
                by_regime[regime] = None
                row[f"{regime}_acc"] = math.nan
                row[f"{regime}_acc_lo"] = math.nan
                row[f"{regime}_acc_hi"] = math.nan
                row[f"{regime}_margin"] = math.nan
                row[f"{regime}_margin_lo"] = math.nan
                row[f"{regime}_margin_hi"] = math.nan
                row[f"{regime}_n_paradigms"] = 0
                row[f"{regime}_n_items"] = 0
                continue
            r = part.iloc[0]
            by_regime[regime] = r
            row[f"{regime}_acc"] = float(r["accuracy"])
            row[f"{regime}_acc_lo"] = float(r["accuracy_lo"])
            row[f"{regime}_acc_hi"] = float(r["accuracy_hi"])
            row[f"{regime}_margin"] = float(r["margin_lp"])
            row[f"{regime}_margin_lo"] = float(r["margin_lp_lo"])
            row[f"{regime}_margin_hi"] = float(r["margin_lp_hi"])
            row[f"{regime}_n_paradigms"] = int(r["n_paradigms"])
            row[f"{regime}_n_items"] = int(r["n_items"])
        if by_regime.get("head") is not None and by_regime.get("xtail") is not None:
            row["head_to_xtail_delta_acc"] = row["xtail_acc"] - row["head_acc"]
            row["head_to_xtail_delta_margin"] = row["xtail_margin"] - row["head_margin"]
        else:
            row["head_to_xtail_delta_acc"] = math.nan
            row["head_to_xtail_delta_margin"] = math.nan
        rows.append(row)

    table = pd.DataFrame(rows)
    requested_cols = [
        "model",
        "original_acc",
        "head_acc",
        "tail_acc",
        "xtail_acc",
        "head_to_xtail_delta_acc",
        "head_to_xtail_delta_margin",
    ]
    extras = [col for col in table.columns if col not in requested_cols]
    return table[requested_cols + extras], generated_summary


def _write_manifest(files: Sequence[ScoreFile], path: Path) -> None:
    rows = [
        {
            "path": str(meta.path),
            "model": meta.model,
            "method": meta.method,
            "regime": meta.regime,
            "paradigm": meta.paradigm,
            "variant": meta.variant,
            "dataset_path": meta.dataset_path,
            "source_rank": meta.source_rank,
            "mtime": datetime.fromtimestamp(meta.mtime).isoformat(timespec="seconds"),
        }
        for meta in files
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
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
        help="Raw acceptability JSONL files, directories, or glob patterns.",
    )
    parser.add_argument("--method", default="nll", help="Scoring method to plot; use 'any' to include all methods.")
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=["Instruct", "-it"],
        help=(
            "Exclude models whose full name, label, or slug contains this token. "
            "Repeatable. Defaults to excluding instruct/IT models."
        ),
    )
    parser.add_argument(
        "--latest-days",
        type=float,
        default=0.0,
        help="Keep files modified within this many days; use 0 to disable the freshness filter.",
    )
    parser.add_argument(
        "--dataset-contains",
        action="append",
        default=None,
        help=(
            "Keep files whose dataset path or filename contains this token. "
            "Repeatable. Default keeps latest full-3-regime generated runs."
        ),
    )
    parser.add_argument(
        "--no-dataset-filter",
        action="store_true",
        help="Disable the default dataset substring filter.",
    )
    parser.add_argument(
        "--no-latest-per-key",
        dest="latest_per_key",
        action="store_false",
        help="Do not collapse to the latest file per model/method/regime/paradigm.",
    )
    parser.set_defaults(latest_per_key=True)
    parser.add_argument("--bins", type=int, default=12, help="Number of Zipf bins.")
    parser.add_argument("--bin-strategy", choices=["quantile", "equal"], default="quantile")
    parser.add_argument("--zipf-source", choices=["good_content", "pair_content"], default="good_content")
    parser.add_argument("--zipf-stat", choices=["median", "mean"], default="median")
    parser.add_argument("--unique-words", action="store_true", help="Count each content word once per item.")
    parser.add_argument("--min-paradigms", type=int, default=2)
    parser.add_argument("--min-items-per-paradigm-bin", type=int, default=1)
    parser.add_argument(
        "--original-scores",
        nargs="*",
        default=["results/acceptability_pair_scores"],
        help="Original BLiMP acceptability JSONL files, directories, or globs for the summary table.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    if args.dataset_contains is None:
        args.dataset_contains = ["full_3regimes"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / ".mplconfig").mkdir(exist_ok=True)
    _style()

    paths = _iter_score_paths(args.scores)
    if not paths:
        raise SystemExit("No raw acceptability JSONL files found.")
    files = _latest_score_files(paths, args)
    if args.exclude_model:
        files = [meta for meta in files if not _model_matches_exclude(meta.model, args.exclude_model)]
    if not files:
        raise SystemExit(
            "No recent generated-regime score files matched. "
            "Try --latest-days 0 or check --method."
        )

    points = _collect_points(files, args)
    summary = _balanced_binned(points, args)
    behavior_points = _collect_behavior_points(files, allowed_variants=["freq"])
    original_files = _latest_original_files(_iter_score_paths(args.original_scores), args.method)
    if args.exclude_model:
        original_files = [meta for meta in original_files if not _model_matches_exclude(meta.model, args.exclude_model)]
    original_points = _collect_behavior_points(original_files, allowed_variants=["original"]) if original_files else pd.DataFrame()
    regime_table, regime_detail = _make_regime_table(behavior_points, original_points)

    suffix = f"{args.method}_{args.zipf_source}_{args.zipf_stat}_{args.bin_strategy}{args.bins}"
    points_path = args.out_dir / f"latest_zipf_accuracy_margin_points_{suffix}.csv"
    summary_path = args.out_dir / f"latest_zipf_accuracy_margin_binned_{suffix}.csv"
    manifest_path = args.out_dir / f"latest_zipf_accuracy_margin_manifest_{suffix}.csv"
    original_manifest_path = args.out_dir / f"latest_zipf_accuracy_margin_original_manifest_{suffix}.csv"
    table_path = args.out_dir / f"latest_zipf_accuracy_margin_regime_table_{suffix}.csv"
    regime_detail_path = args.out_dir / f"latest_zipf_accuracy_margin_regime_detail_{suffix}.csv"
    spearman_path = args.out_dir / f"latest_zipf_accuracy_margin_binned_spearman_{suffix}.csv"
    effects_path = args.out_dir / f"latest_zipf_accuracy_margin_binned_effects_{suffix}.csv"
    plot_path = args.out_dir / f"latest_zipf_accuracy_margin_{suffix}.png"
    pdf_path = args.out_dir / f"latest_zipf_accuracy_margin_{suffix}.pdf"

    points.to_csv(points_path, index=False)
    summary.to_csv(summary_path, index=False)
    regime_table.to_csv(table_path, index=False)
    regime_detail.to_csv(regime_detail_path, index=False)
    _write_binned_spearman(summary, spearman_path)
    _write_binned_effects(summary, effects_path)
    _write_manifest(files, manifest_path)
    if original_files:
        _write_manifest(original_files, original_manifest_path)
    _plot(
        summary,
        out_png=plot_path,
        out_pdf=pdf_path,
        title="Balanced Accuracy by Realized Zipf",
    )

    print(f"Loaded {len(points):,} rows from {len(files):,} selected generated-regime file(s).")
    if original_files:
        print(f"Loaded {len(original_points):,} original rows from {len(original_files):,} original file(s).")
    print(f"Models: {', '.join(sorted({_model_label(meta.model) for meta in files}))}")
    print(f"Regimes: {', '.join(REGIMES)}")
    print(f"Saved points to {points_path}")
    print(f"Saved balanced binned means to {summary_path}")
    print(f"Saved regime table to {table_path}")
    print(f"Saved regime detail to {regime_detail_path}")
    print(f"Saved binned Spearman rhos to {spearman_path}")
    print(f"Saved binned effects to {effects_path}")
    print(f"Saved source manifest to {manifest_path}")
    if original_files:
        print(f"Saved original source manifest to {original_manifest_path}")
    print(f"Saved plot to {plot_path}")
    print(f"Saved PDF to {pdf_path}")


if __name__ == "__main__":
    main()
