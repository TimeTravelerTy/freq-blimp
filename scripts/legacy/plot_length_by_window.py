"""
Plot token length distributions (good vs bad) for windowed datasets, and report
length/Zipf imbalances per dataset.

Default windows (original):
- 4.0-5.2 (head)
- 2.2-3.0 (tail)
- 1.2-2.0 (xtail)
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Matplotlib config to avoid ~/.matplotlib permission issues.
MPL_DIR = Path("results/analysis_plots/.mplconfig")
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAIR_SCORES_DIR = Path("results/blimp_pair_scores")
DATA_DIR = Path("data/processed")
PLOT_PATH = Path("results/analysis_plots/length_by_window.png")
STATS_PATH = Path("results/analysis_plots/length_by_window_stats.csv")

_PAIR_RE = re.compile(
    r"(?:^|/)\d{8}-\d{6}_(?P<model>.+?)_(?P<dataset>\d{8}-\d{6}_.+?)_pair-scores\.jsonl$"
)
_WINDOW_RE = re.compile(r"zipf(?P<z>\d+_(?:\d+-\d+_\d+))")


def _parse_pair_scores_path(path: Path) -> Tuple[str, str, str]:
    m = _PAIR_RE.search(str(path))
    if not m:
        raise ValueError(f"Unparseable pair-scores filename: {path}")
    dataset_full = m.group("dataset")
    variant_hint = "original" if dataset_full.endswith("_original") else "rare"
    dataset_base = re.sub(r"_(original|rare)$", "", dataset_full)
    return m.group("model"), dataset_base, variant_hint


def _window_key(dataset_base: str) -> Optional[str]:
    m = _WINDOW_RE.search(dataset_base)
    if not m:
        return None
    return m.group("z").replace("_", ".").replace("-", "–")


def _window_label_for_variant(dataset_base: str, variant: str) -> Optional[str]:
    if variant == "original":
        return "original"
    window = _window_key(dataset_base)
    if window is None:
        return None
    if "4.0–5.2" in window:
        return "4.0–5.2 (head)"
    if "2.2–3.0" in window:
        return "2.2–3.0 (tail)"
    if "1.2–2.0" in window:
        return "1.2–2.0 (xtail)"
    return None


def _load_dataset_meta(dataset_base: str, cache: Dict[str, Dict[int, dict]]) -> Dict[int, dict]:
    if dataset_base not in cache:
        ds_path = DATA_DIR / f"{dataset_base}.jsonl"
        if not ds_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {ds_path}")
        meta_by_idx: Dict[int, dict] = {}
        with ds_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                idx = rec.get("idx")
                if isinstance(idx, int):
                    meta_by_idx[idx] = rec.get("meta") or {}
        cache[dataset_base] = meta_by_idx
    return cache[dataset_base]


def _median_zipf_for_field(meta: dict, field: str) -> Optional[float]:
    aggs = (meta or {}).get("zipf_swapped_position_aggregates") or {}
    return (aggs.get(field) or {}).get("median")


def _collect_rows(
    paths: Iterable[Path],
    *,
    window_labels: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset_cache: Dict[str, Dict[int, dict]] = {}
    rows: List[dict] = []
    stats: List[dict] = []

    for path in paths:
        model, dataset_base, variant_hint = _parse_pair_scores_path(path)
        window_label = window_labels.get(f"{variant_hint}:{dataset_base}")
        if not window_label:
            continue

        meta_by_idx = _load_dataset_meta(dataset_base, dataset_cache)
        for line in path.read_text().splitlines():
            if not line:
                continue
            rec = json.loads(line)
            idx = rec.get("idx")
            meta = meta_by_idx.get(idx, {}) if isinstance(idx, int) else {}
            good_len = rec.get("good_token_count")
            bad_len = rec.get("bad_token_count")
            if good_len is None or bad_len is None:
                continue
            good_zipf = _median_zipf_for_field(meta, rec.get("good_field"))
            bad_zipf = _median_zipf_for_field(meta, rec.get("bad_field"))
            rows.append(
                {
                    "model": model,
                    "dataset": dataset_base,
                    "variant": variant_hint,
                    "window": window_label,
                    "good_len": int(good_len),
                    "bad_len": int(bad_len),
                    "len_diff": int(good_len) - int(bad_len),
                    "good_zipf": good_zipf,
                    "bad_zipf": bad_zipf,
                    "zipf_diff": (good_zipf - bad_zipf) if good_zipf is not None and bad_zipf is not None else None,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No rows collected; check model/variant/windows filters.")

    for (variant, dataset, window), df_sub in df.groupby(["variant", "dataset", "window"], observed=True):
        stats.append(
            {
                "variant": variant,
                "dataset": dataset,
                "window": window,
                "n": len(df_sub),
                "mean_good_len": float(df_sub["good_len"].mean()),
                "mean_bad_len": float(df_sub["bad_len"].mean()),
                "median_good_len": float(df_sub["good_len"].median()),
                "median_bad_len": float(df_sub["bad_len"].median()),
                "mean_len_diff": float(df_sub["len_diff"].mean()),
                "median_len_diff": float(df_sub["len_diff"].median()),
                "mean_zipf_diff": float(df_sub["zipf_diff"].mean()),
                "median_zipf_diff": float(df_sub["zipf_diff"].median()),
            }
        )
    stats_df = pd.DataFrame(stats).sort_values("window")
    return df, stats_df


def _plot_violin(df: pd.DataFrame, out_path: Path) -> None:
    order = ["original", "4.0–5.2 (head)", "2.2–3.0 (tail)", "1.2–2.0 (xtail)"]
    windows = [w for w in order if w in set(df["window"])]
    data_good = [df[df["window"] == w]["good_len"].to_numpy() for w in windows]
    data_bad = [df[df["window"] == w]["bad_len"].to_numpy() for w in windows]

    positions_good = np.arange(len(windows)) * 3.0
    positions_bad = positions_good + 0.9

    fig, ax = plt.subplots(figsize=(9, 5))
    parts_good = ax.violinplot(data_good, positions=positions_good, widths=0.8, showmeans=False, showmedians=True)
    parts_bad = ax.violinplot(data_bad, positions=positions_bad, widths=0.8, showmeans=False, showmedians=True)

    for pc in parts_good["bodies"]:
        pc.set_facecolor("#4C72B0")
        pc.set_alpha(0.6)
    for pc in parts_bad["bodies"]:
        pc.set_facecolor("#DD8452")
        pc.set_alpha(0.6)

    ax.set_xticks(positions_good + 0.45)
    ax.set_xticklabels(windows)
    ax.set_ylabel("Token length")
    ax.set_title("Good vs bad token lengths by Zipf window")
    ax.legend(
        [parts_good["bodies"][0], parts_bad["bodies"][0]],
        ["good", "bad"],
        loc="upper right",
    )
    ax.grid(True, axis="y", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot good/bad token lengths for windowed datasets.")
    ap.add_argument("--model", default="Llama-3_1-8B", help="Model name substring to match.")
    ap.add_argument("--variant", choices=["original", "rare", "both"], default="both", help="Dataset variant to include.")
    ap.add_argument("--out", default=str(PLOT_PATH), help="Output PNG path.")
    ap.add_argument("--stats-out", default=str(STATS_PATH), help="Output CSV stats path.")
    args = ap.parse_args()

    window_labels: Dict[str, str] = {}

    paths = []
    for path in PAIR_SCORES_DIR.glob("*.jsonl"):
        try:
            model, dataset_base, variant_hint = _parse_pair_scores_path(path)
        except ValueError:
            continue
        if args.model not in model:
            continue
        if args.variant != "both" and variant_hint != args.variant:
            continue
        label = _window_label_for_variant(dataset_base, variant_hint)
        if not label:
            continue
        window_labels[f"{variant_hint}:{dataset_base}"] = label
        paths.append(path)

    if not paths:
        raise SystemExit("No pair-score files matched model/variant/window filters.")

    df, stats_df = _collect_rows(paths, window_labels=window_labels)
    _plot_violin(df, Path(args.out))

    stats_path = Path(args.stats_out)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(stats_path, index=False)
    print(stats_df.to_string(index=False))
    print(f"Saved plot to {args.out}")
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
