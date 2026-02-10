"""
Plot average token length vs realised Zipf rarity.

Uses BLiMP pair-score JSONL files and bins swapped_median_zipf into quantiles.
"""

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_MPL_DIR = Path("results/analysis_plots/.mplconfig")
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAIR_SCORES_DIR = Path("results/blimp_pair_scores")
DATA_DIR = Path("data/processed")
PLOT_PATH = Path("results/analysis_plots/zipf_vs_token_len.png")

_PAIR_RE = re.compile(
    r"(?:^|/)\d{8}-\d{6}_(?P<model>.+?)_(?P<dataset>\d{8}-\d{6}_.+?)_pair-scores\.jsonl$"
)


def _ensure_mpl_config(out_dir: Path) -> None:
    cfg = out_dir / ".mplconfig"
    cfg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cfg))


def _parse_pair_scores_path(path: Path) -> Tuple[str, str, str]:
    m = _PAIR_RE.search(str(path))
    if not m:
        raise ValueError(f"Unparseable pair-scores filename: {path}")
    dataset_full = m.group("dataset")
    variant_hint = "freq"
    if dataset_full.endswith("_original"):
        variant_hint = "original"
    elif dataset_full.endswith("_freq") or dataset_full.endswith("_rare"):
        variant_hint = "freq"
    dataset_base = re.sub(r"_(original|freq|rare)$", "", dataset_full)
    return m.group("model"), dataset_base, variant_hint


def _load_dataset_meta(dataset_base: str, cache: Dict[str, Dict[str, dict]]) -> Dict[str, dict]:
    if dataset_base not in cache:
        ds_path = DATA_DIR / f"{dataset_base}.jsonl"
        if not ds_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {ds_path}")
        meta_by_idx: Dict[str, dict] = {}
        with ds_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                idx = rec.get("idx")
                if idx is None:
                    continue
                meta_by_idx[idx] = rec.get("meta") or {}
        cache[dataset_base] = meta_by_idx
    return cache[dataset_base]


def _median_zipf_for_field(meta: dict, field: str) -> Optional[float]:
    aggs = (meta or {}).get("zipf_swapped_position_aggregates") or {}
    return (aggs.get(field) or {}).get("median")


def _swapped_median_zipf(meta: dict, good_field: str, bad_field: str) -> Optional[float]:
    vals = [
        _median_zipf_for_field(meta, good_field),
        _median_zipf_for_field(meta, bad_field),
    ]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _collect_rows(paths: List[Path]) -> pd.DataFrame:
    dataset_cache: Dict[str, Dict[str, dict]] = {}
    rows: List[dict] = []

    for path in paths:
        model, dataset_base, _ = _parse_pair_scores_path(path)
        meta_by_idx = _load_dataset_meta(dataset_base, dataset_cache)

        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                meta = meta_by_idx.get(rec.get("idx")) or {}
                zipf_med = _swapped_median_zipf(meta, rec.get("good_field"), rec.get("bad_field"))
                if zipf_med is None:
                    continue
                good_len = rec.get("good_token_count")
                bad_len = rec.get("bad_token_count")
                if not isinstance(good_len, int) or not isinstance(bad_len, int):
                    continue
                avg_len = 0.5 * (good_len + bad_len)
                rows.append(
                    {
                        "model": model,
                        "swapped_median_zipf": float(zipf_med),
                        "avg_token_len": float(avg_len),
                    }
                )

    if not rows:
        raise SystemExit("No data collected from pair-score files.")
    return pd.DataFrame(rows)


def _per_model_binned(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for model, df_model in df.groupby("model", observed=True):
        stats_rows = []
        for bin_label, df_bin in df_model.groupby("zipf_bin", observed=True):
            if df_bin.empty or pd.isna(bin_label):
                continue
            mid = float(bin_label.mid)
            n_bin = int(len(df_bin))
            vals = df_bin["avg_token_len"].to_numpy()
            mean = float(np.mean(vals))
            stats_rows.append({"mid": mid, "n": n_bin, "mean": mean})
        out[model] = pd.DataFrame(stats_rows).sort_values("mid")
    return out


def _display_model(model: str) -> str:
    if model.startswith("Llama-"):
        return "Llama series"
    return model.replace("_", ".")


def _merge_series(per_model: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    grouped: Dict[str, List[pd.DataFrame]] = {}
    for model, stats in per_model.items():
        label = _display_model(model)
        grouped.setdefault(label, []).append(stats)

    merged: Dict[str, pd.DataFrame] = {}
    for label, frames in grouped.items():
        if len(frames) == 1:
            merged[label] = frames[0]
            continue
        combo = pd.concat(frames, ignore_index=True)
        if combo.empty:
            merged[label] = combo
            continue
        # Weighted average by bin counts to collapse identical series.
        agg = (
            combo.groupby("mid", as_index=False)
            .apply(lambda df: pd.Series({"mean": np.average(df["mean"], weights=df["n"]), "n": df["n"].sum()}))
            .reset_index(drop=True)
            .sort_values("mid")
        )
        merged[label] = agg
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot average token length vs realised Zipf rarity.")
    ap.add_argument("--pattern", default="results/blimp_pair_scores/*.jsonl", help="Glob for pair-score JSONL files.")
    ap.add_argument("--out", default=str(PLOT_PATH), help="Output PNG path.")
    ap.add_argument("--out-pdf", default=None, help="Optional PDF output path.")
    ap.add_argument("--variant", choices=["freq", "rare", "original", "any"], default="any", help="Filter on dataset variant.")
    ap.add_argument(
        "--dataset-contains",
        default=None,
        help="Optional substring that dataset name must contain.",
    )
    ap.add_argument(
        "--window-only",
        action="store_true",
        help="Keep only windowed datasets (names containing zipfX_a-b_c pattern).",
    )
    ap.add_argument(
        "--min-bin-n",
        type=int,
        default=25,
        help="Minimum number of samples required per Zipf bin.",
    )
    ap.add_argument(
        "--quantile-bins",
        type=int,
        default=20,
        help="Number of quantile bins for swapped_median_zipf (uses pd.qcut).",
    )
    args = ap.parse_args()
    if args.variant == "rare":
        args.variant = "freq"

    _ensure_mpl_config(Path(args.out).parent)

    paths = sorted(Path().glob(args.pattern))
    paths = [p for p in paths if p.is_file()]
    window_re = re.compile(r"zipf\d+_\d+-\d+_\d+")
    filtered = []
    for p in paths:
        try:
            _, dataset_base, variant_hint = _parse_pair_scores_path(p)
        except ValueError:
            continue
        if args.dataset_contains and args.dataset_contains not in dataset_base:
            continue
        if args.window_only and not window_re.search(dataset_base):
            continue
        if args.variant != "any" and variant_hint != args.variant:
            continue
        filtered.append(p)
    paths = filtered
    if not paths:
        raise SystemExit(
            f"No pair-score files match filters (pattern={args.pattern}, variant={args.variant}, dataset_contains={args.dataset_contains}, window_only={args.window_only})"
        )

    df = _collect_rows(paths)
    try:
        df = df.assign(zipf_bin=pd.qcut(df["swapped_median_zipf"], q=args.quantile_bins, duplicates="drop"))
    except ValueError:
        raise SystemExit("Unable to create quantile bins; try reducing --quantile-bins.")
    per_model = _per_model_binned(df)
    if args.min_bin_n > 1:
        for model, stats in per_model.items():
            if not stats.empty:
                per_model[model] = stats[stats["n"] >= args.min_bin_n]

    per_label = _merge_series(per_model)

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.2))
    for model, stats in per_label.items():
        if stats.empty:
            continue
        ax.plot(stats["mid"], stats["mean"], marker="o", label=model)

    ax.set_ylabel("Avg token length (mean of good/bad)")
    ax.set_xlabel("Realised median Zipf of swapped lemmas")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plotted_mids = pd.concat([stats["mid"] for stats in per_model.values() if not stats.empty], ignore_index=True)
    if not plotted_mids.empty:
        x_max = float(plotted_mids.max())
        x_min = float(plotted_mids.min())
        ax.set_xlim(x_max, x_min)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    if args.out_pdf:
        fig.savefig(Path(args.out_pdf))

    print(f"Saved plot to {out_path}")
    if args.out_pdf:
        print(f"Saved PDF to {args.out_pdf}")


if __name__ == "__main__":
    main()
