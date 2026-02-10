"""
Plot BLiMP pair-score NLLs vs realised Zipf rarity.

Definitions:
- delta_nll = bad_total_nll - good_total_nll (grammar preference; larger = stronger)
- total_nll = (bad_total_nll + good_total_nll) / 2 (overall difficulty)
- swapped_median_zipf = average of the median Zipf of swapped tokens in the good/bad
  sentences for the scored variant (pulled from the processed dataset metadata).

Outputs:
- results/analysis_plots/zipf_vs_nll.png  (top: delta, bottom: total)
- results/analysis_plots/zipf_vs_nll_correlations.txt (Spearman r and p)
"""

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Force matplotlib config/cache into a writable directory and use a non-GUI backend.
DEFAULT_MPL_DIR = Path("results/analysis_plots/.mplconfig")
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)


PAIR_SCORES_DIR = Path("results/blimp_pair_scores")
DATA_DIR = Path("data/processed")
PLOT_PATH = Path("results/analysis_plots/zipf_vs_nll.png")
CORR_PATH = Path("results/analysis_plots/zipf_vs_nll_correlations.txt")

_PAIR_RE = re.compile(
    r"(?:^|/)\d{8}-\d{6}_(?P<model>.+?)_(?P<dataset>\d{8}-\d{6}_.+?)_pair-scores\.jsonl$"
)

DEFAULT_MPL_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_mpl_config(out_dir: Path) -> None:
    """
    Avoid ~/.matplotlib permission issues by forcing a writable config dir.
    """
    cfg = out_dir / ".mplconfig"
    cfg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cfg))


def _parse_pair_scores_path(path: Path) -> Tuple[str, str, str]:
    """
    Returns (model, dataset_base, variant_hint)
    dataset_base strips the trailing _original/_freq/_rare suffix.
    """
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
    """
    Returns idx -> meta mapping for the processed dataset.
    """
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


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = 600, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    if values.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    n = values.size
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        means[i] = float(np.mean(sample))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def _spearman_r_p(x: Iterable[float], y: Iterable[float]) -> Tuple[float, float, int]:
    xs = pd.Series(list(x))
    ys = pd.Series(list(y))
    mask = xs.notna() & ys.notna()
    xs = xs[mask]
    ys = ys[mask]
    n = len(xs)
    if n < 3:
        return math.nan, math.nan, n
    rx = xs.rank(method="average")
    ry = ys.rank(method="average")
    r = float(rx.corr(ry))
    # Approximate p-value using Fisher z
    r = max(min(r, 0.999999), -0.999999)
    z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(max(n - 3, 1))
    p = math.erfc(abs(z) / math.sqrt(2))
    return r, p, n


def _per_model_binned(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for model, df_model in df.groupby("model", observed=True):
        stats_rows = []
        for bin_label, df_bin in df_model.groupby("zipf_bin", observed=True):
            if df_bin.empty or pd.isna(bin_label):
                continue
            mid = float(bin_label.mid)
            n_bin = int(len(df_bin))
            d_vals = df_bin["delta_nll"].to_numpy()
            t_vals = df_bin["total_nll"].to_numpy()
            d_mean = float(np.mean(d_vals))
            t_mean = float(np.mean(t_vals))
            d_lo, d_hi = _bootstrap_mean_ci(d_vals)
            t_lo, t_hi = _bootstrap_mean_ci(t_vals)
            stats_rows.append(
                {
                    "mid": mid,
                    "n": n_bin,
                    "delta_mean": d_mean,
                    "delta_lo": d_lo,
                    "delta_hi": d_hi,
                    "total_mean": t_mean,
                    "total_lo": t_lo,
                    "total_hi": t_hi,
                }
            )
        out[model] = pd.DataFrame(stats_rows).sort_values("mid")
    return out


def _collect_rows(paths: List[Path], char_normalize: bool = False) -> pd.DataFrame:
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
                good_nll = rec.get("good_total_nll")
                bad_nll = rec.get("bad_total_nll")
                if not isinstance(good_nll, (int, float)) or not isinstance(bad_nll, (int, float)):
                    continue
                if char_normalize:
                    good_len = _char_len(rec.get("good_text"))
                    bad_len = _char_len(rec.get("bad_text"))
                    if not good_len or not bad_len:
                        continue
                    good_nll = good_nll / good_len
                    bad_nll = bad_nll / bad_len
                delta_nll = bad_nll - good_nll
                total_nll = 0.5 * (bad_nll + good_nll)
                rows.append(
                    {
                        "model": model,
                        "swapped_median_zipf": float(zipf_med),
                        "delta_nll": float(delta_nll),
                        "total_nll": float(total_nll),
                    }
                )

    if not rows:
        raise SystemExit("No data collected from pair-score files.")
    return pd.DataFrame(rows)


def _char_len(text: Optional[str]) -> Optional[int]:
    if not isinstance(text, str):
        return None
    return len(text)


def _token_len(value: Optional[object]) -> Optional[int]:
    if isinstance(value, int):
        return value
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot delta/total NLL vs realised Zipf rarity using BLiMP pair scores.")
    ap.add_argument("--pattern", default="results/blimp_pair_scores/*.jsonl", help="Glob for pair-score JSONL files.")
    ap.add_argument("--out", default=str(PLOT_PATH), help="Output PNG path.")
    ap.add_argument("--out-pdf", default=None, help="Optional PDF output path.")
    ap.add_argument(
        "--char-normalize",
        action="store_true",
        help="Normalize NLL by character length (uses len(good_text)/len(bad_text)).",
    )
    ap.add_argument(
        "--token-normalize",
        action="store_true",
        help="Normalize NLL by token count (uses good_token_count/bad_token_count).",
    )
    ap.add_argument("--variant", choices=["freq", "rare", "original", "any"], default="any", help="Filter on dataset variant inferred from filename.")
    ap.add_argument(
        "--dataset-contains",
        default=None,
        help="Optional substring that dataset name must contain (e.g., a timestamp like 20251225-004036 for window-sampled sets).",
    )
    ap.add_argument(
        "--window-only",
        action="store_true",
        help="Keep only windowed datasets (names containing zipfX_a-b_c pattern, e.g., zipf1_2-2_0).",
    )
    ap.add_argument(
        "--min-bin-n",
        type=int,
        default=25,
        help="Minimum number of samples required per Zipf bin (bins with fewer samples are dropped).",
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

    if args.char_normalize and args.token_normalize:
        raise SystemExit("Choose only one: --char-normalize or --token-normalize.")
    df = _collect_rows(paths, char_normalize=args.char_normalize)
    if args.token_normalize:
        df = _collect_rows(paths, char_normalize=False)
        df = df.assign(
            _token_normalize=True,
        )
        # Recompute using token counts.
        rows = []
        dataset_cache: Dict[str, Dict[str, dict]] = {}
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
                    good_len = _token_len(rec.get("good_token_count"))
                    bad_len = _token_len(rec.get("bad_token_count"))
                    if not good_len or not bad_len:
                        continue
                    good_nll = rec.get("good_total_nll")
                    bad_nll = rec.get("bad_total_nll")
                    if not isinstance(good_nll, (int, float)) or not isinstance(bad_nll, (int, float)):
                        continue
                    good_nll = good_nll / good_len
                    bad_nll = bad_nll / bad_len
                    delta_nll = bad_nll - good_nll
                    total_nll = 0.5 * (bad_nll + good_nll)
                    rows.append(
                        {
                            "model": model,
                            "swapped_median_zipf": float(zipf_med),
                            "delta_nll": float(delta_nll),
                            "total_nll": float(total_nll),
                        }
                    )
        if not rows:
            raise SystemExit("No data collected after token normalization.")
        df = pd.DataFrame(rows)
    try:
        df = df.assign(zipf_bin=pd.qcut(df["swapped_median_zipf"], q=args.quantile_bins, duplicates="drop"))
    except ValueError:
        raise SystemExit("Unable to create quantile bins; try reducing --quantile-bins.")
    per_model = _per_model_binned(df)
    if args.min_bin_n > 1:
        for model, stats in per_model.items():
            if not stats.empty:
                per_model[model] = stats[stats["n"] >= args.min_bin_n]

    correlations: List[str] = []
    for model, stats in per_model.items():
        r, p, n = _spearman_r_p(
            df[df["model"] == model]["swapped_median_zipf"],
            df[df["model"] == model]["delta_nll"],
        )
        correlations.append(f"{model}: r_s={r:.4f}, p≈{p:.2e}, n={n}")

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 10), sharex=True)
    ax_delta, ax_total = axes

    for model, stats in per_model.items():
        if stats.empty:
            continue
        ax_delta.plot(stats["mid"], stats["delta_mean"], marker="o", label=model.replace("_", "."))
        ax_delta.fill_between(stats["mid"], stats["delta_lo"], stats["delta_hi"], alpha=0.2)

        ax_total.plot(stats["mid"], stats["total_mean"], marker="o", label=model.replace("_", "."))
        ax_total.fill_between(stats["mid"], stats["total_lo"], stats["total_hi"], alpha=0.2)

    norm_suffix = ""
    if args.char_normalize:
        norm_suffix = " / char"
    if args.token_normalize:
        norm_suffix = " / token"
    ax_delta.set_ylabel(f"Δ NLL (bad - good){norm_suffix}")
    ax_delta.legend()
    ax_delta.grid(True, alpha=0.3)

    ax_total.set_ylabel(f"Total NLL (mean of pair){norm_suffix}")
    ax_total.set_xlabel("Realised median Zipf")
    ax_total.grid(True, alpha=0.3)

    # Higher Zipf = more common, so flip the x-axis to show rarity increasing left→right.
    plotted_mids = pd.concat([stats["mid"] for stats in per_model.values() if not stats.empty], ignore_index=True)
    x_max = float(plotted_mids.max())
    x_min = float(plotted_mids.min())
    ax_delta.set_xlim(x_max, x_min)
    ax_total.set_xlim(x_max, x_min)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    if args.out_pdf:
        fig.savefig(Path(args.out_pdf))

    corr_text = "\n".join(correlations) + "\n"
    CORR_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORR_PATH.write_text(corr_text)

    for line in correlations:
        print(line)
    print(f"Saved plot to {out_path}")
    if args.out_pdf:
        print(f"Saved PDF to {args.out_pdf}")
    print(f"Saved correlations to {CORR_PATH}")


if __name__ == "__main__":
    main()
