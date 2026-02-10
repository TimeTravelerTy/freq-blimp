import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _ensure_mpl_config(out_dir: Path) -> None:
    """
    Matplotlib sometimes tries to create ~/.matplotlib which can fail in restricted environments.
    Force it to a writable directory under out_dir.
    """
    cfg = out_dir / ".mplconfig"
    cfg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cfg))


def _model_slug(model: str) -> str:
    return (model or "unknown").split("/")[-1].replace(" ", "_")


_MODEL_SIZE_RE = re.compile(r"(?P<size>[0-9]+(?:\.[0-9]+)?)B\b")


def _model_sort_key(model_slug: str) -> Tuple[int, float, str]:
    """
    Sort models by parameter scale when possible (e.g., 1B < 3B < 8B), otherwise lexicographically.
    """
    m = _MODEL_SIZE_RE.search(model_slug)
    if not m:
        return (1, float("inf"), model_slug)
    try:
        size = float(m.group("size"))
    except ValueError:
        return (1, float("inf"), model_slug)
    return (0, size, model_slug)


_ZIPF_RE = re.compile(r"(?:^|_)zipf(\d+)_(\d+)(?:_|$)")


def _zipf_threshold_from_data_path(data_path: str) -> Optional[float]:
    m = _ZIPF_RE.search(Path(data_path).name)
    if not m:
        return None
    whole = int(m.group(1))
    frac_s = m.group(2)
    frac = int(frac_s) / (10 ** len(frac_s))
    return whole + frac


@dataclass(frozen=True)
class PairPoint:
    run_file: str
    model: str
    zipf_thr: Optional[float]
    row: int
    k_swaps: int
    delta_z: float  # Z_typ - Z_rare (mean Zipf over swapped positions)
    delta_nll_per_swap: float  # (NLL_rare - NLL_typ) / k


@dataclass(frozen=True)
class GrammarPoint:
    run_file: str
    model: str
    zipf_thr: Optional[float]
    row: int
    delta_z: float  # Z_typ - Z_rare
    z_rare: float
    g_typ: float  # bad_original - good_original
    g_rare: float  # bad_rare - good_rare
    delta_g: float  # g_rare - g_typ


def _iter_run_paths(pattern: str) -> List[Path]:
    paths = sorted(Path().glob(pattern))
    return [p for p in paths if p.is_file() and p.name.endswith(".json") and "sentence-nll" in p.name]


def _build_per_row(details: List[dict]) -> Dict[int, Dict[str, dict]]:
    per_row: Dict[int, Dict[str, dict]] = {}
    for it in details:
        row = it.get("row")
        variant = it.get("variant")
        if not isinstance(row, int) or not isinstance(variant, str):
            continue
        per_row.setdefault(row, {})[variant] = it
    return per_row


def _zipf_mean(item: dict) -> Optional[float]:
    agg = item.get("zipf_swapped_position_agg")
    if not isinstance(agg, dict):
        return None
    v = agg.get("mean")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _k_swaps(item: dict) -> int:
    k = item.get("k_swaps")
    if isinstance(k, int):
        return k
    try:
        return int(k) if k is not None else 0
    except (TypeError, ValueError):
        return 0


def _nll(item: dict) -> Optional[float]:
    v = item.get("total_nll")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _collect_points_from_obj(
    run_path: Path, obj: dict
) -> Tuple[List[PairPoint], List[GrammarPoint], Optional[Tuple[float, float]]]:
    model = str(obj.get("model") or "unknown")
    data_path = str(obj.get("data") or "")
    zipf_thr = _zipf_threshold_from_data_path(data_path)

    run_level_point = None
    rpps = obj.get("rare_penalty_per_swap") or {}
    if isinstance(rpps, dict):
        mean_per_swap = rpps.get("mean_per_swap_nll")
        if zipf_thr is not None and mean_per_swap is not None:
            try:
                run_level_point = (zipf_thr, float(mean_per_swap))
            except (TypeError, ValueError):
                run_level_point = None

    details = obj.get("details") or []
    if not isinstance(details, list) or not details:
        return [], [], run_level_point

    per_row = _build_per_row(details)
    pair_points: List[PairPoint] = []
    grammar_points: List[GrammarPoint] = []

    for row, variants in per_row.items():
        gt = variants.get("good_original") or variants.get("good_typical")
        gr = variants.get("good_rare")
        bt = variants.get("bad_original") or variants.get("bad_typical")
        br = variants.get("bad_rare")

        if isinstance(gt, dict) and isinstance(gr, dict):
            zt = _zipf_mean(gt)
            zr = _zipf_mean(gr)
            nll_t = _nll(gt)
            nll_r = _nll(gr)
            k = _k_swaps(gt) or _k_swaps(gr)
            if zt is not None and zr is not None and nll_t is not None and nll_r is not None and k > 0:
                pair_points.append(
                    PairPoint(
                        run_file=str(run_path),
                        model=model,
                        zipf_thr=zipf_thr,
                        row=row,
                        k_swaps=k,
                        delta_z=zt - zr,
                        delta_nll_per_swap=(nll_r - nll_t) / k,
                    )
                )

        if isinstance(gt, dict) and isinstance(gr, dict) and isinstance(bt, dict) and isinstance(br, dict):
            zt = _zipf_mean(gt)
            zr = _zipf_mean(gr)
            nll_gt = _nll(gt)
            nll_gr = _nll(gr)
            nll_bt = _nll(bt)
            nll_br = _nll(br)
            if (
                zt is not None
                and zr is not None
                and nll_gt is not None
                and nll_gr is not None
                and nll_bt is not None
                and nll_br is not None
            ):
                g_typ = nll_bt - nll_gt
                g_rare = nll_br - nll_gr
                grammar_points.append(
                    GrammarPoint(
                        run_file=str(run_path),
                        model=model,
                        zipf_thr=zipf_thr,
                        row=row,
                        delta_z=zt - zr,
                        z_rare=zr,
                        g_typ=g_typ,
                        g_rare=g_rare,
                        delta_g=g_rare - g_typ,
                    )
                )

    return pair_points, grammar_points, run_level_point


def _binned_mean_curve(x: np.ndarray, y: np.ndarray, q: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.empty:
        return np.array([]), np.array([])
    try:
        bins = pd.qcut(df["x"], q=q, duplicates="drop")
    except ValueError:
        return np.array([]), np.array([])
    grouped = df.groupby(bins, observed=True).agg(x_mean=("x", "mean"), y_mean=("y", "mean")).dropna()
    return grouped["x_mean"].to_numpy(), grouped["y_mean"].to_numpy()


def _bootstrap_mean_ci(values: np.ndarray, *, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = values.size
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        means[i] = float(np.mean(sample))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def _plot_delta_nll_vs_delta_zipf(df_pairs: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    models = sorted(df_pairs["model_slug"].unique().tolist(), key=_model_sort_key)
    if not models:
        return
    ncols = min(3, len(models))
    nrows = int(math.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 4.2 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    xlim = (float(df_pairs["delta_z"].min()), float(df_pairs["delta_z"].max()))
    ylim = (float(df_pairs["delta_nll_per_swap"].min()), float(df_pairs["delta_nll_per_swap"].max()))

    for ax, model in zip(axes, models):
        sub = df_pairs[df_pairs["model_slug"] == model]
        ax.hexbin(
            sub["delta_z"],
            sub["delta_nll_per_swap"],
            gridsize=55,
            mincnt=1,
            linewidths=0.0,
            cmap="viridis",
        )
        x_curve, y_curve = _binned_mean_curve(sub["delta_z"].to_numpy(), sub["delta_nll_per_swap"].to_numpy(), q=10)
        if x_curve.size:
            ax.plot(x_curve, y_curve, color="white", linewidth=2.0)
            ax.plot(x_curve, y_curve, color="black", linewidth=1.0)
        ax.set_title(model)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("ΔZ = Z_typ − Z_rare (mean Zipf)")
        ax.set_ylabel("ΔNLL/k = (NLL_rare − NLL_typ)/k")
        ax.grid(alpha=0.15)

    for ax in axes[len(models) :]:
        ax.axis("off")

    fig.suptitle("Rare penalty per swap vs lexical rarity shift (good pairs)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_penalty_vs_zipf_threshold(df_run: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    models = sorted(df_run["model_slug"].unique().tolist(), key=_model_sort_key)
    if not models:
        return
    ncols = min(3, len(models))
    nrows = int(math.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 3.8 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)

    # Use canonical Zipf threshold ticks (4.0, 3.8, 3.6, ...) based on run coverage.
    all_thr = df_run["zipf_thr"].dropna().to_numpy(dtype=float)
    if all_thr.size:
        max_thr = float(np.max(all_thr))
        min_thr = float(np.min(all_thr))
        # Snap to the 0.2 grid.
        start = round(max_thr * 5.0) / 5.0
        end = round(min_thr * 5.0) / 5.0
        ticks = np.arange(start, end - 1e-9, -0.2, dtype=float)
    else:
        ticks = None

    for ax, model in zip(axes, models):
        sub = df_run[df_run["model_slug"] == model].sort_values("zipf_thr", ascending=False)
        ax.plot(sub["zipf_thr"], sub["mean_per_swap_nll"], marker="o", linewidth=1.5)
        ax.invert_xaxis()
        if ticks is not None:
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.set_title(model)
        ax.set_xlabel("Zipf threshold used to build dataset")
        ax.set_ylabel("Run mean rare penalty per swap (NLL)")
        ax.grid(alpha=0.2)

    for ax in axes[len(models) :]:
        ax.axis("off")

    fig.suptitle("Sanity check: run-level rare penalty per swap vs Zipf threshold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_good_token_length_vs_zipf_threshold(df_run: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    needed = {"zipf_thr", "model_slug", "mean_tokens_good_original", "mean_tokens_good_rare"}
    if not needed.issubset(set(df_run.columns)):
        return

    models = sorted(df_run["model_slug"].unique().tolist(), key=_model_sort_key)
    if not models:
        return

    ncols = min(3, len(models))
    nrows = int(math.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 3.8 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)

    all_thr = df_run["zipf_thr"].dropna().to_numpy(dtype=float)
    if all_thr.size:
        max_thr = float(np.max(all_thr))
        min_thr = float(np.min(all_thr))
        start = round(max_thr * 5.0) / 5.0
        end = round(min_thr * 5.0) / 5.0
        ticks = np.arange(start, end - 1e-9, -0.2, dtype=float)
    else:
        ticks = None

    for ax, model in zip(axes, models):
        sub = df_run[df_run["model_slug"] == model].dropna(subset=["zipf_thr"])
        sub = sub.sort_values("zipf_thr", ascending=False)
        ax.plot(
            sub["zipf_thr"],
            sub["mean_tokens_good_original"],
            marker="o",
            linewidth=1.5,
            label="good_original",
        )
        ax.plot(
            sub["zipf_thr"],
            sub["mean_tokens_good_rare"],
            marker="o",
            linewidth=1.5,
            label="good_rare",
        )
        ax.invert_xaxis()
        if ticks is not None:
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.set_title(model)
        ax.set_xlabel("Zipf threshold used to build dataset")
        ax.set_ylabel("Mean token count")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)

    for ax in axes[len(models) :]:
        ax.axis("off")

    fig.suptitle("Token length sanity check (good sentences)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_grammar_sensitivity(df_g: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    models = sorted(df_g["model_slug"].unique().tolist(), key=_model_sort_key)
    if not models:
        return
    ncols = min(3, len(models))
    nrows = int(math.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 4.2 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    xlim = (float(df_g["delta_z"].min()), float(df_g["delta_z"].max()))
    ylim = (float(df_g["delta_g"].min()), float(df_g["delta_g"].max()))

    for ax, model in zip(axes, models):
        sub = df_g[df_g["model_slug"] == model]
        ax.hexbin(
            sub["delta_z"],
            sub["delta_g"],
            gridsize=55,
            mincnt=1,
            linewidths=0.0,
            cmap="magma",
        )
        x_curve, y_curve = _binned_mean_curve(sub["delta_z"].to_numpy(), sub["delta_g"].to_numpy(), q=10)
        if x_curve.size:
            ax.plot(x_curve, y_curve, color="white", linewidth=2.0)
            ax.plot(x_curve, y_curve, color="black", linewidth=1.0)
        ax.set_title(model)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("ΔZ = Z_typ − Z_rare (mean Zipf)")
        ax.set_ylabel("ΔG = (bad−good)_rare − (bad−good)_typ")
        ax.grid(alpha=0.15)

    for ax in axes[len(models) :]:
        ax.axis("off")

    fig.suptitle("Grammar sensitivity shift vs lexical rarity shift (per item)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_grammar_robustness(df_g: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    models = sorted(df_g["model_slug"].unique().tolist(), key=_model_sort_key)
    if not models:
        return
    ncols = min(3, len(models))
    nrows = int(math.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 3.8 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    for ax, model in zip(axes, models):
        sub = df_g[df_g["model_slug"] == model].dropna(subset=["z_rare", "g_rare"])
        if sub.empty:
            ax.axis("off")
            continue

        # Bin by Z_rare quantiles, then bootstrap mean CI for G_rare.
        try:
            bins = pd.qcut(sub["z_rare"], q=12, duplicates="drop")
        except ValueError:
            bins = None

        if bins is None:
            ax.axis("off")
            continue

        rows = []
        for bin_interval, grp in sub.groupby(bins, observed=True):
            vals = grp["g_rare"].to_numpy(dtype=float)
            if vals.size < 20:
                continue
            mean = float(np.mean(vals))
            lo, hi = _bootstrap_mean_ci(vals, n_boot=600, alpha=0.05, seed=0)
            center = float(np.mean(grp["z_rare"].to_numpy(dtype=float)))
            rows.append((center, mean, lo, hi, int(vals.size)))

        if not rows:
            ax.axis("off")
            continue

        rows.sort(key=lambda r: r[0], reverse=True)
        xs = np.array([r[0] for r in rows], dtype=float)
        ys = np.array([r[1] for r in rows], dtype=float)
        los = np.array([r[2] for r in rows], dtype=float)
        his = np.array([r[3] for r in rows], dtype=float)

        ax.plot(xs, ys, marker="o", linewidth=1.5)
        ax.fill_between(xs, los, his, alpha=0.25)
        ax.invert_xaxis()
        ax.set_title(model)
        ax.set_xlabel("Z_rare (mean Zipf, binned)")
        ax.set_ylabel("Mean G_rare = NLL(bad_rare) − NLL(good_rare)")
        ax.grid(alpha=0.2)

    for ax in axes[len(models) :]:
        ax.axis("off")

    fig.suptitle("Grammar robustness vs lexical rarity (bootstrap 95% CI)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Zipf/NLL analyses from results/sentence_nll_runs.")
    ap.add_argument("--pattern", default="results/sentence_nll_runs/*sentence-nll.json")
    ap.add_argument("--out-dir", default="results/analysis_plots")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    _ensure_mpl_config(out_dir)
    # Ensure a non-GUI backend (avoids crashes in headless / sandboxed environments).
    os.environ.setdefault("MPLBACKEND", "Agg")

    run_paths = _iter_run_paths(args.pattern)
    if not run_paths:
        raise SystemExit(f"No run files matched: {args.pattern}")

    all_pairs: List[PairPoint] = []
    all_grammar: List[GrammarPoint] = []
    run_level_rows: List[dict] = []

    for rp in run_paths:
        obj = json.loads(rp.read_text(encoding="utf-8"))
        model = str(obj.get("model") or "unknown")
        model_slug = _model_slug(model)
        variant_stats = obj.get("variant_stats") or {}

        # Reuse parsed obj for points collection.
        pairs, grams, run_level = _collect_points_from_obj(rp, obj)
        all_pairs.extend(pairs)
        all_grammar.extend(grams)
        if run_level is not None:
            zipf_thr, mean_per_swap = run_level
            gt_tokens = None
            gr_tokens = None
            try:
                gt_tokens = float((variant_stats.get("good_original") or variant_stats.get("good_typical") or {}).get("mean_tokens"))
            except (TypeError, ValueError):
                gt_tokens = None
            try:
                gr_tokens = float((variant_stats.get("good_rare") or {}).get("mean_tokens"))
            except (TypeError, ValueError):
                gr_tokens = None
            run_level_rows.append(
                {
                    "run_file": str(rp),
                    "model": model,
                    "model_slug": model_slug,
                    "zipf_thr": float(zipf_thr),
                    "mean_per_swap_nll": float(mean_per_swap),
                    "mean_tokens_good_original": gt_tokens,
                    "mean_tokens_good_rare": gr_tokens,
                }
            )

    df_pairs = pd.DataFrame([p.__dict__ for p in all_pairs])
    df_g = pd.DataFrame([g.__dict__ for g in all_grammar])
    df_run = pd.DataFrame(run_level_rows)

    if not df_pairs.empty:
        df_pairs["model_slug"] = df_pairs["model"].map(_model_slug)
    if not df_g.empty:
        df_g["model_slug"] = df_g["model"].map(_model_slug)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Write panel order so it's easy to verify which model is which in figures.
    model_slugs = set()
    if not df_pairs.empty:
        model_slugs |= set(df_pairs["model_slug"].unique().tolist())
    if not df_g.empty:
        model_slugs |= set(df_g["model_slug"].unique().tolist())
    if not df_run.empty:
        model_slugs |= set(df_run["model_slug"].unique().tolist())
    ordered = sorted(model_slugs, key=_model_sort_key)
    (out_dir / "panel_order.txt").write_text("\n".join(ordered) + "\n", encoding="utf-8")
    if not df_pairs.empty:
        df_pairs.to_csv(out_dir / "pair_points.csv", index=False)
    if not df_g.empty:
        df_g.to_csv(out_dir / "grammar_points.csv", index=False)
    if not df_run.empty:
        df_run.to_csv(out_dir / "run_level_points.csv", index=False)

    if not df_pairs.empty:
        _plot_delta_nll_vs_delta_zipf(df_pairs, out_dir / "delta_nll_per_swap_vs_delta_zipf.png")
    if not df_run.empty:
        _plot_penalty_vs_zipf_threshold(df_run, out_dir / "rare_penalty_vs_zipf_threshold.png")
        _plot_good_token_length_vs_zipf_threshold(df_run, out_dir / "good_token_length_vs_zipf_threshold.png")
    if not df_g.empty:
        _plot_grammar_sensitivity(df_g, out_dir / "grammar_sensitivity_vs_delta_zipf.png")
        _plot_grammar_robustness(df_g, out_dir / "grammar_robustness_vs_zrare.png")

    print(f"Wrote plots + CSVs to {out_dir}")


if __name__ == "__main__":
    main()
