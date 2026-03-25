"""Correlate Qwen2.5-72B disruption measures with model accuracy on FreqBLiMP.

Hypothesis: the ~8pp BLiMP → FreqBLiMP accuracy gap is driven by a minority of
items where lexical swapping severely disrupts plausibility (high delta_pair),
not a uniform frequency effect.

Required extra packages (not in requirements.txt):
    pip install statsmodels

Inputs:
  1. Qwen plausibility scores:  results/plausibility_scores/*.jsonl
                                (output of score_plausibility.py)
  2. Eval model pair scores:    results/blimp_pair_scores/*_rare_pair-scores.jsonl
                                (existing results for Llama-3.2-1B/3B, Llama-3.1-8B,
                                 Mistral-7B-v0.3 across head/tail/xtail regimes)

Analyses:
  1. Logistic regression: correct ~ disruption_measure  (per model, per measure)
  2. Pooled regression with model as fixed effect
  3. Binned accuracy plots (quartile bins of delta_pair) — main paper figure
  4. Per-model sensitivity bar chart (regression slopes)
  5. Regime breakdown (head/tail/xtail split)

Outputs (results/plausibility_analysis/):
  figures/
    binned_accuracy_delta_pair.pdf      main paper figure
    binned_accuracy_all_measures.pdf    supplementary
    sensitivity_by_model.pdf            effect-size comparison
    regime_breakdown.pdf                head/tail/xtail split
  regression_results.csv               raw regression output
  regression_table.tex                 LaTeX table for paper
  summary_stats.csv                    descriptive stats per regime

Usage:
  python scripts/analyze_plausibility.py

  # Point to specific Qwen scores:
  python scripts/analyze_plausibility.py \\
      --qwen-dir results/plausibility_scores \\
      --pair-scores-dir results/blimp_pair_scores \\
      --out-dir results/plausibility_analysis
"""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print(
        "[Warn] statsmodels not found — regression analysis will be skipped.\n"
        "       Install with:  pip install statsmodels\n"
    )

# ---------------------------------------------------------------------------
# Matplotlib style — publication-quality defaults
# ---------------------------------------------------------------------------

_FIGSTYLE = {
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

_PALETTE = ["#2166ac", "#d6604d", "#4dac26", "#b2abd2"]  # one per model

# Disruption measures we analyse
MEASURES = ["delta_pair", "delta_g", "delta_b", "asymmetry"]
MEASURE_LABELS = {
    "delta_pair": r"$\Delta_\mathrm{pair}$",
    "delta_g": r"$\Delta_g$",
    "delta_b": r"$\Delta_b$",
    "asymmetry": r"Asymmetry ($\Delta_g - \Delta_b$)",
}

# Regime → human label
REGIME_LABELS = {
    "head": "Head (Zipf 3.6–5.0)",
    "tail": "Tail (Zipf 2.2–3.0)",
    "xtail": "X-Tail (Zipf 1.2–2.0)",
}

# Pair-score filename → regime (based on NLP2026 naming)
_REGIME_PATTERNS = [
    (r"zipf3_6|zipf4_0", "head"),
    (r"zipf2_2|zipf2_4", "tail"),
    (r"zipf1_2",         "xtail"),
]


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------

def _iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def load_qwen_scores(qwen_dir: Path) -> pd.DataFrame:
    """Load all plausibility score JSONL files into a single DataFrame."""
    files = sorted(qwen_dir.glob("*_plausibility_scores.jsonl"))
    if not files:
        sys.exit(f"[Error] No plausibility score files found in {qwen_dir}\n"
                 f"        Run score_plausibility.py first.")
    dfs = []
    for f in files:
        rows = list(_iter_jsonl(str(f)))
        if rows:
            df = pd.DataFrame(rows)
            print(f"[Qwen] Loaded {len(df):,} items from {f.name}")
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(f"[Qwen] Total: {len(df):,} items across {df['regime'].nunique()} regimes\n")
    return df


def _parse_pair_score_filename(name: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (model_slug, regime) from a pair-score filename."""
    # Model: second segment after first timestamp
    m = re.match(r"^\d{8}-\d{6}_([^_]+(?:_[^_]+)*)_\d{8}-\d{6}", name)
    if m:
        raw = m.group(1)
        # Normalise underscores used in place of dots/hyphens
        model_slug = raw.replace("_", "-")
    else:
        model_slug = None

    # Regime from Zipf window in filename
    regime = None
    for pattern, label in _REGIME_PATTERNS:
        if re.search(pattern, name):
            regime = label
            break

    return model_slug, regime


def load_pair_scores(pair_scores_dir: Path) -> pd.DataFrame:
    """Load existing eval-model pair scores → per-item correctness DataFrame."""
    files = sorted(pair_scores_dir.glob("*_rare_pair-scores.jsonl"))
    if not files:
        sys.exit(
            f"[Error] No *_rare_pair-scores.jsonl files found in {pair_scores_dir}"
        )

    rows = []
    for f in files:
        model_slug, regime = _parse_pair_score_filename(f.name)
        if regime is None:
            print(f"[Warn] Could not infer regime from {f.name} — skipping.")
            continue
        n = 0
        for rec in _iter_jsonl(str(f)):
            correct = int(rec["good_total_nll"] < rec["bad_total_nll"])
            rows.append({
                "model":      model_slug,
                "regime":     regime,
                "phenomenon": rec.get("phenomenon"),
                "subtask":    rec.get("subtask"),
                "idx":        rec.get("idx"),
                "correct":    correct,
                "good_total_nll": rec["good_total_nll"],
                "bad_total_nll":  rec["bad_total_nll"],
            })
            n += 1
        print(f"[Pair] {model_slug:<24} regime={regime:<6}  {n:,} items  ({f.name[:30]}...)")

    df = pd.DataFrame(rows)
    print(
        f"[Pair] Total: {len(df):,} rows, "
        f"{df['model'].nunique()} models, "
        f"{df['regime'].nunique()} regimes\n"
    )
    return df


# ---------------------------------------------------------------------------
# Join Qwen scores with eval-model correctness
# ---------------------------------------------------------------------------

def merge_scores(qwen_df: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
    """Inner join on (regime, phenomenon, subtask, idx)."""
    key = ["regime", "phenomenon", "subtask", "idx"]
    merged = pair_df.merge(qwen_df[key + MEASURES], on=key, how="inner")
    n_qwen = len(qwen_df)
    n_pair = len(pair_df)
    n_merged = len(merged)
    print(
        f"[Merge] {n_qwen:,} Qwen items × {n_pair:,} pair-score rows → "
        f"{n_merged:,} matched rows"
    )
    if n_merged < 0.8 * min(n_qwen, n_pair):
        print(
            "[Merge] WARNING: fewer than 80% of items matched. "
            "Check that the same datasets were used for both scoring runs."
        )
    print()
    return merged


# ---------------------------------------------------------------------------
# Regression analysis
# ---------------------------------------------------------------------------

def _fit_logit(formula: str, data: pd.DataFrame) -> Optional[dict]:
    """Fit a logistic regression with statsmodels. Returns a result dict."""
    if not HAS_STATSMODELS:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = smf.logit(formula, data=data).fit(disp=False, maxiter=200)
        # McFadden pseudo-R²
        pseudo_r2 = 1.0 - result.llf / result.llnull
        return {
            "formula":   formula,
            "n":         int(result.nobs),
            "coef":      result.params.to_dict(),
            "pval":      result.pvalues.to_dict(),
            "ci_low":    result.conf_int()[0].to_dict(),
            "ci_high":   result.conf_int()[1].to_dict(),
            "pseudo_r2": pseudo_r2,
            "aic":       result.aic,
            "llf":       result.llf,
            "llnull":    result.llnull,
            "converged": result.mle_retvals.get("converged", True),
        }
    except Exception as e:
        print(f"[Warn] Regression failed for '{formula}': {e}")
        return None


def run_regressions(merged: pd.DataFrame) -> List[dict]:
    """
    Run logistic regressions:
      - Univariate: correct ~ measure  per (model, measure)
      - Multivariate: correct ~ delta_g + delta_b + asymmetry  per model
      - Pooled: correct ~ delta_pair + C(model)
    """
    if not HAS_STATSMODELS:
        print("[Skip] Regression analysis skipped (statsmodels not installed).")
        return []

    results = []
    models = merged["model"].unique()

    print("[Regression] Univariate logistic regressions (per model × measure) ...")
    for model in sorted(models):
        df_m = merged[merged["model"] == model].copy()
        for measure in MEASURES:
            r = _fit_logit(f"correct ~ {measure}", df_m)
            if r:
                r.update({"model": model, "measure": measure, "scope": "per_model_univariate"})
                results.append(r)

    print("[Regression] Multivariate per-model regressions ...")
    for model in sorted(models):
        df_m = merged[merged["model"] == model].copy()
        # delta_pair is a linear combo of delta_g + delta_b; use delta_g + delta_b + asymmetry
        r = _fit_logit("correct ~ delta_g + delta_b + asymmetry", df_m)
        if r:
            r.update({"model": model, "measure": "multivariate", "scope": "per_model_multivariate"})
            results.append(r)

    print("[Regression] Pooled regression (delta_pair + model fixed effects) ...")
    r = _fit_logit("correct ~ delta_pair + C(model)", merged.copy())
    if r:
        r.update({"model": "pooled", "measure": "delta_pair", "scope": "pooled"})
        results.append(r)

    print(f"[Regression] Completed {len(results)} regressions.\n")
    return results


def _results_to_df(results: List[dict]) -> pd.DataFrame:
    """Flatten regression results into a tidy DataFrame for saving."""
    rows = []
    for r in results:
        for param, coef in r["coef"].items():
            if param == "Intercept":
                continue
            rows.append({
                "scope":     r["scope"],
                "model":     r["model"],
                "measure":   r["measure"],
                "param":     param,
                "coef":      coef,
                "OR":        np.exp(coef),
                "ci_low":    r["ci_low"].get(param),
                "ci_high":   r["ci_high"].get(param),
                "pval":      r["pval"].get(param),
                "pseudo_r2": r["pseudo_r2"],
                "aic":       r["aic"],
                "n":         r["n"],
                "converged": r.get("converged", True),
            })
    return pd.DataFrame(rows)


def _make_latex_table(reg_df: pd.DataFrame) -> str:
    """Return a LaTeX table for univariate per-model × delta_pair regressions."""
    sub = reg_df[
        (reg_df["scope"] == "per_model_univariate") &
        (reg_df["measure"] == "delta_pair")
    ].copy()
    if sub.empty:
        return "% No regression results available.\n"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Logistic regression: $\Pr(\text{correct}) \sim \Delta_\mathrm{pair}$ per model}",
        r"\label{tab:plausibility_regression}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & $\beta$ & OR & 95\% CI & $p$ & Pseudo-$R^2$ \\",
        r"\midrule",
    ]
    for _, row in sub.sort_values("model").iterrows():
        ci = f"[{row['ci_low']:.3f},\\ {row['ci_high']:.3f}]"
        p_str = f"{row['pval']:.3f}" if row["pval"] >= 0.001 else "$<$0.001"
        lines.append(
            f"{row['model']} & "
            f"{row['coef']:.4f} & "
            f"{row['OR']:.3f} & "
            f"{ci} & "
            f"{p_str} & "
            f"{row['pseudo_r2']:.4f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary_stats(merged: pd.DataFrame) -> pd.DataFrame:
    """Descriptive stats for disruption measures and accuracy per (regime, model)."""
    rows = []
    for (regime, model), grp in merged.groupby(["regime", "model"]):
        row = {
            "regime": regime,
            "model":  model,
            "n":      len(grp),
            "accuracy": grp["correct"].mean(),
            "accuracy_se": np.sqrt(
                grp["correct"].mean() * (1 - grp["correct"].mean()) / len(grp)
            ),
        }
        for m in MEASURES:
            row[f"{m}_mean"] = grp[m].mean()
            row[f"{m}_std"]  = grp[m].std()
            row[f"{m}_p25"]  = grp[m].quantile(0.25)
            row[f"{m}_p75"]  = grp[m].quantile(0.75)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["regime", "model"])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _se_prop(correct: np.ndarray) -> float:
    p = correct.mean()
    return np.sqrt(p * (1 - p) / len(correct)) if len(correct) > 0 else 0.0


def _quartile_bins(series: pd.Series) -> pd.Categorical:
    """Bin into quartiles, labelled Q1–Q4."""
    try:
        return pd.qcut(series, q=4, labels=["Q1\n(least)", "Q2", "Q3", "Q4\n(most)"])
    except ValueError:
        # Fallback if ties prevent clean quartiles
        return pd.qcut(series, q=4, labels=["Q1", "Q2", "Q3", "Q4"],
                       duplicates="drop")


def fig_binned_accuracy(
    merged: pd.DataFrame,
    measure: str,
    out_path: Path,
    pooled: bool = False,
):
    """Accuracy per quartile bin of `measure`, one panel per model (+ optional pooled panel)."""
    models = sorted(merged["model"].unique())
    subjects = models + (["pooled"] if pooled else [])
    n_panels = len(subjects)

    with plt.rc_context(_FIGSTYLE):
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(2.8 * n_panels, 3.2),
            sharey=True,
        )
        if n_panels == 1:
            axes = [axes]

        # Bin by measure across ALL data (consistent bins)
        merged = merged.copy()
        merged["bin"] = _quartile_bins(merged[measure])

        for ax, subject in zip(axes, subjects):
            if subject == "pooled":
                df_s = merged
                title = "Pooled"
            else:
                df_s = merged[merged["model"] == subject]
                title = subject

            bin_acc = df_s.groupby("bin", observed=False)["correct"].agg(
                ["mean", "count"]
            ).reset_index()
            bin_acc["se"] = df_s.groupby("bin", observed=False)["correct"].apply(
                lambda x: _se_prop(x.values)
            ).values

            color = _PALETTE[subjects.index(subject) % len(_PALETTE)]
            x = np.arange(len(bin_acc))
            ax.bar(x, bin_acc["mean"], color=color, alpha=0.85, width=0.6)
            ax.errorbar(
                x, bin_acc["mean"], yerr=1.96 * bin_acc["se"],
                fmt="none", color="black", capsize=3, linewidth=1,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(bin_acc["bin"].astype(str), fontsize=7)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(f"{MEASURE_LABELS[measure]} quartile", fontsize=8)
            if ax is axes[0]:
                ax.set_ylabel("Accuracy (swapped)")
            ax.set_ylim(
                max(0.0, bin_acc["mean"].min() - 0.15),
                min(1.0, bin_acc["mean"].max() + 0.12),
            )
            # Annotate n per bin
            for xi, row in zip(x, bin_acc.itertuples()):
                ax.text(
                    xi, row.mean + 1.96 * row.se + 0.005,
                    f"n={row.count}", ha="center", va="bottom", fontsize=6,
                )

        fig.suptitle(
            f"Accuracy by {MEASURE_LABELS[measure]} quartile",
            fontsize=10, y=1.02,
        )
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    print(f"[Fig] Saved → {out_path}")


def fig_sensitivity_by_model(reg_df: pd.DataFrame, out_path: Path):
    """Grouped bar chart of regression slopes (beta) per model × measure."""
    sub = reg_df[reg_df["scope"] == "per_model_univariate"].copy()
    if sub.empty:
        print("[Fig] No regression results — skipping sensitivity chart.")
        return

    models  = sorted(sub["model"].unique())
    measures = [m for m in MEASURES if m in sub["measure"].unique()]
    x = np.arange(len(models))
    width = 0.2

    with plt.rc_context(_FIGSTYLE):
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        for j, measure in enumerate(measures):
            coefs = []
            errs_low, errs_high = [], []
            for model in models:
                row = sub[(sub["model"] == model) & (sub["measure"] == measure)]
                if row.empty:
                    coefs.append(np.nan)
                    errs_low.append(0)
                    errs_high.append(0)
                else:
                    row = row.iloc[0]
                    coefs.append(row["coef"])
                    # Asymmetric error bars from CI
                    errs_low.append(row["coef"] - row["ci_low"])
                    errs_high.append(row["ci_high"] - row["coef"])

            offset = (j - len(measures) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, coefs, width,
                label=MEASURE_LABELS[measure],
                alpha=0.85,
            )
            ax.errorbar(
                x + offset, coefs,
                yerr=[errs_low, errs_high],
                fmt="none", color="black", capsize=2, linewidth=0.8,
            )

        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel(r"Logistic regression $\beta$ (disruption effect on accuracy)")
        ax.set_title("Per-model sensitivity to plausibility disruption")
        ax.legend(loc="upper right", frameon=False)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    print(f"[Fig] Saved → {out_path}")


def fig_regime_breakdown(merged: pd.DataFrame, measure: str, out_path: Path):
    """Binned accuracy split by regime (head/tail/xtail), pooled across models."""
    regimes = sorted(merged["regime"].unique())
    n_regimes = len(regimes)

    with plt.rc_context(_FIGSTYLE):
        fig, axes = plt.subplots(1, n_regimes, figsize=(3 * n_regimes, 3.2), sharey=True)
        if n_regimes == 1:
            axes = [axes]

        # Consistent bins across regimes
        merged = merged.copy()
        merged["bin"] = _quartile_bins(merged[measure])

        for ax, regime in zip(axes, regimes):
            df_r = merged[merged["regime"] == regime]
            bin_acc = df_r.groupby("bin", observed=False)["correct"].agg(
                ["mean", "count"]
            ).reset_index()
            bin_acc["se"] = df_r.groupby("bin", observed=False)["correct"].apply(
                lambda x: _se_prop(x.values)
            ).values

            x = np.arange(len(bin_acc))
            ax.bar(x, bin_acc["mean"], color="#5e9acd", alpha=0.85, width=0.6)
            ax.errorbar(
                x, bin_acc["mean"], yerr=1.96 * bin_acc["se"],
                fmt="none", color="black", capsize=3, linewidth=1,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(bin_acc["bin"].astype(str), fontsize=7)
            ax.set_title(REGIME_LABELS.get(regime, regime), fontsize=9)
            ax.set_xlabel(f"{MEASURE_LABELS[measure]} quartile", fontsize=8)
            if ax is axes[0]:
                ax.set_ylabel("Accuracy (swapped, pooled models)")

        fig.suptitle(
            f"Regime breakdown — accuracy by {MEASURE_LABELS[measure]} quartile",
            fontsize=10, y=1.02,
        )
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    print(f"[Fig] Saved → {out_path}")


def fig_all_measures_grid(merged: pd.DataFrame, out_path: Path):
    """4×1 grid of binned accuracy plots for all measures (pooled across models)."""
    with plt.rc_context(_FIGSTYLE):
        fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)

        for ax, measure in zip(axes, MEASURES):
            df = merged.copy()
            df["bin"] = _quartile_bins(df[measure])
            bin_acc = df.groupby("bin", observed=False)["correct"].agg(
                ["mean", "count"]
            ).reset_index()
            bin_acc["se"] = df.groupby("bin", observed=False)["correct"].apply(
                lambda x: _se_prop(x.values)
            ).values

            x = np.arange(len(bin_acc))
            ax.bar(x, bin_acc["mean"], color="#4dac26", alpha=0.85, width=0.6)
            ax.errorbar(
                x, bin_acc["mean"], yerr=1.96 * bin_acc["se"],
                fmt="none", color="black", capsize=3, linewidth=1,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(bin_acc["bin"].astype(str), fontsize=7)
            ax.set_title(MEASURE_LABELS[measure], fontsize=9)
            ax.set_xlabel("Quartile")
            if ax is axes[0]:
                ax.set_ylabel("Accuracy (pooled)")

        fig.suptitle("Accuracy by disruption quartile — all measures (pooled)", y=1.02)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    print(f"[Fig] Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Correlate Qwen plausibility disruption with FreqBLiMP accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--qwen-dir",
        default="results/plausibility_scores",
        help="Directory with *_plausibility_scores.jsonl files. Default: results/plausibility_scores",
    )
    ap.add_argument(
        "--pair-scores-dir",
        default="results/blimp_pair_scores",
        help="Directory with *_rare_pair-scores.jsonl files. Default: results/blimp_pair_scores",
    )
    ap.add_argument(
        "--out-dir",
        default="results/plausibility_analysis",
        help="Output directory for figures and tables. Default: results/plausibility_analysis",
    )
    ap.add_argument(
        "--no-regime-breakdown",
        action="store_true",
        help="Skip the per-regime figure (saves time if regimes are all similar).",
    )
    ap.add_argument(
        "--pooled-in-main-fig",
        action="store_true",
        help="Add a pooled panel to the main binned accuracy figure.",
    )
    args = ap.parse_args()

    qwen_dir        = Path(args.qwen_dir)
    pair_scores_dir = Path(args.pair_scores_dir)
    out_dir         = Path(args.out_dir)
    fig_dir         = out_dir / "figures"

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading data")
    print("=" * 60)
    qwen_df = load_qwen_scores(qwen_dir)
    pair_df = load_pair_scores(pair_scores_dir)
    merged  = merge_scores(qwen_df, pair_df)

    if merged.empty:
        sys.exit("[Error] No items matched between Qwen scores and pair scores. "
                 "Check that regime/phenomenon/subtask/idx keys align.")

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Summary statistics")
    print("=" * 60)
    summary = compute_summary_stats(merged)
    print(summary[["regime", "model", "n", "accuracy", "delta_pair_mean", "delta_pair_std"]].to_string(index=False))
    print()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary_stats.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[Saved] {summary_path}\n")

    # ------------------------------------------------------------------
    # Regression analysis
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Regression analysis")
    print("=" * 60)
    regression_results = run_regressions(merged)

    if regression_results:
        reg_df = _results_to_df(regression_results)
        reg_path = out_dir / "regression_results.csv"
        reg_df.to_csv(reg_path, index=False)
        print(f"[Saved] {reg_path}")

        # Print univariate delta_pair results
        print("\n--- Univariate: correct ~ delta_pair (per model) ---")
        uni = reg_df[
            (reg_df["scope"] == "per_model_univariate") &
            (reg_df["measure"] == "delta_pair")
        ].sort_values("model")
        print(uni[["model", "coef", "OR", "pval", "pseudo_r2", "aic", "n"]].to_string(index=False))

        # LaTeX table
        latex = _make_latex_table(reg_df)
        tex_path = out_dir / "regression_table.tex"
        tex_path.write_text(latex, encoding="utf-8")
        print(f"\n[Saved] {tex_path}\n")
    else:
        reg_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Generating figures")
    print("=" * 60)

    # Main figure: binned accuracy by delta_pair (one panel per model)
    fig_binned_accuracy(
        merged, "delta_pair",
        fig_dir / "binned_accuracy_delta_pair.pdf",
        pooled=args.pooled_in_main_fig,
    )

    # All measures grid (supplementary)
    fig_all_measures_grid(merged, fig_dir / "binned_accuracy_all_measures.pdf")

    # Per-model sensitivity (effect size comparison)
    if not reg_df.empty:
        fig_sensitivity_by_model(reg_df, fig_dir / "sensitivity_by_model.pdf")

    # Regime breakdown (skip if all regimes look similar or only 1 regime present)
    if not args.no_regime_breakdown and merged["regime"].nunique() > 1:
        fig_regime_breakdown(
            merged, "delta_pair",
            fig_dir / "regime_breakdown.pdf",
        )
    elif merged["regime"].nunique() == 1:
        print(
            "[Info] Only one regime in data — skipping regime breakdown figure.\n"
            "       Run with data from all three regimes for a breakdown."
        )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Analysis complete")
    print("=" * 60)
    print(f"Outputs in: {out_dir}")
    print(f"  figures/          — PDF plots")
    print(f"  summary_stats.csv — descriptive stats")
    if regression_results:
        print(f"  regression_results.csv — full regression table")
        print(f"  regression_table.tex   — LaTeX table for paper")


if __name__ == "__main__":
    main()
