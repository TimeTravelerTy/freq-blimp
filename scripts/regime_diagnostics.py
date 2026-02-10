import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_MPL_DIR = Path("results/analysis_plots/.mplconfig")
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAIR_SCORES_DIR = Path("results/blimp_pair_scores")
DATA_DIR = Path("data/processed")

_PAIR_RE = re.compile(
    r"(?:^|/)\d{8}-\d{6}_(?P<model>.+?)_(?P<dataset>\d{8}-\d{6}_.+?)_pair-scores\.jsonl$"
)
_ZIPF_WINDOW_RE = re.compile(r"zipf(?P<low>\d+_\d+)-(?P<high>\d+_\d+)")
_ZIPF_SINGLE_RE = re.compile(r"zipf(?P<value>\d+_\d+)(?:_|$)")

_WINDOW_TO_GROUP = {
    "4.0-5.2": "head",
    "3.6-5.0": "head",
    "2.2-3.0": "tail",
    "1.2-2.0": "xtail",
}
_GROUP_ORDER = ["original", "head", "tail", "xtail"]


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


def _group_label(dataset_base: str) -> Optional[str]:
    m = _ZIPF_WINDOW_RE.search(dataset_base)
    if m:
        low = m.group("low").replace("_", ".")
        high = m.group("high").replace("_", ".")
        window = f"{low}-{high}"
        return _WINDOW_TO_GROUP.get(window)

    m = _ZIPF_SINGLE_RE.search(dataset_base)
    if not m:
        return None
    value = float(m.group("value").replace("_", "."))
    if 3.6 <= value <= 5.2:
        return "head"
    if 2.2 <= value <= 3.0:
        return "tail"
    if 1.2 <= value <= 2.0:
        return "xtail"
    return None


def _load_dataset_meta(dataset_base: str, cache: Dict[str, Dict[str, dict]]) -> Dict[str, dict]:
    if dataset_base not in cache:
        path = DATA_DIR / f"{dataset_base}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        meta_by_idx: Dict[str, dict] = {}
        with path.open() as f:
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


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _median_zipf(meta: dict, field: str) -> Optional[float]:
    aggs = (meta or {}).get("zipf_swapped_position_aggregates") or {}
    return (aggs.get(field) or {}).get("median")


def _swapped_median_zipf(meta: dict, good_field: str, bad_field: str) -> Optional[float]:
    vals = [_median_zipf(meta, good_field), _median_zipf(meta, bad_field)]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _collect_swapped_lemmas(meta: dict) -> List[str]:
    lemmas: List[str] = []
    for key in ("g_swaps", "b_swaps", "g_verb_swaps", "b_verb_swaps", "g_adj_swaps", "b_adj_swaps"):
        swaps = meta.get(key) or []
        for item in swaps:
            lemma = item.get("lemma")
            if lemma:
                lemmas.append(str(lemma))
    return lemmas


def _format_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]

    def _fmt(row: List[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)
    out = [_fmt(rows[0]), sep]
    out.extend(_fmt(r) for r in rows[1:])
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Regime diagnostics for freq BLiMP datasets.")
    ap.add_argument("--pattern", default="results/blimp_pair_scores/*.jsonl")
    ap.add_argument("--out", default="results/analysis_plots/regime_diagnostics.png")
    ap.add_argument("--out-pdf", default=None, help="Optional PDF output path.")
    ap.add_argument(
        "--tables-out",
        default="results/analysis_plots/regime_diagnostics_tables.txt",
        help="Output path for diagnostic tables.",
    )
    ap.add_argument(
        "--dataset-contains",
        default=None,
        help="Optional substring that dataset name must contain.",
    )
    args = ap.parse_args()

    DEFAULT_MPL_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(Path().glob(args.pattern))
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise SystemExit(f"No pair-score files found for {args.pattern}")

    dataset_cache: Dict[str, Dict[str, dict]] = {}
    zipf_by_regime: Dict[str, List[float]] = defaultdict(list)
    lemma_counts: Dict[str, Counter] = defaultdict(Counter)
    token_len_stats: Dict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)
    zipf_imbalance: Dict[str, List[float]] = defaultdict(list)
    swap_failures: Dict[str, Counter] = defaultdict(Counter)
    swap_totals: Dict[str, int] = defaultdict(int)

    for path in paths:
        try:
            model, dataset_base, variant_hint = _parse_pair_scores_path(path)
        except ValueError:
            continue
        if variant_hint == "original":
            regime = "original"
        elif variant_hint == "freq":
            regime = _group_label(dataset_base)
        else:
            continue
        if args.dataset_contains and args.dataset_contains not in dataset_base:
            continue
        if regime not in _GROUP_ORDER:
            continue
        meta_by_idx = _load_dataset_meta(dataset_base, dataset_cache)

        for rec in _iter_jsonl(path):
            meta = meta_by_idx.get(rec.get("idx")) or {}
            good_field = rec.get("good_field")
            bad_field = rec.get("bad_field")
            if good_field and bad_field:
                zipf_med = _swapped_median_zipf(meta, good_field, bad_field)
                if zipf_med is not None:
                    zipf_by_regime[regime].append(zipf_med)
                good_zipf = _median_zipf(meta, good_field)
                bad_zipf = _median_zipf(meta, bad_field)
                if good_zipf is not None and bad_zipf is not None:
                    zipf_imbalance[regime].append(float(good_zipf - bad_zipf))

            good_len = rec.get("good_token_count")
            bad_len = rec.get("bad_token_count")
            if isinstance(good_len, int) and isinstance(bad_len, int):
                token_len_stats[(model, regime)].append((good_len, bad_len))

            lemmas = _collect_swapped_lemmas(meta)
            if lemmas:
                lemma_counts[regime].update(lemmas)

            reason = meta.get("swap_failed_reason")
            swap_totals[regime] += 1
            if reason:
                swap_failures[regime][str(reason)] += 1

    # Plot: histogram/KDE of median zipf + top-20 lemma share.
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.2))
    ax_hist, ax_bar = axes

    bins = 20
    for regime in _GROUP_ORDER:
        vals = zipf_by_regime.get(regime, [])
        if not vals:
            continue
        ax_hist.hist(vals, bins=bins, density=True, alpha=0.4, label=regime)
    # ax_hist.set_title("Realised median Zipf per regime")
    ax_hist.set_xlabel("Realised median Zipf")
    ax_hist.set_ylabel("Density")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    ax_hist.invert_xaxis()

    shares = []
    for regime in _GROUP_ORDER:
        counts = lemma_counts.get(regime, Counter())
        total = sum(counts.values())
        if total == 0:
            shares.append(0.0)
            continue
        top20 = sum(c for _, c in counts.most_common(20))
        shares.append(top20 / total)
    ax_bar.bar(_GROUP_ORDER, shares, color="#4C78A8")
    max_share = max(shares) if shares else 0.0
    ax_bar.set_ylim(0, min(1.0, max_share * 1.25 if max_share > 0 else 0.25))
    # ax_bar.set_title("Top-20 lemma share by regime")
    ax_bar.set_ylabel("Top-20 lemma share")
    ax_bar.grid(True, axis="y", alpha=0.3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    if args.out_pdf:
        fig.savefig(Path(args.out_pdf))

    # Tables.
    tables: List[str] = []

    # Token length table.
    rows = [["Model", "Regime", "Good avg", "Bad avg", "N"]]
    for (model, regime), vals in sorted(token_len_stats.items()):
        goods = [v[0] for v in vals]
        bads = [v[1] for v in vals]
        rows.append(
            [
                model,
                regime,
                f"{np.mean(goods):.2f}",
                f"{np.mean(bads):.2f}",
                str(len(vals)),
            ]
        )
    tables.append("Avg token length per model/regime\n" + _format_table(rows))

    # Zipf imbalance table.
    rows = [["Regime", "Mean(good-bad)", "Mean|good-bad|", "N"]]
    for regime in _GROUP_ORDER:
        vals = zipf_imbalance.get(regime, [])
        if not vals:
            rows.append([regime, "-", "-", "0"])
            continue
        rows.append(
            [
                regime,
                f"{np.mean(vals):.3f}",
                f"{np.mean(np.abs(vals)):.3f}",
                str(len(vals)),
            ]
        )
    tables.append("Realised Zipf imbalance per regime\n" + _format_table(rows))

    # Swap success table.
    rows = [["Regime", "Success rate", "Top failure reasons"]]
    for regime in _GROUP_ORDER:
        total = swap_totals.get(regime, 0)
        failures = sum(swap_failures.get(regime, Counter()).values())
        success_rate = (total - failures) / total if total else 0.0
        top_reasons = ", ".join(r for r, _ in swap_failures.get(regime, Counter()).most_common(3))
        rows.append([regime, f"{success_rate:.3f}", top_reasons or "-"])
    tables.append("Swap success rate and top failure reasons\n" + _format_table(rows))

    tables_out = Path(args.tables_out)
    tables_out.parent.mkdir(parents=True, exist_ok=True)
    tables_out.write_text("\n\n".join(tables) + "\n")

    print(f"Saved plot to {out_path}")
    if args.out_pdf:
        print(f"Saved PDF to {args.out_pdf}")
    print(f"Saved tables to {tables_out}")


if __name__ == "__main__":
    main()
