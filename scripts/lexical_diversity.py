"""
Lexical diversity analysis for FreqBLiMP datasets.

Computes per-sentence-type and per-phenomenon/field breakdowns of:
  - Token count, vocabulary size, unique lemmas
  - Type-Token Ratio (TTR) and Moving-Average TTR (MATTR)
  - Zipf frequency profile (mean, median, % rare, % OOV)
  - Swap lemma pool diversity (for generated datasets)

Usage:
  python scripts/lexical_diversity.py data/processed/blimp_original.jsonl
  python scripts/lexical_diversity.py data/processed/*.jsonl --compare
  python scripts/lexical_diversity.py data/processed/my_dataset.jsonl --by phenomenon
  python scripts/lexical_diversity.py data/processed/my_dataset.jsonl --csv results/lex_diversity.csv
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import spacy
from wordfreq import zipf_frequency

# ---------------------------------------------------------------------------
# Sentence type columns present in each record type
# ---------------------------------------------------------------------------
ORIGINAL_SENT_KEYS = ["good_original", "bad_original"]
GENERATED_SENT_KEYS = [
    "good_original", "bad_original",
    "good_rare", "bad_rare",
    "good_freq", "bad_freq",
]

RARE_ZIPF_THRESHOLD = 3.0  # words with Zipf < this are "rare"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _iter_jsonl(path: str, limit: Optional[int] = None) -> Iterable[dict]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _detect_sent_keys(records: List[dict]) -> List[str]:
    """Return the sentence keys that actually appear in this dataset."""
    if not records:
        return ORIGINAL_SENT_KEYS
    first = records[0]
    return [k for k in GENERATED_SENT_KEYS if k in first and first[k]]


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def _tokenize(nlp, text: str) -> Tuple[List[str], List[str]]:
    """Return (tokens, lemmas) for a sentence, lowercased, alpha only."""
    doc = nlp(text)
    tokens, lemmas = [], []
    for tok in doc:
        if tok.is_alpha:
            tokens.append(tok.text.lower())
            lemmas.append(tok.lemma_.lower())
    return tokens, lemmas


# ---------------------------------------------------------------------------
# Lexical diversity metrics
# ---------------------------------------------------------------------------

def ttr(tokens: List[str]) -> float:
    if not tokens:
        return float("nan")
    return len(set(tokens)) / len(tokens)


def mattr(tokens: List[str], window: int = 100) -> float:
    """Moving-Average Type-Token Ratio (Covington & McFall, 2010).

    Averages TTR across all sliding windows of `window` tokens.
    More robust to text length than plain TTR.
    Returns NaN if fewer than `window` tokens.
    """
    n = len(tokens)
    if n < window:
        # Fall back to plain TTR for short texts
        return ttr(tokens)
    ratios = []
    for i in range(n - window + 1):
        w = tokens[i : i + window]
        ratios.append(len(set(w)) / window)
    return sum(ratios) / len(ratios)


def yule_k(tokens: List[str]) -> float:
    """Yule's K — lower = more diverse vocabulary."""
    if not tokens:
        return float("nan")
    freq = Counter(tokens)
    freq_of_freq = Counter(freq.values())
    N = len(tokens)
    S2 = sum(v * v * count for v, count in freq_of_freq.items())
    return 10_000 * (S2 - N) / (N * N)


# ---------------------------------------------------------------------------
# Zipf / frequency profile
# ---------------------------------------------------------------------------

def zipf_profile(tokens: List[str], lang: str = "en") -> dict:
    """Compute aggregate Zipf statistics over a token list."""
    scores = [zipf_frequency(t, lang) for t in tokens]
    n = len(scores)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "pct_rare": float("nan"), "pct_oov": float("nan")}
    scores_sorted = sorted(scores)
    mid = n // 2
    median = (scores_sorted[mid - 1] + scores_sorted[mid]) / 2 if n % 2 == 0 else scores_sorted[mid]
    oov_count = sum(1 for s in scores if s == 0.0)
    rare_count = sum(1 for s in scores if s < RARE_ZIPF_THRESHOLD)
    return {
        "n": n,
        "mean": sum(scores) / n,
        "median": median,
        "pct_rare": rare_count / n,
        "pct_oov": oov_count / n,
    }


# ---------------------------------------------------------------------------
# Per-corpus statistics accumulator
# ---------------------------------------------------------------------------

class CorpusStats:
    """Accumulates tokens/lemmas across many sentences."""

    def __init__(self, name: str):
        self.name = name
        self.all_tokens: List[str] = []
        self.all_lemmas: List[str] = []
        self.sentence_count = 0

    def add(self, tokens: List[str], lemmas: List[str]):
        self.all_tokens.extend(tokens)
        self.all_lemmas.extend(lemmas)
        self.sentence_count += 1

    def summarise(self, mattr_window: int = 100) -> dict:
        tok = self.all_tokens
        lem = self.all_lemmas
        n_tok = len(tok)
        n_types = len(set(tok))
        n_lemmas = len(set(lem))
        return {
            "name": self.name,
            "sentences": self.sentence_count,
            "tokens": n_tok,
            "types": n_types,
            "unique_lemmas": n_lemmas,
            "ttr": ttr(tok),
            "mattr": mattr(tok, mattr_window),
            "yule_k": yule_k(tok),
            **zipf_profile(tok),
        }


# ---------------------------------------------------------------------------
# Swap pool analysis (generated datasets only)
# ---------------------------------------------------------------------------

def _collect_swap_lemmas(records: List[dict]) -> Dict[str, List[str]]:
    """Extract lemmas of swapped words, grouped by POS."""
    pool: Dict[str, List[str]] = defaultdict(list)
    for rec in records:
        meta = rec.get("meta") or {}
        for swap_key in ("g_swaps", "b_swaps"):
            for swap in meta.get(swap_key) or []:
                pos = swap.get("pos", "unknown")
                lemma = swap.get("lemma") or swap.get("new")
                if lemma:
                    pool[pos].append(lemma.lower())
    return pool


def _swap_pool_stats(pool: Dict[str, List[str]]) -> dict:
    stats = {}
    for pos, lemmas in pool.items():
        stats[pos] = {
            "total_swaps": len(lemmas),
            "unique_lemmas": len(set(lemmas)),
            "diversity_ratio": len(set(lemmas)) / len(lemmas) if lemmas else float("nan"),
            "top10": [w for w, _ in Counter(lemmas).most_common(10)],
        }
    return stats


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_W = 16  # column width


def _fmt(v) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "   n/a"
        return f"{v:6.4f}"
    if isinstance(v, int):
        return f"{v:>6}"
    return str(v)


def _print_summary_table(summaries: List[dict], title: str = ""):
    COLS = [
        ("sentences",    "sents"),
        ("tokens",       "tokens"),
        ("types",        "types"),
        ("unique_lemmas","lemmas"),
        ("ttr",          "TTR"),
        ("mattr",        "MATTR"),
        ("yule_k",       "Yule-K"),
        ("mean",         "Zipf-mean"),
        ("median",       "Zipf-med"),
        ("pct_rare",     "pct_rare"),
        ("pct_oov",      "pct_oov"),
    ]
    col_w = _W
    if title:
        print(f"\n{'='*80}\n{title}\n{'='*80}")
    header = f"{'sent_type':<{col_w}}" + "".join(f"{h:>12}" for _, h in COLS)
    print(header)
    print("-" * len(header))
    for s in summaries:
        row = f"{s['name']:<{col_w}}" + "".join(f"{_fmt(s.get(k, float('nan'))):>12}" for k, _ in COLS)
        print(row)


# ---------------------------------------------------------------------------
# Main analysis routines
# ---------------------------------------------------------------------------

def analyse_dataset(
    path: str,
    nlp,
    sent_keys: Optional[List[str]] = None,
    mattr_window: int = 100,
    group_by: Optional[str] = None,   # "phenomenon" | "field" | None
    limit: Optional[int] = None,
) -> Tuple[List[dict], dict]:
    """
    Analyse a single JSONL dataset.

    Returns:
        (corpus_summaries, swap_stats)
    """
    records = list(_iter_jsonl(path, limit))
    if not records:
        print(f"[warn] No records found in {path}")
        return [], {}

    if sent_keys is None:
        sent_keys = _detect_sent_keys(records)

    # ---- Overall corpus stats per sentence type ----
    corpora: Dict[str, CorpusStats] = {k: CorpusStats(k) for k in sent_keys}

    for rec in records:
        for key in sent_keys:
            text = rec.get(key)
            if not text:
                continue
            tokens, lemmas = _tokenize(nlp, text)
            corpora[key].add(tokens, lemmas)

    summaries = [corpora[k].summarise(mattr_window) for k in sent_keys]

    # ---- Swap pool stats ----
    swap_stats = _swap_pool_stats(_collect_swap_lemmas(records))

    # ---- Optional group-by breakdown ----
    if group_by:
        _print_group_breakdown(records, nlp, sent_keys, group_by, mattr_window)

    return summaries, swap_stats


def _print_group_breakdown(
    records: List[dict],
    nlp,
    sent_keys: List[str],
    group_by: str,
    mattr_window: int,
):
    groups: Dict[str, Dict[str, CorpusStats]] = defaultdict(
        lambda: {k: CorpusStats(k) for k in sent_keys}
    )
    for rec in records:
        gval = rec.get(group_by) or "unknown"
        for key in sent_keys:
            text = rec.get(key)
            if text:
                tokens, lemmas = _tokenize(nlp, text)
                groups[gval][key].add(tokens, lemmas)

    for gval, corpora in sorted(groups.items()):
        sums = [corpora[k].summarise(mattr_window) for k in sent_keys]
        _print_summary_table(sums, title=f"{group_by} = {gval}")


# ---------------------------------------------------------------------------
# Cross-dataset comparison
# ---------------------------------------------------------------------------

def compare_datasets(paths: List[str], nlp, mattr_window: int = 100, limit: Optional[int] = None):
    """Print a side-by-side comparison across multiple dataset files."""
    COLS = [
        ("tokens",       "tokens"),
        ("types",        "types"),
        ("unique_lemmas","lemmas"),
        ("ttr",          "TTR"),
        ("mattr",        "MATTR"),
        ("mean",         "Zipf-mean"),
        ("pct_rare",     "pct_rare"),
    ]
    col_w = 22
    # Prefer good_original; fall back to good_freq or good_rare for generated files
    GOOD_SENT_CANDIDATES = ["good_original", "good_freq", "good_rare"]

    print(f"\n{'='*80}\nCross-dataset comparison (good sentences)\n{'='*80}")
    header = f"{'dataset':<{col_w}}" + "".join(f"{h:>12}" for _, h in COLS) + f"  {'sent_key'}"
    print(header)
    print("-" * len(header))

    for path in paths:
        records = list(_iter_jsonl(path, limit))
        if not records:
            continue
        # Detect which good-sentence key is present
        sent_key = next(
            (k for k in GOOD_SENT_CANDIDATES if records[0].get(k)),
            None,
        )
        corpus = CorpusStats(Path(path).name)
        if sent_key:
            for rec in records:
                text = rec.get(sent_key)
                if text:
                    tokens, lemmas = _tokenize(nlp, text)
                    corpus.add(tokens, lemmas)
        s = corpus.summarise(mattr_window)
        label = Path(path).stem[:col_w - 1]
        row = (
            f"{label:<{col_w}}"
            + "".join(f"{_fmt(s.get(k, float('nan'))):>12}" for k, _ in COLS)
            + f"  {sent_key or 'none'}"
        )
        print(row)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _summaries_to_csv(summaries: List[dict], path: str):
    if not summaries:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"[csv] Written to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Lexical diversity analysis for BLiMP-RARE datasets."
    )
    p.add_argument("datasets", nargs="+", help="One or more .jsonl dataset paths.")
    p.add_argument(
        "--compare", action="store_true",
        help="Side-by-side comparison of multiple datasets instead of per-dataset detail."
    )
    p.add_argument(
        "--by", choices=["phenomenon", "field"],
        help="Break down results by phenomenon or field."
    )
    p.add_argument(
        "--mattr-window", type=int, default=100,
        help="Window size for MATTR (default: 100)."
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of records to read per file."
    )
    p.add_argument(
        "--csv", metavar="PATH",
        help="Export summary table to CSV."
    )
    p.add_argument(
        "--no-swap-stats", action="store_true",
        help="Skip swap pool analysis."
    )
    return p


def main():
    args = _build_parser().parse_args()

    print("Loading spaCy model (en_core_web_sm)...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    if args.compare:
        compare_datasets(args.datasets, nlp, args.mattr_window, args.limit)
        return

    all_summaries = []
    for path in args.datasets:
        print(f"\nAnalysing: {path}")
        summaries, swap_stats = analyse_dataset(
            path,
            nlp,
            sent_keys=None,
            mattr_window=args.mattr_window,
            group_by=args.by,
            limit=args.limit,
        )
        _print_summary_table(summaries, title=Path(path).name)
        all_summaries.extend(summaries)

        if swap_stats and not args.no_swap_stats:
            print("\n  -- Swap pool diversity --")
            for pos, st in swap_stats.items():
                print(
                    f"  {pos:<10} total={st['total_swaps']:>5}  "
                    f"unique_lemmas={st['unique_lemmas']:>5}  "
                    f"diversity={st['diversity_ratio']:.4f}  "
                    f"top10: {', '.join(st['top10'])}"
                )

    if args.csv and all_summaries:
        _summaries_to_csv(all_summaries, args.csv)


if __name__ == "__main__":
    main()
