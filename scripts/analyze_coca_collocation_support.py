#!/usr/bin/env python3
"""Estimate external local-collocation support for BLiMP/FreqBLiMP items.

The main analysis is deliberately target-driven: extract the lemma pairs that
occur in the benchmark, then stream COCA WLP once and count only those pairs.
This avoids building a global COCA collocation index.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "results/coca_collocation_support/.mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from wordfreq import zipf_frequency


SCRIPT_DIR = Path(__file__).resolve().parent
BLIMP_RARE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = BLIMP_RARE_ROOT.parent
FREQ_BLIMP_ROOT = WORKSPACE_ROOT / "freq-blimp"

DEFAULT_ORIGINAL_DATA = BLIMP_RARE_ROOT / "data/processed/blimp_original.jsonl"
DEFAULT_FREQ_DATA_ROOT = FREQ_BLIMP_ROOT / "outputs/full_3regimes_current"
DEFAULT_CURRENT_SCORE_GLOB = (
    FREQ_BLIMP_ROOT
    / "outputs/blimp_rare_results/acceptability_pair_scores/20260429_drop_argument_diverse_full/*_nll_acceptability.jsonl"
)
DEFAULT_ORIGINAL_SCORE_GLOB = (
    BLIMP_RARE_ROOT
    / "results/acceptability_pair_scores/20260408-*blimp_original_original_nll_acceptability.jsonl"
)
DEFAULT_COCA_WLP = Path("/Users/tyronewhite/Downloads/COCA/coca-wlp.tar")
DEFAULT_OUT_DIR = BLIMP_RARE_ROOT / "results/coca_collocation_support"

REGIMES = ("original", "head", "tail", "xtail")
FREQ_REGIMES = ("head", "tail", "xtail")
SIDES = ("good", "bad")

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
WORD_RE = re.compile(r"^[a-z][a-z'-]*$")

STOP_LEMMAS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
    "be",
    "do",
    "have",
    "will",
    "would",
    "can",
    "could",
    "should",
    "may",
    "might",
    "must",
    "not",
    "no",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "from",
    "as",
    "than",
    "and",
    "or",
    "but",
    "if",
    "because",
    "while",
    "before",
    "after",
    "so",
    "very",
    "just",
    "only",
    "there",
    "here",
    "what",
    "who",
    "whom",
    "whose",
    "which",
    "where",
    "when",
    "why",
    "how",
}

SPACY_CONTENT_POS = {"VERB", "NOUN", "ADJ", "ADV"}
COCA_CONTENT_PREFIXES = ("vv", "nn", "jj", "rr")


def _norm_lemma(value: str) -> str:
    lemma = str(value or "").strip().lower()
    lemma = lemma.replace("_", " ")
    if " " in lemma:
        lemma = lemma.split()[0]
    return lemma


def _valid_lemma(lemma: str) -> bool:
    return bool(lemma) and lemma not in STOP_LEMMAS and WORD_RE.match(lemma) is not None


def _pair_key(a: str, b: str) -> Optional[Tuple[str, str]]:
    a = _norm_lemma(a)
    b = _norm_lemma(b)
    if a == b or not _valid_lemma(a) or not _valid_lemma(b):
        return None
    return tuple(sorted((a, b)))


def _model_short(model: str) -> str:
    return (model or "unknown").split("/")[-1].replace(".", "_")


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for row_i, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["_row_i"] = row_i
            yield rec


def _collect_paths(values: Sequence[str]) -> List[Path]:
    paths = set()
    for value in values:
        if any(ch in str(value) for ch in "*?[]"):
            paths.update(Path(p) for p in glob.glob(str(value)))
        else:
            p = Path(value)
            if p.is_dir():
                paths.update(p.rglob("*.jsonl"))
            elif p.is_file():
                paths.add(p)
    return sorted(p.resolve() for p in paths if p.is_file())


def _infer_regime(rec: dict, path: Path) -> str:
    text = " ".join(str(part) for part in (path.name, rec.get("dataset_path", ""), rec.get("dataset_name", "")))
    for regime in FREQ_REGIMES:
        if re.search(rf"(^|[/_-]){regime}($|[/_.-])", text):
            return regime
    if rec.get("variant") == "original":
        return "original"
    return "unknown"


def _canonical_metadata(original_data: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for rec in _iter_jsonl(original_data):
        uid = rec.get("subtask")
        if isinstance(uid, str) and uid not in out:
            out[uid] = {
                "field": rec.get("field", ""),
                "phenomenon": rec.get("phenomenon", ""),
            }
    return out


def _load_items(
    original_data: Path,
    freq_data_root: Path,
    max_items_per_regime: Optional[int] = None,
    uid_filter: Optional[set] = None,
) -> pd.DataFrame:
    metadata = _canonical_metadata(original_data)
    rows: List[dict] = []
    regime_counts: Counter = Counter()

    for rec in _iter_jsonl(original_data):
        if max_items_per_regime is not None and regime_counts["original"] >= max_items_per_regime:
            continue
        uid = str(rec.get("subtask"))
        if uid_filter is not None and uid not in uid_filter:
            continue
        pair_id = str(rec.get("idx"))
        rows.append(
            {
                "regime": "original",
                "uid": uid,
                "pair_id": pair_id,
                "field": rec.get("field", ""),
                "phenomenon": rec.get("phenomenon", ""),
                "good_text": rec.get("good_original", ""),
                "bad_text": rec.get("bad_original", ""),
                "source_path": str(original_data),
            }
        )
        regime_counts["original"] += 1

    for regime in FREQ_REGIMES:
        for path in sorted((freq_data_root / regime).glob("*.jsonl")):
            uid = path.stem
            if uid_filter is not None and uid not in uid_filter:
                continue
            meta = metadata.get(uid, {})
            for rec in _iter_jsonl(path):
                if max_items_per_regime is not None and regime_counts[regime] >= max_items_per_regime:
                    break
                pair_id = str(rec.get("pairID", rec["_row_i"]))
                rows.append(
                    {
                        "regime": regime,
                        "uid": uid,
                        "pair_id": pair_id,
                        "field": meta.get("field", rec.get("field", "")),
                        "phenomenon": meta.get("phenomenon", rec.get("linguistics_term", "")),
                        "good_text": rec.get("sentence_good", ""),
                        "bad_text": rec.get("sentence_bad", ""),
                        "source_path": str(path),
                    }
                )
                regime_counts[regime] += 1
            if max_items_per_regime is not None and regime_counts[regime] >= max_items_per_regime:
                break
    return pd.DataFrame(rows)


def _spacy_content(tok) -> bool:
    if tok.pos_ not in SPACY_CONTENT_POS:
        return False
    if tok.pos_ == "PROPN" or tok.ent_type_ == "PERSON":
        return False
    lemma = _norm_lemma(tok.lemma_)
    return _valid_lemma(lemma)


def _lemma(tok) -> str:
    return _norm_lemma(tok.lemma_)


def _add_probe(
    probes: Dict[Tuple[str, str, str], dict],
    item: dict,
    side: str,
    relation: str,
    a: str,
    b: str,
) -> None:
    key = _pair_key(a, b)
    if key is None:
        return
    probe_key = (relation, key[0], key[1])
    probes[probe_key] = {
        "regime": item["regime"],
        "uid": item["uid"],
        "pair_id": item["pair_id"],
        "side": side,
        "relation": relation,
        "lemma1": key[0],
        "lemma2": key[1],
        "sentence_text": item[f"{side}_text"],
    }


def _extract_doc_probes(doc, item: dict, side: str) -> List[dict]:
    probes: Dict[Tuple[str, str, str], dict] = {}
    content = [tok for tok in doc if _spacy_content(tok)]
    content_set = set(content)

    for tok in content:
        head = tok.head
        if head not in content_set or tok.i == head.i:
            continue
        dep = tok.dep_.lower()
        if dep in {"nsubj", "nsubjpass", "csubj"} and head.pos_ == "VERB":
            relation = "verb_subject"
        elif dep in {"dobj", "obj", "iobj", "dative"} and head.pos_ == "VERB":
            relation = "verb_object"
        elif dep in {"amod", "acomp"} and (head.pos_ == "NOUN" or tok.pos_ == "ADJ"):
            relation = "adjective_noun"
        elif dep == "compound" and head.pos_ == "NOUN":
            relation = "compound_noun"
        elif dep in {"advmod"} and head.pos_ == "VERB":
            relation = "verb_adverb"
        else:
            relation = "generic_dependency"
        _add_probe(probes, item, side, relation, _lemma(tok), _lemma(head))

    for prep in doc:
        if prep.dep_.lower() != "prep" or prep.head not in content_set:
            continue
        for pobj in prep.children:
            if pobj.dep_.lower() in {"pobj", "pcomp"} and pobj in content_set:
                _add_probe(probes, item, side, "prep_object", _lemma(prep.head), _lemma(pobj))

    # Fallback: adjacent content pairs catch collocations that dependency
    # parsing misses in short artificial benchmark sentences.
    ordered = sorted(content, key=lambda tok: tok.i)
    for left, right in zip(ordered, ordered[1:]):
        if right.i - left.i <= 3:
            _add_probe(probes, item, side, "adjacent_content", _lemma(left), _lemma(right))

    return list(probes.values())


def extract_benchmark_probes(items: pd.DataFrame, nlp, batch_size: int) -> pd.DataFrame:
    item_dicts = items.to_dict("records")
    texts: List[str] = []
    contexts: List[Tuple[dict, str]] = []
    for item in item_dicts:
        for side in SIDES:
            texts.append(str(item[f"{side}_text"]))
            contexts.append((item, side))

    rows: List[dict] = []
    start = time.time()
    next_log = start + 10.0
    for doc_i, (doc, (item, side)) in enumerate(zip(nlp.pipe(texts, batch_size=batch_size), contexts), start=1):
        rows.extend(_extract_doc_probes(doc, item, side))
        now = time.time()
        if now >= next_log:
            elapsed = max(now - start, 0.001)
            print(f"\rspaCy sentences {doc_i:,}/{len(texts):,} | {doc_i / elapsed:.1f}/s", end="", flush=True)
            next_log = now + 10.0
    if texts:
        print()
    return pd.DataFrame(rows)


def _coca_is_content(lemma: str, pos: str) -> bool:
    lemma = _norm_lemma(lemma)
    pos = str(pos or "").strip().lower()
    if not _valid_lemma(lemma):
        return False
    if pos.startswith("np"):
        return False
    return pos.startswith(COCA_CONTENT_PREFIXES)


def _format_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds_i = int(seconds)
    hours, rem = divmod(seconds_i, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _iter_wlp_rows(tar_path: Path) -> Iterator[Tuple[str, str, str, str, float]]:
    with tarfile.open(tar_path) as tar:
        members = [member for member in tar.getmembers() if member.name.lower().endswith(".zip")]
        total_zip_bytes = sum(max(member.size, 1) for member in members)
        completed_zip_bytes = 0
        for member in members:
            with tar.extractfile(member) as fzip:
                if fzip is None:
                    completed_zip_bytes += max(member.size, 1)
                    continue
                with zipfile.ZipFile(fzip) as zf:
                    for fname in sorted(zf.namelist()):
                        with zf.open(fname) as handle:
                            for raw in handle:
                                parts = raw.decode("utf-8", "ignore").rstrip("\n").split("\t")
                                if len(parts) == 4:
                                    progress = completed_zip_bytes / total_zip_bytes if total_zip_bytes else 0.0
                                    yield parts[0], parts[1], parts[2], parts[3], progress
            completed_zip_bytes += max(member.size, 1)


def _sentence_break(surface: str, pos: str) -> bool:
    return (pos or "").lower().startswith("y") and surface in {".", "!", "?"}


def _scan_coca_wlp(
    tar_path: Path,
    target_pairs: set,
    target_lemmas: set,
    *,
    window: int,
    max_sentences: Optional[int],
    log_interval_sec: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    pair_token_counts: Counter = Counter()
    pair_doc_counts: Counter = Counter()
    lemma_token_counts: Counter = Counter()
    lemma_doc_counts: Counter = Counter()
    docs_seen = set()
    doc_seen_pairs = set()
    doc_seen_lemmas = set()
    sentence_tokens: List[Tuple[str, str]] = []
    current_doc = None
    processed_sentences = 0
    progress_fraction = 0.0
    start = time.time()
    next_log = start + log_interval_sec

    def flush_sentence() -> bool:
        nonlocal sentence_tokens, processed_sentences, next_log
        if not sentence_tokens:
            return True
        content = [(lemma, pos) for lemma, pos in sentence_tokens if _coca_is_content(lemma, pos)]
        for lemma, _pos in content:
            if lemma in target_lemmas:
                lemma_token_counts[lemma] += 1
                if lemma not in doc_seen_lemmas:
                    lemma_doc_counts[lemma] += 1
                    doc_seen_lemmas.add(lemma)
        for i, (lemma_i, _pos_i) in enumerate(content):
            for lemma_j, _pos_j in content[i + 1 : i + 1 + window]:
                key = _pair_key(lemma_i, lemma_j)
                if key is None or key not in target_pairs:
                    continue
                pair_token_counts[key] += 1
                if key not in doc_seen_pairs:
                    pair_doc_counts[key] += 1
                    doc_seen_pairs.add(key)
        processed_sentences += 1
        sentence_tokens = []
        now = time.time()
        if now >= next_log:
            elapsed = max(now - start, 0.001)
            if max_sentences is None and progress_fraction > 0:
                eta = _format_eta(elapsed * (1.0 - progress_fraction) / progress_fraction)
                progress_text = f" | {progress_fraction * 100:.1f}% | ETA {eta}"
            else:
                progress_text = ""
            print(
                f"\rCOCA sentences {processed_sentences:,} | {processed_sentences / elapsed:.1f}/s{progress_text}",
                end="",
                flush=True,
            )
            next_log = now + log_interval_sec
        if max_sentences is not None and processed_sentences >= max_sentences:
            return False
        return True

    for doc_id, surface, lemma, pos, progress_fraction in _iter_wlp_rows(tar_path):
        if current_doc is None:
            current_doc = doc_id
            docs_seen.add(doc_id)
        if doc_id != current_doc:
            if not flush_sentence():
                break
            current_doc = doc_id
            docs_seen.add(doc_id)
            doc_seen_pairs = set()
            doc_seen_lemmas = set()
        if surface == "<p>" or str(lemma).startswith("@@"):
            if not flush_sentence():
                break
            continue
        if _sentence_break(surface, pos):
            sentence_tokens.append((_norm_lemma(lemma), pos))
            if not flush_sentence():
                break
            continue
        sentence_tokens.append((_norm_lemma(lemma), pos))

    if max_sentences is None or processed_sentences < max_sentences:
        flush_sentence()
    if max_sentences is None:
        progress_fraction = 1.0
    print()

    pair_rows = [
        {
            "lemma1": pair[0],
            "lemma2": pair[1],
            "pair_token_count": pair_token_counts.get(pair, 0),
            "pair_doc_count": pair_doc_counts.get(pair, 0),
        }
        for pair in sorted(target_pairs)
    ]
    lemma_rows = [
        {
            "lemma": lemma,
            "lemma_token_count": lemma_token_counts.get(lemma, 0),
            "lemma_doc_count": lemma_doc_counts.get(lemma, 0),
        }
        for lemma in sorted(target_lemmas)
    ]
    meta = {
        "coca_sentences_scanned": processed_sentences,
        "coca_docs_seen": len(docs_seen),
        "coca_progress_fraction": progress_fraction,
        "window": window,
        "max_sentences": max_sentences,
    }
    return pd.DataFrame(pair_rows), pd.DataFrame(lemma_rows), meta


def _support_from_counts(probes: pd.DataFrame, pair_counts: pd.DataFrame, lemma_counts: pd.DataFrame, total_docs: int) -> pd.DataFrame:
    out = probes.merge(pair_counts, on=["lemma1", "lemma2"], how="left")
    out[["pair_token_count", "pair_doc_count"]] = out[["pair_token_count", "pair_doc_count"]].fillna(0)
    lemma_docs = dict(zip(lemma_counts["lemma"], lemma_counts["lemma_doc_count"]))
    alpha = 0.5
    denom = max(total_docs, 1)

    def pmi(row: pd.Series) -> float:
        c12 = float(row["pair_doc_count"])
        c1 = float(lemma_docs.get(row["lemma1"], 0))
        c2 = float(lemma_docs.get(row["lemma2"], 0))
        return math.log((c12 + alpha) / (denom + alpha)) - math.log((c1 + alpha) / (denom + alpha)) - math.log((c2 + alpha) / (denom + alpha))

    out["local_log_doc_count"] = np.log1p(out["pair_doc_count"].astype(float))
    out["local_log_token_count"] = np.log1p(out["pair_token_count"].astype(float))
    out["local_pair_attested"] = (out["pair_doc_count"].astype(float) > 0).astype(int)
    out["local_pmi_doc"] = out.apply(pmi, axis=1)
    return out


def _aggregate_side_support(probe_support: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["regime", "uid", "pair_id", "side"]
    agg = probe_support.groupby(group_cols, as_index=False).agg(
        probe_count=("relation", "size"),
        local_support_mean=("local_log_doc_count", "mean"),
        local_support_min=("local_log_doc_count", "min"),
        local_support_zero_rate=("local_pair_attested", lambda s: 1.0 - float(np.mean(s))),
        local_support_pmi_mean=("local_pmi_doc", "mean"),
    )
    return agg


def _pivot_item_support(side_support: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    wide = side_support.pivot_table(
        index=["regime", "uid", "pair_id"],
        columns="side",
        values=[
            "probe_count",
            "local_support_mean",
            "local_support_min",
            "local_support_zero_rate",
            "local_support_pmi_mean",
        ],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{side}" for metric, side in wide.columns]
    wide = wide.reset_index()
    base = items.copy()
    for side in SIDES:
        base[f"{side}_char_count"] = base[f"{side}_text"].astype(str).str.len()
        base[f"{side}_zipf_mean"] = base[f"{side}_text"].map(_zipf_mean)
    merged = base.merge(wide, on=["regime", "uid", "pair_id"], how="left")
    for metric in ("local_support_mean", "local_support_min", "local_support_zero_rate", "local_support_pmi_mean"):
        merged[f"{metric}_delta_good_minus_bad"] = merged.get(f"{metric}_good") - merged.get(f"{metric}_bad")
    return merged


def _zipf_mean(text: str) -> float:
    vals = []
    for tok in TOKEN_RE.findall(str(text).lower()):
        if tok in STOP_LEMMAS:
            continue
        z = zipf_frequency(tok, "en")
        if z > 0:
            vals.append(float(z))
    return float(np.mean(vals)) if vals else float("nan")


def _read_scores(paths: Sequence[Path]) -> pd.DataFrame:
    rows: List[dict] = []
    for path in paths:
        for rec in _iter_jsonl(path):
            if rec.get("method") != "nll":
                continue
            regime = _infer_regime(rec, path)
            uid = rec.get("subtask") or Path(str(rec.get("dataset_name") or rec.get("dataset_path"))).stem
            if regime == "original":
                pair_id = str(rec.get("idx"))
            else:
                pair_id = str(rec.get("idx") if rec.get("idx") is not None else rec["_row_i"])
            score_good = rec.get("score_good")
            score_bad = rec.get("score_bad")
            if isinstance(score_good, (int, float)) and isinstance(score_bad, (int, float)):
                margin = float(score_good) - float(score_bad)
            else:
                good_nll = rec.get("good_total_nll")
                bad_nll = rec.get("bad_total_nll")
                margin = float(bad_nll) - float(good_nll) if isinstance(good_nll, (int, float)) and isinstance(bad_nll, (int, float)) else float("nan")
            rows.append(
                {
                    "regime": regime,
                    "uid": str(uid),
                    "pair_id": pair_id,
                    "model": rec.get("model", ""),
                    "model_slug": _model_short(rec.get("model", "")),
                    "correctness": rec.get("correctness"),
                    "margin_logprob": margin,
                    "good_token_count": rec.get("good_token_count"),
                    "bad_token_count": rec.get("bad_token_count"),
                    "score_source": str(path),
                }
            )
    return pd.DataFrame(rows)


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_outputs(item_support: pd.DataFrame, score_support: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    regime_order = ["original", "head", "tail", "xtail"]
    colors = {"original": "#6B7280", "head": "#4C78A8", "tail": "#F58518", "xtail": "#E45756"}

    plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 250, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    data = [item_support.loc[item_support["regime"].eq(r), "local_support_mean_good"].dropna().values for r in regime_order]
    box_labels = [r.capitalize() if r != "xtail" else "XTail" for r in regime_order]
    try:
        bp = ax.boxplot(data, tick_labels=box_labels, patch_artist=True, showfliers=False)
    except TypeError:
        bp = ax.boxplot(data, labels=box_labels, patch_artist=True, showfliers=False)
    for patch, regime in zip(bp["boxes"], regime_order):
        patch.set(facecolor=colors[regime], alpha=0.45, edgecolor=colors[regime])
    ax.set_ylabel("Mean log(1 + COCA doc count) for local pairs")
    ax.set_title("Local Collocation Support by Dataset Regime")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "local_support_by_regime.png")
    fig.savefig(out_dir / "local_support_by_regime.pdf")
    plt.close(fig)

    _plot_support_heatmap(item_support, "field", regime_order, out_dir)
    _plot_support_heatmap(item_support, "phenomenon", regime_order, out_dir)

    scored = score_support.dropna(subset=["local_support_mean_good", "correctness", "margin_logprob"]).copy()
    if not scored.empty:
        scored["support_quartile"] = scored.groupby(["uid", "regime"])["local_support_mean_good"].transform(_quartile_labels)
        q = scored.dropna(subset=["support_quartile"]).groupby("support_quartile", as_index=False).agg(
            accuracy=("correctness", "mean"),
            margin=("margin_logprob", "mean"),
            n=("correctness", "size"),
        )
        q.to_csv(out_dir / "model_behavior_by_support_quartile.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
        axes[0].plot(q["support_quartile"], q["accuracy"] * 100, marker="o")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_xlabel("Within UID/regime support quartile")
        axes[0].grid(alpha=0.3)
        axes[1].plot(q["support_quartile"], q["margin"], marker="o", color="#F58518")
        axes[1].set_ylabel("Mean logprob margin")
        axes[1].set_xlabel("Within UID/regime support quartile")
        axes[1].grid(alpha=0.3)
        fig.suptitle("Model Behavior by Local-Support Quartile")
        fig.tight_layout()
        fig.savefig(out_dir / "model_behavior_by_support_quartile.png")
        fig.savefig(out_dir / "model_behavior_by_support_quartile.pdf")
        plt.close(fig)


def _plot_support_heatmap(item_support: pd.DataFrame, group_col: str, regime_order: Sequence[str], out_dir: Path) -> None:
    if group_col not in item_support.columns:
        return
    grouped = item_support.groupby([group_col, "regime"])["local_support_mean_good"].mean().reset_index()
    piv = grouped.pivot(index=group_col, columns="regime", values="local_support_mean_good").reindex(columns=regime_order)
    piv = piv.sort_index()
    piv.to_csv(out_dir / f"local_support_by_{group_col}.csv")
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.42 * len(piv) + 1.2)))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(regime_order)))
    ax.set_xticklabels([r.capitalize() if r != "xtail" else "XTail" for r in regime_order])
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_title(f"Mean Local Support by {group_col.replace('_', ' ').title()}")
    fig.colorbar(im, ax=ax, label="Mean log(1 + doc count)")
    fig.tight_layout()
    fig.savefig(out_dir / f"local_support_by_{group_col}_heatmap.png")
    fig.savefig(out_dir / f"local_support_by_{group_col}_heatmap.pdf")
    plt.close(fig)


def _quartile_labels(values: pd.Series) -> pd.Series:
    if values.nunique(dropna=True) < 4:
        return pd.Series(np.nan, index=values.index)
    try:
        return pd.qcut(values, q=4, labels=[1, 2, 3, 4], duplicates="drop").astype(float)
    except ValueError:
        return pd.Series(np.nan, index=values.index)


def _spearman_no_scipy(left: pd.Series, right: pd.Series) -> float:
    ranked = pd.DataFrame({"left": left, "right": right}).rank(method="average")
    return float(ranked["left"].corr(ranked["right"], method="pearson"))


def _correlation_tables(score_support: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    columns = ["model", "outcome", "feature", "n", "pearson", "spearman"]
    features = [
        "local_support_mean_good",
        "local_support_min_good",
        "local_support_zero_rate_good",
        "local_support_pmi_mean_good",
        "local_support_mean_delta_good_minus_bad",
        "good_zipf_mean",
        "good_char_count",
        "good_token_count",
    ]
    for model, part in score_support.groupby("model_slug"):
        for outcome in ("correctness", "margin_logprob"):
            for feature in features:
                sub = part[[outcome, feature]].dropna()
                if len(sub) < 10 or sub[feature].nunique() < 2:
                    continue
                rows.append(
                    {
                        "model": model,
                        "outcome": outcome,
                        "feature": feature,
                        "n": len(sub),
                        "pearson": sub[outcome].corr(sub[feature], method="pearson"),
                        "spearman": _spearman_no_scipy(sub[outcome], sub[feature]),
                    }
                )
    pd.DataFrame(rows, columns=columns).to_csv(out_dir / "support_behavior_correlations.csv", index=False)


def _residualize(values: pd.Series, controls: pd.DataFrame) -> pd.Series:
    data = pd.concat([values.rename("target"), controls], axis=1).dropna()
    if data.empty:
        return pd.Series(dtype=float)
    y = data["target"].astype(float).to_numpy()
    x = data.drop(columns=["target"]).astype(float).to_numpy()
    if x.size == 0:
        resid = y - np.mean(y)
    else:
        x = np.column_stack([np.ones(len(x)), x])
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ coef
    return pd.Series(resid, index=data.index)


def _within_group_residualize(values: pd.Series, groups: pd.Series) -> pd.Series:
    return values - values.groupby(groups).transform("mean")


def _controlled_correlation_tables(score_support: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    columns = [
        "model",
        "outcome",
        "feature",
        "n",
        "fixed_effects",
        "numeric_controls",
        "partial_pearson",
        "partial_spearman",
    ]
    features = [
        "local_support_mean_good",
        "local_support_min_good",
        "local_support_zero_rate_good",
        "local_support_pmi_mean_good",
        "local_support_mean_delta_good_minus_bad",
    ]
    controls = ["good_zipf_mean", "good_char_count", "good_token_count"]
    scored = score_support.dropna(subset=["model_slug", "uid"]).copy()
    scored["_uid_regime"] = scored["uid"].astype(str) + "::" + scored["regime"].astype(str)

    for model, part in scored.groupby("model_slug"):
        for outcome in ("correctness", "margin_logprob"):
            for feature in features:
                cols = [outcome, feature, "_uid_regime"] + controls
                sub = part[cols].dropna().copy()
                if len(sub) < 50 or sub[feature].nunique() < 2 or sub[outcome].nunique() < 2:
                    continue
                y_within = _within_group_residualize(sub[outcome].astype(float), sub["_uid_regime"])
                x_within = _within_group_residualize(sub[feature].astype(float), sub["_uid_regime"])
                numeric_controls = sub[controls].astype(float)
                y_resid = _residualize(y_within, numeric_controls)
                x_resid = _residualize(x_within, numeric_controls)
                aligned = pd.concat([y_resid.rename("y"), x_resid.rename("x")], axis=1).dropna()
                if len(aligned) < 50 or aligned["x"].nunique() < 2:
                    continue
                rows.append(
                    {
                        "model": model,
                        "outcome": outcome,
                        "feature": feature,
                        "n": len(aligned),
                        "fixed_effects": "uid+regime",
                        "numeric_controls": "+".join(controls),
                        "partial_pearson": aligned["y"].corr(aligned["x"], method="pearson"),
                        "partial_spearman": _spearman_no_scipy(aligned["y"], aligned["x"]),
                    }
                )
    pd.DataFrame(rows, columns=columns).to_csv(out_dir / "support_behavior_controlled_correlations.csv", index=False)


def _uid_gap_analysis(score_support: pd.DataFrame, out_dir: Path) -> None:
    cols = ["model_slug", "uid", "field", "phenomenon", "regime", "correctness", "margin_logprob", "local_support_mean_good"]
    scored = score_support[cols].dropna(subset=["model_slug", "uid", "regime", "local_support_mean_good"]).copy()
    if scored.empty:
        return
    agg = scored.groupby(["model_slug", "uid", "field", "phenomenon", "regime"], as_index=False).agg(
        accuracy=("correctness", "mean"),
        margin=("margin_logprob", "mean"),
        local_support=("local_support_mean_good", "mean"),
        n=("correctness", "size"),
    )
    original = agg[agg["regime"].eq("original")].rename(
        columns={
            "accuracy": "accuracy_original",
            "margin": "margin_original",
            "local_support": "local_support_original",
            "n": "n_original",
        }
    )
    freq = agg[agg["regime"].isin(FREQ_REGIMES)].rename(
        columns={
            "accuracy": "accuracy_freq",
            "margin": "margin_freq",
            "local_support": "local_support_freq",
            "n": "n_freq",
        }
    )
    merged = freq.merge(
        original[
            [
                "model_slug",
                "uid",
                "accuracy_original",
                "margin_original",
                "local_support_original",
                "n_original",
            ]
        ],
        on=["model_slug", "uid"],
        how="inner",
    )
    if merged.empty:
        return
    merged["accuracy_gap_original_minus_freq"] = merged["accuracy_original"] - merged["accuracy_freq"]
    merged["margin_gap_original_minus_freq"] = merged["margin_original"] - merged["margin_freq"]
    merged["support_gap_original_minus_freq"] = merged["local_support_original"] - merged["local_support_freq"]
    merged.to_csv(out_dir / "uid_original_freq_support_behavior_gaps.csv", index=False)

    corr_rows = []
    corr_columns = ["model", "regime", "outcome_gap", "n_uid", "pearson", "spearman"]
    for model, part in merged.groupby("model_slug"):
        for regime, sub in part.groupby("regime"):
            if len(sub) < 5 or sub["support_gap_original_minus_freq"].nunique() < 2:
                continue
            for outcome in ("accuracy_gap_original_minus_freq", "margin_gap_original_minus_freq"):
                corr_rows.append(
                    {
                        "model": model,
                        "regime": regime,
                        "outcome_gap": outcome,
                        "n_uid": len(sub),
                        "pearson": sub[outcome].corr(sub["support_gap_original_minus_freq"], method="pearson"),
                        "spearman": _spearman_no_scipy(sub[outcome], sub["support_gap_original_minus_freq"]),
                    }
                )
    pd.DataFrame(corr_rows, columns=corr_columns).to_csv(out_dir / "uid_gap_correlations.csv", index=False)

    models = sorted(merged["model_slug"].unique())
    regimes = [r for r in FREQ_REGIMES if r in set(merged["regime"])]
    colors = {"head": "#4C78A8", "tail": "#F58518", "xtail": "#E45756"}
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.8), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        part = merged[merged["model_slug"].eq(model)]
        for regime in regimes:
            sub = part[part["regime"].eq(regime)]
            ax.scatter(
                sub["support_gap_original_minus_freq"],
                sub["accuracy_gap_original_minus_freq"] * 100,
                s=24,
                alpha=0.75,
                label=regime,
                color=colors.get(regime),
            )
        trend = _linear_trend(part["support_gap_original_minus_freq"], part["accuracy_gap_original_minus_freq"] * 100)
        if not trend.empty:
            ax.plot(trend["x"], trend["y"], color="#111827", lw=2.2, alpha=0.9, label="linear trend")
        ax.axhline(0, color="#374151", lw=0.8, alpha=0.5)
        ax.axvline(0, color="#374151", lw=0.8, alpha=0.5)
        ax.set_title(model)
        ax.set_xlabel("Original - Freq local support")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Original - Freq accuracy (pp)")
    axes[-1].legend(frameon=False, title="Regime")
    fig.suptitle("UID-Level Accuracy Gap vs Local-Support Gap")
    fig.tight_layout()
    fig.savefig(out_dir / "uid_gap_accuracy_vs_support_gap.png")
    fig.savefig(out_dir / "uid_gap_accuracy_vs_support_gap.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.8), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        part = merged[merged["model_slug"].eq(model)]
        for regime in regimes:
            sub = part[part["regime"].eq(regime)].copy()
            ax.scatter(
                sub["support_gap_original_minus_freq"],
                sub["accuracy_gap_original_minus_freq"] * 100,
                s=20,
                alpha=0.35,
                color=colors.get(regime),
            )
        trend = _linear_trend(part["support_gap_original_minus_freq"], part["accuracy_gap_original_minus_freq"] * 100)
        if not trend.empty:
            ax.plot(trend["x"], trend["y"], color="#111827", lw=2.4, alpha=0.9, label="linear trend")
        ax.axhline(0, color="#374151", lw=0.8, alpha=0.5)
        ax.axvline(0, color="#374151", lw=0.8, alpha=0.5)
        ax.set_title(model)
        ax.set_xlabel("Original - Freq local support")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Original - Freq accuracy (pp)")
    axes[-1].legend(frameon=False, title="Regime")
    fig.suptitle("UID-Level Accuracy Gap vs Local-Support Gap, Linear Trend")
    fig.tight_layout()
    fig.savefig(out_dir / "uid_gap_accuracy_vs_support_gap_linear_trend.png")
    fig.savefig(out_dir / "uid_gap_accuracy_vs_support_gap_linear_trend.pdf")
    plt.close(fig)


def _linear_trend(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data["x"].nunique() < 2:
        return pd.DataFrame(columns=["x", "y"])
    slope, intercept = np.polyfit(data["x"].astype(float), data["y"].astype(float), deg=1)
    xs = np.linspace(float(data["x"].min()), float(data["x"].max()), 100)
    return pd.DataFrame({"x": xs, "y": intercept + slope * xs})


def _frame_support(item_support: pd.DataFrame, frame_counts_path: Optional[Path], out_dir: Path) -> None:
    if frame_counts_path is None or not frame_counts_path.exists():
        return
    counts = pd.read_csv(frame_counts_path)
    if not {"lemma", "frame", "doc_count"}.issubset(counts.columns):
        return
    # Lightweight secondary analysis: count support for the first content verb
    # in each side, with a coarse intr/trans heuristic from the sentence text.
    rows = []
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    for item in item_support.to_dict("records"):
        for side in SIDES:
            doc = nlp(str(item.get(f"{side}_text", "")))
            verbs = [tok for tok in doc if tok.pos_ == "VERB" and _valid_lemma(_lemma(tok))]
            if not verbs:
                continue
            verb = verbs[0]
            has_obj = any(child.dep_.lower() in {"obj", "dobj", "iobj", "dative"} for child in verb.children)
            has_prep = any(child.dep_.lower() == "prep" for child in verb.children)
            frame = "trans" if has_obj else ("intr_pp" if has_prep else "intr")
            rows.append(
                {
                    "regime": item["regime"],
                    "uid": item["uid"],
                    "pair_id": item["pair_id"],
                    "side": side,
                    "verb_lemma": _lemma(verb),
                    "coarse_frame": frame,
                }
            )
    frames = pd.DataFrame(rows)
    if frames.empty:
        return
    merged = frames.merge(
        counts[["lemma", "frame", "doc_count"]].rename(columns={"lemma": "verb_lemma", "frame": "coarse_frame", "doc_count": "coca_frame_doc_count"}),
        on=["verb_lemma", "coarse_frame"],
        how="left",
    )
    merged["coca_frame_doc_count"] = merged["coca_frame_doc_count"].fillna(0)
    merged["coca_frame_log_doc_count"] = np.log1p(merged["coca_frame_doc_count"].astype(float))
    merged.to_csv(out_dir / "secondary_frame_support.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-data", type=Path, default=DEFAULT_ORIGINAL_DATA)
    parser.add_argument("--freq-data-root", type=Path, default=DEFAULT_FREQ_DATA_ROOT)
    parser.add_argument("--current-score-glob", default=str(DEFAULT_CURRENT_SCORE_GLOB))
    parser.add_argument("--original-score-glob", default=str(DEFAULT_ORIGINAL_SCORE_GLOB))
    parser.add_argument("--coca-wlp", type=Path, default=DEFAULT_COCA_WLP)
    parser.add_argument("--frame-counts", type=Path, default=None, help="Optional COCA frame-count CSV for secondary frame support.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--spacy-batch-size", type=int, default=500)
    parser.add_argument(
        "--uid",
        action="append",
        default=[],
        help="Optional UID filter for targeted/debug runs. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--max-items-per-regime",
        type=int,
        default=None,
        help="Debug cap applied separately to original/head/tail/xtail before probe extraction.",
    )
    parser.add_argument("--max-coca-sentences", type=int, default=None, help="Smoke-test limit for COCA sentence scanning.")
    parser.add_argument("--reuse-counts", action="store_true", help="Reuse pair/lemma count CSVs in out-dir if present.")
    parser.add_argument(
        "--reuse-derived",
        action="store_true",
        help="Reuse existing item/model support CSVs and only regenerate summary tables/figures.",
    )
    parser.add_argument("--log-interval-sec", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / ".mplconfig").mkdir(exist_ok=True)
    uid_filter = {uid.strip() for value in args.uid for uid in str(value).split(",") if uid.strip()} or None

    if args.reuse_derived:
        item_support_path = args.out_dir / "item_local_support.csv"
        score_support_path = args.out_dir / "model_item_support_scores.csv"
        if not item_support_path.exists() or not score_support_path.exists():
            raise FileNotFoundError("--reuse-derived requires item_local_support.csv and model_item_support_scores.csv in out-dir")
        print("Reusing existing derived support CSVs.")
        item_support = pd.read_csv(item_support_path)
        score_usecols = [
            "model_slug",
            "uid",
            "field",
            "phenomenon",
            "regime",
            "correctness",
            "margin_logprob",
            "local_support_mean_good",
            "local_support_min_good",
            "local_support_zero_rate_good",
            "local_support_pmi_mean_good",
            "local_support_mean_delta_good_minus_bad",
            "good_zipf_mean",
            "good_char_count",
            "good_token_count",
        ]
        score_support = pd.read_csv(score_support_path, usecols=lambda col: col in set(score_usecols))
        _plot_outputs(item_support, score_support, args.out_dir)
        _correlation_tables(score_support, args.out_dir)
        _controlled_correlation_tables(score_support, args.out_dir)
        _uid_gap_analysis(score_support, args.out_dir)
        print(f"Regenerated tables/figures from existing derived CSVs in {args.out_dir}")
        return

    print("Loading benchmark items...")
    items = _load_items(
        args.original_data,
        args.freq_data_root,
        max_items_per_regime=args.max_items_per_regime,
        uid_filter=uid_filter,
    )
    items.to_csv(args.out_dir / "benchmark_items.csv", index=False)
    print(f"Items: {len(items):,}")

    print("Extracting local collocation probes with spaCy...")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    probes = extract_benchmark_probes(items, nlp, args.spacy_batch_size)
    probes.to_csv(args.out_dir / "local_collocation_probes.csv", index=False)
    target_pairs = set(zip(probes["lemma1"], probes["lemma2"]))
    target_lemmas = set(probes["lemma1"]).union(set(probes["lemma2"]))
    print(f"Target pairs: {len(target_pairs):,}; target lemmas: {len(target_lemmas):,}; probes: {len(probes):,}")

    pair_counts_path = args.out_dir / "coca_target_pair_counts.csv"
    lemma_counts_path = args.out_dir / "coca_target_lemma_counts.csv"
    meta_path = args.out_dir / "coca_scan_meta.json"
    if args.reuse_counts and pair_counts_path.exists() and lemma_counts_path.exists() and meta_path.exists():
        print("Reusing existing COCA target counts.")
        pair_counts = pd.read_csv(pair_counts_path)
        lemma_counts = pd.read_csv(lemma_counts_path)
        meta = json.loads(meta_path.read_text())
    else:
        print("Scanning COCA WLP for target pairs...")
        pair_counts, lemma_counts, meta = _scan_coca_wlp(
            args.coca_wlp,
            target_pairs,
            target_lemmas,
            window=args.window,
            max_sentences=args.max_coca_sentences,
            log_interval_sec=args.log_interval_sec,
        )
        pair_counts.to_csv(pair_counts_path, index=False)
        lemma_counts.to_csv(lemma_counts_path, index=False)
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    probe_support = _support_from_counts(probes, pair_counts, lemma_counts, int(meta.get("coca_docs_seen", 1)))
    probe_support.to_csv(args.out_dir / "local_collocation_probe_support.csv", index=False)
    side_support = _aggregate_side_support(probe_support)
    side_support.to_csv(args.out_dir / "item_side_local_support.csv", index=False)
    item_support = _pivot_item_support(side_support, items)
    item_support.to_csv(args.out_dir / "item_local_support.csv", index=False)

    print("Reading and merging score files...")
    score_paths = _collect_paths([args.current_score_glob, args.original_score_glob])
    scores = _read_scores(score_paths)
    scores.to_csv(args.out_dir / "nll_scores_for_support_merge.csv", index=False)
    score_support = scores.merge(item_support, on=["regime", "uid", "pair_id"], how="left")
    score_support.to_csv(args.out_dir / "model_item_support_scores.csv", index=False)

    _plot_outputs(item_support, score_support, args.out_dir)
    _correlation_tables(score_support, args.out_dir)
    _controlled_correlation_tables(score_support, args.out_dir)
    _uid_gap_analysis(score_support, args.out_dir)
    _frame_support(item_support, args.frame_counts, args.out_dir)

    summary = {
        "items": int(len(items)),
        "probes": int(len(probes)),
        "target_pairs": int(len(target_pairs)),
        "target_lemmas": int(len(target_lemmas)),
        "scores": int(len(scores)),
        **meta,
    }
    (args.out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
