#!/usr/bin/env python3
"""Paper-ready frequency diagnostics for the current 3-regime BLiMP datasets.

The script deliberately does not read the overlay manifest or frequency cache.
It streams vocabulary CSVs and keeps only rows whose surface expression occurs
in the dataset sentences.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lemminflect import getLemma
from wordfreq import zipf_frequency


REGIME_ORDER = ["original", "head", "tail", "xtail"]
GENERATED_REGIMES = ["head", "tail", "xtail"]
POS_ORDER = ["noun", "verb", "adjective"]
MODEL_ORDER = [
    "Gemma-4-E4B",
    "Gemma-4-31B",
    "Llama-3.1-8B",
    "Mistral-7B-v0.1",
    "Llama-3.1-70B",
    "Qwen2.5-7B",
    "Qwen2.5-72B",
]
MODEL_COLORS = {
    "Gemma-4-E4B": "#1b9e77",
    "Gemma-4-31B": "#66a61e",
    "Llama-3.1-8B": "#d95f02",
    "Mistral-7B-v0.1": "#7570b3",
    "Llama-3.1-70B": "#e7298a",
    "Qwen2.5-7B": "#a6761d",
    "Qwen2.5-72B": "#1f78b4",
    "Llama-3.3-70B-Instruct": "#666666",
}
STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "many",
    "all", "each", "every", "no", "more", "most", "less", "least", "much",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "who", "what",
    "which", "where", "when", "why", "how", "there", "here",
    "am", "is", "are", "was", "were", "be", "been", "being", "do", "does",
    "did", "done", "have", "has", "had", "having", "will", "would", "can",
    "could", "may", "might", "must", "shall", "should", "to", "of", "in",
    "on", "for", "with", "at", "by", "from", "as", "than", "and", "or",
    "but", "if", "because", "while", "although", "not", "nt", "s", "d",
    "ll", "re", "ve", "m",
}
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


@dataclass(frozen=True)
class LexEntry:
    surface: str
    lemma: str
    pos: str
    zipf: float
    width: int
    is_proper: bool = False


def norm_text(text: str) -> str:
    return " ".join(TOKEN_RE.findall(str(text).lower().replace("n't", " not")))


def tokens(text: str) -> list[str]:
    return norm_text(text).split()


def source_rank(path: Path) -> str:
    match = re.match(r"(?P<rank>\d{8}(?:-\d{6})?(?:-[A-Za-z0-9]+)?)", path.name)
    return match.group("rank") if match else path.name


def ngrams(toks: list[str], max_n: int) -> Iterable[str]:
    for width in range(1, max_n + 1):
        for idx in range(0, len(toks) - width + 1):
            yield " ".join(toks[idx : idx + width])


def row_pos(row: dict[str, str]) -> str | None:
    if row.get("noun") == "1":
        return "noun"
    if row.get("verb") == "1":
        return "verb"
    if row.get("category_2") in {"Adj", "adjective"} or row.get("adjs") == "1":
        return "adjective"
    return None


def lemmatize_expression(expression: str, pos: str) -> str:
    expression = norm_text(expression)
    if not expression:
        return ""
    if " " in expression:
        return expression
    upos = {"noun": "NOUN", "verb": "VERB", "adjective": "ADJ"}.get(pos)
    if upos:
        try:
            lemmas = getLemma(expression, upos=upos) or ()
        except Exception:
            lemmas = ()
        if lemmas:
            return norm_text(lemmas[0])
    return expression


def source_lemma(row: dict[str, str], pos: str) -> str:
    root = (row.get("root") or "").strip()
    if root and "_overlay_" in root:
        return norm_text(root.split("_overlay_", 1)[0])
    if pos == "noun" and (row.get("singularform") or "").strip():
        return norm_text(row["singularform"])
    return lemmatize_expression(row.get("expression", ""), pos)


def better_entry(old: LexEntry | None, new: LexEntry) -> LexEntry:
    if old is None:
        return new
    # Prefer the entry with a usable Zipf value, then the shorter source lemma.
    return max(
        [old, new],
        key=lambda e: (e.zipf > 0, -len(e.lemma), e.pos == "noun", e.surface),
    )


def load_items(dataset_dir: Path, original_path: Path) -> tuple[pd.DataFrame, set[str]]:
    current_uids = sorted(path.stem for path in (dataset_dir / "head").glob("*.jsonl"))
    rows: list[dict[str, object]] = []

    def append_item(
        regime: str,
        uid: str,
        line_no: int,
        obj: dict[str, object],
        good_field: str = "sentence_good",
        bad_field: str = "sentence_bad",
    ) -> None:
        rows.append({
            "regime": regime,
            "uid": obj.get("UID") or uid,
            "pair_id": str(obj.get("pairID", line_no)),
            "idx": line_no,
            "sentence_good": obj[good_field],
            "sentence_bad": obj[bad_field],
            "field": obj.get("field", ""),
            "phenomenon": obj.get("linguistics_term", obj.get("phenomenon", "")),
        })

    def read_regime(regime: str, base_dir: Path, uids: list[str]) -> None:
        for uid in uids:
            path = base_dir / uid if base_dir.name == "blimp" else base_dir / regime / uid
            path = path.with_suffix(".jsonl")
            if not path.exists():
                continue
            with path.open() as handle:
                for line_no, line in enumerate(handle):
                    obj = json.loads(line)
                    append_item(regime, uid, line_no, obj)

    def read_original_file(path: Path, uids: list[str]) -> None:
        wanted = set(uids)
        per_uid_seen: Counter[str] = Counter()
        with path.open() as handle:
            for line in handle:
                obj = json.loads(line)
                uid = obj.get("subtask", obj.get("UID", ""))
                if uid not in wanted:
                    continue
                line_no = per_uid_seen[uid]
                per_uid_seen[uid] += 1
                good_field = "good_original" if "good_original" in obj else "sentence_good"
                bad_field = "bad_original" if "bad_original" in obj else "sentence_bad"
                append_item("original", uid, line_no, obj, good_field, bad_field)

    def read_original(uids: list[str]) -> None:
        if original_path.is_file():
            read_original_file(original_path, uids)
            return
        if original_path.is_dir():
            combined = original_path / "blimp.jsonl"
            if combined.exists():
                read_original_file(combined, uids)
                return

    if original_path.exists():
        read_original(current_uids)
    for regime in GENERATED_REGIMES:
        read_regime(regime, dataset_dir, current_uids)
    return pd.DataFrame(rows), set(current_uids)


def collect_sentence_ngrams(items: pd.DataFrame, max_n: int) -> set[str]:
    seen: set[str] = set()
    for sentence in items["sentence_good"]:
        toks = tokens(sentence)
        seen.update(ng for ng in ngrams(toks, max_n) if ng and ng not in STOPWORDS)
    return seen


def build_lexicon(vocab_paths: list[Path], wanted: set[str]) -> dict[str, dict[str, LexEntry]]:
    by_surface: dict[str, dict[str, LexEntry]] = defaultdict(dict)
    for path in vocab_paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                surface = norm_text(row.get("expression", ""))
                if not surface or surface not in wanted:
                    continue
                pos = row_pos(row)
                if pos is None:
                    continue
                lemma = source_lemma(row, pos)
                z = float(zipf_frequency(surface, "en"))
                is_proper = row.get("properNoun") == "1" or row.get("locale") == "1"
                entry = LexEntry(surface=surface, lemma=lemma, pos=pos, zipf=z, width=len(surface.split()), is_proper=is_proper)
                by_surface[surface][pos] = better_entry(by_surface[surface].get(pos), entry)
    return by_surface


def choose_entry(candidates: dict[str, LexEntry]) -> LexEntry:
    for pos in POS_ORDER:
        if pos in candidates:
            return candidates[pos]
    return next(iter(candidates.values()))


def match_sentence(sentence: str, lexicon: dict[str, dict[str, LexEntry]], max_n: int) -> list[LexEntry]:
    toks = tokens(sentence)
    matches: list[LexEntry] = []
    used = [False] * len(toks)
    for width in range(max_n, 0, -1):
        for idx in range(0, len(toks) - width + 1):
            if any(used[idx : idx + width]):
                continue
            surface = " ".join(toks[idx : idx + width])
            if surface in STOPWORDS:
                continue
            candidates = lexicon.get(surface)
            if not candidates:
                continue
            entry = choose_entry(candidates)
            matches.append(entry)
            for pos in range(idx, idx + width):
                used[pos] = True
    return matches


def quantile(values: pd.Series, q: float) -> float:
    return float(values.quantile(q)) if len(values) else math.nan


def inverse_simpson(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return math.nan
    return float(total * total / sum(v * v for v in counts.values()))


def summarize_realized(item_points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, grp in item_points.groupby("regime", sort=False):
        values = grp["realized_zipf"].dropna()
        rows.append({
            "regime": regime,
            "median_realized_zipf": quantile(values, 0.50),
            "iqr": quantile(values, 0.75) - quantile(values, 0.25),
            "p05": quantile(values, 0.05),
            "p95": quantile(values, 0.95),
            "pair_count": int(len(grp)),
            "matched_pair_count": int(values.shape[0]),
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
        })
    return pd.DataFrame(rows)


def lexical_diversity(occ: pd.DataFrame, by_pos: bool) -> pd.DataFrame:
    rows = []
    data = occ
    group_cols = ["regime", "pos"] if by_pos else ["regime"]
    for key, grp in data.groupby(group_cols, sort=False):
        key_vals = key if isinstance(key, tuple) else (key,)
        lemma_counts = Counter(grp["lemma"])
        total = sum(lemma_counts.values())
        rows.append({
            **dict(zip(group_cols, key_vals)),
            "lexical_occurrences": int(total),
            "unique_surface_forms": int(grp["surface"].nunique()),
            "unique_source_lemmas": int(len(lemma_counts)),
            "top20_lemma_share": float(sum(v for _, v in lemma_counts.most_common(20)) / total) if total else math.nan,
            "effective_lemma_count_inverse_simpson": inverse_simpson(lemma_counts),
        })
    return pd.DataFrame(rows)


def load_proper_name_set(vocab_paths: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in vocab_paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("properNoun") == "1" or row.get("locale") == "1":
                    surface = norm_text(row.get("expression", ""))
                    if surface:
                        names.add(surface)
    return names


def slot_pos(value: str) -> str | None:
    key = str(value or "").strip().lower()
    if key in {"noun", "n", "propn", "proper_noun"}:
        return "noun"
    if key in {"verb", "v"}:
        return "verb"
    if key in {"adjective", "adj"}:
        return "adjective"
    return None


def load_original_slot_occurrences(original_path: Path, current_uids: set[str], proper_names: set[str]) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    rows: list[dict[str, object]] = []
    fixed_lemmas: dict[str, set[str]] = defaultdict(set)
    if not original_path.exists():
        return pd.DataFrame(rows), fixed_lemmas

    path = original_path if original_path.is_file() else original_path / "blimp.jsonl"
    per_uid_seen: Counter[str] = Counter()
    with path.open() as handle:
        for line in handle:
            obj = json.loads(line)
            uid = obj.get("subtask", obj.get("UID", ""))
            if uid not in current_uids:
                continue
            idx = per_uid_seen[uid]
            per_uid_seen[uid] += 1
            swaps = obj.get("meta", {}).get("g_swaps", ())
            swap_surfaces = {norm_text(swap.get("old", "")) for swap in swaps}
            for swap in swaps:
                pos = slot_pos(swap.get("pos", ""))
                surface = norm_text(swap.get("old", ""))
                if pos is None or not surface or surface in STOPWORDS or surface in proper_names:
                    continue
                lemma = norm_text(swap.get("lemma", "")) or lemmatize_expression(surface, pos)
                if not lemma or lemma in proper_names:
                    continue
                rows.append({
                    "regime": "original",
                    "uid": uid,
                    "idx": idx,
                    "pair_id": str(obj.get("idx", idx)),
                    "surface": surface,
                    "lemma": lemma,
                    "pos": pos,
                    "zipf": float(zipf_frequency(surface, "en")),
                    "source": "original_g_swaps",
                })
            # These matched content words are not swapped in the paper source, so
            # treat them as fixed template lexical material for the same paradigm.
            good = obj.get("good_original", obj.get("sentence_good", ""))
            for token in tokens(good):
                if token in STOPWORDS or token in swap_surfaces or token in proper_names:
                    continue
                z = zipf_frequency(token, "en")
                if z > 0:
                    fixed_lemmas[uid].add(norm_text(token))
    return pd.DataFrame(rows), fixed_lemmas


def extract_generated_slot_occurrences(
    items: pd.DataFrame,
    lexicon: dict[str, dict[str, LexEntry]],
    fixed_lemmas_by_uid: dict[str, set[str]],
    max_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    generated = items[items["regime"].isin(GENERATED_REGIMES)]
    for row in generated.itertuples(index=False):
        fixed = fixed_lemmas_by_uid.get(row.uid, set())
        for m in match_sentence(row.sentence_good, lexicon, max_n):
            if (
                m.zipf <= 0
                or m.is_proper
                or m.surface in STOPWORDS
                or m.lemma in STOPWORDS
                or m.lemma in fixed
                or m.surface in fixed
            ):
                continue
            rows.append({
                "regime": row.regime,
                "uid": row.uid,
                "idx": row.idx,
                "pair_id": row.pair_id,
                "surface": m.surface,
                "lemma": m.lemma,
                "pos": m.pos,
                "zipf": m.zipf,
                "source": "generated_vocab_match",
            })
    return pd.DataFrame(rows)


def load_lp_points(score_dirs: list[Path], item_points: pd.DataFrame) -> pd.DataFrame:
    lookup = {
        (row.regime, row.uid, row.idx): row
        for row in item_points[item_points["regime"].isin(GENERATED_REGIMES)].itertuples(index=False)
    }
    latest_files: dict[tuple[str, str, str], tuple[float, str, Path, str, str]] = {}
    seen_files: set[Path] = set()
    for score_dir in score_dirs:
        for path in sorted(score_dir.glob("*_freq_in_template_lp_acceptability.jsonl")):
            if path in seen_files:
                continue
            seen_files.add(path)
            regime = next((r for r in GENERATED_REGIMES if f"-{r}-" in path.name), None)
            if regime is None:
                continue
            with path.open() as handle:
                first = handle.readline()
            if not first:
                continue
            obj = json.loads(first)
            model = str(obj.get("model", ""))
            uid = Path(obj.get("dataset_name", "")).stem
            if not model or not uid:
                continue
            key = (model, regime, uid)
            candidate = (path.stat().st_mtime, source_rank(path), path, regime, uid)
            previous = latest_files.get(key)
            if previous is None or candidate[:3] > previous[:3]:
                latest_files[key] = candidate

    rows = []
    for _, _, path, regime, uid in sorted(latest_files.values(), key=lambda row: row[2].name):
        with path.open() as handle:
            for idx, line in enumerate(handle):
                obj = json.loads(line)
                key = (regime, uid, idx)
                item = lookup.get(key)
                if item is None:
                    continue
                good_lp = float(obj.get("good_total_logprob", obj["score_good"]))
                bad_lp = float(obj.get("bad_total_logprob", obj["score_bad"]))
                pair_lp = (good_lp + bad_lp) / 2.0
                good_text = obj.get("good_text", item.sentence_good)
                bad_text = obj.get("bad_text", item.sentence_bad)
                good_words = len(tokens(good_text))
                bad_words = len(tokens(bad_text))
                good_chars = len(str(good_text).replace(" ", ""))
                bad_chars = len(str(bad_text).replace(" ", ""))
                rows.append({
                    "model": obj.get("model", ""),
                    "model_short": model_short(obj.get("model", "")),
                    "regime": regime,
                    "uid": uid,
                    "idx": idx,
                    "realized_zipf": float(item.realized_zipf) if pd.notna(item.realized_zipf) else math.nan,
                    "good_lp": good_lp,
                    "good_lp_per_word": good_lp / good_words,
                    "good_lp_per_char": good_lp / good_chars,
                    "pair_lp": pair_lp,
                    "pair_lp_per_word": pair_lp / ((good_words + bad_words) / 2.0),
                    "pair_lp_per_char": pair_lp / ((good_chars + bad_chars) / 2.0),
                    "correctness": obj.get("correctness"),
                    "source_file": str(path),
                })
    return pd.DataFrame(rows).dropna(subset=["realized_zipf", "pair_lp_per_word"])


def model_short(model: str) -> str:
    low = model.lower()
    if "llama-3.3-70b" in low or "llama-3_3-70b" in low:
        return "Llama-3.3-70B-Instruct"
    if "llama-3.1-70b" in low or "llama-3_1-70b" in low:
        return "Llama-3.1-70B"
    if "gemma-4-31b" in low or "gemma-4_31b" in low:
        return "Gemma-4-31B"
    if "gemma-4-e4b" in low or "gemma-4_e4b" in low or "gemma" in low:
        return "Gemma-4-E4B"
    if "qwen2.5-72b" in low or "qwen2_5-72b" in low:
        return "Qwen2.5-72B"
    if "qwen2.5-7b" in low or "qwen2_5-7b" in low:
        return "Qwen2.5-7B"
    if "llama" in low:
        return "Llama-3.1-8B"
    if "mistral" in low:
        return "Mistral-7B-v0.1"
    return model.rsplit("/", 1)[-1]


def model_matches_exclude(model: str, patterns: list[str]) -> bool:
    model_low = str(model).lower()
    short_low = model_short(str(model)).lower()
    return any(pattern.lower() in model_low or pattern.lower() in short_low for pattern in patterns)


def slope_ci(points: pd.DataFrame, metric: str, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for model, grp in points.groupby("model_short", sort=False):
        x = grp["realized_zipf"].to_numpy(float)
        y = grp[metric].to_numpy(float)
        slope = float(np.polyfit(x, y, 1)[0])
        boots = []
        n = len(x)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            boots.append(float(np.polyfit(x[idx], y[idx], 1)[0]))
        lo, hi = np.quantile(boots, [0.025, 0.975])
        rows.append({"model_short": model, "metric": metric, "n": n, "slope": slope, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def binned_lp(points: pd.DataFrame, n_boot: int, seed: int, bins: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    edges = np.linspace(points["realized_zipf"].min(), points["realized_zipf"].max(), bins + 1)
    rows = []
    for model, grp in points.groupby("model_short", sort=False):
        bin_ids = np.digitize(grp["realized_zipf"], edges, right=False) - 1
        bin_ids = np.clip(bin_ids, 0, bins - 1)
        for bin_id in range(bins):
            ys = grp.loc[bin_ids == bin_id, "good_lp_per_word"].to_numpy(float)
            xs = grp.loc[bin_ids == bin_id, "realized_zipf"].to_numpy(float)
            if len(ys) < 50:
                continue
            means = [ys[rng.integers(0, len(ys), len(ys))].mean() for _ in range(n_boot)]
            lo, hi = np.quantile(means, [0.025, 0.975])
            rows.append({
                "model_short": model,
                "zipf_mid": float(xs.mean()),
                "n": int(len(ys)),
                "good_lp_per_word_mean": float(ys.mean()),
                "ci_low": float(lo),
                "ci_high": float(hi),
            })
    return pd.DataFrame(rows)


def binned_lp_uid_balanced(points: pd.DataFrame, metric: str, n_boot: int, seed: int, bins: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    edges = np.linspace(points["realized_zipf"].min(), points["realized_zipf"].max(), bins + 1)
    work = points.copy()
    work["bin_id"] = np.digitize(work["realized_zipf"], edges, right=False) - 1
    work["bin_id"] = work["bin_id"].clip(0, bins - 1)
    uid_means = (
        work.groupby(["model_short", "uid", "bin_id"], observed=True)
        .agg(
            zipf_mid=("realized_zipf", "mean"),
            value=(metric, "mean"),
            item_count=(metric, "size"),
        )
        .reset_index()
    )
    rows = []
    for (model, bin_id), grp in uid_means.groupby(["model_short", "bin_id"], sort=False):
        if len(grp) < 5:
            continue
        values = grp["value"].to_numpy(float)
        means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
        lo, hi = np.quantile(means, [0.025, 0.975])
        rows.append({
            "model_short": model,
            "bin_id": int(bin_id),
            "zipf_mid": float(grp["zipf_mid"].mean()),
            "paradigm_count": int(len(grp)),
            "item_count": int(grp["item_count"].sum()),
            "metric": metric,
            "value_mean": float(values.mean()),
            "ci_low": float(lo),
            "ci_high": float(hi),
        })
    return pd.DataFrame(rows)


def binned_lp_raw(points: pd.DataFrame, n_boot: int, seed: int, bins: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    edges = np.linspace(points["realized_zipf"].min(), points["realized_zipf"].max(), bins + 1)
    rows = []
    for model, grp in points.groupby("model_short", sort=False):
        bin_ids = np.digitize(grp["realized_zipf"], edges, right=False) - 1
        bin_ids = np.clip(bin_ids, 0, bins - 1)
        for bin_id in range(bins):
            ys = grp.loc[bin_ids == bin_id, "good_lp"].to_numpy(float)
            xs = grp.loc[bin_ids == bin_id, "realized_zipf"].to_numpy(float)
            if len(ys) < 50:
                continue
            means = [ys[rng.integers(0, len(ys), len(ys))].mean() for _ in range(n_boot)]
            lo, hi = np.quantile(means, [0.025, 0.975])
            rows.append({
                "model_short": model,
                "zipf_mid": float(xs.mean()),
                "n": int(len(ys)),
                "good_lp_mean": float(ys.mean()),
                "ci_low": float(lo),
                "ci_high": float(hi),
            })
    return pd.DataFrame(rows)


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
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
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def savefig(fig: plt.Figure, out_base: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"))
    plt.close(fig)


def set_descending_xlim(ax, values: pd.Series | np.ndarray, pad: float = 0.12) -> None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return
    ax.set_xlim(float(arr.max()) + pad, float(arr.min()) - pad)


def set_half_step_zipf_ticks(ax, values: pd.Series | np.ndarray) -> None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return
    lo = math.floor(float(arr.min()) * 2.0) / 2.0
    hi = math.ceil(float(arr.max()) * 2.0) / 2.0
    ax.set_xticks(np.arange(lo, hi + 0.25, 0.5))


def spearman_rho(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    xr = pd.Series(x, dtype="float64").rank(method="average")
    yr = pd.Series(y, dtype="float64").rank(method="average")
    return float(xr.corr(yr, method="pearson"))


def write_binned_spearman(binned: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for metric, grp_metric in binned.groupby("metric", sort=False):
        for model, grp in grp_metric.groupby("model_short", sort=False):
            rows.append({
                "model_short": model,
                "metric": metric,
                "spearman_rho": spearman_rho(grp["zipf_mid"], grp["value_mean"]),
                "n_bins": int(len(grp)),
                "item_count": int(grp["item_count"].sum()),
                "paradigm_count_mean": float(grp["paradigm_count"].mean()),
            })
        rows.append({
            "model_short": "ALL_BINS_POOLED",
            "metric": metric,
            "spearman_rho": spearman_rho(grp_metric["zipf_mid"], grp_metric["value_mean"]),
            "n_bins": int(len(grp_metric)),
            "item_count": int(grp_metric["item_count"].sum()),
            "paradigm_count_mean": float(grp_metric["paradigm_count"].mean()),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_binned_effects(binned: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for metric, grp_metric in binned.groupby("metric", sort=False):
        for model, grp in grp_metric.groupby("model_short", sort=False):
            part = grp.sort_values("zipf_mid")
            low = part.iloc[0]
            high = part.iloc[-1]
            slope_per_zipf_increase = float(np.polyfit(part["zipf_mid"].to_numpy(float), part["value_mean"].to_numpy(float), 1)[0])
            rows.append({
                "model_short": model,
                "metric": metric,
                "lowest_zipf_mid": float(low["zipf_mid"]),
                "highest_zipf_mid": float(high["zipf_mid"]),
                "lowest_bin_value": float(low["value_mean"]),
                "highest_bin_value": float(high["value_mean"]),
                "change_highest_to_lowest": float(low["value_mean"] - high["value_mean"]),
                "slope_per_1_zipf_decrease": -slope_per_zipf_increase,
                "n_bins": int(len(part)),
                "item_count": int(part["item_count"].sum()),
                "paradigm_count_mean": float(part["paradigm_count"].mean()),
            })

        avg = (
            grp_metric.groupby("zipf_mid", observed=True)
            .agg(value_mean=("value_mean", "mean"), item_count=("item_count", "sum"), paradigm_count=("paradigm_count", "mean"))
            .reset_index()
            .sort_values("zipf_mid")
        )
        low = avg.iloc[0]
        high = avg.iloc[-1]
        slope_per_zipf_increase = float(np.polyfit(avg["zipf_mid"].to_numpy(float), avg["value_mean"].to_numpy(float), 1)[0])
        rows.append({
            "model_short": "AVERAGE",
            "metric": metric,
            "lowest_zipf_mid": float(low["zipf_mid"]),
            "highest_zipf_mid": float(high["zipf_mid"]),
            "lowest_bin_value": float(low["value_mean"]),
            "highest_bin_value": float(high["value_mean"]),
            "change_highest_to_lowest": float(low["value_mean"] - high["value_mean"]),
            "slope_per_1_zipf_decrease": -slope_per_zipf_increase,
            "n_bins": int(len(avg)),
            "item_count": int(avg["item_count"].sum()),
            "paradigm_count_mean": float(avg["paradigm_count"].mean()),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def line_subset_for_summary(
    binned: pd.DataFrame,
    value_col: str,
    higher_is_better: bool = True,
) -> list[tuple[str, pd.DataFrame, str, float, str]]:
    average_rows = []
    for zipf_mid, grp in binned.groupby("zipf_mid", sort=True):
        average_rows.append({
            "zipf_mid": float(zipf_mid),
            value_col: float(grp[value_col].mean()),
            "ci_low": float(grp[value_col].mean()),
            "ci_high": float(grp[value_col].mean()),
        })
    average = pd.DataFrame(average_rows)

    model_means = binned.groupby("model_short", observed=True)[value_col].mean().sort_values(ascending=not higher_is_better)
    less_model = str(model_means.index[0])
    more_model = str(model_means.index[-1])
    return [
        ("Average", average, "#111111", 2.4, "--"),
        (f"Less Sensitive: {less_model}", binned[binned["model_short"].eq(less_model)], "#2563A6", 2.1, "-"),
        (f"More Sensitive: {more_model}", binned[binned["model_short"].eq(more_model)], "#C2410C", 2.1, "-"),
    ]


def plot_realized(item_points: pd.DataFrame, out_dir: Path) -> None:
    colors = {"original": "#777777", "head": "#1b9e77", "tail": "#d95f02", "xtail": "#7570b3"}
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    bins = np.linspace(0.8, 6.2, 32)
    for regime in REGIME_ORDER:
        vals = item_points.loc[item_points["regime"] == regime, "realized_zipf"].dropna()
        if len(vals):
            ax.hist(vals, bins=bins, density=True, histtype="bar", alpha=0.42, linewidth=0.7, edgecolor="white", label=regime, color=colors[regime])
    ax.set_xlabel("Median content-word Zipf frequency")
    ax.set_ylabel("Density")
    set_descending_xlim(ax, item_points["realized_zipf"], pad=0.18)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    savefig(fig, out_dir / "fig_realized_zipf_density")


def plot_pos_occurrences(occ: pd.DataFrame, out_dir: Path) -> None:
    colors = {"original": "#777777", "head": "#1b9e77", "tail": "#d95f02", "xtail": "#7570b3"}
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), sharey=True)
    bins = np.linspace(0.8, 6.2, 28)
    for ax, pos in zip(axes, POS_ORDER):
        data = occ[occ["pos"] == pos]
        for regime in REGIME_ORDER:
            vals = data.loc[data["regime"] == regime, "zipf"].dropna()
            if len(vals):
                ax.hist(vals, bins=bins, density=True, histtype="bar", alpha=0.38, linewidth=0.5, edgecolor="white", label=regime, color=colors[regime])
        ax.set_title(pos.capitalize())
        ax.set_xlabel("Median Zipf")
        set_descending_xlim(ax, data["zipf"], pad=0.18)
    axes[0].set_ylabel("Density")
    axes[-1].legend(fontsize=8, loc="upper right")
    savefig(fig, out_dir / "fig_appendix_zipf_by_pos")


def plot_model_lp(binned: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for model in MODEL_ORDER:
        grp = binned[binned["model_short"] == model].sort_values("zipf_mid")
        if grp.empty:
            continue
        color = MODEL_COLORS.get(model)
        ax.plot(grp["zipf_mid"], grp["good_lp_per_word_mean"], marker="o", markersize=3, linewidth=1.6, label=model, color=color)
        ax.fill_between(grp["zipf_mid"].to_numpy(float), grp["ci_low"].to_numpy(float), grp["ci_high"].to_numpy(float), color=color, alpha=0.16, linewidth=0)
    ax.set_xlabel("Median content-word Zipf frequency")
    ax.set_ylabel("LP/word")
    set_descending_xlim(ax, binned["zipf_mid"], pad=0.15)
    ax.legend(ncol=2)
    savefig(fig, out_dir / "fig_model_good_lp_by_zipf")


def plot_model_lp_raw(binned: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for model in MODEL_ORDER:
        grp = binned[binned["model_short"] == model].sort_values("zipf_mid")
        if grp.empty:
            continue
        color = MODEL_COLORS.get(model)
        ax.plot(grp["zipf_mid"], grp["good_lp_mean"], marker="o", markersize=3, linewidth=1.6, label=model, color=color)
        ax.fill_between(grp["zipf_mid"].to_numpy(float), grp["ci_low"].to_numpy(float), grp["ci_high"].to_numpy(float), color=color, alpha=0.16, linewidth=0)
    ax.set_xlabel("Median content-word Zipf frequency")
    ax.set_ylabel("LP")
    set_descending_xlim(ax, binned["zipf_mid"], pad=0.15)
    ax.legend(ncol=2)
    savefig(fig, out_dir / "fig_model_good_lp_raw_by_zipf")


def plot_model_lp_uid_balanced(binned: pd.DataFrame, out_dir: Path, metric_label: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for label, grp, color, linewidth, linestyle in line_subset_for_summary(binned, "value_mean", higher_is_better=True):
        grp = grp.sort_values("zipf_mid")
        if grp.empty:
            continue
        ax.plot(
            grp["zipf_mid"],
            grp["value_mean"],
            marker="o",
            markersize=3.5,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
            color=color,
        )
        if not label.startswith("Average"):
            ax.fill_between(
                grp["zipf_mid"].to_numpy(float),
                grp["ci_low"].to_numpy(float),
                grp["ci_high"].to_numpy(float),
                color=color,
                alpha=0.16,
                linewidth=0,
            )
    ax.set_xlabel("Median content-word Zipf frequency")
    ax.set_ylabel(metric_label)
    set_descending_xlim(ax, binned["zipf_mid"], pad=0.15)
    set_half_step_zipf_ticks(ax, binned["zipf_mid"])
    ax.legend(loc="upper right", fontsize=10.5, handlelength=2.0)
    savefig(fig, out_dir / filename)


def markdown_table(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(headers[col_idx]), *(len(row[col_idx]) for row in rows))
        for col_idx in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")
    return "\n".join(lines)


def write_paper_tables(out_dir: Path, realized: pd.DataFrame, lexical: pd.DataFrame, slopes: pd.DataFrame) -> None:
    realized_cols = {
        "regime": "Regime",
        "median_realized_zipf": "Median Zipf",
        "iqr": "IQR",
        "p05": "P5",
        "p95": "P95",
        "pair_count": "Pairs",
        "mean": "Mean",
        "sd": "SD",
    }
    lexical_cols = {
        "regime": "Regime",
        "unique_surface_forms": "Unique surfaces",
        "unique_source_lemmas": "Unique lemmas",
        "top20_lemma_share": "Top-20 lemma share",
        "effective_lemma_count_inverse_simpson": "Effective lemma count",
    }
    slope_cols = {
        "model_short": "Model",
        "slope": "Slope",
        "ci_low": "95% CI low",
        "ci_high": "95% CI high",
        "n": "Pairs",
    }

    realized_tbl = realized[list(realized_cols)].rename(columns=realized_cols).round(3)
    lexical_tbl = lexical[list(lexical_cols)].rename(columns=lexical_cols).round(3)
    slopes_tbl = slopes[list(slope_cols)].rename(columns=slope_cols).round(3)

    with (out_dir / "paper_tables.md").open("w") as handle:
        handle.write("## Realized Zipf Separation\n\n")
        handle.write(markdown_table(realized_tbl))
        handle.write("\n\n## Lexical Diversity\n\n")
        handle.write(markdown_table(lexical_tbl))
        handle.write("\n\n## Word-Normalized Good-Sentence LP Slopes\n\n")
        handle.write(markdown_table(slopes_tbl))
        handle.write("\n")

    with (out_dir / "paper_tables.tex").open("w") as handle:
        handle.write("% Realized Zipf separation\n")
        handle.write(realized_tbl.to_latex(index=False, escape=True))
        handle.write("\n% Lexical diversity\n")
        handle.write(lexical_tbl.to_latex(index=False, escape=True))
        handle.write("\n% Word-normalized good-sentence LP slopes\n")
        handle.write(slopes_tbl.to_latex(index=False, escape=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/Users/tyronewhite/masters_research_code/freq-blimp/outputs/full_3regimes_current"),
    )
    parser.add_argument("--original-path", type=Path, default=Path("/Users/tyronewhite/masters_research_code/blimp-rare/data/processed/blimp_original.jsonl"))
    parser.add_argument(
        "--score-dir",
        action="append",
        type=Path,
        default=[
            Path("/Users/tyronewhite/masters_research_code/freq-blimp/outputs/blimp_rare_results/acceptability_pair_scores/20260429_drop_argument_diverse_full"),
            Path("/Users/tyronewhite/masters_research_code/blimp-rare/results/acceptability_pair_scores"),
        ],
    )
    parser.add_argument(
        "--vocab-path",
        action="append",
        type=Path,
        default=[
            Path("/Users/tyronewhite/masters_research_code/freq-blimp/vocabulary.csv"),
            Path("/Users/tyronewhite/masters_research_code/freq-blimp/vocabulary_overlay.csv"),
        ],
        help="Vocabulary CSV path. Repeatable.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/frequency_figures_current"))
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=["Instruct", "-it"],
        help=(
            "Exclude models whose full name or short label contains this token. "
            "Repeatable. Defaults to excluding instruct/IT models."
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-ngram", type=int, default=4)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    items, current_uids = load_items(args.dataset_dir, args.original_path)
    wanted = collect_sentence_ngrams(items, args.max_ngram)
    lexicon = build_lexicon(args.vocab_path, wanted)
    proper_names = load_proper_name_set(args.vocab_path)

    item_rows = []
    occ_rows = []
    for row in items.itertuples(index=False):
        matches = match_sentence(row.sentence_good, lexicon, args.max_ngram)
        values = [m.zipf for m in matches if m.zipf > 0]
        item_rows.append({
            **row._asdict(),
            "realized_zipf": float(np.median(values)) if values else math.nan,
            "zipf_mean": float(np.mean(values)) if values else math.nan,
            "matched_lexical_count": len(values),
            "word_count_good": len(tokens(row.sentence_good)),
            "char_count_good": len(str(row.sentence_good).replace(" ", "")),
        })
        for m in matches:
            if m.zipf <= 0:
                continue
            occ_rows.append({
                "regime": row.regime,
                "uid": row.uid,
                "idx": row.idx,
                "pair_id": row.pair_id,
                "surface": m.surface,
                "lemma": m.lemma,
                "pos": m.pos,
                "zipf": m.zipf,
            })

    item_points = pd.DataFrame(item_rows)
    occ = pd.DataFrame(occ_rows)
    item_points.to_csv(args.out_dir / "realized_zipf_item_points.csv", index=False)
    occ.to_csv(args.out_dir / "lexical_occurrences.csv", index=False)

    original_slots, fixed_lemmas_by_uid = load_original_slot_occurrences(args.original_path, current_uids, proper_names)
    generated_slots = extract_generated_slot_occurrences(items, lexicon, fixed_lemmas_by_uid, args.max_ngram)
    slot_occ = pd.concat([original_slots, generated_slots], ignore_index=True)
    slot_occ.to_csv(args.out_dir / "lexical_slot_occurrences.csv", index=False)

    realized_summary = summarize_realized(item_points)
    realized_summary.to_csv(args.out_dir / "realized_zipf_summary.csv", index=False)
    occ.groupby(["regime", "pos"])["zipf"].agg(["count", "mean", "std", "median"]).reset_index().to_csv(args.out_dir / "appendix_zipf_by_pos_summary.csv", index=False)

    lexical_main = lexical_diversity(slot_occ, by_pos=False)
    lexical_main.to_csv(args.out_dir / "lexical_diversity_main.csv", index=False)
    lexical_diversity(slot_occ, by_pos=True).to_csv(args.out_dir / "lexical_diversity_by_pos.csv", index=False)

    plot_realized(item_points, args.out_dir)
    plot_pos_occurrences(occ, args.out_dir)

    lp_points = load_lp_points(args.score_dir, item_points)
    if args.exclude_model:
        lp_points = lp_points.loc[
            ~lp_points["model"].apply(lambda model: model_matches_exclude(str(model), args.exclude_model))
        ].copy()
    lp_points.to_csv(args.out_dir / "model_good_lp_points.csv", index=False)
    if not lp_points.empty:
        slopes = slope_ci(lp_points, "good_lp_per_word", args.bootstrap, args.seed)
        slopes.to_csv(args.out_dir / "model_good_lp_slope_ci.csv", index=False)
        slope_ci(lp_points, "good_lp_per_char", args.bootstrap, args.seed).to_csv(args.out_dir / "model_good_lp_char_slope_ci.csv", index=False)
        binned = binned_lp(lp_points, args.bootstrap, args.seed)
        binned.to_csv(args.out_dir / "model_good_lp_binned.csv", index=False)
        raw_slopes = slope_ci(lp_points, "good_lp", args.bootstrap, args.seed)
        raw_slopes.to_csv(args.out_dir / "model_good_lp_raw_slope_ci.csv", index=False)
        raw_binned = binned_lp_raw(lp_points, args.bootstrap, args.seed)
        raw_binned.to_csv(args.out_dir / "model_good_lp_raw_binned.csv", index=False)
        uid_balanced_word = binned_lp_uid_balanced(lp_points, "good_lp_per_word", args.bootstrap, args.seed)
        uid_balanced_word.to_csv(args.out_dir / "model_good_lp_paradigm_balanced_binned.csv", index=False)
        uid_balanced_raw = binned_lp_uid_balanced(lp_points, "good_lp", args.bootstrap, args.seed)
        uid_balanced_raw.to_csv(args.out_dir / "model_good_lp_raw_paradigm_balanced_binned.csv", index=False)
        write_binned_spearman(uid_balanced_word, args.out_dir / "model_good_lp_paradigm_balanced_binned_spearman.csv")
        write_binned_spearman(uid_balanced_raw, args.out_dir / "model_good_lp_raw_paradigm_balanced_binned_spearman.csv")
        write_binned_effects(uid_balanced_word, args.out_dir / "model_good_lp_paradigm_balanced_binned_effects.csv")
        write_binned_effects(uid_balanced_raw, args.out_dir / "model_good_lp_raw_paradigm_balanced_binned_effects.csv")
        lp_points.groupby("model_short")["good_lp_per_char"].agg(["count", "mean", "std"]).reset_index().to_csv(args.out_dir / "model_good_lp_char_normalized_summary.csv", index=False)
        plot_model_lp(binned, args.out_dir)
        plot_model_lp_raw(raw_binned, args.out_dir)
        plot_model_lp_uid_balanced(uid_balanced_word, args.out_dir, "LP/word, paradigm-balanced", "fig_model_good_lp_paradigm_balanced_by_zipf")
        plot_model_lp_uid_balanced(uid_balanced_raw, args.out_dir, "LP, paradigm-balanced", "fig_model_good_lp_raw_paradigm_balanced_by_zipf")
        write_paper_tables(args.out_dir, realized_summary, lexical_main, slopes)

    with (args.out_dir / "README.md").open("w") as handle:
        handle.write(
            "# Frequency Diagnostics\n\n"
            f"Dataset: `{args.dataset_dir}`\n\n"
            f"Original BLiMP source: `{args.original_path}` restricted to the {len(current_uids)} current UIDs.\n\n"
            "Zipf and lexical diversity are computed from matched controlled vocabulary "
            "items in the grammatical sentence. LP checks use generated-regime grammatical-sentence LP, "
            "normalized by word count for the main metric "
            "and by character count as a robustness table.\n"
        )


if __name__ == "__main__":
    main()
