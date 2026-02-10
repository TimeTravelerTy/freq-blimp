import statistics
from typing import Any, Dict, Iterable, List, Optional

from wordfreq import zipf_frequency


def _zipf(word: str) -> float:
    return float(zipf_frequency(word, "en"))


def _values_to_aggs(values: List[float]) -> Dict[str, Optional[float]]:
    oov_count = sum(1 for v in values if v <= 0.0)
    in_vocab = [v for v in values if v > 0.0]
    if not in_vocab:
        return {"n": 0, "oov_count": oov_count, "mean": None, "median": None, "min": None}
    return {
        "n": len(in_vocab),
        "oov_count": oov_count,
        "mean": sum(in_vocab) / len(in_vocab),
        "median": float(statistics.median(in_vocab)),
        "min": float(min(in_vocab)),
    }


def _extract_words_from_swaps(
    swap_items: Iterable[Dict[str, Any]],
    *,
    which: str,
) -> List[str]:
    words: List[str] = []
    for s in swap_items:
        w = s.get(which)
        if isinstance(w, str) and w:
            words.append(w)
    return words


def _all_swaps(meta: Dict[str, Any], prefix: str) -> List[Dict[str, Any]]:
    swaps = list(meta.get(f"{prefix}_swaps") or [])
    # Backward compatibility for legacy records with split POS keys.
    swaps.extend(meta.get(f"{prefix}_adj_swaps") or [])
    swaps.extend(meta.get(f"{prefix}_verb_swaps") or [])
    return swaps


def add_zipf_aggregates(record: Dict[str, Any]) -> Dict[str, Any]:
    meta = record.get("meta") or {}

    g_swaps = _all_swaps(meta, "g")
    b_swaps = _all_swaps(meta, "b")

    good_original_words = (
        _extract_words_from_swaps(g_swaps, which="old")
    )
    bad_original_words = (
        _extract_words_from_swaps(b_swaps, which="old")
    )
    good_freq_words = (
        _extract_words_from_swaps(g_swaps, which="new")
    )
    bad_freq_words = (
        _extract_words_from_swaps(b_swaps, which="new")
    )

    zipf_values = {
        "good_original": [_zipf(w) for w in good_original_words],
        "bad_original": [_zipf(w) for w in bad_original_words],
        "good_freq": [_zipf(w) for w in good_freq_words],
        "bad_freq": [_zipf(w) for w in bad_freq_words],
    }
    aggs = {k: _values_to_aggs(v) for k, v in zipf_values.items()}

    meta_out = dict(meta)
    meta_out["zipf_swapped_position_aggregates"] = aggs

    out = dict(record)
    out["meta"] = meta_out
    return out
