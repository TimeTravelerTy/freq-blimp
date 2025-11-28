import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.sentence_nll import LlamaNLLScorer  # noqa: E402


Variant = str
VARIANTS: Sequence[Variant] = ("good_typical", "bad_typical", "good_rare", "bad_rare")


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    sorted_xs = sorted(xs)
    mid = len(sorted_xs) // 2
    if len(sorted_xs) % 2:
        return float(sorted_xs[mid])
    return float((sorted_xs[mid - 1] + sorted_xs[mid]) / 2)


def load_records(path: str, limit: Optional[int] = None) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec["_row"] = i
            records.append(rec)
            if limit is not None and len(records) >= limit:
                break
    return records


def _build_items(records: List[dict]):
    items = []
    for pos, rec in enumerate(records):
        meta = rec.get("meta") or {}
        g_swaps = meta.get("g_swaps") or []
        g_adj_swaps = meta.get("g_adj_swaps") or []
        g_verb_swaps = meta.get("g_verb_swaps") or []
        k_swaps = len(g_swaps) + len(g_adj_swaps) + len(g_verb_swaps)
        for variant in VARIANTS:
            text = rec.get(variant)
            if not text:
                continue
            char_len = len(text)
            items.append(
                {
                    "variant": variant,
                    "text": text,
                    "record_idx": rec.get("idx", pos),
                    "row": rec.get("_row", pos),
                    "group": rec.get("group"),
                    "phenomenon": rec.get("phenomenon"),
                    "subtask": rec.get("subtask"),
                    "char_len": char_len,
                    "k_swaps": k_swaps,
                }
            )
    return items


def _aggregate_by_variant(scored_items: List[dict]) -> Dict[Variant, Dict[str, float]]:
    stats: Dict[Variant, Dict[str, float]] = {}
    for variant in VARIANTS:
        subset = [it for it in scored_items if it["variant"] == variant]
        totals = [it["total_nll"] for it in subset]
        per_char = [it["nll_per_char"] for it in subset]
        tokens = [it["token_count"] for it in subset]
        stats[variant] = {
            "count": len(subset),
            "mean_total_nll": _mean(totals),
            "median_total_nll": _median(totals),
            "mean_nll_per_char": _mean(per_char),
            "median_nll_per_char": _median(per_char),
            "mean_tokens": _mean(tokens),
        }
    return stats


def _pairwise_stats(per_record: Dict[int, Dict[Variant, dict]], typical: Variant, rare: Variant, field: str):
    deltas = []
    rare_higher = 0
    for variants in per_record.values():
        if typical not in variants or rare not in variants:
            continue
        t = variants[typical][field]
        r = variants[rare][field]
        deltas.append(r - t)
        rare_higher += int(r > t)
    total = len(deltas)
    return {
        "pairs": total,
        "pct_rare_higher": (rare_higher / total * 100.0) if total else float("nan"),
        "mean_delta": _mean(deltas),
        "median_delta": _median(deltas),
    }


def _good_bad_stats(per_record: Dict[int, Dict[Variant, dict]], good: Variant, bad: Variant, field: str):
    deltas = []
    bad_higher = 0
    for variants in per_record.values():
        if good not in variants or bad not in variants:
            continue
        g = variants[good][field]
        b = variants[bad][field]
        deltas.append(b - g)
        bad_higher += int(b > g)
    total = len(deltas)
    return {
        "pairs": total,
        "pct_bad_higher": (bad_higher / total * 100.0) if total else float("nan"),
        "mean_delta": _mean(deltas),
        "median_delta": _median(deltas),
    }


def _subtask_deltas(scored_items: List[dict], typical: Variant, rare: Variant, top_k: int = 8):
    grouped: Dict[str, Dict[Variant, List[float]]] = defaultdict(lambda: defaultdict(list))
    for item in scored_items:
        grouped[item.get("subtask") or "unknown"][item["variant"]].append(item["total_nll"])
    rows = []
    for subtask, variants in grouped.items():
        if typical not in variants or rare not in variants:
            continue
        t_mean = _mean(variants[typical])
        r_mean = _mean(variants[rare])
        rows.append(
            {
                "subtask": subtask,
                "typical_mean": t_mean,
                "rare_mean": r_mean,
                "delta": r_mean - t_mean,
            }
        )
    rows.sort(key=lambda r: -r["delta"])
    return rows[:top_k]


def _rare_penalty_stats(per_record: Dict[int, Dict[Variant, dict]]):
    per_swap_total = []
    per_swap_char = []
    for variants in per_record.values():
        if "good_typical" not in variants or "good_rare" not in variants:
            continue
        rare = variants["good_rare"]
        typical = variants["good_typical"]
        k = rare.get("k_swaps") or typical.get("k_swaps") or 0
        if k <= 0:
            continue
        delta_total = rare["total_nll"] - typical["total_nll"]
        delta_char = rare["nll_per_char"] - typical["nll_per_char"]
        per_swap_total.append(delta_total / k)
        per_swap_char.append(delta_char / k)
    return {
        "count": len(per_swap_total),
        "mean_per_swap_nll": _mean(per_swap_total),
        "median_per_swap_nll": _median(per_swap_total),
        "mean_per_swap_nll_per_char": _mean(per_swap_char),
        "median_per_swap_nll_per_char": _median(per_swap_char),
    }


def _default_out_path(model: str, limit: Optional[int]) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    model_slug = model.split("/")[-1].replace(".", "_")
    limit_part = f"n{limit}" if limit is not None else "all"
    name = f"{ts}_{model_slug}_{limit_part}_sentence-nll.json"
    return Path("results") / "sentence_nll_runs" / name


def _save_single_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _save_json(path: Optional[str], obj):
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _model_slug(model: str) -> str:
    return model.split("/")[-1].replace(".", "_")


def _typical_cache_path(model: str) -> Path:
    return Path("results") / "sentence_nll_runs" / "cache" / f"{_model_slug(model)}_typical.json"


def _typical_fingerprint(items: List[dict]) -> str:
    typical = [it for it in items if it.get("variant") in ("good_typical", "bad_typical")]
    typical.sort(key=lambda it: (it.get("row"), it.get("variant")))
    payload = "\n".join(f"{it.get('row')}|{it.get('variant')}|{it.get('text')}" for it in typical)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_typical_cache(model: str, fingerprint: str) -> Dict[Tuple[int, str], dict]:
    path = _typical_cache_path(model)
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if data.get("fingerprint") != fingerprint:
        return {}
    cached = {}
    for entry in data.get("scores", []):
        key = (entry.get("row"), entry.get("variant"))
        if key[0] is None or key[1] is None:
            continue
        cached[key] = {
            "total_nll": entry.get("total_nll"),
            "token_count": entry.get("token_count"),
            "nll_per_char": entry.get("nll_per_char"),
            "char_len": entry.get("char_len"),
        }
    return cached


def _save_typical_cache(model: str, fingerprint: str, items: List[dict]) -> None:
    path = _typical_cache_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    typical = [
        {
            "row": it.get("row"),
            "variant": it.get("variant"),
            "text": it.get("text"),
            "total_nll": it.get("total_nll"),
            "token_count": it.get("token_count"),
            "nll_per_char": it.get("nll_per_char"),
            "char_len": it.get("char_len", len(it.get("text") or "")),
        }
        for it in items
        if it.get("variant") in ("good_typical", "bad_typical")
    ]
    payload = {"fingerprint": fingerprint, "scores": typical}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/pilot_tierA.jsonl")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of records to score.")
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available).")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument(
        "--device-map",
        default=None,
        help="Pass through to transformers.from_pretrained device_map (e.g., 'auto' for multi-GPU).",
    )
    ap.add_argument("--compile", action="store_true", help="Try torch.compile for extra throughput on CUDA.")
    ap.add_argument(
        "--out",
        default=None,
        help="Optional JSON output path; defaults to results/sentence_nll_runs/<timestamp>_<model>_<n>_sentence-nll.json",
    )
    args = ap.parse_args()

    records = load_records(args.data, args.limit)
    print(f"Loaded {len(records)} records from {args.data}.")
    items = _build_items(records)
    print(f"Scoring {len(items)} variants across {len(VARIANTS)} buckets...")

    fingerprint = _typical_fingerprint(items)
    typical_cache = _load_typical_cache(args.model, fingerprint)

    scorer = LlamaNLLScorer(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        compile_model=args.compile,
    )
    missing_indices = []
    missing_texts = []
    for idx, item in enumerate(items):
        key = (item.get("row"), item.get("variant"))
        cached = typical_cache.get(key)
        if cached:
            char_len = max(1, item.get("char_len", len(item.get("text", "")) or 0))
            total_nll = cached.get("total_nll")
            token_count = cached.get("token_count")
            if total_nll is None or token_count is None:
                # If cache is incomplete, fall back to scoring.
                missing_indices.append(idx)
                missing_texts.append(item["text"])
                continue
            item.update(
                {
                    "total_nll": total_nll,
                    "token_count": token_count,
                    "nll_per_char": total_nll / char_len,
                }
            )
        else:
            missing_indices.append(idx)
            missing_texts.append(item["text"])

    if missing_texts:
        scores = scorer.score_texts(
            missing_texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        for (idx, item_text), score in zip(zip(missing_indices, missing_texts), scores):
            item = items[idx]
            char_len = max(1, item.get("char_len", len(item.get("text", "")) or 0))
            item.update(
                {
                    "total_nll": score.total_nll,
                    "token_count": score.token_count,
                    "nll_per_char": score.total_nll / char_len,
                }
            )

    _save_typical_cache(args.model, fingerprint, items)

    per_record: Dict[int, Dict[Variant, dict]] = defaultdict(dict)
    for item in items:
        per_record[item["row"]][item["variant"]] = item

    variant_stats = _aggregate_by_variant(items)
    rare_good = _pairwise_stats(per_record, "good_typical", "good_rare", "total_nll")
    rare_bad = _pairwise_stats(per_record, "bad_typical", "bad_rare", "total_nll")
    good_vs_bad_typical = _good_bad_stats(per_record, "good_typical", "bad_typical", "total_nll")
    good_vs_bad_rare = _good_bad_stats(per_record, "good_rare", "bad_rare", "total_nll")
    rare_good_char = _pairwise_stats(per_record, "good_typical", "good_rare", "nll_per_char")
    rare_bad_char = _pairwise_stats(per_record, "bad_typical", "bad_rare", "nll_per_char")
    good_vs_bad_typical_char = _good_bad_stats(per_record, "good_typical", "bad_typical", "nll_per_char")
    good_vs_bad_rare_char = _good_bad_stats(per_record, "good_rare", "bad_rare", "nll_per_char")
    rare_penalty = _rare_penalty_stats(per_record)
    subtask_rows = _subtask_deltas(items, "good_typical", "good_rare")

    out_path = Path(args.out) if args.out else _default_out_path(args.model, args.limit)

    summary = {
        "data": args.data,
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "limit": args.limit,
        "variant_stats": variant_stats,
        "rare_vs_typical": {"good": rare_good, "bad": rare_bad},
        "good_vs_bad": {"typical": good_vs_bad_typical, "rare": good_vs_bad_rare},
        "rare_vs_typical_char": {"good": rare_good_char, "bad": rare_bad_char},
        "good_vs_bad_char": {"typical": good_vs_bad_typical_char, "rare": good_vs_bad_rare_char},
        "rare_penalty_per_swap": rare_penalty,
        "subtask_gaps_good": subtask_rows,
        "details": items,
    }

    print("\nVariant sentence-level NLL:")
    for variant in VARIANTS:
        stats = variant_stats[variant]
        print(
            f"  {variant:12} count={stats['count']:5d} "
            f"mean_total_nll={stats['mean_total_nll']:.4f} median_total_nll={stats['median_total_nll']:.4f} "
            f"mean_nll_per_char={stats['mean_nll_per_char']:.6f} median_nll_per_char={stats['median_nll_per_char']:.6f} "
            f"mean_tokens={stats['mean_tokens']:.1f}"
        )

    print("\nRare vs typical deltas:")
    print(
        f"  good: pairs={rare_good['pairs']:5d} pct_rare_higher={rare_good['pct_rare_higher']:.1f}% "
        f"mean_delta={rare_good['mean_delta']:.4f} median_delta={rare_good['median_delta']:.4f}"
    )
    print(
        f"  bad : pairs={rare_bad['pairs']:5d} pct_rare_higher={rare_bad['pct_rare_higher']:.1f}% "
        f"mean_delta={rare_bad['mean_delta']:.4f} median_delta={rare_bad['median_delta']:.4f}"
    )
    print("\nRare vs typical deltas (per char):")
    print(
        f"  good: pairs={rare_good_char['pairs']:5d} pct_rare_higher={rare_good_char['pct_rare_higher']:.1f}% "
        f"mean_delta={rare_good_char['mean_delta']:.6f} median_delta={rare_good_char['median_delta']:.6f}"
    )
    print(
        f"  bad : pairs={rare_bad_char['pairs']:5d} pct_rare_higher={rare_bad_char['pct_rare_higher']:.1f}% "
        f"mean_delta={rare_bad_char['mean_delta']:.6f} median_delta={rare_bad_char['median_delta']:.6f}"
    )

    print("\nGood vs bad checks:")
    print(
        f"  typical: pairs={good_vs_bad_typical['pairs']:5d} pct_bad_higher={good_vs_bad_typical['pct_bad_higher']:.1f}% "
        f"mean_delta={good_vs_bad_typical['mean_delta']:.4f} median_delta={good_vs_bad_typical['median_delta']:.4f}"
    )
    print(
        f"  rare   : pairs={good_vs_bad_rare['pairs']:5d} pct_bad_higher={good_vs_bad_rare['pct_bad_higher']:.1f}% "
        f"mean_delta={good_vs_bad_rare['mean_delta']:.4f} median_delta={good_vs_bad_rare['median_delta']:.4f}"
    )
    print("\nGood vs bad checks (per char):")
    print(
        f"  typical: pairs={good_vs_bad_typical_char['pairs']:5d} pct_bad_higher={good_vs_bad_typical_char['pct_bad_higher']:.1f}% "
        f"mean_delta={good_vs_bad_typical_char['mean_delta']:.6f} median_delta={good_vs_bad_typical_char['median_delta']:.6f}"
    )
    print(
        f"  rare   : pairs={good_vs_bad_rare_char['pairs']:5d} pct_bad_higher={good_vs_bad_rare_char['pct_bad_higher']:.1f}% "
        f"mean_delta={good_vs_bad_rare_char['mean_delta']:.6f} median_delta={good_vs_bad_rare_char['median_delta']:.6f}"
    )

    print("\nRare penalty per swapped site (good pairs only):")
    print(
        f"  count={rare_penalty['count']:5d} "
        f"mean_per_swap_nll={rare_penalty['mean_per_swap_nll']:.4f} "
        f"median_per_swap_nll={rare_penalty['median_per_swap_nll']:.4f} "
        f"mean_per_swap_nll_per_char={rare_penalty['mean_per_swap_nll_per_char']:.6f} "
        f"median_per_swap_nll_per_char={rare_penalty['median_per_swap_nll_per_char']:.6f}"
    )

    if subtask_rows:
        print("\nLargest rare/typical NLL gaps by subtask (good sentences):")
        for row in subtask_rows:
            print(
                f"  {row['subtask']}: delta={row['delta']:.4f} "
                f"(typical_mean={row['typical_mean']:.4f}, rare_mean={row['rare_mean']:.4f})"
            )

    _save_single_json(out_path, summary)
    print(f"\nWrote combined summary+details to {out_path}")


if __name__ == "__main__":
    main()
