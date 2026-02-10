import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


VARIANTS: Sequence[str] = ("good_original", "bad_original", "good_rare", "bad_rare")

_LEGACY_VARIANTS = {
    "good_original": "good_typical",
    "bad_original": "bad_typical",
}

def _canon_variant(variant: str) -> str:
    # Keep the `details[].variant` field as-is; this is for indexing into
    # per-entry Zipf aggregate maps inside the dataset meta.
    return _LEGACY_VARIANTS.get(variant, variant)


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    return float(statistics.median(xs))


def _iter_jsonl_rows(path: Path, wanted_rows: Optional[set[int]] = None) -> Iterable[Tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if wanted_rows is not None and i not in wanted_rows:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield i, rec


def _load_zipf_meta_by_row(data_path: Path, rows: set[int]) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    if not rows:
        return out
    for i, rec in _iter_jsonl_rows(data_path, wanted_rows=rows):
        meta = rec.get("meta") or {}
        out[i] = {
            "zipf_swapped_position_aggregates": meta.get("zipf_swapped_position_aggregates") or {},
            "zipf_swapped_position_deltas": meta.get("zipf_swapped_position_deltas") or {},
        }
    return out


def _zipf_variant_stats(details: List[dict]) -> Dict[str, dict]:
    """
    Aggregate per-entry Zipf summaries across the run, split by sentence variant.

    We summarize the distribution of per-entry `mean`/`median`/`min` Zipf over swapped positions.
    """
    stats: Dict[str, dict] = {}
    for variant in VARIANTS:
        subset = [it for it in details if it.get("variant") == variant]
        aggs = [it.get("zipf_swapped_position_agg") for it in subset]
        aggs = [a for a in aggs if isinstance(a, dict)]

        means = [a["mean"] for a in aggs if a.get("mean") is not None]
        medians = [a["median"] for a in aggs if a.get("median") is not None]
        mins = [a["min"] for a in aggs if a.get("min") is not None]
        ns = [a["n"] for a in aggs if isinstance(a.get("n"), int)]
        oovs = [a["oov_count"] for a in aggs if isinstance(a.get("oov_count"), int)]

        stats[variant] = {
            "count": len(subset),
            "count_with_iv": len(means),
            "mean_of_mean_zipf": _mean(means),
            "median_of_mean_zipf": _median(means),
            "mean_of_median_zipf": _mean(medians),
            "median_of_median_zipf": _median(medians),
            "mean_of_min_zipf": _mean(mins),
            "median_of_min_zipf": _median(mins),
            "mean_iv_token_count": _mean([float(x) for x in ns]),
            "mean_oov_count": _mean([float(x) for x in oovs]),
            "total_iv_token_count": int(sum(ns)) if ns else 0,
            "total_oov_count": int(sum(oovs)) if oovs else 0,
        }
    return stats


def _pairwise_zipf_stats(
    per_row: Dict[int, Dict[str, dict]],
    typical: str,
    rare: str,
    metric: str,
) -> dict:
    deltas: List[float] = []
    rare_higher = 0
    for variants in per_row.values():
        t = variants.get(typical)
        r = variants.get(rare)
        if not isinstance(t, dict) or not isinstance(r, dict):
            continue
        t_val = t.get(metric)
        r_val = r.get(metric)
        if t_val is None or r_val is None:
            continue
        d = float(r_val) - float(t_val)
        deltas.append(d)
        rare_higher += int(d > 0.0)
    total = len(deltas)
    return {
        "pairs": total,
        "pct_rare_higher": (rare_higher / total * 100.0) if total else float("nan"),
        "mean_delta": _mean(deltas),
        "median_delta": _median(deltas),
    }


def _zipf_rare_vs_typical(details: List[dict]) -> dict:
    per_row: Dict[int, Dict[str, dict]] = {}
    for it in details:
        row = it.get("row")
        variant = it.get("variant")
        agg = it.get("zipf_swapped_position_agg")
        if not isinstance(row, int) or variant not in VARIANTS or not isinstance(agg, dict):
            continue
        per_row.setdefault(row, {})[variant] = agg

    return {
        "good": {
            "mean_zipf": _pairwise_zipf_stats(per_row, "good_original", "good_rare", "mean"),
            "median_zipf": _pairwise_zipf_stats(per_row, "good_original", "good_rare", "median"),
            "min_zipf": _pairwise_zipf_stats(per_row, "good_original", "good_rare", "min"),
        },
        "bad": {
            "mean_zipf": _pairwise_zipf_stats(per_row, "bad_original", "bad_rare", "mean"),
            "median_zipf": _pairwise_zipf_stats(per_row, "bad_original", "bad_rare", "median"),
            "min_zipf": _pairwise_zipf_stats(per_row, "bad_original", "bad_rare", "min"),
        },
    }


def _zipf_good_vs_bad(details: List[dict]) -> dict:
    per_row: Dict[int, Dict[str, dict]] = {}
    for it in details:
        row = it.get("row")
        variant = it.get("variant")
        agg = it.get("zipf_swapped_position_agg")
        if not isinstance(row, int) or variant not in VARIANTS or not isinstance(agg, dict):
            continue
        per_row.setdefault(row, {})[variant] = agg

    def good_bad(variant_good: str, variant_bad: str, metric: str) -> dict:
        deltas: List[float] = []
        bad_higher = 0
        for variants in per_row.values():
            g = variants.get(variant_good)
            b = variants.get(variant_bad)
            if not isinstance(g, dict) or not isinstance(b, dict):
                continue
            g_val = g.get(metric)
            b_val = b.get(metric)
            if g_val is None or b_val is None:
                continue
            d = float(b_val) - float(g_val)
            deltas.append(d)
            bad_higher += int(d > 0.0)
        total = len(deltas)
        return {
            "pairs": total,
            "pct_bad_higher": (bad_higher / total * 100.0) if total else float("nan"),
            "mean_delta": _mean(deltas),
            "median_delta": _median(deltas),
        }

    return {
        "original": {
            "mean_zipf": good_bad("good_original", "bad_original", "mean"),
            "median_zipf": good_bad("good_original", "bad_original", "median"),
            "min_zipf": good_bad("good_original", "bad_original", "min"),
        },
        "rare": {
            "mean_zipf": good_bad("good_rare", "bad_rare", "mean"),
            "median_zipf": good_bad("good_rare", "bad_rare", "median"),
            "min_zipf": good_bad("good_rare", "bad_rare", "min"),
        },
    }


def update_run(path: Path, *, inplace: bool, backup_suffix: str) -> bool:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "details" not in obj or "data" not in obj:
        return False

    data_path = Path(obj["data"])
    if not data_path.exists():
        raise FileNotFoundError(f"{path}: data path not found: {data_path}")

    details: List[dict] = obj["details"]
    rows = {it["row"] for it in details if isinstance(it.get("row"), int)}
    zipf_meta_by_row = _load_zipf_meta_by_row(data_path, rows)

    changed = False
    for it in details:
        row = it.get("row")
        variant = it.get("variant")
        if not isinstance(row, int) or variant not in VARIANTS:
            continue
        meta = zipf_meta_by_row.get(row) or {}
        per_variant = (meta.get("zipf_swapped_position_aggregates") or {}).get(variant)
        if per_variant is None:
            per_variant = (meta.get("zipf_swapped_position_aggregates") or {}).get(_canon_variant(variant))
        if isinstance(per_variant, dict):
            if it.get("zipf_swapped_position_agg") != per_variant:
                it["zipf_swapped_position_agg"] = per_variant
                changed = True

        deltas = meta.get("zipf_swapped_position_deltas") or {}
        delta_key = None
        if variant in ("good_original", "good_rare"):
            delta_key = "good_rare_minus_original"
        elif variant in ("bad_original", "bad_rare"):
            delta_key = "bad_rare_minus_original"
        if delta_key is not None and delta_key not in deltas:
            # Backward compat for older datasets.
            if delta_key == "good_rare_minus_original":
                delta_key = "good_rare_minus_typical"
            elif delta_key == "bad_rare_minus_original":
                delta_key = "bad_rare_minus_typical"
        if delta_key is not None:
            d = deltas.get(delta_key)
            if isinstance(d, dict) and it.get("zipf_swapped_position_delta") != d:
                it["zipf_swapped_position_delta"] = d
                changed = True

    # Top-level rollups.
    zipf_variant_stats = _zipf_variant_stats(details)
    zipf_rare_vs_typical = _zipf_rare_vs_typical(details)
    zipf_good_vs_bad = _zipf_good_vs_bad(details)

    if obj.get("zipf_variant_stats") != zipf_variant_stats:
        obj["zipf_variant_stats"] = zipf_variant_stats
        changed = True
    if obj.get("zipf_rare_vs_typical") != zipf_rare_vs_typical:
        obj["zipf_rare_vs_typical"] = zipf_rare_vs_typical
        changed = True
    if obj.get("zipf_good_vs_bad") != zipf_good_vs_bad:
        obj["zipf_good_vs_bad"] = zipf_good_vs_bad
        changed = True

    if not changed:
        return False

    if inplace:
        if backup_suffix:
            backup_path = path.with_suffix(path.suffix + backup_suffix)
            backup_path.write_bytes(path.read_bytes())
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return True

    out_path = path.with_name(path.stem + ".with_zipf_aggs.json")
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add Zipf swapped-position aggregates/deltas to existing results/sentence_nll_runs/*.json outputs."
    )
    ap.add_argument(
        "--pattern",
        default="results/sentence_nll_runs/*.json",
        help="Glob pattern for run JSON files to update.",
    )
    ap.add_argument("--inplace", action="store_true", help="Update run JSON files in place.")
    ap.add_argument(
        "--backup-suffix",
        default=".bak",
        help="If --inplace, write backups with this suffix (set to empty to disable).",
    )
    args = ap.parse_args()

    paths = sorted(Path().glob(args.pattern))
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise SystemExit(f"No files matched: {args.pattern}")

    updated = 0
    for p in paths:
        try:
            changed = update_run(p, inplace=args.inplace, backup_suffix=args.backup_suffix)
        except FileNotFoundError as e:
            print(f"[Skip] {e}")
            continue
        if changed:
            updated += 1
            print(f"[Updated] {p}")
        else:
            print(f"[No-op]  {p}")
    print(f"\nDone. Updated {updated}/{len(paths)} files.")


if __name__ == "__main__":
    main()
