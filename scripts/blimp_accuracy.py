import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _iter_jsonl(path: str, limit: Optional[int]) -> Iterable[dict]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec["_row"] = i
            yield rec


def _pick_field(explicit: Optional[str], candidates: List[str], rec: dict) -> Optional[str]:
    if explicit:
        return explicit
    for c in candidates:
        if c in rec and rec[c] is not None:
            return c
    return None


def _group_key(rec: dict, field: Optional[str], fallback: str) -> str:
    if field:
        val = rec.get(field)
        if val is not None and val != "":
            return str(val)
    return fallback


def _update_stats(stats: Dict[str, Dict[str, int]], key: str, correct: int) -> None:
    bucket = stats.setdefault(key, {"correct": 0, "total": 0})
    bucket["correct"] += correct
    bucket["total"] += 1


def _accuracy(stats: Dict[str, int]) -> float:
    total = stats.get("total", 0)
    if total <= 0:
        return float("nan")
    return stats.get("correct", 0) / total


def _print_stats(title: str, stats: Dict[str, Dict[str, int]]) -> None:
    print(f"\n{title}")
    for key, bucket in sorted(stats.items(), key=lambda kv: (-kv[1]["total"], kv[0])):
        acc = _accuracy(bucket)
        print(f"  {key}: acc={acc:.4f} ({bucket['correct']}/{bucket['total']})")


def _default_out_path(scores_path: str) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = Path(scores_path).name.rsplit(".", 1)[0]
    name = f"{ts}_{base}_pair-scores-accuracy.json"
    return Path("results") / "blimp_accuracy_runs" / name


def _to_rows(stats: Dict[str, Dict[str, int]]) -> List[dict]:
    rows = []
    for key, bucket in stats.items():
        rows.append(
            {
                "key": key,
                "accuracy": _accuracy(bucket),
                "correct": bucket["correct"],
                "total": bucket["total"],
            }
        )
    rows.sort(key=lambda r: (-r["accuracy"], -r["total"], r["key"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute BLiMP accuracy from a blimp_pair_scores JSONL file."
    )
    ap.add_argument("--scores", required=True, help="Path to a blimp_pair_scores JSONL file.")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of rows.")
    ap.add_argument("--good-nll-field", default=None, help="Field for good sentence NLL.")
    ap.add_argument("--bad-nll-field", default=None, help="Field for bad sentence NLL.")
    ap.add_argument(
        "--normalize-by",
        default="none",
        choices=["none", "char", "token"],
        help="Normalization for NLL comparison: none (raw), char, or token.",
    )
    ap.add_argument("--phenomenon-field", default=None, help="Field for phenomenon.")
    ap.add_argument("--subtask-field", default=None, help="Field for subtask/config.")
    ap.add_argument("--output", default=None, help="Optional JSON output path for metrics.")
    args = ap.parse_args()

    overall = {"correct": 0, "total": 0}
    by_phenomenon: Dict[str, Dict[str, int]] = {}
    by_subtask: Dict[str, Dict[str, int]] = {}
    by_pair: Dict[str, Dict[str, int]] = {}

    good_field = args.good_nll_field
    bad_field = args.bad_nll_field
    phenomenon_field = args.phenomenon_field
    subtask_field = args.subtask_field
    variant = None
    skipped = 0
    scored = 0

    for rec in _iter_jsonl(args.scores, args.limit):
        if variant is None and rec.get("variant") is not None:
            variant = str(rec.get("variant"))

        if good_field is None:
            good_field = _pick_field(good_field, ["good_total_nll"], rec)
        if bad_field is None:
            bad_field = _pick_field(bad_field, ["bad_total_nll"], rec)
        if phenomenon_field is None:
            phenomenon_field = _pick_field(phenomenon_field, ["phenomenon", "group"], rec)
        if subtask_field is None:
            subtask_field = _pick_field(subtask_field, ["subtask"], rec)

        if not good_field or not bad_field:
            skipped += 1
            continue

        good_score = rec.get(good_field)
        bad_score = rec.get(bad_field)
        if not isinstance(good_score, (int, float)) or not isinstance(bad_score, (int, float)):
            skipped += 1
            continue

        if args.normalize_by == "char":
            good_len = len(rec.get("good_text") or "") if isinstance(rec.get("good_text"), str) else 0
            bad_len = len(rec.get("bad_text") or "") if isinstance(rec.get("bad_text"), str) else 0
            if good_len <= 0 or bad_len <= 0:
                skipped += 1
                continue
            good_score = good_score / good_len
            bad_score = bad_score / bad_len
        elif args.normalize_by == "token":
            good_len = rec.get("good_token_count")
            bad_len = rec.get("bad_token_count")
            if not isinstance(good_len, int) or not isinstance(bad_len, int) or good_len <= 0 or bad_len <= 0:
                skipped += 1
                continue
            good_score = good_score / good_len
            bad_score = bad_score / bad_len

        correct = 1 if good_score < bad_score else 0
        overall["correct"] += correct
        overall["total"] += 1
        phen = _group_key(rec, phenomenon_field, "unknown")
        subtask = _group_key(rec, subtask_field, "unknown")
        _update_stats(by_phenomenon, phen, correct)
        _update_stats(by_subtask, subtask, correct)
        _update_stats(by_pair, f"{phen}|{subtask}", correct)
        scored += 1

    if scored == 0:
        raise ValueError("No valid score rows found in the scores file.")

    print("BLiMP accuracy (good_total_nll < bad_total_nll):")
    print(f"  overall: acc={_accuracy(overall):.4f} ({overall['correct']}/{overall['total']})")
    if skipped:
        print(f"  skipped: {skipped}")
    _print_stats("By phenomenon", by_phenomenon)
    _print_stats("By subtask", by_subtask)
    _print_stats("By phenomenon|subtask", by_pair)

    out = {
        "scores_path": args.scores,
        "variant": variant,
        "normalize_by": args.normalize_by,
        "good_nll_field": good_field,
        "bad_nll_field": bad_field,
        "phenomenon_field": phenomenon_field,
        "subtask_field": subtask_field,
        "overall_accuracy": _accuracy(overall),
        "overall_correct": overall["correct"],
        "overall_total": overall["total"],
        "by_phenomenon": _to_rows(by_phenomenon),
        "by_subtask": _to_rows(by_subtask),
        "by_phenomenon_subtask": _to_rows(by_pair),
        "skipped": skipped,
    }
    out_path = Path(args.output) if args.output else _default_out_path(args.scores)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
