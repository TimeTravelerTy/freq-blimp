#!/usr/bin/env python3
"""Compare per-paradigm FreqBLiMP accuracy against original BLiMP accuracy.

The script is intentionally JSONL-streaming: it does not load score files into
memory. It is meant for acceptability score outputs from
``scripts/score_acceptability_methods.py``.
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


REGIMES = ("head", "tail", "xtail")

KNOWN_ARTIFACT_NOTES = {
    "inchoative": "known audit target: curated inchoative verb choices can make good/bad side accidentally acceptable",
    "causative": "known audit target: curated causative/inchoative contrast can leak accidental acceptability",
}


@dataclass
class Bucket:
    correct: int = 0
    total: int = 0
    phenomenon: str = ""
    field: str = ""
    source: str = ""
    source_rank: str = ""

    @property
    def accuracy(self) -> float:
        return float("nan") if self.total <= 0 else self.correct / self.total


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc


def collect_paths(patterns: Sequence[str]) -> List[Path]:
    paths = set()
    for pattern in patterns:
        matched = list(Path().glob(pattern))
        if matched:
            paths.update(p for p in matched if p.is_file())
        else:
            p = Path(pattern)
            if p.is_file():
                paths.add(p)
    return sorted(paths)


def source_rank(path: Path) -> str:
    """Return a sortable run identifier, falling back to the filename."""
    name = path.name
    match = re.match(r"(?P<rank>\d{8}(?:-\d{6})?(?:-[A-Za-z0-9]+)?)", name)
    return match.group("rank") if match else name


def infer_regime(rec: dict, path: Path) -> str:
    text = " ".join(
        str(part)
        for part in (
            rec.get("dataset_path", ""),
            rec.get("dataset_name", ""),
            path.name,
        )
    )
    for regime in REGIMES:
        if re.search(rf"(^|[/_-]){regime}($|[/_-])", text):
            return regime
        if f"freq_blimp_{regime}_" in text:
            return regime
    return "unknown"


def infer_paradigm(rec: dict) -> str:
    subtask = rec.get("subtask")
    if isinstance(subtask, str) and subtask:
        return subtask
    dataset_name = rec.get("dataset_name")
    if isinstance(dataset_name, str) and dataset_name.endswith(".jsonl"):
        return Path(dataset_name).stem
    dataset_path = rec.get("dataset_path")
    if isinstance(dataset_path, str) and dataset_path:
        stem = Path(dataset_path).stem
        if stem:
            return stem
    return "unknown"


def add_record(bucket: Bucket, rec: dict, path: Path, rank: str) -> None:
    correct = rec.get("correctness")
    if not isinstance(correct, int):
        return
    bucket.correct += correct
    bucket.total += 1
    phenomenon = rec.get("phenomenon")
    field = rec.get("field")
    if isinstance(phenomenon, str) and phenomenon:
        bucket.phenomenon = phenomenon
    if isinstance(field, str) and field:
        bucket.field = field
    bucket.source = str(path)
    bucket.source_rank = rank


def read_original(paths: Sequence[Path], method: Optional[str]) -> Dict[Tuple[str, str, str], Bucket]:
    buckets: DefaultDict[Tuple[str, str, str], Bucket] = defaultdict(Bucket)
    for path in paths:
        rank = source_rank(path)
        for rec in iter_jsonl(path):
            if method and rec.get("method") != method:
                continue
            if rec.get("variant") not in (None, "original"):
                continue
            model = rec.get("model")
            rec_method = rec.get("method")
            if not isinstance(model, str) or not isinstance(rec_method, str):
                continue
            paradigm = infer_paradigm(rec)
            add_record(buckets[(model, rec_method, paradigm)], rec, path, rank)
    return dict(buckets)


def read_freq(paths: Sequence[Path], method: Optional[str], latest_per_key: bool) -> Dict[Tuple[str, str, str, str], Bucket]:
    all_buckets: DefaultDict[Tuple[str, str, str, str, str], Bucket] = defaultdict(Bucket)
    for path in paths:
        rank = source_rank(path)
        for rec in iter_jsonl(path):
            if method and rec.get("method") != method:
                continue
            if rec.get("variant") not in (None, "freq"):
                continue
            model = rec.get("model")
            rec_method = rec.get("method")
            if not isinstance(model, str) or not isinstance(rec_method, str):
                continue
            regime = infer_regime(rec, path)
            paradigm = infer_paradigm(rec)
            add_record(all_buckets[(model, rec_method, regime, paradigm, rank)], rec, path, rank)

    if not latest_per_key:
        merged: DefaultDict[Tuple[str, str, str, str], Bucket] = defaultdict(Bucket)
        for (model, rec_method, regime, paradigm, _rank), bucket in all_buckets.items():
            target = merged[(model, rec_method, regime, paradigm)]
            target.correct += bucket.correct
            target.total += bucket.total
            target.phenomenon = target.phenomenon or bucket.phenomenon
            target.field = target.field or bucket.field
            target.source = ";".join(filter(None, [target.source, bucket.source]))
            target.source_rank = ";".join(filter(None, [target.source_rank, bucket.source_rank]))
        return dict(merged)

    latest: Dict[Tuple[str, str, str, str], Bucket] = {}
    for (model, rec_method, regime, paradigm, rank), bucket in all_buckets.items():
        key = (model, rec_method, regime, paradigm)
        prev = latest.get(key)
        if prev is None or rank > prev.source_rank:
            latest[key] = bucket
    return latest


def fmt_float(value: float, digits: int = 4) -> str:
    return "nan" if math.isnan(value) else f"{value:.{digits}f}"


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return float("nan") if not vals else sum(vals) / len(vals)


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: Sequence[dict], group_fields: Sequence[str]) -> List[dict]:
    grouped: DefaultDict[Tuple[str, ...], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in group_fields)].append(row)

    out: List[dict] = []
    for key, group in grouped.items():
        original_accs = [float(row["original_accuracy"]) for row in group]
        freq_accs = [float(row["freq_accuracy"]) for row in group]
        gap_pps = [float(row["gap_pp"]) for row in group]
        drop_pps = [float(row["drop_pp"]) for row in group]
        base = {field: value for field, value in zip(group_fields, key)}
        base.update(
            {
                "phenomenon": next((row["phenomenon"] for row in group if row["phenomenon"]), ""),
                "field": next((row["field"] for row in group if row["field"]), ""),
                "n_comparisons": len(group),
                "models": ",".join(sorted({row["model"] for row in group})),
                "regimes": ",".join(sorted({row["regime"] for row in group})),
                "original_accuracy_mean": fmt_float(mean(original_accs)),
                "freq_accuracy_mean": fmt_float(mean(freq_accs)),
                "gap_pp_mean": fmt_float(mean(gap_pps), 2),
                "drop_pp_mean": fmt_float(mean(drop_pps), 2),
                "drop_pp_max": fmt_float(max(drop_pps), 2),
                "drop_pp_min": fmt_float(min(drop_pps), 2),
                "artifact_note": next((row["artifact_note"] for row in group if row["artifact_note"]), ""),
            }
        )
        out.append(base)
    out.sort(key=lambda row: (-float(row["drop_pp_mean"]), row.get("paradigm", ""), row.get("regime", "")))
    return out


def print_top(rows: Sequence[dict], top: int) -> None:
    if top <= 0:
        return
    shown = rows[:top]
    if not shown:
        print("No comparable rows found.")
        return
    cols = ["paradigm", "phenomenon", "regimes", "models", "original_accuracy_mean", "freq_accuracy_mean", "drop_pp_mean", "drop_pp_max", "artifact_note"]
    widths = {col: len(col) for col in cols}
    for row in shown:
        for col in cols:
            widths[col] = min(44, max(widths[col], len(str(row.get(col, "")))))
    print("Top mean drops by paradigm:")
    print("  ".join(col.ljust(widths[col]) for col in cols))
    print("  ".join("-" * widths[col] for col in cols))
    for row in shown:
        print("  ".join(str(row.get(col, ""))[: widths[col]].ljust(widths[col]) for col in cols))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare per-paradigm FreqBLiMP acceptability accuracy against original BLiMP."
    )
    ap.add_argument(
        "--freq-pattern",
        action="append",
        default=None,
        help="Glob or file for FreqBLiMP acceptability JSONL. Repeatable.",
    )
    ap.add_argument(
        "--original-pattern",
        action="append",
        default=None,
        help="Glob or file for original BLiMP acceptability JSONL. Repeatable.",
    )
    ap.add_argument("--method", default="nll", help="Scoring method to compare, or empty string for all methods.")
    ap.add_argument(
        "--no-latest-per-key",
        action="store_true",
        help="Merge all matching freq files instead of keeping only the latest run per model/regime/paradigm.",
    )
    ap.add_argument("--output-dir", default="results/accuracy_gap", help="Directory for output CSVs.")
    ap.add_argument("--prefix", default="freq_vs_original", help="Output filename prefix.")
    ap.add_argument("--top", type=int, default=20, help="Number of top-drop aggregate rows to print.")
    args = ap.parse_args()

    method = args.method or None
    freq_patterns = args.freq_pattern or ["results/acceptability_pair_scores/*_freq_nll_acceptability.jsonl"]
    original_patterns = args.original_pattern or [
        "results/acceptability_pair_scores/20260408-*blimp_original_original_nll_acceptability.jsonl"
    ]
    freq_paths = collect_paths(freq_patterns)
    original_paths = collect_paths(original_patterns)
    if not freq_paths:
        raise SystemExit("No freq score files found.")
    if not original_paths:
        raise SystemExit("No original score files found.")

    original = read_original(original_paths, method)
    freq = read_freq(freq_paths, method, latest_per_key=not args.no_latest_per_key)

    detail_rows: List[dict] = []
    missing_original = 0
    for (model, rec_method, regime, paradigm), freq_bucket in sorted(freq.items()):
        orig_bucket = original.get((model, rec_method, paradigm))
        if orig_bucket is None:
            missing_original += 1
            continue
        freq_acc = freq_bucket.accuracy
        orig_acc = orig_bucket.accuracy
        gap_pp = (freq_acc - orig_acc) * 100
        drop_pp = (orig_acc - freq_acc) * 100
        detail_rows.append(
            {
                "model": model,
                "method": rec_method,
                "regime": regime,
                "paradigm": paradigm,
                "phenomenon": freq_bucket.phenomenon or orig_bucket.phenomenon,
                "field": freq_bucket.field or orig_bucket.field,
                "original_accuracy": fmt_float(orig_acc),
                "freq_accuracy": fmt_float(freq_acc),
                "gap_pp": fmt_float(gap_pp, 2),
                "drop_pp": fmt_float(drop_pp, 2),
                "original_correct": orig_bucket.correct,
                "original_total": orig_bucket.total,
                "freq_correct": freq_bucket.correct,
                "freq_total": freq_bucket.total,
                "original_source": orig_bucket.source,
                "freq_source": freq_bucket.source,
                "freq_source_rank": freq_bucket.source_rank,
                "artifact_note": KNOWN_ARTIFACT_NOTES.get(paradigm, ""),
            }
        )

    detail_rows.sort(key=lambda row: (-float(row["drop_pp"]), row["paradigm"], row["regime"], row["model"]))
    per_paradigm = aggregate_rows(detail_rows, ["paradigm"])
    per_paradigm_regime = aggregate_rows(detail_rows, ["paradigm", "regime"])

    out_dir = Path(args.output_dir)
    detail_fields = [
        "model",
        "method",
        "regime",
        "paradigm",
        "phenomenon",
        "field",
        "original_accuracy",
        "freq_accuracy",
        "gap_pp",
        "drop_pp",
        "original_correct",
        "original_total",
        "freq_correct",
        "freq_total",
        "freq_source_rank",
        "original_source",
        "freq_source",
        "artifact_note",
    ]
    agg_fields = [
        "paradigm",
        "phenomenon",
        "field",
        "n_comparisons",
        "models",
        "regimes",
        "original_accuracy_mean",
        "freq_accuracy_mean",
        "gap_pp_mean",
        "drop_pp_mean",
        "drop_pp_max",
        "drop_pp_min",
        "artifact_note",
    ]
    agg_regime_fields = ["paradigm", "regime"] + [f for f in agg_fields if f != "paradigm"]

    write_csv(out_dir / f"{args.prefix}_detail.csv", detail_rows, detail_fields)
    write_csv(out_dir / f"{args.prefix}_per_paradigm.csv", per_paradigm, agg_fields)
    write_csv(out_dir / f"{args.prefix}_per_paradigm_regime.csv", per_paradigm_regime, agg_regime_fields)

    print(
        f"Compared {len(detail_rows)} model/regime/paradigm rows "
        f"from {len(freq_paths)} freq files against {len(original_paths)} original files."
    )
    if missing_original:
        print(f"Skipped {missing_original} freq groups without a matching original model/method/paradigm.")
    print(f"Wrote: {out_dir / f'{args.prefix}_detail.csv'}")
    print(f"Wrote: {out_dir / f'{args.prefix}_per_paradigm.csv'}")
    print(f"Wrote: {out_dir / f'{args.prefix}_per_paradigm_regime.csv'}")
    print()
    print_top(per_paradigm, args.top)


if __name__ == "__main__":
    main()
