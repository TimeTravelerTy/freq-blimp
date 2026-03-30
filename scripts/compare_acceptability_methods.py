import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _collect_paths(inputs: Sequence[str], pattern: Optional[str]) -> List[Path]:
    paths = [Path(p) for p in inputs]
    if pattern:
        paths.extend(Path().glob(pattern))
    files = sorted({p for p in paths if p.is_file()})
    if not files:
        raise SystemExit("No score files found. Use --inputs and/or --pattern.")
    return files


def _safe_accuracy(correct: int, total: int) -> float:
    if total <= 0:
        return float("nan")
    return correct / total


def _fmt_acc(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def _print_table(rows: List[dict]) -> None:
    header = ["model", "dataset", "variant", "method", "accuracy", "correct", "total"]
    widths = {key: len(key) for key in header}
    for row in rows:
        for key in header:
            widths[key] = max(widths[key], len(str(row[key])))
    print("  ".join(key.ljust(widths[key]) for key in header))
    print("  ".join("-" * widths[key] for key in header))
    for row in rows:
        print("  ".join(str(row[key]).ljust(widths[key]) for key in header))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare minimal-pair acceptability-scoring outputs across methods/models/datasets."
    )
    ap.add_argument("--inputs", nargs="*", default=[], help="Score JSONL files.")
    ap.add_argument(
        "--pattern",
        default="results/acceptability_pair_scores/*_acceptability.jsonl",
        help="Glob pattern for score files.",
    )
    ap.add_argument("--output-csv", default=None, help="Optional summary CSV path.")
    ap.add_argument("--output-json", default=None, help="Optional summary JSON path.")
    args = ap.parse_args()

    paths = _collect_paths(args.inputs, args.pattern)
    grouped: Dict[Tuple[str, str, str, str], Dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )

    for path in paths:
        for rec in _iter_jsonl(path):
            method = rec.get("method")
            model = rec.get("model")
            dataset = rec.get("dataset_name") or Path(str(rec.get("dataset_path", ""))).name
            variant = rec.get("variant")
            correct = rec.get("correctness")
            if not isinstance(method, str) or not isinstance(model, str) or not isinstance(dataset, str):
                continue
            if not isinstance(variant, str):
                variant = "unknown"
            if not isinstance(correct, int):
                continue
            key = (model, dataset, variant, method)
            grouped[key]["correct"] += correct
            grouped[key]["total"] += 1

    rows: List[dict] = []
    for (model, dataset, variant, method), bucket in grouped.items():
        acc = _safe_accuracy(bucket["correct"], bucket["total"])
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "variant": variant,
                "method": method,
                "accuracy": _fmt_acc(acc),
                "correct": bucket["correct"],
                "total": bucket["total"],
                "_accuracy_sort": -1.0 if math.isnan(acc) else acc,
            }
        )
    rows.sort(key=lambda row: (row["dataset"], row["model"], -row["_accuracy_sort"], row["method"]))

    if not rows:
        raise SystemExit("No valid score rows found in the provided files.")

    _print_table(rows)

    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["model", "dataset", "variant", "method", "accuracy", "correct", "total"],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row[k] for k in writer.fieldnames})
        print(f"\nSaved CSV summary to {csv_path}")

    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {k: row[k] for k in ["model", "dataset", "variant", "method", "accuracy", "correct", "total"]}
            for row in rows
        ]
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Saved JSON summary to {json_path}")


if __name__ == "__main__":
    main()
