import os
import sys
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.zipf_aggregates import add_zipf_aggregates


def _iter_paths(pattern: str) -> List[Path]:
    paths = sorted(Path(p) for p in Path().glob(pattern) if Path(p).is_file())
    if not paths:
        raise SystemExit(f"No files found for pattern: {pattern}")
    return paths


def _process_file(path: Path) -> Tuple[int, int, Path]:
    total = 0
    changed = 0
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with path.open("r", encoding="utf-8") as r, tmp_path.open("w", encoding="utf-8") as w:
        for line in r:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            before_meta = rec.get("meta") or {}
            before = (
                before_meta.get("zipf_swapped_position_aggregates"),
                before_meta.get("zipf_swapped_position_deltas"),
            )
            rec2 = add_zipf_aggregates(rec)
            after_meta = rec2.get("meta") or {}
            after = (
                after_meta.get("zipf_swapped_position_aggregates"),
                after_meta.get("zipf_swapped_position_deltas"),
            )
            if before != after:
                changed += 1
            w.write(json.dumps(rec2, ensure_ascii=False) + "\n")
    return total, changed, tmp_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add Zipf aggregates (mean/median/min) over swapped positions to processed rare-BLIMP JSONL files."
    )
    ap.add_argument(
        "--pattern",
        default="data/processed/*rare_blimp*.jsonl",
        help="Glob pattern for JSONL files to update.",
    )
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Replace files in-place (writes via a temporary file then renames).",
    )
    ap.add_argument(
        "--backup-suffix",
        default=".bak",
        help="If --inplace, write a backup copy with this suffix (set to empty to disable).",
    )
    args = ap.parse_args()

    paths = _iter_paths(args.pattern)
    for path in paths:
        total, changed, tmp_path = _process_file(path)
        if args.inplace:
            if args.backup_suffix:
                backup_path = path.with_suffix(path.suffix + args.backup_suffix)
                backup_path.write_bytes(path.read_bytes())
            tmp_path.replace(path)
            out_path = path
        else:
            out_path = path.with_name(path.stem + ".with_zipf_aggs.jsonl")
            tmp_path.replace(out_path)
        print(f"{path} -> {out_path} ({changed}/{total} records updated)")


if __name__ == "__main__":
    main()
