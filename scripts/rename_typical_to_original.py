import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def _rename_key(d: Dict[str, Any], old: str, new: str) -> bool:
    if old not in d or new in d:
        return False
    d[new] = d.pop(old)
    return True


def _rename_in_meta(meta: Dict[str, Any]) -> bool:
    changed = False

    aggs = meta.get("zipf_swapped_position_aggregates")
    if isinstance(aggs, dict):
        changed |= _rename_key(aggs, "good_typical", "good_original")
        changed |= _rename_key(aggs, "bad_typical", "bad_original")

    deltas = meta.get("zipf_swapped_position_deltas")
    if isinstance(deltas, dict):
        changed |= _rename_key(deltas, "good_rare_minus_typical", "good_rare_minus_original")
        changed |= _rename_key(deltas, "bad_rare_minus_typical", "bad_rare_minus_original")

    return changed


def _rename_record(rec: Dict[str, Any]) -> bool:
    changed = False
    changed |= _rename_key(rec, "good_typical", "good_original")
    changed |= _rename_key(rec, "bad_typical", "bad_original")
    meta = rec.get("meta")
    if isinstance(meta, dict):
        changed |= _rename_in_meta(meta)
    return changed


def _iter_paths(pattern: str) -> Iterable[Path]:
    for p in sorted(Path().glob(pattern)):
        if p.is_file():
            yield p


def _process_file(path: Path) -> Tuple[int, int]:
    total = 0
    changed = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with path.open("r", encoding="utf-8") as r, tmp.open("w", encoding="utf-8") as w:
        for line in r:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            if isinstance(rec, dict) and _rename_record(rec):
                changed += 1
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return total, changed


def main() -> int:
    ap = argparse.ArgumentParser(description="Rename good/bad_typical -> good/bad_original in rare BLiMP JSONL files.")
    ap.add_argument("--pattern", default="data/processed/*rare_blimp*.jsonl", help="Glob pattern of JSONL files to update in-place.")
    args = ap.parse_args()

    paths = list(_iter_paths(args.pattern))
    if not paths:
        raise SystemExit(f"No files found for pattern: {args.pattern}")
    for path in paths:
        total, changed = _process_file(path)
        print(f"{path} ({changed}/{total} records updated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

