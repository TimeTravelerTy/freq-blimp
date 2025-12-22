import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


PRIMARY_ORDER = (
    "group",
    "phenomenon",
    "subtask",
    "idx",
    "good_original",
    "bad_original",
    "good_rare",
    "bad_rare",
    "meta",
)


def _reorder_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in PRIMARY_ORDER:
        if k in rec:
            out[k] = rec[k]
    for k, v in rec.items():
        if k in out:
            continue
        out[k] = v
    return out


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
            if not isinstance(rec, dict):
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            reordered = _reorder_record(rec)
            if list(reordered.keys()) != list(rec.keys()):
                changed += 1
            w.write(json.dumps(reordered, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return total, changed


def main() -> int:
    ap = argparse.ArgumentParser(description="Reorder top-level keys in rare BLiMP JSONL files.")
    ap.add_argument("--pattern", default="data/processed/*rare_blimp*.jsonl")
    args = ap.parse_args()

    paths = list(_iter_paths(args.pattern))
    if not paths:
        raise SystemExit(f"No files found for pattern: {args.pattern}")
    for path in paths:
        total, changed = _process_file(path)
        print(f"{path} ({changed}/{total} lines reordered)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

