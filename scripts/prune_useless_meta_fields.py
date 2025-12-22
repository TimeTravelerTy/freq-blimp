import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def _prune_verb_swap(item: Dict[str, Any]) -> bool:
    changed = False
    for k in ("particle_i", "particle_old", "particle_new"):
        if k in item:
            item.pop(k, None)
            changed = True
    return changed


def _prune_record(rec: Dict[str, Any]) -> bool:
    meta = rec.get("meta")
    if not isinstance(meta, dict):
        return False
    changed = False
    for key in ("g_verb_swaps", "b_verb_swaps"):
        swaps = meta.get(key)
        if not isinstance(swaps, list):
            continue
        for it in swaps:
            if isinstance(it, dict):
                changed |= _prune_verb_swap(it)
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
            if isinstance(rec, dict) and _prune_record(rec):
                changed += 1
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return total, changed


def main() -> int:
    ap = argparse.ArgumentParser(description="Prune useless always-null fields from rare BLiMP processed datasets.")
    ap.add_argument("--pattern", default="data/processed/*rare_blimp*.jsonl", help="Glob of JSONL files to update in-place.")
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

