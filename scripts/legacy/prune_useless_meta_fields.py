import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_KEEP_SWAP_KEYS = ("i", "old", "new", "tag", "lemma")


def _slim_swap(entry: Dict[str, Any], pos_label: str) -> Optional[Dict[str, Any]]:
    out: Dict[str, Any] = {}
    for key in _KEEP_SWAP_KEYS:
        val = entry.get(key)
        if val is not None:
            out[key] = val
    if "old" not in out or "new" not in out:
        return None
    out["pos"] = pos_label
    return out


def _collect_swaps(meta: Dict[str, Any], prefix: str) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []

    base = meta.get(f"{prefix}_swaps")
    if isinstance(base, list):
        for item in base:
            if not isinstance(item, dict):
                continue
            pos = item.get("pos")
            if not isinstance(pos, str) or not pos.strip():
                pos = "noun"
            slim = _slim_swap(item, pos.strip().lower())
            if slim is not None:
                merged.append(slim)

    for pos_label, key in (
        ("adjective", f"{prefix}_adj_swaps"),
        ("verb", f"{prefix}_verb_swaps"),
    ):
        swaps = meta.get(key)
        if not isinstance(swaps, list):
            continue
        for item in swaps:
            if not isinstance(item, dict):
                continue
            slim = _slim_swap(item, pos_label)
            if slim is not None:
                merged.append(slim)

    merged.sort(key=lambda it: (it.get("i", -1), it.get("pos", "")))
    return merged


def _slim_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "g_swaps": _collect_swaps(meta, "g"),
        "b_swaps": _collect_swaps(meta, "b"),
    }
    reason = meta.get("swap_failed_reason")
    if reason:
        out["swap_failed_reason"] = str(reason)
    for key in ("zipf_swapped_position_aggregates", "zipf_swapped_position_deltas"):
        val = meta.get(key)
        if isinstance(val, dict):
            out[key] = val
    return out


def _prune_record(rec: Dict[str, Any]) -> bool:
    meta = rec.get("meta")
    if not isinstance(meta, dict):
        return False
    slim_meta = _slim_meta(meta)
    if slim_meta == meta:
        return False
    rec["meta"] = slim_meta
    return True


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
    ap = argparse.ArgumentParser(
        description="Slim processed freq BLiMP metadata to unified swaps + Zipf aggregates."
    )
    ap.add_argument(
        "--pattern",
        default="data/processed/*freq_blimp*.jsonl",
        help="Glob of JSONL files to update in-place.",
    )
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
