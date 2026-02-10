"""
Prune verb inventory frames using a prune suggestions CSV.

The CSV is expected to have columns: lemma, frame, prep, doc_count, status.
Frames whose status matches --drop_status are removed; verbs with no
remaining frames are dropped entirely.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def _norm(text: str) -> str:
    return (text or "").strip().lower()


def _load_inventory(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save_inventory(path: Path, data: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_prune_set(path: Path, drop_status: Set[str]) -> Set[Tuple[str, str, str]]:
    targets: Set[Tuple[str, str, str]] = set()
    drop_status = {_norm(s) for s in drop_status}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = _norm(row.get("status", ""))
            if status not in drop_status:
                continue
            lemma = _norm(row.get("lemma", ""))
            frame = _norm(row.get("frame", ""))
            prep = _norm(row.get("prep", ""))
            if not lemma or not frame:
                continue
            targets.add((lemma, frame, prep))
    return targets


def prune_inventory(
    inventory: Iterable[Dict],
    prune_targets: Set[Tuple[str, str, str]],
) -> Tuple[List[Dict], int, int]:
    """
    Returns (new_inventory, removed_frames, removed_verbs)
    """
    new_entries: List[Dict] = []
    removed_frames = 0
    removed_verbs = 0

    for entry in inventory:
        lemma = _norm(entry.get("lemma", ""))
        frames = entry.get("frames") or []
        kept_frames = []
        for frame in frames:
            kind = _norm(frame.get("type", ""))
            prep = _norm(frame.get("prep", ""))
            if (lemma, kind, prep) in prune_targets:
                removed_frames += 1
                continue
            kept_frames.append(frame)

        if not kept_frames:
            removed_verbs += 1
            continue

        new_entry = dict(entry)
        new_entry["frames"] = kept_frames
        new_entries.append(new_entry)

    new_entries.sort(key=lambda x: x.get("lemma", ""))
    return new_entries, removed_frames, removed_verbs


def main():
    ap = argparse.ArgumentParser(description="Prune verb inventory using a prune suggestions CSV.")
    ap.add_argument(
        "--inventory",
        default=".cache/verb_inventory/verb_inventory_b808c317202955bf.json",
        help="Input inventory JSON.",
    )
    ap.add_argument(
        "--prune_csv",
        default="results/coca_frames/inventory_prune_suggestions.csv",
        help="CSV produced by scripts/coca_frame_counts.py.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Path to write pruned inventory. Defaults to overwriting --inventory.",
    )
    ap.add_argument(
        "--drop_status",
        nargs="+",
        default=["unattested", "low"],
        help="Statuses to prune from the inventory.",
    )
    args = ap.parse_args()

    inv_path = Path(args.inventory)
    prune_path = Path(args.prune_csv)
    out_path = Path(args.out) if args.out else inv_path

    inventory = _load_inventory(inv_path)
    prune_targets = _load_prune_set(prune_path, set(args.drop_status))
    new_inventory, removed_frames, removed_verbs = prune_inventory(inventory, prune_targets)
    _save_inventory(out_path, new_inventory)

    print(
        f"Pruned inventory written to {out_path} | removed frames: {removed_frames} | "
        f"removed verbs: {removed_verbs}"
    )


if __name__ == "__main__":
    main()
