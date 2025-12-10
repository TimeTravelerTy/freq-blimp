"""
Augment verb inventory intr_pp frames using COCA missing-candidate CSV.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _load_inventory(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save_inventory(path: Path, data: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _has_intr_pp(entry: Dict, prep: str) -> bool:
    for frame in entry.get("frames", ()):
        if frame.get("type") == "intr_pp" and (frame.get("prep") or "").lower() == prep:
            return True
    return False


def augment_inventory(
    inventory: List[Dict],
    missing_csv: Path,
    *,
    min_doc: int,
    restrict_existing: bool,
    allow_new_verbs: bool,
) -> Tuple[List[Dict], int, int]:
    """
    Returns (new_inventory, added_frames, added_verbs)
    """
    by_lemma: Dict[str, Dict] = {}
    for entry in inventory:
        lemma = (entry.get("lemma") or "").strip().lower()
        if not lemma:
            continue
        by_lemma[lemma] = entry

    added_frames = 0
    added_verbs = 0

    with missing_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lemma = (row.get("verb") or "").strip().lower()
            prep = (row.get("prep") or "").strip().lower()
            try:
                doc_count = int(row.get("doc_count", "0"))
            except ValueError:
                continue
            if not lemma or not prep:
                continue
            if doc_count < min_doc:
                continue

            entry = by_lemma.get(lemma)
            if entry is None:
                if restrict_existing and not allow_new_verbs:
                    continue
                entry = {"lemma": lemma, "frames": []}
                by_lemma[lemma] = entry
                added_verbs += 1

            if _has_intr_pp(entry, prep):
                continue

            frames = entry.setdefault("frames", [])
            frames.append({"type": "intr_pp", "prep": prep})
            added_frames += 1

    # Stable sorted output by lemma for readability.
    new_inventory = [by_lemma[lemma] for lemma in sorted(by_lemma.keys())]
    return new_inventory, added_frames, added_verbs


def main():
    ap = argparse.ArgumentParser(
        description="Augment verb inventory with high-confidence intr_pp pairs from COCA."
    )
    ap.add_argument(
        "--inventory",
        default=".cache/verb_inventory/verb_inventory_b808c317202955bf.json",
        help="Input inventory JSON.",
    )
    ap.add_argument(
        "--missing_csv",
        default="results/coca_intr_pp/missing_intr_pp_candidates.csv",
        help="CSV produced by scripts/coca_intr_pp_validate.py.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Path to write augmented inventory. Defaults to overwriting --inventory.",
    )
    ap.add_argument(
        "--min_doc",
        type=int,
        default=25,
        help="Minimum doc_count to accept a missing intr_pp candidate.",
    )
    ap.add_argument(
        "--restrict_existing",
        action="store_true",
        default=False,
        help="Only add frames for verbs already present in the inventory.",
    )
    ap.add_argument(
        "--allow_new_verbs",
        action="store_true",
        default=False,
        help="Permit adding new verb entries if restrict_existing is False.",
    )
    args = ap.parse_args()

    inv_path = Path(args.inventory)
    missing_path = Path(args.missing_csv)
    out_path = Path(args.out) if args.out else None

    inventory = _load_inventory(inv_path)
    new_inventory, added_frames, added_verbs = augment_inventory(
        inventory,
        missing_path,
        min_doc=args.min_doc,
        restrict_existing=args.restrict_existing,
        allow_new_verbs=args.allow_new_verbs,
    )
    target_path = out_path if out_path is not None else inv_path
    _save_inventory(target_path, new_inventory)

    print(
        f"Augmented inventory written to {target_path} | "
        f"added intr_pp frames: {added_frames} | new verbs: {added_verbs}"
    )


if __name__ == "__main__":
    main()
