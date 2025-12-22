import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


VARIANTS = ("good_original", "bad_original", "good_rare", "bad_rare")

_LEGACY_VARIANTS = {
    "good_original": "good_typical",
    "bad_original": "bad_typical",
}


def _get_variant(rec: dict, key: str):
    val = rec.get(key)
    if val is None and key in _LEGACY_VARIANTS:
        val = rec.get(_LEGACY_VARIANTS[key])
    return val


def load_jsonl(path: Path, limit: Optional[int] = None) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec["_row"] = i
            records.append(rec)
            if limit is not None and len(records) >= limit:
                break
    return records


def _short(rec: dict) -> str:
    g = _get_variant(rec, "good_original")
    b = _get_variant(rec, "bad_original")
    gr = _get_variant(rec, "good_rare")
    br = _get_variant(rec, "bad_rare")
    return (
        f"[{rec.get('group')}|{rec.get('subtask')}|row={rec.get('_row')}|idx={rec.get('idx')}]\n"
        f"  good_original: {g}\n"
        f"  bad_original : {b}\n"
        f"  good_rare   : {gr}\n"
        f"  bad_rare    : {br}\n"
    )


def _by_subtask(recs: Iterable[dict]) -> Counter:
    return Counter((r.get("group"), r.get("phenomenon"), r.get("subtask")) for r in recs)


def spotcheck(records: List[dict], *, sample_k: int, seed: int) -> int:
    rng = random.Random(seed)

    def has_text(x) -> bool:
        return isinstance(x, str) and bool(x.strip())

    same_typical: List[dict] = []
    same_rare: List[dict] = []
    missing_rare: List[dict] = []
    no_terminal_punct: List[Tuple[str, dict]] = []
    plural_possessive_merges: List[Tuple[str, dict]] = []

    re_plural_merge = re.compile(r"\b\w+s'(?!s)\w")
    re_no_terminal = re.compile(r"[^.!?]$")

    for rec in records:
        gt = _get_variant(rec, "good_original")
        bt = _get_variant(rec, "bad_original")
        gr = _get_variant(rec, "good_rare")
        br = _get_variant(rec, "bad_rare")

        if has_text(gt) and has_text(bt) and gt == bt:
            same_typical.append(rec)

        if not has_text(gr) or not has_text(br):
            missing_rare.append(rec)
            continue

        if gr == br:
            same_rare.append(rec)

        for variant in ("good_rare", "bad_rare"):
            s = rec.get(variant)
            if not has_text(s):
                continue
            s2 = s.strip()
            if re_no_terminal.search(s2):
                no_terminal_punct.append((variant, rec))
            if re_plural_merge.search(s2):
                plural_possessive_merges.append((variant, rec))

    print(f"records: {len(records)}")
    print(f"rare missing (good_rare/bad_rare None/empty): {len(missing_rare)}")
    print(f"good_original == bad_original: {len(same_typical)}")
    print(f"good_rare == bad_rare (when present): {len(same_rare)}")
    print(f"rare w/o terminal punctuation: {len(no_terminal_punct)} (counting variants)")
    print(f"rare plural-possessive spacing merge (e.g., customers'wife): {len(plural_possessive_merges)} (counting variants)")

    def _print_issue(name: str, recs: List[dict]) -> None:
        if not recs:
            return
        print(f"\n[{name}] count={len(recs)}")
        counts = _by_subtask(recs)
        for (group, phen, subtask), c in counts.most_common(8):
            print(f"  {c:5d}  {group}|{phen}|{subtask}")
        for rec in rng.sample(recs, min(sample_k, len(recs))):
            print(_short(rec), end="")

    def _print_issue_variants(name: str, items: List[Tuple[str, dict]]) -> None:
        if not items:
            return
        print(f"\n[{name}] count={len(items)}")
        counts = _by_subtask([rec for _, rec in items])
        for (group, phen, subtask), c in counts.most_common(8):
            print(f"  {c:5d}  {group}|{phen}|{subtask}")
        for variant, rec in rng.sample(items, min(sample_k, len(items))):
            print(f"variant={variant}")
            print(_short(rec), end="")

    _print_issue("same_typical", same_typical)
    _print_issue("missing_rare", missing_rare)
    _print_issue("same_rare", same_rare)
    _print_issue_variants("no_terminal_punct", no_terminal_punct)
    _print_issue_variants("plural_possessive_merge", plural_possessive_merges)

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Spot-check a generated RARE-BLiMP JSONL dataset.")
    ap.add_argument("--path", required=True, help="Path to the JSONL dataset.")
    ap.add_argument("--limit", type=int, default=None, help="Only read the first N records.")
    ap.add_argument("--sample", type=int, default=3, help="Examples to print per issue.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = Path(args.path)
    records = load_jsonl(path, args.limit)
    if not records:
        raise SystemExit(f"No records loaded from {path}")
    return spotcheck(records, sample_k=args.sample, seed=args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
