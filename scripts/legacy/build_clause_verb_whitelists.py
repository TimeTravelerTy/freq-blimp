import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _load_inventory_lemmas(path: Path) -> Set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    lemmas = set()
    for item in data:
        lemma = (item.get("lemma") or "").strip().lower()
        if lemma:
            lemmas.add(lemma)
    return lemmas


def _is_ok_lemma(lemma: str, inventory_lemmas: Set[str]) -> bool:
    if not lemma or not isinstance(lemma, str):
        return False
    if not lemma.isalpha():
        return False
    return lemma in inventory_lemmas


def main() -> int:
    ap = argparse.ArgumentParser(description="Build clause-embedding verb whitelists from COCA marker counts CSV.")
    ap.add_argument("--csv", default="results/coca_clause_markers/verb_clause_marker_counts.csv")
    ap.add_argument("--inventory", default="data/processed/verb_inventory_pruned_particles.json")
    ap.add_argument("--out", default="data/processed/clause_verb_whitelists.json")
    ap.add_argument(
        "--blocklist",
        default="data/processed/clause_verb_blocklist.json",
        help="Optional JSON file with {'global':[...]} lemmas to exclude from all lists.",
    )

    ap.add_argument("--doc_total_min", type=int, default=25)
    ap.add_argument("--doc_verb_total_min", type=int, default=1000)

    ap.add_argument("--doc_rate_that_min", type=float, default=0.005)
    ap.add_argument("--doc_rate_wh_total_min", type=float, default=0.005)
    ap.add_argument("--doc_rate_wh_what_min", type=float, default=0.005)
    ap.add_argument("--doc_rate_wh_who_min", type=float, default=0.005)
    ap.add_argument("--token_rate_that_min", type=float, default=0.0)

    args = ap.parse_args()

    csv_path = Path(args.csv)
    inv_path = Path(args.inventory)
    out_path = Path(args.out)
    blocklist_path = Path(args.blocklist) if args.blocklist else None

    inventory_lemmas = _load_inventory_lemmas(inv_path)
    blocked: Set[str] = set()
    if blocklist_path and blocklist_path.is_file():
        try:
            raw = json.loads(blocklist_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for lemma in raw.get("global") or []:
                    if isinstance(lemma, str) and lemma.strip():
                        blocked.add(lemma.strip().lower())
        except Exception:
            blocked = set()

    that: List[str] = []
    wh_total: List[str] = []
    wh: Dict[str, List[str]] = {"what": [], "who": []}
    both: List[str] = []

    def add_unique(lst: List[str], lemma: str) -> None:
        if lemma not in lst:
            lst.append(lemma)

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lemma = (row.get("verb") or "").strip().lower()
            if not _is_ok_lemma(lemma, inventory_lemmas):
                continue
            if lemma in blocked:
                continue
            doc_total = int(float(row.get("doc_total") or 0))
            doc_verb_total = int(float(row.get("doc_verb_total") or 0))
            if doc_total < args.doc_total_min:
                continue
            if doc_verb_total < args.doc_verb_total_min:
                continue

            doc_rate_that = float(row.get("doc_rate_that") or 0.0)
            doc_rate_wh_total = float(row.get("doc_rate_wh_total") or 0.0)
            doc_rate_wh_what = float(row.get("doc_rate_wh_what") or 0.0)
            doc_rate_wh_who = float(row.get("doc_rate_wh_who") or 0.0)
            token_rate_that = float(row.get("token_rate_that") or 0.0)

            is_that = doc_rate_that >= args.doc_rate_that_min and token_rate_that >= args.token_rate_that_min
            is_wh_total = doc_rate_wh_total >= args.doc_rate_wh_total_min
            is_wh_what = doc_rate_wh_what >= args.doc_rate_wh_what_min
            is_wh_who = doc_rate_wh_who >= args.doc_rate_wh_who_min

            if is_that:
                add_unique(that, lemma)
            if is_wh_total:
                add_unique(wh_total, lemma)
            if is_wh_what:
                add_unique(wh["what"], lemma)
            if is_wh_who:
                add_unique(wh["who"], lemma)
            if is_that and is_wh_total:
                add_unique(both, lemma)

    payload = {
        "meta": {
            "source_csv": str(csv_path),
            "inventory": str(inv_path),
            "blocklist": str(blocklist_path) if blocklist_path else None,
            "blocked_count": len(blocked),
            "doc_total_min": args.doc_total_min,
            "doc_verb_total_min": args.doc_verb_total_min,
            "doc_rate_that_min": args.doc_rate_that_min,
            "doc_rate_wh_total_min": args.doc_rate_wh_total_min,
            "doc_rate_wh_what_min": args.doc_rate_wh_what_min,
            "doc_rate_wh_who_min": args.doc_rate_wh_who_min,
            "token_rate_that_min": args.token_rate_that_min,
        },
        "that": sorted(that),
        "wh_total": sorted(wh_total),
        "wh": {k: sorted(v) for k, v in wh.items()},
        "both": sorted(both),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Wrote {out_path} | that={len(payload['that'])} wh_total={len(payload['wh_total'])} both={len(payload['both'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
