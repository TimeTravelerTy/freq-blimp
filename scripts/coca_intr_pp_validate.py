"""
Validate intr_pp verb–preposition frames against COCA wlp files and surface
high-frequency missing collocations.
"""

import argparse
import json
import sys
import tarfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence, Tuple

# CLAWS tags for lexical verbs; exclude modals (vm) to focus on content verbs.
_VERB_PREFIXES = ("vv", "vd", "vh", "vb")
# Preposition tags in COCA wlp use the CLAWS "i" prefix.
_PREP_PREFIX = "i"
# Align with src/verb_inventory.py
_PREP_KEYWORDS = {
    "about",
    "across",
    "after",
    "against",
    "along",
    "around",
    "at",
    "before",
    "behind",
    "by",
    "down",
    "for",
    "from",
    "in",
    "into",
    "near",
    "of",
    "off",
    "on",
    "onto",
    "over",
    "through",
    "to",
    "toward",
    "under",
    "up",
    "with",
}
_PREP_KEYWORDS_SET = set(_PREP_KEYWORDS)


def _iter_wlp_tokens(tar_path: Path) -> Iterator[Tuple[str, str, str]]:
    """
    Yield (doc_id, lemma, pos) tokens from all COCA wlp archives.
    """
    with tarfile.open(tar_path) as tar:
        for member in tar:
            if not member.name.lower().endswith(".zip"):
                continue
            with tar.extractfile(member) as fzip:
                if fzip is None:
                    continue
                with zipfile.ZipFile(fzip) as zf:
                    # Deterministic order for reproducibility.
                    for fname in sorted(zf.namelist()):
                        with zf.open(fname) as fin:
                            current_doc = None
                            for raw_line in fin:
                                parts = raw_line.decode("utf-8", "ignore").rstrip("\n").split("\t")
                                if len(parts) != 4:
                                    continue
                                doc_id, _surface, lemma, pos = parts
                                if doc_id != current_doc:
                                    current_doc = doc_id
                                yield current_doc, lemma, pos


def _is_verb(pos: str) -> bool:
    return pos.startswith(_VERB_PREFIXES)


def _is_prep(pos: str) -> bool:
    return pos.startswith(_PREP_PREFIX)


def _token_breaks_sequence(lemma: str, pos: str) -> bool:
    """
    Return True if the token should break adjacency (punctuation, markup, sentinels).
    """
    if not lemma or lemma.startswith("@@"):
        return True
    if lemma.startswith("<"):
        return True
    if pos and pos[0] == "y":  # punctuation in CLAWS
        return True
    return False


def load_inventory_intr_pp(path: Path) -> Dict[Tuple[str, str], bool]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    intr_pp = {}
    for item in data:
        lemma = (item.get("lemma") or "").strip().lower()
        if not lemma:
            continue
        for frame in item.get("frames", ()):
            if frame.get("type") != "intr_pp":
                continue
            prep = (frame.get("prep") or "").strip().lower()
            if prep:
                intr_pp[(lemma, prep)] = True
    return intr_pp


def _collect_zip_file_lists(tar_path: Path) -> Tuple[Dict[str, Sequence[str]], int]:
    """
    Pre-scan to count files for ETA and keep deterministic ordering.
    """
    file_lists: Dict[str, Sequence[str]] = {}
    total = 0
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            if not member.name.lower().endswith(".zip"):
                continue
            with tar.extractfile(member) as fzip:
                if fzip is None:
                    continue
                with zipfile.ZipFile(fzip) as zf:
                    names = sorted(zf.namelist())
                    file_lists[member.name] = names
                    total += len(names)
    return file_lists, total


def tally_v_prep(
    wlp_tar: Path,
    allowed_preps: Optional[Sequence[str]] = None,
    log_interval_sec: float = 10.0,
) -> Tuple[Counter, Counter]:
    """
    Count adjacent verb+preposition pairs across all wlp files.
    Returns (token_counts, doc_counts).
    """
    allowed = set(p.lower() for p in allowed_preps) if allowed_preps else None
    token_counts: Counter = Counter()
    doc_counts: Counter = Counter()

    file_lists, total_files = _collect_zip_file_lists(wlp_tar)
    start = time.time()
    next_log = start
    processed_files = 0
    bar_width = 30

    def _print_progress(force: bool = False) -> None:
        now = time.time()
        nonlocal next_log
        if not force and now < next_log:
            return
        elapsed = now - start
        rate = processed_files / elapsed if elapsed > 0 else 0
        remaining = max(total_files - processed_files, 0)
        eta = (remaining / rate) if rate > 0 else 0
        frac = processed_files / total_files if total_files else 1.0
        filled = int(bar_width * frac)
        bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"
        msg = (
            f"\r{bar} {processed_files}/{total_files} files "
            f"elapsed {elapsed/60:.1f}m eta {eta/60:.1f}m"
        )
        print(msg, end="", file=sys.stderr, flush=True)
        next_log = now + log_interval_sec

    with tarfile.open(wlp_tar) as tar:
        members = {m.name: m for m in tar.getmembers() if m.name in file_lists}
        for member_name, names in file_lists.items():
            member = members.get(member_name)
            if member is None:
                continue
            with tar.extractfile(member) as fzip:
                if fzip is None:
                    continue
                with zipfile.ZipFile(fzip) as zf:
                    for fname in names:
                        processed_files += 1
                        _print_progress(force=(processed_files == 1))

                        with zf.open(fname) as fin:
                            current_doc = None
                            doc_seen = set()
                            prev_verb: Optional[str] = None
                            for raw_line in fin:
                                parts = raw_line.decode("utf-8", "ignore").rstrip("\n").split("\t")
                                if len(parts) != 4:
                                    continue
                                doc_id, _surface, lemma, pos = parts
                                if doc_id != current_doc:
                                    current_doc = doc_id
                                    doc_seen = set()
                                    prev_verb = None

                                lemma_norm = lemma.lower()

                                if _token_breaks_sequence(lemma_norm, pos):
                                    prev_verb = None
                                    continue

                                if _is_verb(pos):
                                    prev_verb = lemma_norm
                                    continue

                                if _is_prep(pos):
                                    if not prev_verb:
                                        continue
                                    prep = lemma_norm
                                    if allowed and prep not in allowed:
                                        prev_verb = None
                                        continue
                                    key = (prev_verb, prep)
                                    token_counts[key] += 1
                                    if key not in doc_seen:
                                        doc_counts[key] += 1
                                        doc_seen.add(key)
                                    prev_verb = None
                                    continue

                                prev_verb = None

    _print_progress(force=True)
    print(file=sys.stderr)  # newline for progress bar

    return token_counts, doc_counts


def write_csv(path: Path, rows: Iterable[Tuple[str, ...]], header: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(col) for col in row) + "\n")


def _add_default_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--wlp_tar", default="data/external/coca-wlp.tar")
    ap.add_argument(
        "--inventory",
        default=".cache/verb_inventory/verb_inventory_b808c317202955bf.json",
        help="Verb inventory JSON (built via scripts/build_verb_inventory.py).",
    )
    ap.add_argument("--out_dir", default="results/coca_intr_pp")
    ap.add_argument(
        "--missing_min_doc",
        type=int,
        default=10,
        help="Doc frequency threshold for proposing missing collocations.",
    )
    ap.add_argument(
        "--log_interval_sec",
        type=float,
        default=10.0,
        help="Seconds between progress bar updates.",
    )


def main():
    ap = argparse.ArgumentParser(description="Validate intr_pp frames against COCA wlp.")
    _add_default_args(ap)
    args = ap.parse_args()

    wlp_tar = Path(args.wlp_tar)
    if not wlp_tar.exists():
        raise SystemExit(f"Missing COCA wlp archive: {wlp_tar}")

    inventory_intr = load_inventory_intr_pp(Path(args.inventory))
    token_counts, doc_counts = tally_v_prep(
        wlp_tar,
        allowed_preps=sorted(_PREP_KEYWORDS),
        log_interval_sec=args.log_interval_sec,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Report counts for inventory pairs.
    inv_rows = []
    for (verb, prep) in sorted(inventory_intr):
        inv_rows.append(
            (
                verb,
                prep,
                token_counts.get((verb, prep), 0),
                doc_counts.get((verb, prep), 0),
            )
        )
    write_csv(out_dir / "inventory_intr_pp_counts.csv", inv_rows, ["verb", "prep", "token_count", "doc_count"])

    # Propose missing pairs above the threshold.
    missing_rows = []
    for (verb, prep), tok_count in token_counts.items():
        if (verb, prep) in inventory_intr:
            continue
        doc_freq = doc_counts.get((verb, prep), 0)
        if doc_freq < args.missing_min_doc:
            continue
        missing_rows.append((verb, prep, tok_count, doc_freq))

    missing_rows.sort(key=lambda row: (-row[3], -row[2], row[0], row[1]))
    write_csv(out_dir / "missing_intr_pp_candidates.csv", missing_rows, ["verb", "prep", "token_count", "doc_count"])

    print(f"Wrote {len(inv_rows)} inventory pairs to {out_dir/'inventory_intr_pp_counts.csv'}")
    print(f"Wrote {len(missing_rows)} missing candidates to {out_dir/'missing_intr_pp_candidates.csv'}")


if __name__ == "__main__":
    main()
