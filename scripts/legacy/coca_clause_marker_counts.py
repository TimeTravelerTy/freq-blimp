"""
Count simple verb->clause-marker collocations in COCA wlp.

This is a lightweight heuristic to approximate which verbs commonly embed
finite clauses headed by complementizers like "that" or wh-words (what, who,
which, where, when, why, how).

It scans COCA wlp tokens in document order and counts cases where a lexical verb
is followed by a clause marker within a small token window (default=2), with
punctuation/markup breaking sequences.
"""

import argparse
import csv
import sys
import tarfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple


# CLAWS tags for lexical verbs; exclude modals (vm) to focus on content verbs.
_VERB_PREFIXES = ("vv", "vd", "vh", "vb")

# Common COCA CLAWS tag for complementizer "that".
_THAT_COMP_PREFIXES = ("cst",)

# Default wh markers to count. For BLiMP filler-gap tasks in this repo's
# generated data, this is typically just {"what", "who"}.
_DEFAULT_WH_LEMMAS = ("what", "who")

# Bytes constants for fast scanning.
_B_VERB_PREFIXES = (b"vv", b"vd", b"vh", b"vb")
_B_THAT = b"that"
_B_CST = b"cst"
_B_WH_PREFIX = b"wh:"


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
                    for fname in sorted(zf.namelist()):
                        with zf.open(fname) as fin:
                            current_doc = None
                            for raw_line in fin:
                                parts = raw_line.decode("utf-8", "ignore").rstrip("\n").split("\t")
                                if len(parts) != 4:
                                    continue
                                doc_id, _surface, lemma, pos = parts
                                doc_id = doc_id.strip()
                                lemma = (lemma or "").strip().lower()
                                pos = (pos or "").strip().lower()
                                if doc_id != current_doc:
                                    current_doc = doc_id
                                yield current_doc, lemma, pos


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


def _is_verb(pos: str) -> bool:
    return bool(pos) and pos.startswith(_VERB_PREFIXES)


def _is_that_complementizer(lemma: str, pos: str) -> bool:
    if lemma != "that":
        return False
    return bool(pos) and pos.startswith(_THAT_COMP_PREFIXES)


def _is_wh_marker(lemma: str) -> bool:
    return False


def _token_breaks_sequence(lemma: str, pos: str) -> bool:
    if not lemma or lemma.startswith("@@"):
        return True
    if lemma.startswith("<"):
        return True
    if pos and pos[0] == "y":  # punctuation in CLAWS
        return True
    return False


def tally_clause_markers(
    wlp_tar: Path,
    *,
    window: int = 2,
    focus_verbs: Optional[Sequence[str]] = None,
    wh_lemmas: Optional[Sequence[str]] = None,
    log_interval_sec: float = 10.0,
) -> Tuple[Counter, Counter, Counter, Counter]:
    """
    Count (verb, marker) pairs where marker is "that" (complementizer only) or a wh-lemma.
    Returns (marker_token_counts, marker_doc_counts, verb_token_counts, verb_doc_counts).

    window:
        Maximum number of intervening tokens (gap) allowed between the verb and
        the clause marker. For direct adjacency, use window=0.
    """
    marker_token_counts: Counter = Counter()
    marker_doc_counts: Counter = Counter()
    verb_token_counts: Counter = Counter()
    verb_doc_counts: Counter = Counter()

    wh_lemmas = tuple(wh_lemmas or _DEFAULT_WH_LEMMAS)
    wh_bytes = {w.strip().lower().encode("utf-8") for w in wh_lemmas if w and w.strip()}

    focus_bytes = None
    if focus_verbs:
        focus_bytes = {
            v.strip().lower().encode("utf-8")
            for v in focus_verbs
            if v and v.strip()
        }

    # Progress / ETA tracking (by file).
    file_lists, total_files = _collect_zip_file_lists(wlp_tar)
    start = time.time()
    next_log = start
    processed_files = 0
    bar_width = 30

    def _print_progress(force: bool = False) -> None:
        nonlocal next_log
        now = time.time()
        if not force and now < next_log:
            return
        elapsed = now - start
        rate = processed_files / elapsed if elapsed > 0 else 0.0
        remaining = max(total_files - processed_files, 0)
        eta = (remaining / rate) if rate > 0 else 0.0
        frac = (processed_files / total_files) if total_files else 1.0
        filled = int(bar_width * frac)
        bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"
        msg = (
            f"\r{bar} {processed_files}/{total_files} files"
            f" | elapsed {elapsed/60:.1f}m"
            f" | eta {eta/60:.1f}m"
        )
        print(msg, end="", file=sys.stderr, flush=True)
        next_log = now + log_interval_sec

    window = max(0, int(window))
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
                            doc_seen_verbs = set()
                            prev_verb = None
                            gap = 0
                            for raw_line in fin:
                                parts = raw_line.split(b"\t")
                                if len(parts) != 4:
                                    continue
                                doc_id_b, _surface_b, lemma_b, pos_b = parts
                                doc_id_b = doc_id_b.strip()
                                pos_b = pos_b.strip().lower()
                                lemma_b = lemma_b.strip().lower()

                                if doc_id_b != current_doc:
                                    current_doc = doc_id_b
                                    doc_seen = set()
                                    doc_seen_verbs = set()
                                    prev_verb = None
                                    gap = 0

                                if (
                                    not lemma_b
                                    or lemma_b.startswith(b"@@")
                                    or lemma_b.startswith(b"<")
                                    or (pos_b and pos_b[:1] == b"y")
                                ):
                                    prev_verb = None
                                    gap = 0
                                    continue

                                is_verb = pos_b.startswith(_B_VERB_PREFIXES)
                                if is_verb:
                                    if focus_bytes is not None and lemma_b not in focus_bytes:
                                        prev_verb = None
                                        gap = 0
                                        continue
                                    verb_token_counts[lemma_b] += 1
                                    if lemma_b not in doc_seen_verbs:
                                        verb_doc_counts[lemma_b] += 1
                                        doc_seen_verbs.add(lemma_b)
                                    prev_verb = lemma_b
                                    gap = 0
                                    continue

                                if not prev_verb:
                                    continue

                                if gap > window:
                                    prev_verb = None
                                    gap = 0
                                    continue

                                marker = None
                                # Count complementizer "that" only when it is tagged as CLAWS cst.
                                # This avoids folding in demonstrative determiner uses (often tagged
                                # as cst_dd1 / cst_dd1 variants).
                                if lemma_b == _B_THAT and pos_b == _B_CST:
                                    marker = b"that"
                                elif lemma_b in wh_bytes:
                                    marker = _B_WH_PREFIX + lemma_b

                                if marker is not None:
                                    key = (prev_verb, marker)
                                    marker_token_counts[key] += 1
                                    if key not in doc_seen:
                                        marker_doc_counts[key] += 1
                                        doc_seen.add(key)
                                    prev_verb = None
                                    gap = 0
                                    continue

                                gap += 1

                        _print_progress()

    _print_progress(force=True)
    print(file=sys.stderr)  # newline for progress bar

    return marker_token_counts, marker_doc_counts, verb_token_counts, verb_doc_counts


def _aggregate_rows(
    marker_token_counts: Counter,
    marker_doc_counts: Counter,
    verb_token_counts: Counter,
    verb_doc_counts: Counter,
    *,
    wh_lemmas: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    by_verb: Dict[str, Dict[str, int]] = {}
    wh_cols = [w.strip().lower() for w in wh_lemmas if w and w.strip()]
    # Seed rows with total verb counts so missing marker rows still have totals.
    for verb_b, tok_total in verb_token_counts.items():
        verb_s = verb_b.decode("utf-8", "ignore") if isinstance(verb_b, (bytes, bytearray)) else str(verb_b)
        by_verb.setdefault(
            verb_s,
            {
                "token_verb_total": int(tok_total),
                "doc_verb_total": int(verb_doc_counts.get(verb_b, 0)),
                "token_that": 0,
                "doc_that": 0,
                "token_wh_total": 0,
                "doc_wh_total": 0,
                **{f"token_wh_{w}": 0 for w in wh_cols},
                **{f"doc_wh_{w}": 0 for w in wh_cols},
            },
        )

    for (verb, marker), tok_count in marker_token_counts.items():
        doc_count = int(marker_doc_counts.get((verb, marker), 0))
        verb_s = verb.decode("utf-8", "ignore") if isinstance(verb, (bytes, bytearray)) else str(verb)
        marker_s = marker.decode("utf-8", "ignore") if isinstance(marker, (bytes, bytearray)) else str(marker)
        rec = by_verb.setdefault(
            verb_s,
            {
                "token_verb_total": int(verb_token_counts.get(verb, 0)),
                "doc_verb_total": int(verb_doc_counts.get(verb, 0)),
                "token_that": 0,
                "doc_that": 0,
                "token_wh_total": 0,
                "doc_wh_total": 0,
                **{f"token_wh_{w}": 0 for w in wh_cols},
                **{f"doc_wh_{w}": 0 for w in wh_cols},
            },
        )
        if marker_s == "that":
            rec["token_that"] += int(tok_count)
            rec["doc_that"] += doc_count
        elif marker_s.startswith("wh:"):
            wh = marker_s.split(":", 1)[1] if ":" in marker_s else ""
            if wh:
                rec["token_wh_total"] += int(tok_count)
                rec["doc_wh_total"] += doc_count
                tok_key = f"token_wh_{wh}"
                doc_key = f"doc_wh_{wh}"
                if tok_key in rec:
                    rec[tok_key] += int(tok_count)
                if doc_key in rec:
                    rec[doc_key] += doc_count
    return by_verb


def write_csv(path: Path, rows: Sequence[Tuple[str, ...]], header: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def main() -> int:
    ap = argparse.ArgumentParser(description="Count verb->that/wh marker collocations in COCA wlp.")
    ap.add_argument("--wlp_tar", default="data/external/coca-wlp.tar")
    ap.add_argument("--window", type=int, default=2, help="Max intervening tokens (gap) between verb and marker; use 0 for adjacency.")
    ap.add_argument("--min_doc", type=int, default=25, help="Minimum doc count to report.")
    ap.add_argument("--out_csv", default="results/coca_clause_markers/verb_clause_marker_counts.csv")
    ap.add_argument("--focus_verbs", default=None, help="Optional newline-separated verb lemmas to include.")
    ap.add_argument(
        "--wh_lemmas",
        default=",".join(_DEFAULT_WH_LEMMAS),
        help="Comma-separated wh lemmas to count (default matches BLiMP filler-gap tasks).",
    )
    args = ap.parse_args()

    wlp_tar = Path(args.wlp_tar)
    focus = None
    if args.focus_verbs:
        focus_path = Path(args.focus_verbs)
        focus = [line.strip() for line in focus_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    wh_lemmas = [w.strip().lower() for w in (args.wh_lemmas or "").split(",") if w.strip()]
    marker_tok, marker_doc, verb_tok, verb_doc = tally_clause_markers(
        wlp_tar, window=args.window, focus_verbs=focus, wh_lemmas=wh_lemmas
    )
    by_verb = _aggregate_rows(marker_tok, marker_doc, verb_tok, verb_doc, wh_lemmas=wh_lemmas)

    rows = []
    for verb, rec in by_verb.items():
        doc_total = int(rec["doc_that"]) + int(rec["doc_wh_total"])
        if doc_total < int(args.min_doc):
            continue
        doc_verb_total = int(rec.get("doc_verb_total", 0))
        token_verb_total = int(rec.get("token_verb_total", 0))
        doc_rate_that = (int(rec["doc_that"]) / doc_verb_total) if doc_verb_total else 0.0
        doc_rate_wh_total = (int(rec["doc_wh_total"]) / doc_verb_total) if doc_verb_total else 0.0
        token_rate_that = (int(rec["token_that"]) / token_verb_total) if token_verb_total else 0.0
        token_rate_wh_total = (int(rec["token_wh_total"]) / token_verb_total) if token_verb_total else 0.0
        row = [
            verb,
            str(doc_verb_total),
            str(token_verb_total),
            str(rec["doc_that"]),
            str(rec["token_that"]),
            str(rec["doc_wh_total"]),
            str(rec["token_wh_total"]),
            f"{doc_rate_that:.6f}",
            f"{doc_rate_wh_total:.6f}",
            f"{token_rate_that:.6f}",
            f"{token_rate_wh_total:.6f}",
        ]
        for wh in wh_lemmas:
            doc_wh = int(rec.get(f"doc_wh_{wh}", 0))
            tok_wh = int(rec.get(f"token_wh_{wh}", 0))
            doc_rate_wh = (doc_wh / doc_verb_total) if doc_verb_total else 0.0
            tok_rate_wh = (tok_wh / token_verb_total) if token_verb_total else 0.0
            row.append(str(doc_wh))
            row.append(str(tok_wh))
            row.append(f"{doc_rate_wh:.6f}")
            row.append(f"{tok_rate_wh:.6f}")
        row.append(str(doc_total))
        rows.append(tuple(row))
    rows.sort(key=lambda r: int(r[-1]), reverse=True)

    out_path = Path(args.out_csv)
    wh_headers = []
    for wh in wh_lemmas:
        wh_headers.extend((f"doc_wh_{wh}", f"token_wh_{wh}", f"doc_rate_wh_{wh}", f"token_rate_wh_{wh}"))
    write_csv(
        out_path,
        rows,
        header=(
            "verb",
            "doc_verb_total",
            "token_verb_total",
            "doc_that",
            "token_that",
            "doc_wh_total",
            "token_wh_total",
            "doc_rate_that",
            "doc_rate_wh_total",
            "token_rate_that",
            "token_rate_wh_total",
            *wh_headers,
            "doc_total",
        ),
    )
    print(f"Wrote {len(rows)} verbs to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
