"""
Count verb frame attestations in COCA wlp and flag unattested inventory frames.

Frames:
- intr: verb with no objects/PP complements
- trans: verb with one direct object
- ditrans: verb with two NP objects (direct + indirect)
- intr_pp: verb with a single PP complement (records prep)
- ditrans_pp: verb with one NP object + one PP complement (records prep)
"""

import argparse
import csv
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import os
import sys
import time
import warnings

# Suppress LibreSSL warnings emitted by urllib3 (inherited by child processes).
os.environ.setdefault("PYTHONWARNINGS", "ignore:::urllib3.exceptions.NotOpenSSLWarning")
warnings.filterwarnings("ignore", message=".*OpenSSL.*", category=Warning, module="urllib3")

import spacy

# Dependencies considered as direct objects.
_OBJ_DEPS = {"dobj", "obj"}
_IOBJ_DEPS = {"iobj"}
_PREP_DEPS = {"prep"}


def _iter_wlp_rows(tar_path: Path):
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
                            for raw in fin:
                                parts = raw.decode("utf-8", "ignore").rstrip("\n").split("\t")
                                if len(parts) != 4:
                                    continue
                                yield parts  # doc_id, surface, lemma, pos


def _flush_sentence(tokens: List[Tuple[str, str, str]]) -> Tuple[str, set]:
    if not tokens:
        return "", set()
    out: List[str] = []
    lemmas = set()
    for surface, pos, lemma in tokens:
        if surface == "<p>":
            continue
        if lemma:
            lemmas.add(lemma.lower())
        if pos.startswith("y"):
            if out:
                out[-1] = out[-1] + surface
            else:
                out.append(surface)
        else:
            out.append(surface)
    return " ".join(out), lemmas


def _iter_wlp_sentences(tar_path: Path):
    """
    Yield (doc_id, sentence_text, lemma_set) reconstructed from wlp rows.
    Sentence boundary heuristic: flush on ., !, ? punctuation tokens.
    """
    current_doc = None
    sent_tokens: List[Tuple[str, str, str]] = []
    for doc_id, surface, lemma, pos in _iter_wlp_rows(tar_path):
        if current_doc is None:
            current_doc = doc_id
        if doc_id != current_doc:
            text, lemmas = _flush_sentence(sent_tokens)
            if text:
                yield current_doc, text, lemmas
            sent_tokens = []
            current_doc = doc_id

        sent_tokens.append((surface, pos, lemma))
        if pos.startswith("y") and surface in {".", "!", "?"}:
            text, lemmas = _flush_sentence(sent_tokens)
            if text:
                yield doc_id, text, lemmas
            sent_tokens = []

    if sent_tokens:
        text, lemmas = _flush_sentence(sent_tokens)
        if text:
            yield current_doc, text, lemmas


def count_target_sentences(tar_path: Path, focus_verbs: Optional[set]) -> int:
    total = 0
    for _doc_id, _sent, lemmas in _iter_wlp_sentences(tar_path):
        if focus_verbs is not None and focus_verbs.isdisjoint(lemmas):
            continue
        total += 1
    return total


def _frame_from_parse(token) -> Optional[Tuple[str, Optional[str]]]:
    """
    Return (frame_kind, prep) for a parsed verb token.
    """
    objs = [child for child in token.children if child.dep_ in _OBJ_DEPS]
    # Ignore complementizer "that" mis-tagged as object.
    objs = [c for c in objs if not (c.dep_ == "dobj" and c.text.lower() == "that")]
    iobjs = [child for child in token.children if child.dep_ in _IOBJ_DEPS]
    preps = [child for child in token.children if child.dep_ in _PREP_DEPS and any(grand.dep_ == "pobj" for grand in child.children)]

    prep = preps[0].text.lower() if preps else None

    if len(objs) > 1 or len(iobjs) > 1:
        return None
    if len(preps) > 1:
        return None

    if objs and preps:
        return "ditrans_pp", prep
    if objs and iobjs:
        return "ditrans", None
    if objs:
        return "trans", None
    if preps:
        return "intr_pp", prep
    return "intr", None


def tally_frames(
    tar_path: Path,
    nlp,
    *,
    batch_size: int,
    n_process: int,
    focus_verbs: Optional[set] = None,
    log_interval_sec: float = 10.0,
    estimated_total: Optional[int] = None,
    max_sentences: Optional[int] = None,
) -> Tuple[Counter, Counter]:
    token_counts: Counter = Counter()
    doc_counts: Counter = Counter()

    doc_buffer: List[str] = []
    doc_id_current: Optional[str] = None
    doc_seen: set = set()

    processed_sent = 0
    start = time.time()
    next_log = start
    total_sents = estimated_total

    def _log(force: bool = False):
        nonlocal next_log
        now = time.time()
        if not force and now < next_log:
            return
        elapsed = now - start
        rate = processed_sent / elapsed if elapsed > 0 else 0
        if total_sents and rate > 0:
            remaining = max(total_sents - processed_sent, 0)
            eta_sec = remaining / rate
            eta_msg = f" | eta {eta_sec/60:.1f}m"
        else:
            eta_msg = ""
        msg = (
            f"\rProcessed {processed_sent} sentences"
            f" | elapsed {elapsed/60:.1f}m"
            f" | {rate:.1f} sents/s{eta_msg}"
        )
        print(msg, end="", file=sys.stderr, flush=True)
        next_log = now + log_interval_sec

    def _process_doc_sentences(sents: List[str]):
        nonlocal processed_sent, doc_seen
        if not sents:
            return
        doc_seen = set()
        for parsed in nlp.pipe(sents, n_process=n_process, batch_size=batch_size):
            for tok in parsed:
                if tok.pos_ != "VERB" and not tok.tag_.startswith("VB"):
                    continue
                frame_info = _frame_from_parse(tok)
                if not frame_info:
                    continue
                kind, prep = frame_info
                key = (tok.lemma_.lower(), kind, prep or "")
                token_counts[key] += 1
                if key not in doc_seen:
                    doc_counts[key] += 1
                    doc_seen.add(key)
        processed_sent += len(sents)
        _log()

    for doc_id, sent, lemmas in _iter_wlp_sentences(tar_path):
        if focus_verbs is not None and focus_verbs.isdisjoint(lemmas):
            continue
        if doc_id_current is None:
            doc_id_current = doc_id
        if doc_id != doc_id_current:
            _process_doc_sentences(doc_buffer)
            doc_buffer = []
            doc_id_current = doc_id
        doc_buffer.append(sent)
        if max_sentences is not None and processed_sent + len(doc_buffer) >= max_sentences:
            break

    _process_doc_sentences(doc_buffer)
    _log(force=True)
    print(file=sys.stderr)
    return token_counts, doc_counts


def load_inventory(path: Path) -> List[Tuple[str, str, str]]:
    with path.open(encoding="utf-8") as f:
        data = f.read()
    inv = []
    import json

    items = json.loads(data)
    for entry in items:
        lemma = (entry.get("lemma") or "").strip().lower()
        if not lemma:
            continue
        for frame in entry.get("frames", ()):
            kind = frame.get("type")
            if kind not in {"intr", "trans", "ditrans", "intr_pp", "ditrans_pp"}:
                continue
            prep = (frame.get("prep") or "").strip().lower()
            inv.append((lemma, kind, prep))
    return inv


def write_counts_csv(path: Path, counts: Counter, doc_counts: Counter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lemma", "frame", "prep", "token_count", "doc_count"])
        for (lemma, frame, prep), tok in sorted(counts.items()):
            writer.writerow([lemma, frame, prep, tok, doc_counts.get((lemma, frame, prep), 0)])


def write_prune_csv(path: Path, inv_frames: List[Tuple[str, str, str]], doc_counts: Counter, min_doc: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lemma", "frame", "prep", "doc_count", "status"])
        for lemma, frame, prep in sorted(inv_frames):
            dc = doc_counts.get((lemma, frame, prep), 0)
            status = "unattested" if dc == 0 else ("low" if dc < min_doc else "ok")
            if status == "ok":
                continue
            writer.writerow([lemma, frame, prep, dc, status])


def main():
    ap = argparse.ArgumentParser(description="Count COCA verb frames and flag unattested inventory entries.")
    ap.add_argument("--wlp_tar", default="data/external/coca-wlp.tar")
    ap.add_argument("--inventory", default=".cache/verb_inventory/verb_inventory_b808c317202955bf.json")
    ap.add_argument("--out_dir", default="results/coca_frames")
    ap.add_argument("--min_doc", type=int, default=5, help="Doc-count threshold for pruning suggestions.")
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--n_process", type=int, default=1)
    ap.add_argument(
        "--focus_inventory_verbs",
        action="store_true",
        default=False,
        help="Only parse sentences containing lemmas from the inventory (faster).",
    )
    ap.add_argument(
        "--log_interval_sec",
        type=float,
        default=10.0,
        help="Seconds between progress updates.",
    )
    ap.add_argument(
        "--no_eta",
        action="store_true",
        default=False,
        help="Skip the pre-pass to count target sentences; disables ETA but saves time.",
    )
    ap.add_argument(
        "--max_sentences",
        type=int,
        default=None,
        help="Stop after processing this many sentences (for quick sampling).",
    )
    args = ap.parse_args()

    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

    inv_frames = load_inventory(Path(args.inventory))
    inv_verbs = {lemma for lemma, _kind, _prep in inv_frames}

    wlp_path = Path(args.wlp_tar)
    total_sents = None
    if not args.no_eta:
        print("Counting target sentences for ETA...", file=sys.stderr)
        total_sents = count_target_sentences(wlp_path, inv_verbs if args.focus_inventory_verbs else None)
        print(f"Target sentences: {total_sents}", file=sys.stderr)

    token_counts, doc_counts = tally_frames(
        wlp_path,
        nlp,
        batch_size=args.batch_size,
        n_process=args.n_process,
        focus_verbs=inv_verbs if args.focus_inventory_verbs else None,
        log_interval_sec=args.log_interval_sec,
        estimated_total=total_sents,
        max_sentences=args.max_sentences,
    )

    out_dir = Path(args.out_dir)
    write_counts_csv(out_dir / "coca_frame_counts.csv", token_counts, doc_counts)

    write_prune_csv(out_dir / "inventory_prune_suggestions.csv", inv_frames, doc_counts, args.min_doc)

    print(f"Wrote counts to {out_dir/'coca_frame_counts.csv'}")
    print(f"Wrote prune suggestions to {out_dir/'inventory_prune_suggestions.csv'}")


if __name__ == "__main__":
    main()
