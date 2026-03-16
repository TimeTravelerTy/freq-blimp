import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.edits import _starts_with_vowel_sound
from src.verb_inventory import load_frame_blocklist, load_verb_inventory


_PARTICLE_WORDS = {
    "about",
    "across",
    "ahead",
    "along",
    "around",
    "aside",
    "away",
    "back",
    "by",
    "down",
    "in",
    "off",
    "on",
    "open",
    "out",
    "over",
    "round",
    "through",
    "together",
    "under",
    "up",
}

_TRANSITIVE_GOOD_SUBTASKS = {"causative", "transitive", "passive_1", "passive_2"}
_INTRANSITIVE_GOOD_SUBTASKS = {"drop_argument", "inchoative", "intransitive"}


def _iter_records(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _article_mismatches(text: str):
    if not isinstance(text, str):
        return []
    toks = re.findall(r"[A-Za-z][A-Za-z'.-]*|[^\w\s]", text)
    mismatches = []
    for i, tok in enumerate(toks[:-1]):
        lower = tok.lower()
        if lower not in {"a", "an"}:
            continue
        nxt = toks[i + 1]
        if not any(ch.isalpha() for ch in nxt):
            continue
        desired = "an" if _starts_with_vowel_sound(nxt) else "a"
        if lower != desired:
            mismatches.append((tok, nxt))
    return mismatches


def _verb_swaps(rec: dict):
    meta = rec.get("meta") or {}
    return [entry for entry in (meta.get("g_swaps") or []) if entry.get("pos") == "verb"]


def _is_trans_only(inv, lemma: str) -> bool:
    entry = inv.entry_for_lemma(lemma)
    if entry is None:
        return False
    kinds = {frame.kind for frame in entry.frames}
    has_trans = any(kind.startswith("trans") or kind.startswith("ditrans") for kind in kinds)
    has_intr = any(kind.startswith("intr") for kind in kinds)
    return has_trans and not has_intr


def _looks_like_partial_particle_swap(rec: dict, inv) -> bool:
    good_original = rec.get("good_original")
    good_freq = rec.get("good_freq")
    if not isinstance(good_original, str) or not isinstance(good_freq, str):
        return False
    for entry in _verb_swaps(rec):
        old = entry.get("old")
        new = entry.get("new")
        lemma = (entry.get("lemma") or "").strip().lower()
        if not old or not new or not lemma:
            continue
        inv_entry = inv.entry_for_lemma(lemma)
        for particle in _PARTICLE_WORDS:
            pat_old = rf"\b{re.escape(old)}\s+{re.escape(particle)}\b"
            pat_new = rf"\b{re.escape(new)}\s+{re.escape(particle)}\b"
            if not re.search(pat_old, good_original, flags=re.IGNORECASE):
                continue
            if not re.search(pat_new, good_freq, flags=re.IGNORECASE):
                continue
            if inv_entry is None:
                return True
            has_matching_particle = any(
                frame.kind.endswith("_particle") and (frame.particle or "").lower() == particle
                for frame in inv_entry.frames
            )
            if not has_matching_particle:
                return True
    return False


def qc_file(path: Path, inv, blocked_frames):
    counts = Counter()
    examples = {
        "null_pair": [],
        "good_article": [],
        "objectless_transitive_good": [],
        "blocked_lemma_good": [],
        "partial_particle_swap": [],
    }

    for rec in _iter_records(path):
        counts["records"] += 1
        good = rec.get("good_freq")
        bad = rec.get("bad_freq")
        if good is None or bad is None:
            counts["null_pair"] += 1
            if len(examples["null_pair"]) < 3:
                examples["null_pair"].append((rec["phenomenon"], rec["subtask"], rec.get("idx")))
            continue

        mismatches = _article_mismatches(good)
        if mismatches:
            counts["good_article"] += 1
            if len(examples["good_article"]) < 3:
                examples["good_article"].append((rec["subtask"], good))

        if rec.get("phenomenon") == "argument_structure" and rec.get("subtask") in _INTRANSITIVE_GOOD_SUBTASKS:
            for entry in _verb_swaps(rec):
                lemma = (entry.get("lemma") or "").strip().lower()
                if lemma and _is_trans_only(inv, lemma):
                    counts["objectless_transitive_good"] += 1
                    if len(examples["objectless_transitive_good"]) < 3:
                        examples["objectless_transitive_good"].append((rec["subtask"], lemma, good))
                    break

        blocked_hit = False
        for entry in _verb_swaps(rec):
            lemma = (entry.get("lemma") or "").strip().lower()
            if not lemma:
                continue
            if lemma in blocked_frames:
                blocked_hit = True
                break
        if blocked_hit:
            counts["blocked_lemma_good"] += 1
            if len(examples["blocked_lemma_good"]) < 3:
                examples["blocked_lemma_good"].append((rec["subtask"], good))

        if _looks_like_partial_particle_swap(rec, inv):
            counts["partial_particle_swap"] += 1
            if len(examples["partial_particle_swap"]) < 3:
                examples["partial_particle_swap"].append((rec["subtask"], rec.get("good_original"), good))

    return counts, examples


def main():
    ap = argparse.ArgumentParser(description="QC generated freq-BLiMP datasets for common grammaticality/data hygiene issues.")
    ap.add_argument(
        "--pattern",
        default="data/processed/*freq_blimp*_generated.jsonl",
        help="Glob for generated dataset JSONL files.",
    )
    ap.add_argument(
        "--verb-inventory",
        default="data/processed/verb_inventory_pruned_particles.json",
        help="Verb inventory JSON used for swap-time frame checks.",
    )
    args = ap.parse_args()

    inv = load_verb_inventory(args.verb_inventory)
    blocked_frames = load_frame_blocklist()
    paths = sorted(Path().glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files matched {args.pattern}")

    for path in paths:
        counts, examples = qc_file(path, inv, blocked_frames)
        print(f"\n{path}")
        print(f"  records: {counts['records']}")
        print(f"  null_pair: {counts['null_pair']}")
        print(f"  good_article: {counts['good_article']}")
        print(f"  objectless_transitive_good: {counts['objectless_transitive_good']}")
        print(f"  blocked_lemma_good: {counts['blocked_lemma_good']}")
        print(f"  partial_particle_swap: {counts['partial_particle_swap']}")
        for key, vals in examples.items():
            if not vals:
                continue
            print(f"  {key}_examples:")
            for val in vals:
                print(f"    {val}")


if __name__ == "__main__":
    raise SystemExit(main())
