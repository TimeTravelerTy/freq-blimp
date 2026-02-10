"""
Quick sanity checks for VerbInventory transitivity filters.

This is mainly for debugging argument-structure generation where we enforce
`intr_only` / `trans_only` to preserve (mis)matched surface arguments.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

from src.verb_inventory import load_verb_inventory


def _has_kind(entry, prefix: str) -> bool:
    return any(getattr(frame, "kind", "").startswith(prefix) for frame in entry.frames)


def _has_core_intr(entry) -> bool:
    return any(getattr(frame, "kind", "") == "intr" for frame in entry.frames)


def _has_core_trans(entry) -> bool:
    return any(getattr(frame, "kind", "").startswith("trans") for frame in entry.frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default="data/processed/verb_inventory_pruned_particles.json")
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    inv = load_verb_inventory(args.inventory)

    def report(kind: str, restrict: str) -> None:
        choices = inv.choices_for_frame(kind)
        filtered = inv._filter_choices(choices, restrict_transitivity=restrict)  # noqa: SLF001
        filtered = filtered or []
        print(f"{kind:9s} | total {len(choices):5d} | {restrict:9s} {len(filtered):5d}")
        for entry, frame in filtered[: args.show]:
            assert frame.kind == kind
            assert _has_kind(entry, kind.split("_", 1)[0])
            if restrict == "intr_only":
                assert not _has_core_trans(entry), (entry.lemma, [f.kind for f in entry.frames])
            elif restrict == "trans_only":
                assert not _has_core_intr(entry), (entry.lemma, [f.kind for f in entry.frames])
            print(f"  {entry.lemma:20s} -> {[f.kind for f in entry.frames]}")

    # `intr_only` should veto trans* frames but may allow ditrans* frames.
    report("intr", "intr_only")
    report("intr_pp", "intr_only")

    # `trans_only` should veto core `intr` frames (but allow intr_pp frames).
    report("trans", "trans_only")
    report("trans_pp", "trans_only")
    report("ditrans", "trans_only")
    report("ditrans_pp", "trans_only")


if __name__ == "__main__":
    main()
