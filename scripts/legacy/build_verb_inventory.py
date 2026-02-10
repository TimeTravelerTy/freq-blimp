import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.lemma_bank import LemmaBankError
from src.verb_inventory import (
    build_inventory_from_lemmas,
    build_inventory_from_oewn,
    build_inventory_from_verbnet,
    write_verb_inventory,
)


def main():
    ap = argparse.ArgumentParser(description="Build verb-frame inventory (VerbNet preferred).")
    ap.add_argument("--out", default="configs/verb_frames.json")
    ap.add_argument("--oewn_lexicon", default="oewn:2021")
    ap.add_argument("--zipf_max", type=float, default=3.4)
    ap.add_argument("--zipf_min", type=float, default=None)
    ap.add_argument("--min_length", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--no_shuffle",
        dest="shuffle",
        action="store_false",
        default=True,
        help="Disable shuffling before applying --limit.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lemmas", default=None, help="Optional path to newline-separated lemmas.")
    ap.add_argument("--verb_source", choices=["verbnet", "oewn", "manual"], default="verbnet")
    ap.add_argument("--verbnet_dir", default=None, help="Path to VerbNet corpus root (e.g., ~/nltk_data/corpora/verbnet3).")
    args = ap.parse_args()

    if args.verb_source == "manual":
        if not args.lemmas:
            ap.error("Manual source requires --lemmas path.")
        with open(args.lemmas, encoding="utf-8") as f:
            lemmas = [line.strip() for line in f if line.strip()]
        inventory = build_inventory_from_lemmas(lemmas, lexicon=args.oewn_lexicon)
    elif args.verb_source == "verbnet":
        verbnet_dir = args.verbnet_dir or Path.home() / "nltk_data" / "corpora" / "verbnet3"
        inventory = build_inventory_from_verbnet(verbnet_dir)
    else:
        try:
            inventory = build_inventory_from_oewn(
                zipf_max=args.zipf_max,
                zipf_min=args.zipf_min,
                min_length=args.min_length,
                lexicon=args.oewn_lexicon,
                limit=args.limit,
                shuffle=args.shuffle,
                seed=args.seed,
            )
        except LemmaBankError as exc:
            ap.error(str(exc))

    out_path = Path(args.out)
    write_verb_inventory(out_path, inventory)
    print(f"Wrote {len(inventory.entries)} verb entries to {out_path}")


if __name__ == "__main__":
    main()
