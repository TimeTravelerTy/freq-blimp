import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse, json
from src.pipeline import build_pilot
from src.lemma_bank import (
    LemmaBankError,
    sample_rare_nouns_from_oewn,
    sample_rare_adjectives_from_oewn,
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier_cfg", default="configs/tierA.yaml")
    ap.add_argument("--becl_path", default="data/external/becl_lemma.tsv")
    ap.add_argument("--quant_cfg", default="configs/quantifier_map.yaml")
    ap.add_argument("--out", default="data/processed/pilot_tierA.jsonl")
    ap.add_argument(
        "--swap_target",
        dest="swap_targets",
        action="append",
        choices=["nouns", "adjectives", "all"],
        default=["all"],
        help="Choose one or more swap targets; repeat flag. Use 'all' for every available target.",
    )
    ap.add_argument("--noun_mode", choices=["all","k"], default="all")
    ap.add_argument("--adj_mode", choices=["all","k"], default="all")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--zipf", type=float, default=3.4)
    ap.add_argument("--rare_lemmas", default="[]")  # JSON list
    ap.add_argument("--adj_zipf", type=float, default=3.4)
    ap.add_argument("--rare_adj_lemmas", default="[]")  # JSON list
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lemma_source", choices=["manual", "oewn"], default="oewn")
    ap.add_argument("--adj_lemma_source", choices=["manual", "oewn"], default="oewn")
    ap.add_argument("--oewn_lexicon", default="oewn:2021")
    ap.add_argument("--oewn_zipf_min", type=float, default=None)
    ap.add_argument("--oewn_min_len", type=int, default=3)
    ap.add_argument("--oewn_limit", type=int, default=None)
    ap.add_argument("--adj_oewn_lexicon", default="oewn:2021")
    ap.add_argument("--adj_oewn_zipf_min", type=float, default=None)
    ap.add_argument("--adj_oewn_min_len", type=int, default=3)
    ap.add_argument("--adj_oewn_limit", type=int, default=None)
    ap.add_argument("--gender_lexicon", default="data/processed/wiktionary_gender_lemmas.json")
    args = ap.parse_args()

    swap_targets_set = set(args.swap_targets or [])
    wants_nouns = bool({"nouns", "all"} & swap_targets_set) or not swap_targets_set
    wants_adjectives = bool({"adjectives", "all"} & swap_targets_set)

    rare = json.loads(args.rare_lemmas) if args.rare_lemmas else []
    if wants_nouns and not rare and args.lemma_source == "oewn":
        try:
            rare = sample_rare_nouns_from_oewn(
                zipf_max=args.zipf,
                zipf_min=args.oewn_zipf_min,
                min_length=args.oewn_min_len,
                lexicon=args.oewn_lexicon,
                limit=args.oewn_limit,
            )
        except LemmaBankError as exc:
            ap.error(str(exc))
        print(f"[LemmaBank] Loaded {len(rare)} OEWN noun lemmas (zipf < {args.zipf}).")

    rare_adj = json.loads(args.rare_adj_lemmas) if args.rare_adj_lemmas else []

    if wants_adjectives and not rare_adj:
        if args.adj_lemma_source == "oewn":
            try:
                rare_adj = sample_rare_adjectives_from_oewn(
                    zipf_max=args.adj_zipf,
                    zipf_min=args.adj_oewn_zipf_min,
                    min_length=args.adj_oewn_min_len,
                    lexicon=args.adj_oewn_lexicon,
                    limit=args.adj_oewn_limit,
                )
            except LemmaBankError as exc:
                ap.error(str(exc))
            print(f"[LemmaBank] Loaded {len(rare_adj)} OEWN adjective lemmas (zipf < {args.adj_zipf}).")

    build_pilot(args.tier_cfg, args.becl_path, args.quant_cfg, args.out,
                noun_mode=args.noun_mode, k=args.k, zipf_thr=args.zipf,
                rare_lemmas=rare,
                adj_mode=args.adj_mode, adj_zipf_thr=args.adj_zipf,
                rare_adj_lemmas=rare_adj, swap_targets=args.swap_targets,
                seed=args.seed,
                gender_lexicon_path=args.gender_lexicon)
