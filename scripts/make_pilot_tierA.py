import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse, json
from src.pipeline import build_pilot
from src.lemma_bank import LemmaBankError, sample_rare_nouns_from_oewn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier_cfg", default="configs/tierA.yaml")
    ap.add_argument("--becl_path", default="data/external/becl_lemma.tsv")
    ap.add_argument("--quant_cfg", default="configs/quantifier_map.yaml")
    ap.add_argument("--out", default="data/processed/pilot_tierA.jsonl")
    ap.add_argument("--noun_mode", choices=["all","k"], default="all")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--zipf", type=float, default=3.4)
    ap.add_argument("--rare_lemmas", default="[]")  # JSON list
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lemma_source", choices=["manual", "oewn"], default="oewn")
    ap.add_argument("--oewn_lexicon", default="oewn:2021")
    ap.add_argument("--oewn_zipf_min", type=float, default=None)
    ap.add_argument("--oewn_min_len", type=int, default=3)
    ap.add_argument("--oewn_limit", type=int, default=None)
    ap.add_argument("--rare_name_path", default="data/external/rare_names.tsv")
    ap.add_argument("--name_lookup_path", default="data/external/name_gender_lookup.tsv")
    ap.add_argument("--name_conf", type=float, default=0.75)
    args = ap.parse_args()

    rare = json.loads(args.rare_lemmas) if args.rare_lemmas else []
    if not rare and args.lemma_source == "oewn":
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

    build_pilot(args.tier_cfg, args.becl_path, args.quant_cfg, args.out,
                noun_mode=args.noun_mode, k=args.k, zipf_thr=args.zipf,
                rare_lemmas=rare, seed=args.seed,
                rare_name_path=args.rare_name_path,
                name_lookup_path=args.name_lookup_path,
                name_conf=args.name_conf)
