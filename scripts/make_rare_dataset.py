import os, sys, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse, json
from pathlib import Path
from src.pipeline import build_pilot
from src.lemma_bank import (
    LemmaBankError,
    sample_rare_nouns_from_oewn,
    sample_rare_adjectives_from_oewn,
)
from src.verb_inventory import load_verb_inventory

def _fmt_zipf(val) -> str:
    if val is None:
        return "none"
    try:
        return str(val).replace(".", "_")
    except Exception:
        return "unknown"

def _apply_zipf_overrides(args):
    """
    Apply global zipf fallback while keeping per-POS overrides.
    """
    default_zipf = 3.4
    global_zipf = args.zipf_all

    def _pick(val):
        if val is not None:
            return val
        if global_zipf is not None:
            return global_zipf
        return default_zipf

    args.zipf = _pick(args.zipf)
    args.adj_zipf = _pick(args.adj_zipf)
    args.verb_zipf = _pick(args.verb_zipf)

def _default_out_path(args) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    zipf_part = f"zipf{_fmt_zipf(args.zipf)}"
    adj_part = f"adj{_fmt_zipf(args.adj_zipf)}"
    verb_part = f"verb{_fmt_zipf(args.verb_zipf)}"
    name = f"{ts}_rare_blimp_{zipf_part}_{adj_part}_{verb_part}.jsonl"
    return Path("data") / "processed" / name

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier_cfg", default="configs/tierA.yaml")
    ap.add_argument("--becl_path", default="data/external/becl_lemma.tsv")
    ap.add_argument("--quant_cfg", default="configs/quantifier_map.yaml")
    ap.add_argument(
        "--out",
        default=None,
        help="Output JSONL path (defaults to timestamped file with zipf settings in data/processed).",
    )
    ap.add_argument(
        "--swap_target",
        dest="swap_targets",
        action="append",
        choices=["nouns", "adjectives", "verbs", "all"],
        default=["all"],
        help="Choose one or more swap targets; repeat flag. Use 'all' for every available target.",
    )
    ap.add_argument("--noun_mode", choices=["all","k"], default="all")
    ap.add_argument("--adj_mode", choices=["all","k"], default="all")
    ap.add_argument("--verb_mode", choices=["all","k"], default="k")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--zipf_all", type=float, default=3.4, help="Set a single Zipf threshold for nouns/adjectives/verbs (overridden by per-POS flags).")
    ap.add_argument("--zipf", type=float, default=None)
    ap.add_argument("--rare_lemmas", default="[]")  # JSON list
    ap.add_argument("--adj_zipf", type=float, default=None)
    ap.add_argument("--rare_adj_lemmas", default="[]")  # JSON list
    ap.add_argument("--verb_zipf", type=float, default=None)
    ap.add_argument("--rare_verb_lemmas", default="[]")  # JSON list
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lemma_source", choices=["manual", "oewn"], default="oewn")
    ap.add_argument("--adj_lemma_source", choices=["manual", "oewn"], default="oewn")
    ap.add_argument("--verb_lemma_source", choices=["manual"], default="manual")
    ap.add_argument("--verb_inventory", default="data/processed/verb_inventory_pruned.json")
    ap.add_argument("--oewn_lexicon", default="oewn:2021")
    ap.add_argument("--oewn_zipf_min", type=float, default=None)
    ap.add_argument("--oewn_min_len", type=int, default=3)
    ap.add_argument("--oewn_limit", type=int, default=None)
    ap.add_argument("--adj_oewn_lexicon", default="oewn:2021")
    ap.add_argument("--adj_oewn_zipf_min", type=float, default=None)
    ap.add_argument("--adj_oewn_min_len", type=int, default=3)
    ap.add_argument("--adj_oewn_limit", type=int, default=None)
    ap.add_argument("--spacy_n_process", type=int, default=1, help="spaCy n_process for parsing.")
    ap.add_argument("--spacy_batch_size", type=int, default=128, help="spaCy pipe batch size.")
    ap.add_argument("--gender_lexicon", default="data/processed/wiktionary_gender_lemmas.json")
    ap.add_argument(
        "--zipf_weighted_sampling",
        action="store_true",
        default=True,
        help="Bias noun/adj/verb sampling toward lemmas with higher Zipf scores within the allowed range.",
    )
    ap.add_argument(
        "--zipf_temp",
        type=float,
        default=1.0,
        help="Temperature for Zipf-weighted sampling (1.0=Zipf-proportional, >1 softens).",
    )
    args = ap.parse_args()

    _apply_zipf_overrides(args)

    swap_targets_set = set(args.swap_targets or [])
    wants_nouns = bool({"nouns", "all"} & swap_targets_set) or not swap_targets_set
    wants_adjectives = bool({"adjectives", "all"} & swap_targets_set)
    wants_verbs = bool({"verbs", "all"} & swap_targets_set)

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
    rare_verbs = json.loads(args.rare_verb_lemmas) if args.rare_verb_lemmas else []

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

    verb_inventory_obj = None
    if wants_verbs:
        inv_path = Path(args.verb_inventory)
        if not inv_path.is_file():
            ap.error(f"Verb inventory not found at {inv_path}")
        verb_inventory_obj = load_verb_inventory(inv_path)
        print(f"[VerbInventory] Loaded verb inventory ({len(verb_inventory_obj.entries)} entries) from {inv_path}.")
        if verb_inventory_obj.is_empty():
            ap.error("Verb inventory is empty; provide a populated inventory JSON.")

    out_path = Path(args.out) if args.out else _default_out_path(args)
    print(f"[Build] Writing rare BLiMP data to {out_path}")

    build_pilot(args.tier_cfg, args.becl_path, args.quant_cfg, out_path,
                noun_mode=args.noun_mode, k=args.k, zipf_thr=args.zipf,
                rare_lemmas=rare,
                adj_mode=args.adj_mode, adj_zipf_thr=args.adj_zipf,
                rare_adj_lemmas=rare_adj, swap_targets=args.swap_targets,
                verb_mode=args.verb_mode, verb_zipf_thr=args.verb_zipf,
                rare_verb_lemmas=rare_verbs,
                verb_inventory=verb_inventory_obj,
                seed=args.seed,
                gender_lexicon_path=args.gender_lexicon,
                zipf_weighted_sampling=args.zipf_weighted_sampling,
                zipf_temp=args.zipf_temp,
                spacy_n_process=args.spacy_n_process,
                spacy_batch_size=args.spacy_batch_size)
