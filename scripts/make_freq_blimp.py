import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.lemma_bank import (
    LemmaBankError,
    sample_rare_adjectives_from_oewn,
    sample_rare_nouns_from_oewn,
)
from src.pipeline import build_pilot
from src.verb_inventory import VerbInventory, load_verb_inventory

DEFAULT_SWAP_TOKENIZER = "meta-llama/Llama-3.1-8B"


def _fmt_zipf(val) -> str:
    if val is None:
        return "none"
    try:
        return str(val).replace(".", "_")
    except Exception:
        return "unknown"


def _fmt_zipf_window(zipf_max, zipf_min) -> str:
    max_part = _fmt_zipf(zipf_max)
    if zipf_min is None:
        return max_part
    min_part = _fmt_zipf(zipf_min)
    return f"{min_part}-{max_part}"


def _batch_out_path(ts: str, zipf_max: float, zipf_min: float, out_dir: Path) -> Path:
    slug = _fmt_zipf_window(zipf_max, zipf_min)
    name = f"{ts}_freq_blimp_zipf{slug}_adj{slug}_verb{slug}_generated.jsonl"
    return out_dir / name


def _parse_zipf_window(raw: str) -> tuple[float, float]:
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$", raw)
    if not m:
        raise ValueError(f"Invalid window '{raw}'. Expected format LOW-HIGH (e.g. 1.2-2.0).")
    low = float(m.group(1))
    high = float(m.group(2))
    if high < low:
        raise ValueError(f"Invalid window '{raw}'. Expected LOW <= HIGH.")
    return low, high


def _default_out_path(args) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    zipf_part = f"zipf{_fmt_zipf_window(args.zipf, args.oewn_zipf_min)}"
    adj_part = f"adj{_fmt_zipf_window(args.adj_zipf, args.adj_oewn_zipf_min)}"
    verb_part = f"verb{_fmt_zipf_window(args.verb_zipf, args.verb_zipf_min)}"
    name = f"{ts}_freq_blimp_{zipf_part}_{adj_part}_{verb_part}_generated.jsonl"
    return Path("data") / "processed" / name


def _default_original_path() -> Path:
    return Path("data") / "processed" / "blimp_original.jsonl"


def _apply_zipf_overrides(args):
    default_zipf = 3.4
    global_zipf = args.zipf_max_all
    global_zipf_min = args.zipf_min_all

    def _pick_max(val):
        if val is not None:
            return val
        if global_zipf is not None:
            return global_zipf
        return default_zipf

    def _pick_min(val):
        if val is not None:
            return val
        if global_zipf_min is not None:
            return global_zipf_min
        return None

    args.zipf = _pick_max(args.zipf)
    args.adj_zipf = _pick_max(args.adj_zipf)
    args.verb_zipf = _pick_max(args.verb_zipf)
    args.oewn_zipf_min = _pick_min(args.oewn_zipf_min)
    args.adj_oewn_zipf_min = _pick_min(args.adj_oewn_zipf_min)
    args.verb_zipf_min = _pick_min(args.verb_zipf_min)


def _run_generation(args, parser: argparse.ArgumentParser) -> Path:
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
            parser.error(str(exc))
        if args.oewn_zipf_min is not None:
            print(
                f"[LemmaBank] Loaded {len(rare)} OEWN noun lemmas ({args.oewn_zipf_min} <= zipf < {args.zipf})."
            )
        else:
            print(f"[LemmaBank] Loaded {len(rare)} OEWN noun lemmas (zipf < {args.zipf}).")

    rare_adj = json.loads(args.rare_adj_lemmas) if args.rare_adj_lemmas else []
    rare_verbs = json.loads(args.rare_verb_lemmas) if args.rare_verb_lemmas else []

    if wants_adjectives and not rare_adj and args.adj_lemma_source == "oewn":
        try:
            rare_adj = sample_rare_adjectives_from_oewn(
                zipf_max=args.adj_zipf,
                zipf_min=args.adj_oewn_zipf_min,
                min_length=args.adj_oewn_min_len,
                lexicon=args.adj_oewn_lexicon,
                limit=args.adj_oewn_limit,
            )
        except LemmaBankError as exc:
            parser.error(str(exc))
        if args.adj_oewn_zipf_min is not None:
            print(
                f"[LemmaBank] Loaded {len(rare_adj)} OEWN adjective lemmas ({args.adj_oewn_zipf_min} <= zipf < {args.adj_zipf})."
            )
        else:
            print(
                f"[LemmaBank] Loaded {len(rare_adj)} OEWN adjective lemmas (zipf < {args.adj_zipf})."
            )

    verb_inventory_obj = None
    if wants_verbs:
        inv_path = Path(args.verb_inventory)
        if not inv_path.is_file():
            parser.error(f"Verb inventory not found at {inv_path}")
        verb_inventory_obj = load_verb_inventory(inv_path)
        print(f"[VerbInventory] Loaded verb inventory ({len(verb_inventory_obj.entries)} entries) from {inv_path}.")
        if verb_inventory_obj.is_empty():
            parser.error("Verb inventory is empty; provide a populated inventory JSON.")
        if args.verb_limit is not None:
            try:
                limit = max(0, int(args.verb_limit))
            except Exception:
                limit = 0
            if limit and len(verb_inventory_obj.entries) > limit:
                rng = random.Random(args.seed)
                entries = list(verb_inventory_obj.entries)
                rng.shuffle(entries)
                verb_inventory_obj = VerbInventory(tuple(entries[:limit]))

    out_path = Path(args.out) if args.out else _default_out_path(args)
    original_out_path = Path(args.original_out) if args.original_out else _default_original_path()
    write_original = args.force_original_overwrite or not original_out_path.exists()
    print(f"[Build] Writing freq BLiMP generated data to {out_path}")
    if write_original:
        print(f"[Build] Writing canonical BLiMP original data to {original_out_path}")
    else:
        print(f"[Build] Reusing existing canonical BLiMP original data at {original_out_path}")

    build_pilot(
        args.tier_cfg,
        args.becl_path,
        args.quant_cfg,
        out_path,
        noun_mode=args.noun_mode,
        k=args.k,
        zipf_thr=args.zipf,
        rare_lemmas=rare,
        adj_mode=args.adj_mode,
        adj_zipf_thr=args.adj_zipf,
        rare_adj_lemmas=rare_adj,
        swap_targets=args.swap_targets,
        verb_mode=args.verb_mode,
        verb_zipf_thr=args.verb_zipf,
        rare_verb_lemmas=rare_verbs,
        verb_zipf_min=args.verb_zipf_min,
        noun_zipf_min=args.oewn_zipf_min,
        adj_zipf_min=args.adj_oewn_zipf_min,
        verb_min_verb_share=args.verb_min_verb_share,
        verb_inventory=verb_inventory_obj,
        seed=args.seed,
        record_limit=args.limit,
        phenomenon_filter=args.phenomenon,
        exclude_proper_nouns=args.exclude_proper_nouns,
        gender_lexicon_path=args.gender_lexicon,
        zipf_weighted_sampling=args.zipf_weighted_sampling,
        zipf_temp=args.zipf_temp,
        spacy_n_process=args.spacy_n_process,
        spacy_batch_size=args.spacy_batch_size,
        match_token_count=args.match_token_count,
        swap_tokenizer=args.swap_tokenizer,
        original_out_path=(original_out_path if write_original else None),
    )
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate freq BLiMP datasets (single or batch mode).")
    ap.add_argument(
        "--zipf-values",
        "--zipf_values",
        nargs="+",
        type=float,
        default=None,
        help="Generate one dataset per Zipf value in batch mode.",
    )
    ap.add_argument(
        "--zipf-windows",
        "--zipf_windows",
        nargs="+",
        default=None,
        help="Generate one dataset per Zipf window in batch mode (format: LOW-HIGH, e.g. 1.2-2.0).",
    )
    ap.add_argument(
        "--out-dir",
        default="data/processed",
        help="Output directory for batch mode filenames (default: data/processed).",
    )
    ap.add_argument(
        "--timestamp",
        default=None,
        help="Timestamp prefix for batch mode filenames.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print batch mode actions without generating datasets.",
    )

    ap.add_argument("--tier_cfg", default="configs/blimp_all.yaml")
    ap.add_argument(
        "--phenomenon",
        action="append",
        default=None,
        help="Limit generation to a single phenomenon (repeatable).",
    )
    ap.add_argument("--becl_path", default="data/external/becl_lemma.tsv")
    ap.add_argument("--quant_cfg", default="configs/quantifier_map.yaml")
    ap.add_argument(
        "--out",
        default=None,
        help="Generated-dataset JSONL path (single-run mode).",
    )
    ap.add_argument(
        "--original-out",
        default=None,
        help="Original-dataset JSONL path (default: data/processed/blimp_original.jsonl).",
    )
    ap.add_argument(
        "--force-original-overwrite",
        action="store_true",
        help="Overwrite the original dataset even if it already exists.",
    )
    ap.add_argument(
        "--swap_target",
        dest="swap_targets",
        action="append",
        choices=["nouns", "adjectives", "verbs", "all"],
        default=["all"],
        help="Choose one or more swap targets; repeat flag. Use 'all' for every available target.",
    )
    ap.add_argument("--noun_mode", choices=["all", "k"], default="all")
    ap.add_argument("--adj_mode", choices=["all", "k"], default="all")
    ap.add_argument("--verb_mode", choices=["all", "k"], default="k")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument(
        "--zipf_max_all",
        "--zipf_all",
        dest="zipf_max_all",
        type=float,
        default=3.4,
        help="Set one Zipf threshold for nouns/adjectives/verbs (overridden by per-POS flags).",
    )
    ap.add_argument(
        "--zipf_min_all",
        type=float,
        default=None,
        help="Set one Zipf lower bound for nouns/adjectives/verbs (overridden by per-POS flags).",
    )
    ap.add_argument("--zipf", type=float, default=None)
    ap.add_argument("--rare_lemmas", default="[]")
    ap.add_argument("--adj_zipf", type=float, default=None)
    ap.add_argument("--rare_adj_lemmas", default="[]")
    ap.add_argument("--verb_zipf", type=float, default=None)
    ap.add_argument("--rare_verb_lemmas", default="[]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lemma_source", choices=["manual", "oewn"], default="oewn")
    ap.add_argument("--adj_lemma_source", choices=["manual", "oewn"], default="oewn")
    ap.add_argument("--verb_lemma_source", choices=["manual"], default="manual")
    ap.add_argument("--verb_inventory", default="data/processed/verb_inventory_pruned_particles.json")
    ap.add_argument("--oewn_lexicon", default="oewn:2021")
    ap.add_argument("--oewn_zipf_min", type=float, default=None)
    ap.add_argument("--oewn_min_len", type=int, default=3)
    ap.add_argument("--oewn_limit", type=int, default=None)
    ap.add_argument("--adj_oewn_lexicon", default="oewn:2021")
    ap.add_argument("--adj_oewn_zipf_min", type=float, default=None)
    ap.add_argument("--adj_oewn_min_len", type=int, default=3)
    ap.add_argument("--adj_oewn_limit", type=int, default=None)
    ap.add_argument(
        "--verb_zipf_min",
        type=float,
        default=None,
        help="Lower bound for verb Zipf filtering when sampling from an inventory.",
    )
    ap.add_argument(
        "--verb_limit",
        type=int,
        default=None,
        help="Limit the number of verb inventory entries (randomized by --seed).",
    )
    ap.add_argument(
        "--verb_min_verb_share",
        type=float,
        default=1.0,
        help="Minimum WordNet verb-vs-noun ratio (v/(n+1)) when sampling verb lemmas.",
    )
    ap.add_argument("--spacy_n_process", type=int, default=1, help="spaCy n_process for parsing.")
    ap.add_argument("--spacy_batch_size", type=int, default=128, help="spaCy pipe batch size.")
    ap.add_argument("--gender_lexicon", default="data/processed/wiktionary_gender_lemmas.json")
    ap.add_argument(
        "--zipf_weighted_sampling",
        action="store_true",
        default=False,
        help="Bias noun/adj/verb sampling toward higher-Zipf lemmas within the allowed range.",
    )
    ap.add_argument(
        "--zipf_temp",
        type=float,
        default=1.0,
        help="Temperature for Zipf-weighted sampling (1.0=Zipf-proportional, >1 softens).",
    )
    ap.add_argument(
        "--exclude_proper_nouns",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude proper-noun instances from noun swaps.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of output item pairs written to the dataset.",
    )
    ap.add_argument(
        "--match-token-count",
        action="store_true",
        default=False,
        help="Require swapped wordforms to match the original tokenizer token count.",
    )
    ap.add_argument(
        "--swap-tokenizer",
        default=DEFAULT_SWAP_TOKENIZER,
        help=f"Tokenizer name/path for token count matching (default: {DEFAULT_SWAP_TOKENIZER}).",
    )
    return ap


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.zipf_values and args.zipf_windows:
        parser.error("Use either --zipf-values or --zipf-windows, not both.")

    if args.zipf_values or args.zipf_windows:
        if args.out is not None:
            parser.error("--out cannot be used with batch mode; use --out-dir instead.")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = args.timestamp or time.strftime("%Y%m%d-%H%M%S")

        runs: list[tuple[float, float]] = []
        if args.zipf_values:
            runs.extend((zipf_val, zipf_val) for zipf_val in args.zipf_values)
        if args.zipf_windows:
            try:
                runs.extend(_parse_zipf_window(raw) for raw in args.zipf_windows)
            except ValueError as exc:
                parser.error(str(exc))

        completed = []
        for zipf_min, zipf_max in runs:
            run_args = argparse.Namespace(**vars(args))
            run_args.zipf_min_all = zipf_min
            run_args.zipf_max_all = zipf_max
            run_args.out = str(_batch_out_path(ts, zipf_max, zipf_min, out_dir))
            print(f"[Batch] zipf_min_all={zipf_min} zipf_max_all={zipf_max} -> {run_args.out}")
            if args.dry_run:
                continue
            completed.append(_run_generation(run_args, parser))

        if args.dry_run:
            print("Dry run only; no datasets generated.")
        else:
            print("\nCompleted datasets:")
            for path in completed:
                print(f"  {path}")
        return

    if args.dry_run:
        parser.error("--dry-run is only supported with batch mode (--zipf-values or --zipf-windows).")
    _run_generation(args, parser)


if __name__ == "__main__":
    main()
