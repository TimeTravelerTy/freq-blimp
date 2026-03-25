"""Score FreqBLiMP NLP2026 items with Qwen2.5-72B for plausibility analysis.

For each item, scores four sentences and saves per-item disruption measures:
  NLL(good_original), NLL(bad_original), NLL(good_swapped), NLL(bad_swapped)

Derived measures (saved per-item and used downstream by analyze_plausibility.py):
  delta_g    = NLL(good_swapped) - NLL(good_original)   [disruption on grammatical]
  delta_b    = NLL(bad_swapped)  - NLL(bad_original)    [disruption on ungrammatical]
  delta_pair = (delta_g + delta_b) / 2                  [mean disruption]
  asymmetry  = delta_g - delta_b                        [good more disrupted than bad?]

Inputs (old NLP2026 datasets with both original and swapped sentences):
  head  → data/processed/20251228-165517_rare_blimp_zipf3_6-5_0_*.jsonl
  tail  → data/processed/20251228-165513_rare_blimp_zipf2_2-3_0_*.jsonl
  xtail → data/processed/20251228-165511_rare_blimp_zipf1_2-2_0_*.jsonl

Usage:
  # Test run — 100 items from head, verify NLL values and memory:
  python scripts/score_plausibility.py --regime head --limit 100

  # Single regime:
  python scripts/score_plausibility.py --regime tail

  # Full run (all regimes, ~14-20h on 2x A100-80GB):
  python scripts/score_plausibility.py --regime all

  # Resume / skip existing output:
  python scripts/score_plausibility.py --regime all   # skips existing by default

  # Force recompute:
  python scripts/score_plausibility.py --regime head --overwrite
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sentence_nll import LlamaNLLScorer, SequenceScore

# ---------------------------------------------------------------------------
# Dataset registry — old NLP2026 datasets that have both original + swapped
# ---------------------------------------------------------------------------
REGIME_DATASETS: Dict[str, str] = {
    "head": (
        "data/processed/"
        "20251228-165517_rare_blimp_zipf3_6-5_0_adj3_6-5_0_verb3_6-5_0.jsonl"
    ),
    "tail": (
        "data/processed/"
        "20251228-165513_rare_blimp_zipf2_2-3_0_adj2_2-3_0_verb2_2-3_0.jsonl"
    ),
    "xtail": (
        "data/processed/"
        "20251228-165511_rare_blimp_zipf1_2-2_0_adj1_2-2_0_verb1_2-2_0.jsonl"
    ),
}

DEFAULT_MODEL = "Qwen/Qwen2.5-72B"
DEFAULT_OUT_DIR = "results/plausibility_scores"

# Qwen2.5-72B in bf16: 72e9 params × 2 bytes ≈ 144 GB
_MODEL_SIZE_GB = 144.0


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _iter_jsonl(path: str, limit: Optional[int] = None):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _pick_original(rec: dict) -> tuple:
    """Return (good_original, bad_original) strings."""
    g = rec.get("good_original") or rec.get("sentence_good") or rec.get("good")
    b = rec.get("bad_original") or rec.get("sentence_bad") or rec.get("bad")
    return g, b


def _pick_swapped(rec: dict) -> tuple:
    """Return (good_swapped, bad_swapped) strings. Supports good_rare and good_freq."""
    g = rec.get("good_rare") or rec.get("good_freq")
    b = rec.get("bad_rare") or rec.get("bad_freq")
    return g, b


# ---------------------------------------------------------------------------
# Memory estimate
# ---------------------------------------------------------------------------

def _print_memory_info(model_name: str, n_gpus_visible: int = 0):
    """Print a quick memory sanity check before loading the model."""
    print(f"\n[Memory] Model: {model_name}")
    print(f"[Memory] Estimated size (bf16): {_MODEL_SIZE_GB:.0f} GB")
    if n_gpus_visible:
        try:
            import torch
            total_gb = sum(
                torch.cuda.get_device_properties(i).total_memory / 1e9
                for i in range(n_gpus_visible)
            )
            print(
                f"[Memory] Visible GPU memory: {total_gb:.0f} GB "
                f"({n_gpus_visible} × GPU)"
            )
            headroom = total_gb - _MODEL_SIZE_GB
            if headroom < 8:
                print(
                    f"[Memory] WARNING: only {headroom:.1f} GB headroom — "
                    "consider reducing --batch-size or using more GPUs."
                )
            else:
                print(f"[Memory] Headroom for activations: ~{headroom:.1f} GB (OK)")
        except Exception:
            pass
    print()


# ---------------------------------------------------------------------------
# Scoring one regime
# ---------------------------------------------------------------------------

def score_regime(
    scorer: LlamaNLLScorer,
    regime: str,
    data_path: str,
    out_path: Path,
    batch_size: int,
    max_length: int,
    limit: Optional[int],
    overwrite: bool,
) -> int:
    """Score all items in one regime dataset. Returns number of items written."""

    if out_path.exists() and not overwrite:
        print(
            f"[{regime}] Output exists, skipping: {out_path}\n"
            f"           (use --overwrite to recompute)"
        )
        return 0

    # Load records
    records = list(_iter_jsonl(data_path, limit=limit))
    print(f"[{regime}] Loaded {len(records)} records from {data_path}")

    # Build flat list of 4 texts per item: [g_orig, b_orig, g_swap, b_swap, ...]
    texts: List[str] = []
    valid_records: List[dict] = []
    skipped = 0

    for rec in records:
        g_orig, b_orig = _pick_original(rec)
        g_swap, b_swap = _pick_swapped(rec)
        if not all(isinstance(s, str) and s.strip() for s in [g_orig, b_orig, g_swap, b_swap]):
            skipped += 1
            continue
        texts.extend([g_orig, b_orig, g_swap, b_swap])
        valid_records.append(rec)

    if skipped:
        print(f"[{regime}] Skipped {skipped} records with missing sentences.")
    if not texts:
        print(f"[{regime}] No valid records to score — check dataset fields.")
        return 0

    n_items = len(valid_records)
    n_sentences = len(texts)
    print(
        f"[{regime}] Scoring {n_items} items ({n_sentences} sentences), "
        f"batch_size={batch_size} ..."
    )

    t0 = time.time()
    scores: List[SequenceScore] = scorer.score_texts(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        show_progress=True,
    )
    elapsed = time.time() - t0

    if len(scores) != n_sentences:
        raise RuntimeError(
            f"Score count mismatch: expected {n_sentences}, got {len(scores)}"
        )

    rate = n_items / elapsed
    print(f"[{regime}] Scored in {elapsed:.1f}s ({rate:.1f} items/s)")

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(valid_records):
            s_g_orig = scores[4 * i]
            s_b_orig = scores[4 * i + 1]
            s_g_swap = scores[4 * i + 2]
            s_b_swap = scores[4 * i + 3]

            delta_g = s_g_swap.total_nll - s_g_orig.total_nll
            delta_b = s_b_swap.total_nll - s_b_orig.total_nll
            delta_pair = (delta_g + delta_b) / 2.0
            asymmetry = delta_g - delta_b

            row = {
                # Identity
                "regime": regime,
                "phenomenon": rec.get("phenomenon"),
                "subtask": rec.get("subtask"),
                "field": rec.get("field"),
                "idx": rec.get("idx"),
                # Sentences
                "good_original": s_g_orig.text,
                "bad_original": s_b_orig.text,
                "good_swapped": s_g_swap.text,
                "bad_swapped": s_b_swap.text,
                # Raw NLLs
                "nll_g_original": s_g_orig.total_nll,
                "nll_b_original": s_b_orig.total_nll,
                "nll_g_swapped": s_g_swap.total_nll,
                "nll_b_swapped": s_b_swap.total_nll,
                # Token counts (for per-token NLL if needed)
                "tc_g_original": s_g_orig.token_count,
                "tc_b_original": s_b_orig.token_count,
                "tc_g_swapped": s_g_swap.token_count,
                "tc_b_swapped": s_b_swap.token_count,
                # Disruption measures
                "delta_g": delta_g,
                "delta_b": delta_b,
                "delta_pair": delta_pair,
                "asymmetry": asymmetry,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"[{regime}] Wrote {written} items → {out_path}")
    return written


# ---------------------------------------------------------------------------
# Verification helper (test mode)
# ---------------------------------------------------------------------------

def _print_sample_scores(out_path: Path, n: int = 5):
    """Print a few scored items so the user can sanity-check the NLLs."""
    print(f"\n[Verify] First {n} items from {out_path}:")
    print(f"{'idx':>4}  {'delta_g':>9}  {'delta_b':>9}  {'delta_pair':>11}  good_original (truncated)")
    print("-" * 80)
    for i, rec in enumerate(_iter_jsonl(str(out_path), limit=n)):
        snippet = rec.get("good_original", "")[:45]
        print(
            f"{rec.get('idx', '?'):>4}  "
            f"{rec.get('delta_g', 0):>9.2f}  "
            f"{rec.get('delta_b', 0):>9.2f}  "
            f"{rec.get('delta_pair', 0):>11.2f}  "
            f"{snippet}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Score FreqBLiMP items with Qwen2.5-72B for plausibility analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--regime",
        default="all",
        choices=["head", "tail", "xtail", "all"],
        help="Zipf regime to process. Default: all.",
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID. Default: {DEFAULT_MODEL}",
    )
    ap.add_argument(
        "--dataset",
        default=None,
        help="Override the default dataset path for the selected regime.",
    )
    ap.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUT_DIR}",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Scoring batch size (sentences, not items). Default: 8.",
    )
    ap.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Max token length per sentence. BLiMP sentences are short; 128 is safe.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N items per regime (for test runs).",
    )
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Model dtype. Default: bfloat16.",
    )
    ap.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token. Also reads HF_TOKEN env var.",
    )
    ap.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        default=True,
        help="Disable trust_remote_code (Qwen requires it by default).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="After scoring, print a sample of scored items for sanity-checking.",
    )
    args = ap.parse_args()

    # HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
        except ImportError:
            pass

    # Regimes to process
    regimes = list(REGIME_DATASETS.keys()) if args.regime == "all" else [args.regime]

    # Check GPU count before loading
    try:
        import torch
        n_gpus = torch.cuda.device_count()
    except Exception:
        n_gpus = 0

    _print_memory_info(args.model, n_gpus)

    if args.limit:
        print(f"[Info] TEST MODE: processing first {args.limit} items per regime.")
        print(f"[Info] Remove --limit for the full run.\n")

    # Load model once (shared across all regimes)
    print(f"[Model] Loading {args.model} ...")
    print(f"[Model] device_map=auto  dtype={args.dtype}  trust_remote_code={args.trust_remote_code}")
    t_load = time.time()
    scorer = LlamaNLLScorer(
        model_name=args.model,
        dtype=args.dtype,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    print(f"[Model] Ready in {time.time() - t_load:.1f}s\n")

    out_dir = Path(args.out_dir)
    run_ts = time.strftime("%Y%m%d-%H%M%S")
    model_slug = args.model.split("/")[-1]

    total_items = 0
    output_paths = []

    for regime in regimes:
        data_path = (
            args.dataset
            if (args.dataset and len(regimes) == 1)
            else REGIME_DATASETS[regime]
        )
        out_name = f"{run_ts}_{model_slug}_{regime}_plausibility_scores.jsonl"
        out_path = out_dir / out_name

        t_regime = time.time()
        n = score_regime(
            scorer=scorer,
            regime=regime,
            data_path=data_path,
            out_path=out_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        total_items += n
        if n > 0:
            output_paths.append(out_path)
            if args.verify:
                _print_sample_scores(out_path)
        print(f"[{regime}] Finished in {time.time() - t_regime:.1f}s\n")

    print(f"[Done] Total items scored: {total_items}")
    print("Output files:")
    for p in output_paths:
        print(f"  {p}")

    if args.limit:
        print(
            "\n[Next] Test run complete. "
            "Check NLL values above, then run without --limit for the full dataset."
        )


if __name__ == "__main__":
    main()
