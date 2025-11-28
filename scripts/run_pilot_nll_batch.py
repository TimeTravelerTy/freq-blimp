import argparse
import os
from pathlib import Path
import time
from typing import Optional

from sentence_nll_qc import (
    VARIANTS,
    LlamaNLLScorer,
    _aggregate_by_variant,
    _build_items,
    _good_bad_stats,
    _pairwise_stats,
    _rare_penalty_stats,
    _save_single_json,
    _subtask_deltas,
    load_records,
)


def _out_path_for(model: str, data_path: Path, out_dir: Path, limit: Optional[int]) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    model_slug = model.split("/")[-1].replace(".", "_")
    limit_part = f"n{limit}" if limit is not None else "all"
    data_slug = data_path.stem
    name = f"{ts}_{model_slug}_{data_slug}_{limit_part}_sentence-nll.json"
    return out_dir / name


def _score_dataset(
    scorer: LlamaNLLScorer,
    data_path: Path,
    *,
    batch_size: int,
    max_length: int,
    limit: Optional[int],
    model: str,
    device: Optional[str],
    dtype: str,
    out_dir: Path,
) -> Path:
    records = load_records(str(data_path), limit)
    if not records:
        raise RuntimeError(f"No records loaded from {data_path}")
    items = _build_items(records)

    scores = scorer.score_texts(
        [item["text"] for item in items],
        batch_size=batch_size,
        max_length=max_length,
    )
    for item, score in zip(items, scores):
        char_len = max(1, item.get("char_len", len(item.get("text", "")) or 0))
        item.update(
            {
                "total_nll": score.total_nll,
                "token_count": score.token_count,
                "nll_per_char": score.total_nll / char_len,
            }
        )

    per_record = {}
    for item in items:
        per_record.setdefault(item["row"], {})[item["variant"]] = item

    variant_stats = _aggregate_by_variant(items)
    rare_good = _pairwise_stats(per_record, "good_typical", "good_rare", "total_nll")
    rare_bad = _pairwise_stats(per_record, "bad_typical", "bad_rare", "total_nll")
    good_vs_bad_typical = _good_bad_stats(per_record, "good_typical", "bad_typical", "total_nll")
    good_vs_bad_rare = _good_bad_stats(per_record, "good_rare", "bad_rare", "total_nll")
    rare_good_char = _pairwise_stats(per_record, "good_typical", "good_rare", "nll_per_char")
    rare_bad_char = _pairwise_stats(per_record, "bad_typical", "bad_rare", "nll_per_char")
    good_vs_bad_typical_char = _good_bad_stats(per_record, "good_typical", "bad_typical", "nll_per_char")
    good_vs_bad_rare_char = _good_bad_stats(per_record, "good_rare", "bad_rare", "nll_per_char")
    rare_penalty = _rare_penalty_stats(per_record)
    subtask_rows = _subtask_deltas(items, "good_typical", "good_rare")

    out_path = _out_path_for(model, data_path, out_dir, limit)
    summary = {
        "data": str(data_path),
        "model": model,
        "device": device,
        "dtype": dtype,
        "batch_size": batch_size,
        "max_length": max_length,
        "limit": limit,
        "variant_stats": variant_stats,
        "rare_vs_typical": {"good": rare_good, "bad": rare_bad},
        "good_vs_bad": {"typical": good_vs_bad_typical, "rare": good_vs_bad_rare},
        "rare_vs_typical_char": {"good": rare_good_char, "bad": rare_bad_char},
        "good_vs_bad_char": {"typical": good_vs_bad_typical_char, "rare": good_vs_bad_rare_char},
        "rare_penalty_per_swap": rare_penalty,
        "subtask_gaps_good": subtask_rows,
        "details": items,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_single_json(out_path, summary)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Batch NLL scoring over all pilot tier A JSONL datasets in data/processed."
    )
    ap.add_argument("--pattern", default="data/processed/*pilot_tierA*.jsonl", help="Glob pattern for datasets.")
    ap.add_argument("--model", default="microsoft/phi-2")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--compile", action="store_true", help="Try torch.compile for extra throughput on CUDA.")
    ap.add_argument(
        "--out-dir",
        default="results/sentence_nll_runs",
        help="Directory for per-dataset output JSON files.",
    )
    args = ap.parse_args()

    paths = sorted(Path(p) for p in Path().glob(args.pattern) if p.is_file())
    if not paths:
        raise SystemExit(f"No datasets found for pattern {args.pattern}")

    scorer = LlamaNLLScorer(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        compile_model=args.compile,
    )

    out_dir = Path(args.out_dir)
    completed = []
    for data_path in paths:
        print(f"\n[Run] Scoring {data_path} with {args.model} ...")
        out_path = _score_dataset(
            scorer,
            data_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
            limit=args.limit,
            model=args.model,
            device=args.device,
            dtype=args.dtype,
            out_dir=out_dir,
        )
        print(f"[Done] Wrote results to {out_path}")
        completed.append(out_path)

    print("\nCompleted runs:")
    for p in completed:
        print(f"  {p}")


if __name__ == "__main__":
    main()
