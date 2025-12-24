import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from src.sentence_nll import LlamaNLLScorer  # noqa: E402
except ModuleNotFoundError as e:  # pragma: no cover
    LlamaNLLScorer = None  # type: ignore[assignment]
    _SENTENCE_NLL_IMPORT_ERROR = e
else:  # pragma: no cover
    _SENTENCE_NLL_IMPORT_ERROR = None


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec["_row"] = i
            yield rec


def _load_jsonl(path: str, limit: Optional[int] = None) -> List[dict]:
    records: List[dict] = []
    for rec in _iter_jsonl(path):
        records.append(rec)
        if limit is not None and len(records) >= limit:
            break
    return records


def _pick_field(records: List[dict], explicit: Optional[str], candidates: List[str]) -> Optional[str]:
    if explicit:
        return explicit
    for c in candidates:
        for rec in records:
            if c in rec and rec[c] is not None:
                return c
    return None


def _model_slug(model: str) -> str:
    return model.split("/")[-1].replace(".", "_")


def _data_slug(data: str) -> str:
    return Path(data).name.rsplit(".", 1)[0]


def _default_out_path(model: str, data: str, variant: str) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    model_slug = _model_slug(model)
    data_slug = _data_slug(data)
    name = f"{ts}_{model_slug}_{data_slug}_{variant}_pair-scores.jsonl"
    return Path("results") / "blimp_pair_scores" / name


def _swap_counts(meta: dict) -> Dict[str, int]:
    def _len(key: str) -> int:
        val = meta.get(key)
        return len(val) if isinstance(val, list) else 0

    return {
        "g_swaps": _len("g_swaps"),
        "b_swaps": _len("b_swaps"),
        "g_verb_swaps": _len("g_verb_swaps"),
        "b_verb_swaps": _len("b_verb_swaps"),
        "g_adj_swaps": _len("g_adj_swaps"),
        "b_adj_swaps": _len("b_adj_swaps"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score BLiMP/rare-BLiMP pairs and save per-pair logprob data."
    )
    ap.add_argument("--data", required=True, help="Path to a BLiMP/rare-BLiMP JSONL dataset.")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on records.")
    ap.add_argument(
        "--variant",
        default="rare",
        choices=["rare", "original", "auto"],
        help="Which sentence variant to score when both are present (default: rare).",
    )
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--device-map", default=None, help="Optional HF device_map (e.g., auto).")
    ap.add_argument("--compile-model", action="store_true", help="Use torch.compile when available.")
    ap.add_argument("--good-field", default=None, help="Field for grammatical sentences.")
    ap.add_argument("--bad-field", default=None, help="Field for ungrammatical sentences.")
    ap.add_argument("--output", default=None, help="Optional JSONL output path.")
    args = ap.parse_args()

    if LlamaNLLScorer is None:
        raise ModuleNotFoundError(
            "sentence_nll could not be imported; check dependencies"
        ) from _SENTENCE_NLL_IMPORT_ERROR

    records = _load_jsonl(args.data, limit=args.limit)
    if not records:
        raise ValueError("No records loaded.")

    if args.variant == "original":
        good_candidates = ["good_original", "sentence_good", "good", "grammatical"]
        bad_candidates = ["bad_original", "sentence_bad", "bad", "ungrammatical"]
    elif args.variant == "rare":
        good_candidates = ["good_rare", "sentence_good", "good", "grammatical"]
        bad_candidates = ["bad_rare", "sentence_bad", "bad", "ungrammatical"]
    else:
        good_candidates = [
            "good_rare",
            "good_original",
            "sentence_good",
            "good",
            "grammatical",
        ]
        bad_candidates = [
            "bad_rare",
            "bad_original",
            "sentence_bad",
            "bad",
            "ungrammatical",
        ]

    good_field = _pick_field(records, args.good_field, good_candidates)
    bad_field = _pick_field(records, args.bad_field, bad_candidates)
    if not good_field or not bad_field:
        raise ValueError(
            "Could not infer good/bad fields; set --good-field and --bad-field."
        )

    texts: List[str] = []
    items: List[dict] = []
    skipped = 0
    for rec in records:
        good = rec.get(good_field)
        bad = rec.get(bad_field)
        if not isinstance(good, str) or not isinstance(bad, str):
            skipped += 1
            continue
        texts.append(good)
        texts.append(bad)
        items.append(rec)

    if not texts:
        raise ValueError("No valid good/bad pairs found.")

    scorer = LlamaNLLScorer(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        compile_model=args.compile_model,
    )

    scores = scorer.score_texts(
        texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        show_progress=True,
    )
    if len(scores) != len(texts):
        raise RuntimeError("Scoring returned a different number of results.")

    out_path = Path(args.output) if args.output else _default_out_path(
        args.model, args.data, args.variant
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(items):
            meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
            good_score = scores[2 * i]
            bad_score = scores[2 * i + 1]
            row = {
                "idx": rec.get("idx"),
                "group": rec.get("group"),
                "phenomenon": rec.get("phenomenon"),
                "subtask": rec.get("subtask"),
                "variant": args.variant,
                "good_field": good_field,
                "bad_field": bad_field,
                "good_text": rec.get(good_field),
                "bad_text": rec.get(bad_field),
                "good_total_nll": good_score.total_nll,
                "bad_total_nll": bad_score.total_nll,
                "good_token_count": good_score.token_count,
                "bad_token_count": bad_score.token_count,
                "swap_counts": _swap_counts(meta),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved pair scores to {out_path}")
    if skipped:
        print(f"Skipped {skipped} records with missing good/bad text.")


if __name__ == "__main__":
    main()
