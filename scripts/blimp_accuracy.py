import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from src.sentence_nll import LlamaNLLScorer  # noqa: E402
except ModuleNotFoundError as e:  # pragma: no cover
    LlamaNLLScorer = None  # type: ignore[assignment]
    _SENTENCE_NLL_IMPORT_ERROR = e
else:  # pragma: no cover
    _SENTENCE_NLL_IMPORT_ERROR = None

try:
    from datasets import load_dataset  # noqa: E402
except ModuleNotFoundError as e:  # pragma: no cover
    load_dataset = None  # type: ignore[assignment]
    _DATASETS_IMPORT_ERROR = e
else:  # pragma: no cover
    _DATASETS_IMPORT_ERROR = None


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


def _load_hf_blimp(config: str, split: str, limit: Optional[int]) -> List[dict]:
    if load_dataset is None:
        raise ModuleNotFoundError(
            "datasets is required for --hf-config but could not be imported"
        ) from _DATASETS_IMPORT_ERROR
    ds = load_dataset("nyu-mll/blimp", config)[split]
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return [dict(rec) for rec in ds]


def _pick_field(records: List[dict], explicit: Optional[str], candidates: List[str]) -> Optional[str]:
    if explicit:
        return explicit
    for c in candidates:
        for rec in records:
            if c in rec and rec[c] is not None:
                return c
    return None


def _group_key(rec: dict, field: Optional[str], fallback: str) -> str:
    if field:
        val = rec.get(field)
        if val is not None and val != "":
            return str(val)
    return fallback


def _update_stats(stats: Dict[str, Dict[str, int]], key: str, correct: int) -> None:
    bucket = stats.setdefault(key, {"correct": 0, "total": 0})
    bucket["correct"] += correct
    bucket["total"] += 1


def _accuracy(stats: Dict[str, int]) -> float:
    total = stats.get("total", 0)
    if total <= 0:
        return float("nan")
    return stats.get("correct", 0) / total


def _print_stats(title: str, stats: Dict[str, Dict[str, int]]) -> None:
    print(f"\n{title}")
    for key, bucket in sorted(stats.items(), key=lambda kv: (-kv[1]["total"], kv[0])):
        acc = _accuracy(bucket)
        print(f"  {key}: acc={acc:.4f} ({bucket['correct']}/{bucket['total']})")


def _model_slug(model: str) -> str:
    return model.split("/")[-1].replace(".", "_")


def _data_slug(data: Optional[str], hf_config: Optional[str]) -> str:
    if hf_config:
        return hf_config
    if data:
        return Path(data).name.rsplit(".", 1)[0]
    return "blimp"


def _default_out_path(model: str, data: Optional[str], hf_config: Optional[str], variant: str) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    model_slug = _model_slug(model)
    data_slug = _data_slug(data, hf_config)
    name = f"{ts}_{model_slug}_{data_slug}_{variant}_accuracy.json"
    return Path("results") / "blimp_accuracy_runs" / name


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute BLiMP accuracy using a HF causal LM (logPgood > logPbad)."
    )
    ap.add_argument("--data", help="Path to a BLiMP/rare-BLiMP JSONL dataset.")
    ap.add_argument("--hf-config", help="BLiMP config name to load from HF (nyu-mll/blimp).")
    ap.add_argument("--hf-split", default="train", help="HF split name (default: train).")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of records.")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--device-map", default=None, help="Optional HF device_map (e.g., auto).")
    ap.add_argument("--compile-model", action="store_true", help="Use torch.compile when available.")
    ap.add_argument(
        "--variant",
        default="rare",
        choices=["rare", "original", "auto"],
        help="Which sentence variant to score when both are present (default: rare).",
    )
    ap.add_argument("--good-field", default=None, help="Field for grammatical sentences.")
    ap.add_argument("--bad-field", default=None, help="Field for ungrammatical sentences.")
    ap.add_argument("--phenomenon-field", default=None, help="Field for phenomenon.")
    ap.add_argument("--subtask-field", default=None, help="Field for subtask/config.")
    ap.add_argument("--output", default=None, help="Optional JSON output path for metrics.")
    args = ap.parse_args()

    if not args.data and not args.hf_config:
        ap.error("Provide --data (JSONL) or --hf-config (HF BLiMP config).")

    if LlamaNLLScorer is None:
        raise ModuleNotFoundError(
            "sentence_nll could not be imported; check dependencies"
        ) from _SENTENCE_NLL_IMPORT_ERROR

    if args.data:
        records = _load_jsonl(args.data, limit=args.limit)
        default_subtask = "unknown"
    else:
        records = _load_hf_blimp(args.hf_config, args.hf_split, args.limit)
        default_subtask = args.hf_config

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

    phenomenon_field = _pick_field(records, args.phenomenon_field, ["phenomenon", "group"])
    subtask_field = _pick_field(records, args.subtask_field, ["subtask"])

    texts: List[str] = []
    pair_meta: List[Tuple[str, str]] = []
    skipped = 0
    for rec in records:
        good = rec.get(good_field)
        bad = rec.get(bad_field)
        if not isinstance(good, str) or not isinstance(bad, str):
            skipped += 1
            continue
        texts.append(good)
        texts.append(bad)
        phen = _group_key(rec, phenomenon_field, "unknown")
        subtask = _group_key(rec, subtask_field, default_subtask)
        pair_meta.append((phen, subtask))

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

    overall = {"correct": 0, "total": 0}
    by_phenomenon: Dict[str, Dict[str, int]] = {}
    by_subtask: Dict[str, Dict[str, int]] = {}
    by_pair: Dict[str, Dict[str, int]] = {}

    for i, (phen, subtask) in enumerate(pair_meta):
        good_score = scores[2 * i].total_nll
        bad_score = scores[2 * i + 1].total_nll
        correct = 1 if good_score < bad_score else 0
        overall["correct"] += correct
        overall["total"] += 1
        _update_stats(by_phenomenon, phen, correct)
        _update_stats(by_subtask, subtask, correct)
        _update_stats(by_pair, f"{phen}|{subtask}", correct)

    print("BLiMP accuracy (logPgood > logPbad):")
    print(f"  overall: acc={_accuracy(overall):.4f} ({overall['correct']}/{overall['total']})")
    if skipped:
        print(f"  skipped: {skipped}")
    _print_stats("By phenomenon", by_phenomenon)
    _print_stats("By subtask", by_subtask)
    _print_stats("By phenomenon|subtask", by_pair)

    def _to_rows(stats: Dict[str, Dict[str, int]]) -> List[dict]:
        rows = []
        for key, bucket in stats.items():
            rows.append(
                {
                    "key": key,
                    "accuracy": _accuracy(bucket),
                    "correct": bucket["correct"],
                    "total": bucket["total"],
                }
            )
        rows.sort(key=lambda r: (-r["accuracy"], -r["total"], r["key"]))
        return rows

    out = {
        "model": args.model,
        "variant": args.variant,
        "good_field": good_field,
        "bad_field": bad_field,
        "phenomenon_field": phenomenon_field,
        "subtask_field": subtask_field,
        "overall_accuracy": _accuracy(overall),
        "overall_correct": overall["correct"],
        "overall_total": overall["total"],
        "by_phenomenon": _to_rows(by_phenomenon),
        "by_subtask": _to_rows(by_subtask),
        "by_phenomenon_subtask": _to_rows(by_pair),
        "skipped": skipped,
    }
    out_path = Path(args.output) if args.output else _default_out_path(
        args.model, args.data, args.hf_config, args.variant
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
