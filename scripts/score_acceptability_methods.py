from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SUPPORTED_METHODS = (
    "lp_readout",
    "nll",
    "in_template_lp",
    "in_template_meanlp",
    "in_template_penlp",
    "yes_no",
    "ensemble",
)

try:
    from src.acceptability_scoring import (
        expand_required_components,
        format_in_template,
        score_yes_no_probabilities,
        sentence_nll_score,
        template_lp_score,
        template_meanlp_score,
        template_penlp_score,
    )
    from src.sentence_nll import LlamaNLLScorer
except ModuleNotFoundError as e:  # pragma: no cover
    expand_required_components = None  # type: ignore[assignment]
    format_in_template = None  # type: ignore[assignment]
    score_yes_no_probabilities = None  # type: ignore[assignment]
    sentence_nll_score = None  # type: ignore[assignment]
    template_lp_score = None  # type: ignore[assignment]
    template_meanlp_score = None  # type: ignore[assignment]
    template_penlp_score = None  # type: ignore[assignment]
    LlamaNLLScorer = None  # type: ignore[assignment]
    _SCORING_IMPORT_ERROR = e
else:  # pragma: no cover
    _SCORING_IMPORT_ERROR = None


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
    for candidate in candidates:
        for rec in records:
            if candidate in rec and rec[candidate] is not None:
                return candidate
    return None


def _select_fields(
    variant: str,
    records: List[dict],
    good_field: Optional[str],
    bad_field: Optional[str],
) -> Tuple[str, str]:
    if variant == "original":
        good_candidates = ["good_original", "sentence_good", "good", "grammatical"]
        bad_candidates = ["bad_original", "sentence_bad", "bad", "ungrammatical"]
    elif variant in {"freq", "rare"}:
        good_candidates = ["good_freq", "good_rare", "sentence_good", "good", "grammatical"]
        bad_candidates = ["bad_freq", "bad_rare", "sentence_bad", "bad", "ungrammatical"]
    else:
        good_candidates = [
            "good_freq",
            "good_rare",
            "good_original",
            "sentence_good",
            "good",
            "grammatical",
        ]
        bad_candidates = [
            "bad_freq",
            "bad_rare",
            "bad_original",
            "sentence_bad",
            "bad",
            "ungrammatical",
        ]
    picked_good = _pick_field(records, good_field, good_candidates)
    picked_bad = _pick_field(records, bad_field, bad_candidates)
    if not picked_good or not picked_bad:
        raise ValueError("Could not infer good/bad fields; set --good-field and --bad-field.")
    return picked_good, picked_bad


def _variant_has_pairs(
    variant: str,
    records: List[dict],
    good_field: Optional[str],
    bad_field: Optional[str],
) -> bool:
    try:
        picked_good, picked_bad = _select_fields(variant, records, good_field, bad_field)
    except ValueError:
        return False
    for rec in records:
        if isinstance(rec.get(picked_good), str) and isinstance(rec.get(picked_bad), str):
            return True
    return False


def _parse_multi(values: Sequence[str]) -> List[str]:
    if len(values) == 1 and "," in values[0]:
        return [item.strip() for item in values[0].split(",") if item.strip()]
    return [item.strip() for item in values if item.strip()]


def _method_component_key(method: str) -> str:
    return "nll" if method == "lp_readout" else method


def _collect_paths(data: Sequence[str], pattern: Optional[str]) -> List[Path]:
    paths = [Path(p) for p in data]
    if pattern:
        paths.extend(Path().glob(pattern))
    uniq = []
    seen = set()
    for path in paths:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        uniq.append(path)
    files = [p for p in uniq if p.is_file()]
    if not files:
        raise SystemExit("No dataset files found. Use --data and/or --pattern.")
    return sorted(files)


def _model_slug(model: str) -> str:
    return model.split("/")[-1].replace(".", "_")


def _data_slug(data: Path) -> str:
    return data.name.rsplit(".", 1)[0]


def _default_out_path(
    model: str,
    data_path: Path,
    variant: str,
    method: str,
    run_ts: str,
    out_dir: Path,
) -> Path:
    name = (
        f"{run_ts}_{_model_slug(model)}_{_data_slug(data_path)}_{variant}_{method}"
        "_acceptability.jsonl"
    )
    return out_dir / name


def _swap_counts(meta: dict) -> Dict[str, int]:
    def _count(prefix: str) -> int:
        total = 0
        for key in (f"{prefix}_swaps", f"{prefix}_adj_swaps", f"{prefix}_verb_swaps"):
            val = meta.get(key)
            if isinstance(val, list):
                total += len(val)
        return total

    return {"g_swaps": _count("g"), "b_swaps": _count("b")}


def _build_common_row(
    rec: dict,
    model: str,
    dataset_path: Path,
    variant: str,
    method: str,
    good_field: str,
    bad_field: str,
    good_text: str,
    bad_text: str,
    score_good: float,
    score_bad: float,
) -> dict:
    prediction = "good" if score_good > score_bad else "bad"
    meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
    return {
        "idx": rec.get("idx"),
        "phenomenon": rec.get("phenomenon"),
        "subtask": rec.get("subtask"),
        "field": rec.get("field"),
        "model": model,
        "dataset_path": str(dataset_path),
        "dataset_name": dataset_path.name,
        "variant": variant,
        "method": method,
        "good_field": good_field,
        "bad_field": bad_field,
        "good_text": good_text,
        "bad_text": bad_text,
        "score_good": score_good,
        "score_bad": score_bad,
        "score_margin": score_good - score_bad,
        "prediction": prediction,
        "correctness": 1 if prediction == "good" else 0,
        "swap_counts": _swap_counts(meta),
    }


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _score_variant(
    scorer: LlamaNLLScorer,
    model: str,
    data_path: Path,
    records: List[dict],
    variant: str,
    methods: Sequence[str],
    args,
    run_ts: str,
    out_dir: Path,
) -> List[Path]:
    picked_good, picked_bad = _select_fields(variant, records, args.good_field, args.bad_field)
    out_paths = {
        method: _default_out_path(model, data_path, variant, method, run_ts, out_dir)
        for method in methods
    }
    pending_methods = [
        method for method, out_path in out_paths.items() if args.overwrite or not out_path.exists()
    ]
    if not pending_methods:
        print(f"[Skip] All outputs already exist for {data_path.name} ({variant}, {model}).")
        return []

    items: List[dict] = []
    texts: List[str] = []
    skipped = 0
    for rec in records:
        good = rec.get(picked_good)
        bad = rec.get(picked_bad)
        if not isinstance(good, str) or not isinstance(bad, str):
            skipped += 1
            continue
        items.append(rec)
        texts.extend([good, bad])
    if not texts:
        raise ValueError(f"No valid good/bad pairs found in {data_path} for variant={variant}.")

    required = expand_required_components(_method_component_key(method) for method in pending_methods)
    plain_scores = None
    template_scores = None
    yes_no_scores = None
    resolved_prompt_format = None

    if "plain" in required:
        print(f"[Score] plain sentence NLL for {data_path.name} ({variant})")
        plain_scores = scorer.score_texts(
            texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            show_progress=True,
        )
    if "template" in required:
        templated_texts = [format_in_template(text) for text in texts]
        print(f"[Score] in-template LP for {data_path.name} ({variant})")
        template_scores = scorer.score_texts(
            templated_texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            show_progress=True,
        )
    if "yes_no" in required:
        print(f"[Score] Yes/No prompts for {data_path.name} ({variant})")
        resolved_prompt_format, yes_no_scores = score_yes_no_probabilities(
            scorer,
            texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            prompt_format=args.prompt_format,
            show_progress=True,
        )

    completed: List[Path] = []
    for method in pending_methods:
        rows: List[dict] = []
        for i, rec in enumerate(items):
            good_text = texts[2 * i]
            bad_text = texts[2 * i + 1]
            row = None
            if method in {"lp_readout", "nll"}:
                assert plain_scores is not None
                good_seq = plain_scores[2 * i]
                bad_seq = plain_scores[2 * i + 1]
                score_good = sentence_nll_score(good_seq)
                score_bad = sentence_nll_score(bad_seq)
                row = _build_common_row(
                    rec,
                    model,
                    data_path,
                    variant,
                    method,
                    picked_good,
                    picked_bad,
                    good_text,
                    bad_text,
                    score_good,
                    score_bad,
                )
                row.update(
                    {
                        "good_total_nll": good_seq.total_nll,
                        "bad_total_nll": bad_seq.total_nll,
                        "good_token_count": good_seq.token_count,
                        "bad_token_count": bad_seq.token_count,
                    }
                )
            elif method == "in_template_lp":
                assert template_scores is not None
                good_seq = template_scores[2 * i]
                bad_seq = template_scores[2 * i + 1]
                score_good = template_lp_score(good_seq)
                score_bad = template_lp_score(bad_seq)
                row = _build_common_row(
                    rec,
                    model,
                    data_path,
                    variant,
                    method,
                    picked_good,
                    picked_bad,
                    good_text,
                    bad_text,
                    score_good,
                    score_bad,
                )
                row.update(
                    {
                        "good_total_logprob": good_seq.total_logprob,
                        "bad_total_logprob": bad_seq.total_logprob,
                        "good_token_count": good_seq.token_count,
                        "bad_token_count": bad_seq.token_count,
                    }
                )
            elif method == "in_template_meanlp":
                assert template_scores is not None
                good_seq = template_scores[2 * i]
                bad_seq = template_scores[2 * i + 1]
                score_good = template_meanlp_score(good_seq)
                score_bad = template_meanlp_score(bad_seq)
                row = _build_common_row(
                    rec,
                    model,
                    data_path,
                    variant,
                    method,
                    picked_good,
                    picked_bad,
                    good_text,
                    bad_text,
                    score_good,
                    score_bad,
                )
                row.update(
                    {
                        "good_total_logprob": good_seq.total_logprob,
                        "bad_total_logprob": bad_seq.total_logprob,
                        "good_token_count": good_seq.token_count,
                        "bad_token_count": bad_seq.token_count,
                    }
                )
            elif method == "in_template_penlp":
                assert template_scores is not None
                good_seq = template_scores[2 * i]
                bad_seq = template_scores[2 * i + 1]
                score_good = template_penlp_score(good_seq)
                score_bad = template_penlp_score(bad_seq)
                row = _build_common_row(
                    rec,
                    model,
                    data_path,
                    variant,
                    method,
                    picked_good,
                    picked_bad,
                    good_text,
                    bad_text,
                    score_good,
                    score_bad,
                )
                row.update(
                    {
                        "good_total_logprob": good_seq.total_logprob,
                        "bad_total_logprob": bad_seq.total_logprob,
                        "good_token_count": good_seq.token_count,
                        "bad_token_count": bad_seq.token_count,
                    }
                )
            elif method == "yes_no":
                assert yes_no_scores is not None
                good_yes = yes_no_scores[2 * i]
                bad_yes = yes_no_scores[2 * i + 1]
                score_good = good_yes["score"]
                score_bad = bad_yes["score"]
                row = _build_common_row(
                    rec,
                    model,
                    data_path,
                    variant,
                    method,
                    picked_good,
                    picked_bad,
                    good_text,
                    bad_text,
                    score_good,
                    score_bad,
                )
                row.update(
                    {
                        "prompt_format": resolved_prompt_format,
                        "good_p_yes": good_yes["p_yes"],
                        "good_p_no": good_yes["p_no"],
                        "bad_p_yes": bad_yes["p_yes"],
                        "bad_p_no": bad_yes["p_no"],
                    }
                )
            elif method == "ensemble":
                assert template_scores is not None
                assert yes_no_scores is not None
                good_lp = template_lp_score(template_scores[2 * i])
                bad_lp = template_lp_score(template_scores[2 * i + 1])
                good_yes = yes_no_scores[2 * i]
                bad_yes = yes_no_scores[2 * i + 1]
                score_good = good_lp + good_yes["score"]
                score_bad = bad_lp + bad_yes["score"]
                row = _build_common_row(
                    rec,
                    model,
                    data_path,
                    variant,
                    method,
                    picked_good,
                    picked_bad,
                    good_text,
                    bad_text,
                    score_good,
                    score_bad,
                )
                row.update(
                    {
                        "prompt_format": resolved_prompt_format,
                        "good_in_template_lp": good_lp,
                        "bad_in_template_lp": bad_lp,
                        "good_yes_no_score": good_yes["score"],
                        "bad_yes_no_score": bad_yes["score"],
                        "good_p_yes": good_yes["p_yes"],
                        "good_p_no": good_yes["p_no"],
                        "bad_p_yes": bad_yes["p_yes"],
                        "bad_p_no": bad_yes["p_no"],
                        "good_token_count": template_scores[2 * i].token_count,
                        "bad_token_count": template_scores[2 * i + 1].token_count,
                    }
                )
            else:
                raise ValueError(f"Unsupported method: {method}")
            rows.append(row)
        out_path = out_paths[method]
        _write_jsonl(out_path, rows)
        completed.append(out_path)
        print(f"[Saved] {out_path}")

    if skipped:
        print(f"[Warn] Skipped {skipped} records with missing good/bad text for {variant}.")
    return completed


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score BLiMP-style minimal pairs with multiple acceptability-scoring methods."
    )
    ap.add_argument("--data", nargs="*", default=[], help="Dataset JSONL paths.")
    ap.add_argument("--pattern", default=None, help="Optional glob pattern for dataset JSONL files.")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["meta-llama/Llama-3.2-1B"],
        help="Models to score (space-separated or comma-separated).",
    )
    ap.add_argument(
        "--methods",
        nargs="+",
        default=list(SUPPORTED_METHODS),
        help="Methods to run.",
    )
    ap.add_argument(
        "--variant",
        default="auto",
        choices=["freq", "rare", "original", "auto", "both"],
        help="Which sentence variant to score.",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--compile-model", action="store_true")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--use-slow-tokenizer", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--padding-side", default="left", choices=["left", "right"])
    ap.add_argument("--good-field", default=None)
    ap.add_argument("--bad-field", default=None)
    ap.add_argument(
        "--prompt-format",
        default="auto",
        choices=["auto", "base", "chat"],
        help="Yes/No prompt formatting. auto uses chat when tokenizer.chat_template exists.",
    )
    ap.add_argument(
        "--out-dir",
        default="results/acceptability_pair_scores",
        help="Output directory for per-method JSONL files.",
    )
    ap.add_argument("--run-timestamp", default=None)
    ap.add_argument("--manifest-out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if _SCORING_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "Scoring dependencies could not be imported. Activate the project environment "
            "and install requirements before running this script."
        ) from _SCORING_IMPORT_ERROR

    if args.variant == "rare":
        print("[Info] --variant rare is deprecated; using freq naming.")
        args.variant = "freq"

    methods = _parse_multi(args.methods)
    for method in methods:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method: {method}")
    models = _parse_multi(args.models)
    data_paths = _collect_paths(args.data, args.pattern)
    out_dir = Path(args.out_dir)
    run_ts = args.run_timestamp or time.strftime("%Y%m%d-%H%M%S")

    completed: List[Path] = []
    for model in models:
        print(f"\n[Model] Loading {model}")
        scorer = LlamaNLLScorer(
            model_name=model,
            tokenizer_name=args.tokenizer,
            device=args.device,
            dtype=args.dtype,
            device_map=args.device_map,
            compile_model=args.compile_model,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=args.trust_remote_code,
            padding_side=args.padding_side,
        )
        for data_path in data_paths:
            print(f"\n[Dataset] {data_path}")
            records = _load_jsonl(str(data_path), limit=args.limit)
            if not records:
                raise RuntimeError(f"No records loaded from {data_path}")
            if args.variant == "both":
                variants = ["original", "freq"]
            elif args.variant == "auto":
                variants = []
                if _variant_has_pairs("freq", records, args.good_field, args.bad_field):
                    variants.append("freq")
                if _variant_has_pairs("original", records, args.good_field, args.bad_field):
                    variants.append("original")
                if not variants:
                    print(f"[Skip] No scorable original or freq pairs found in {data_path}.")
                    continue
                if len(variants) > 1:
                    variants = variants[:1]
                print(f"[Auto] Resolved variant={variants[0]}")
            else:
                variants = [args.variant]

            for variant in variants:
                if not _variant_has_pairs(variant, records, args.good_field, args.bad_field):
                    print(f"[Skip] No scorable {variant} pairs found in {data_path}.")
                    continue
                completed.extend(
                    _score_variant(
                        scorer=scorer,
                        model=model,
                        data_path=data_path,
                        records=records,
                        variant=variant,
                        methods=methods,
                        args=args,
                        run_ts=run_ts,
                        out_dir=out_dir,
                    )
                )

    if completed:
        print("\nCompleted outputs:")
        for path in completed:
            print(f"  {path}")
    else:
        print("\nNo new outputs were written.")

    if args.manifest_out:
        manifest_path = Path(args.manifest_out)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_timestamp": run_ts,
            "outputs": [str(path) for path in completed],
        }
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
