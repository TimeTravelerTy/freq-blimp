import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

_SCORES_NAME_RE = re.compile(
    r"^(?P<run>\d{8}-\d{6})_(?P<model>.+?)_(?P<dataset>\d{8}-\d{6}_.+)_(?P<variant>rare|original)$"
)


def _run(cmd: List[str]) -> None:
    print("[Run] " + " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True)


def _parse_score_path(path: Path) -> Tuple[str, str, str]:
    stem = path.stem
    suffix = "_pair-scores"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    m = _SCORES_NAME_RE.match(stem)
    if not m:
        return "unknown_model", "unknown_dataset", "unknown"
    return m.group("model"), m.group("dataset"), m.group("variant")


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Streamlined freq-BLiMP evaluation: score -> accuracy -> key analysis plots."
    )
    ap.add_argument(
        "--data-pattern",
        default="data/processed/*freq_blimp*.jsonl",
        help="Glob for processed freq-BLiMP datasets.",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=[
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.2-3B",
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-8B-Instruct",
        ],
        help="Models to evaluate.",
    )
    ap.add_argument("--variant", default="auto", choices=["rare", "original", "auto", "both"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
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
    ap.add_argument("--score-out-dir", default="results/blimp_pair_scores")
    ap.add_argument("--accuracy-out-dir", default="results/blimp_accuracy_runs")
    ap.add_argument("--analysis-out-dir", default="results/analysis_plots")
    ap.add_argument("--summary-out-dir", default="results/eval_runs")
    ap.add_argument("--overwrite-scores", action="store_true")
    ap.add_argument("--run-timestamp", default=None)
    ap.add_argument(
        "--normalize-by",
        default="none",
        choices=["none", "char", "token"],
        help="Normalization used for accuracy and zipf_vs_nll.",
    )
    ap.add_argument(
        "--dataset-contains",
        default=None,
        help="Optional dataset substring filter for plots.",
    )
    args = ap.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    run_ts = args.run_timestamp or time.strftime("%Y%m%d-%H%M%S")

    manifest_path = Path(args.summary_out_dir) / f"{run_ts}_pair_scores_manifest.json"
    score_cmd = [
        sys.executable,
        str(scripts_dir / "blimp_pair_scores_timestamp_batch.py"),
        "--pattern",
        args.data_pattern,
        "--models",
        *args.models,
        "--variant",
        args.variant,
        "--batch-size",
        str(args.batch_size),
        "--max-length",
        str(args.max_length),
        "--dtype",
        args.dtype,
        "--padding-side",
        args.padding_side,
        "--out-dir",
        args.score_out_dir,
        "--run-timestamp",
        run_ts,
        "--manifest-out",
        str(manifest_path),
    ]
    if args.limit is not None:
        score_cmd.extend(["--limit", str(args.limit)])
    if args.device is not None:
        score_cmd.extend(["--device", args.device])
    if args.device_map is not None:
        score_cmd.extend(["--device-map", args.device_map])
    if args.tokenizer is not None:
        score_cmd.extend(["--tokenizer", args.tokenizer])
    if args.good_field is not None:
        score_cmd.extend(["--good-field", args.good_field])
    if args.bad_field is not None:
        score_cmd.extend(["--bad-field", args.bad_field])
    if args.compile_model:
        score_cmd.append("--compile-model")
    if args.use_slow_tokenizer:
        score_cmd.append("--use-slow-tokenizer")
    if args.trust_remote_code:
        score_cmd.append("--trust-remote-code")
    if args.overwrite_scores:
        score_cmd.append("--overwrite")
    _run(score_cmd)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    score_paths = [Path(p) for p in payload.get("outputs", []) if Path(p).is_file()]
    if not score_paths:
        score_paths = sorted(Path(args.score_out_dir).glob(f"{run_ts}_*_pair-scores.jsonl"))
    if not score_paths:
        raise SystemExit("No pair-score outputs found; nothing to evaluate.")

    accuracy_paths: List[Path] = []
    for score_path in score_paths:
        acc_name = f"{run_ts}_{score_path.stem}_{args.normalize_by}_accuracy.json"
        acc_out = Path(args.accuracy_out_dir) / acc_name
        acc_cmd = [
            sys.executable,
            str(scripts_dir / "blimp_accuracy.py"),
            "--scores",
            str(score_path),
            "--normalize-by",
            args.normalize_by,
            "--output",
            str(acc_out),
        ]
        _run(acc_cmd)
        accuracy_paths.append(acc_out)

    overall_rows: List[Dict[str, object]] = []
    phen_rows: List[Dict[str, object]] = []
    for acc_path in accuracy_paths:
        obj = json.loads(acc_path.read_text(encoding="utf-8"))
        score_path = Path(obj.get("scores_path", ""))
        model, dataset, variant = _parse_score_path(score_path)
        overall_rows.append(
            {
                "model": model,
                "dataset": dataset,
                "variant": variant,
                "normalize_by": obj.get("normalize_by"),
                "accuracy": obj.get("overall_accuracy"),
                "correct": obj.get("overall_correct"),
                "total": obj.get("overall_total"),
                "scores_path": str(score_path),
                "accuracy_path": str(acc_path),
            }
        )
        for row in obj.get("by_phenomenon", []) or []:
            phen_rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "variant": variant,
                    "normalize_by": obj.get("normalize_by"),
                    "phenomenon": row.get("key"),
                    "accuracy": row.get("accuracy"),
                    "correct": row.get("correct"),
                    "total": row.get("total"),
                }
            )

    overall_csv = Path(args.summary_out_dir) / f"{run_ts}_overall_accuracy.csv"
    per_phen_csv = Path(args.summary_out_dir) / f"{run_ts}_per_phenomenon_accuracy.csv"
    _write_csv(
        overall_csv,
        overall_rows,
        ["model", "dataset", "variant", "normalize_by", "accuracy", "correct", "total", "scores_path", "accuracy_path"],
    )
    _write_csv(
        per_phen_csv,
        phen_rows,
        ["model", "dataset", "variant", "normalize_by", "phenomenon", "accuracy", "correct", "total"],
    )

    print("\nOverall accuracy results:")
    for row in sorted(overall_rows, key=lambda r: (str(r["model"]), str(r["dataset"]), str(r["variant"]))):
        acc = row.get("accuracy")
        if isinstance(acc, float):
            acc_text = f"{acc:.4f}"
        else:
            acc_text = str(acc)
        print(
            f"  model={row['model']} dataset={row['dataset']} variant={row['variant']} "
            f"acc={acc_text} ({row['correct']}/{row['total']})"
        )

    score_pattern = str(Path(args.score_out_dir) / f"{run_ts}_*_pair-scores.jsonl")
    zipf_vs_nll_cmd = [
        sys.executable,
        str(scripts_dir / "plot_zipf_vs_nll.py"),
        "--pattern",
        score_pattern,
        "--out",
        str(Path(args.analysis_out_dir) / f"{run_ts}_zipf_vs_nll.png"),
        "--out-pdf",
        str(Path(args.analysis_out_dir) / f"{run_ts}_zipf_vs_nll.pdf"),
    ]
    if args.normalize_by == "char":
        zipf_vs_nll_cmd.append("--char-normalize")
    elif args.normalize_by == "token":
        zipf_vs_nll_cmd.append("--token-normalize")
    if args.dataset_contains:
        zipf_vs_nll_cmd.extend(["--dataset-contains", args.dataset_contains])
    _run(zipf_vs_nll_cmd)

    zipf_vs_token_cmd = [
        sys.executable,
        str(scripts_dir / "plot_zipf_vs_token_len.py"),
        "--pattern",
        score_pattern,
        "--out",
        str(Path(args.analysis_out_dir) / f"{run_ts}_zipf_vs_token_len.png"),
        "--out-pdf",
        str(Path(args.analysis_out_dir) / f"{run_ts}_zipf_vs_token_len.pdf"),
    ]
    if args.dataset_contains:
        zipf_vs_token_cmd.extend(["--dataset-contains", args.dataset_contains])
    _run(zipf_vs_token_cmd)

    regime_cmd = [
        sys.executable,
        str(scripts_dir / "regime_diagnostics.py"),
        "--pattern",
        score_pattern,
        "--out",
        str(Path(args.analysis_out_dir) / f"{run_ts}_regime_diagnostics.png"),
        "--out-pdf",
        str(Path(args.analysis_out_dir) / f"{run_ts}_regime_diagnostics.pdf"),
        "--tables-out",
        str(Path(args.analysis_out_dir) / f"{run_ts}_regime_diagnostics_tables.txt"),
    ]
    if args.dataset_contains:
        regime_cmd.extend(["--dataset-contains", args.dataset_contains])
    _run(regime_cmd)

    print("\nSaved key outputs:")
    print(f"  pair-score manifest: {manifest_path}")
    print(f"  overall accuracy csv: {overall_csv}")
    print(f"  per-phenomenon csv: {per_phen_csv}")
    print(f"  analysis directory: {Path(args.analysis_out_dir)}")


if __name__ == "__main__":
    main()
