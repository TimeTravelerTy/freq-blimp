import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _default_out_path(out_dir: Path, scores_path: Path, timestamp: str) -> Path:
    base = scores_path.name.rsplit(".", 1)[0]
    name = f"{timestamp}_{base}_pair-scores-accuracy.json"
    return out_dir / name


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch BLiMP accuracy over all blimp_pair_scores JSONL files in a folder."
    )
    ap.add_argument(
        "--scores-dir",
        default="results/blimp_pair_scores",
        help="Directory containing blimp_pair_scores JSONL files.",
    )
    ap.add_argument(
        "--pattern",
        default="*.jsonl",
        help="Glob pattern within --scores-dir (default: *.jsonl).",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Optional directory for metrics outputs (defaults to blimp_accuracy.py behavior).",
    )
    ap.add_argument(
        "--timestamp",
        default=None,
        help="Optional timestamp prefix for output filenames (defaults to current time).",
    )
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of rows.")
    ap.add_argument("--good-nll-field", default=None)
    ap.add_argument("--bad-nll-field", default=None)
    ap.add_argument("--phenomenon-field", default=None)
    ap.add_argument("--subtask-field", default=None)
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = ap.parse_args()

    scores_dir = Path(args.scores_dir)
    paths = sorted(p for p in scores_dir.glob(args.pattern) if p.is_file())
    if not paths:
        raise SystemExit(f"No scores files found in {scores_dir} with pattern {args.pattern}")

    script_path = Path(__file__).resolve().parent / "blimp_accuracy.py"
    out_dir = Path(args.out_dir) if args.out_dir else None
    timestamp = args.timestamp or time.strftime("%Y%m%d-%H%M%S")
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    completed = []
    for scores_path in paths:
        cmd = [sys.executable, str(script_path), "--scores", str(scores_path)]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.good_nll_field:
            cmd += ["--good-nll-field", args.good_nll_field]
        if args.bad_nll_field:
            cmd += ["--bad-nll-field", args.bad_nll_field]
        if args.phenomenon_field:
            cmd += ["--phenomenon-field", args.phenomenon_field]
        if args.subtask_field:
            cmd += ["--subtask-field", args.subtask_field]
        if out_dir:
            out_path = _default_out_path(out_dir, scores_path, timestamp)
            cmd += ["--output", str(out_path)]

        cmd_text = " ".join(shlex.quote(part) for part in cmd)
        print(f"[Batch] {scores_path}")
        print(f"        {cmd_text}")
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)
        if out_dir:
            completed.append(out_path)

    if args.dry_run:
        print("\nDry run only; no accuracies computed.")
    elif completed:
        print("\nCompleted outputs:")
        for path in completed:
            print(f"  {path}")


if __name__ == "__main__":
    main()
