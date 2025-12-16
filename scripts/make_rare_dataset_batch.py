import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _fmt_zipf(val: float) -> str:
    try:
        return str(val).replace(".", "_")
    except Exception:
        return str(val)


def _out_path(ts: str, zipf_val: float, out_dir: Path) -> Path:
    slug = _fmt_zipf(zipf_val)
    name = f"{ts}_rare_blimp_zipf{slug}_adj{slug}_verb{slug}.jsonl"
    return out_dir / name


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate multiple rare BLiMP datasets for a list of Zipf thresholds."
    )
    ap.add_argument(
        "--zipf_values",
        nargs="+",
        required=True,
        type=float,
        help="One or more Zipf thresholds to pass as --zipf_all to make_rare_dataset.py.",
    )
    ap.add_argument(
        "--out-dir",
        default="data/processed",
        help="Directory to write JSONL outputs (defaults to data/processed).",
    )
    ap.add_argument(
        "--timestamp",
        default=None,
        help="Timestamp prefix for output filenames (defaults to current time, shared across runs).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the per-threshold commands without executing them.",
    )
    args, passthrough = ap.parse_known_args()

    script_path = Path(__file__).resolve().parent / "make_rare_dataset.py"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = args.timestamp or time.strftime("%Y%m%d-%H%M%S")

    completed = []
    for zipf_val in args.zipf_values:
        out_path = _out_path(ts, zipf_val, out_dir)
        cmd = [
            sys.executable,
            str(script_path),
            "--zipf_all",
            str(zipf_val),
            "--out",
            str(out_path),
            *passthrough,
        ]
        cmd_text = " ".join(shlex.quote(part) for part in cmd)
        print(f"[Batch] zipf_all={zipf_val} -> {out_path}")
        print(f"        {cmd_text}")
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)
        completed.append(out_path)

    if args.dry_run:
        print("\nDry run only; no datasets generated.")
    else:
        print("\nCompleted datasets:")
        for path in completed:
            print(f"  {path}")


if __name__ == "__main__":
    main()
