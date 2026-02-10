import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Unified entrypoint for freq BLiMP dataset generation."
    )
    ap.add_argument(
        "--zipf-values",
        nargs="+",
        type=float,
        default=None,
        help="Generate one dataset per Zipf value (batch mode). Omit for single-run mode.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them (batch mode only).",
    )
    args, passthrough = ap.parse_known_args()

    scripts_dir = Path(__file__).resolve().parent
    if args.zipf_values:
        target = scripts_dir / "make_freq_blimp_datasets.py"
        cmd = [
            sys.executable,
            str(target),
            "--zipf_values",
            *[str(v) for v in args.zipf_values],
            *passthrough,
        ]
        if args.dry_run:
            cmd.append("--dry-run")
    else:
        target = scripts_dir / "make_freq_blimp_dataset.py"
        if args.dry_run:
            raise SystemExit("--dry-run is only supported with --zipf-values.")
        cmd = [sys.executable, str(target), *passthrough]

    print("[Run] " + " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
