#!/usr/bin/env python3
"""
Parallel benchmark generation for all 9 FreqBLiMP datasets.

Runs up to --jobs processes simultaneously, logging each to logs/<name>.log.
Already-completed outputs are skipped unless --force is given.

Usage:
    python scripts/generate_benchmark.py
    python scripts/generate_benchmark.py --jobs 4
    python scripts/generate_benchmark.py --force --jobs 2
    python scripts/generate_benchmark.py --dry-run
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

DATASETS = [
    # (name, zipf_min, zipf_max, cond, extra_flags)
    ("xtail_1_2-2_2_cond1",   "1.2", "2.2", 1, []),
    ("xtail_1_2-2_2_cond2",   "1.2", "2.2", 2, ["--semantic-match", "soft"]),
    # cond3 xtail = v19, already done — kept here for reference, skipped by default
    ("xtail_1_2-2_2_cond3",   "1.2", "2.2", 3, ["--semantic-match", "soft", "--consistent-lemma-map"]),
    ("tail_2_4-3_2_cond1",    "2.4", "3.2", 1, []),
    ("tail_2_4-3_2_cond2",    "2.4", "3.2", 2, ["--semantic-match", "soft"]),
    ("tail_2_4-3_2_cond3",    "2.4", "3.2", 3, ["--semantic-match", "soft", "--consistent-lemma-map"]),
    ("head_3_5-5_5_cond1",    "3.5", "5.5", 1, []),
    ("head_3_5-5_5_cond2",    "3.5", "5.5", 2, ["--semantic-match", "soft"]),
    ("head_3_5-5_5_cond3",    "3.5", "5.5", 3, ["--semantic-match", "soft", "--consistent-lemma-map"]),
]

OUT_DIR  = Path("data/processed")
LOG_DIR  = Path("logs")
SEED     = 42
PYTHON   = sys.executable
SCRIPT   = Path("scripts/make_freq_blimp.py")

# xtail cond3 is already done as v19
EXISTING = {
    # xtail cond3 already generated as v19 (with lexname fallback fix)
    "xtail_1_2-2_2_cond3": OUT_DIR / "freq_blimp_xtail_1_2-2_2_cond3_sem_consistent_v19.jsonl",
}

# Datasets whose output file exists on disk but is pre-fix (stale); always regenerate.
STALE = {
    "head_3_5-5_5_cond3",
}


def out_path(name: str) -> Path:
    suffix = ""
    if "_cond2" in name:
        suffix = "_sem"
    elif "_cond3" in name:
        suffix = "_sem_consistent"
    return OUT_DIR / f"freq_blimp_{name}{suffix}.jsonl"


def run_one(name, zipf_min, zipf_max, extra_flags, force=False):
    """Run generation for one dataset. Returns (name, success, elapsed)."""
    log_path = LOG_DIR / f"{name}.log"
    dst = out_path(name)

    if not force and name not in STALE and dst.exists():
        return name, True, 0.0, "skipped (already exists)"

    # Check if this is an existing dataset with a different filename
    existing = EXISTING.get(name)
    if not force and existing and existing.exists():
        # Symlink or copy to canonical name
        if not dst.exists():
            shutil.copy2(existing, dst)
        return name, True, 0.0, f"skipped (exists as {existing.name})"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, str(SCRIPT),
        "--zipf_min_all", zipf_min,
        "--zipf_max_all", zipf_max,
        "--swap_target", "all",
        "--seed", str(SEED),
        "--out", str(dst),
    ] + extra_flags

    t0 = time.time()
    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).parent.parent,
        )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        return name, False, elapsed, f"FAILED (exit {proc.returncode}) — see {log_path}"

    return name, True, elapsed, f"done in {elapsed/60:.1f}m → {dst.name}"


def _worker(args):
    return run_one(*args)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=4, help="Parallel workers (default: 4)")
    ap.add_argument("--force", action="store_true", help="Re-generate even if output exists")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running")
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help="Run only datasets whose name contains any of these substrings")
    args = ap.parse_args()

    datasets = DATASETS
    if args.only:
        datasets = [d for d in datasets if any(k in d[0] for k in args.only)]
        if not datasets:
            print(f"No datasets match --only {args.only}")
            sys.exit(1)

    if args.dry_run:
        for name, zmin, zmax, cond, extra in datasets:
            dst = EXISTING.get(name, out_path(name))
            if name in STALE:
                status = "(stale — will regenerate)"
            elif dst.exists():
                status = "(exists — will skip)"
            else:
                status = ""
            flags = " ".join(extra)
            print(f"  {name:35s}  zipf=[{zmin},{zmax}]  {flags}  {status}")
        return

    # Build work items
    work = [
        (name, zmin, zmax, extra, args.force)
        for name, zmin, zmax, cond, extra in datasets
    ]

    print(f"Generating {len(work)} datasets with --jobs {args.jobs}")
    print(f"Logs → {LOG_DIR}/  Outputs → {OUT_DIR}/\n")

    t_start = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_worker, w): w[0] for w in work}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                _, success, elapsed, msg = fut.result()
            except Exception as exc:
                msg = f"EXCEPTION: {exc}"
                success = False
            status = "✓" if success else "✗"
            completed += 1
            print(f"  [{completed}/{len(work)}] {status} {name}: {msg}")

    total = time.time() - t_start
    print(f"\nAll done in {total/60:.1f}m")

    # Run QA on completed outputs
    print("\nRunning QA...")
    qa_paths = []
    for name, _, _, cond, _ in datasets:
        p = EXISTING.get(name, out_path(name))
        if p.exists():
            qa_paths.append(str(p))

    if qa_paths:
        consistency_paths = [p for p in qa_paths if "cond3" in p]
        other_paths       = [p for p in qa_paths if "cond3" not in p]

        if other_paths:
            subprocess.run([PYTHON, "scripts/qa_dataset.py"] + other_paths)
        if consistency_paths:
            subprocess.run([PYTHON, "scripts/qa_dataset.py", "--consistency"] + consistency_paths)


if __name__ == "__main__":
    main()
