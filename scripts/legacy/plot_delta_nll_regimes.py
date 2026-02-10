import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_MPL_DIR = Path("results/analysis_plots/.mplconfig")
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_PAIR_RE = re.compile(
    r"(?:^|/)(?P<run_ts>\d{8}-\d{6})_(?P<model>.+?)_(?P<dataset>\d{8}-\d{6}_.+?)_pair-scores\.jsonl$"
)
_ZIPF_RE = re.compile(r"zipf(?P<low>\d+_\d+)-(?P<high>\d+_\d+)")

_WINDOW_TO_GROUP = {
    "4.0-5.2": "head",
    "3.6-5.0": "head",
    "2.2-3.0": "tail",
    "1.2-2.0": "xtail",
}
_GROUP_ORDER = ["original", "head", "tail", "xtail"]


def _parse_pair_scores_path(path: Path) -> Tuple[str, str, str, str]:
    m = _PAIR_RE.search(str(path))
    if not m:
        raise ValueError(f"Unparseable pair-scores filename: {path}")
    dataset_full = m.group("dataset")
    variant_hint = "original" if dataset_full.endswith("_original") else "rare"
    dataset_base = re.sub(r"_(original|rare)$", "", dataset_full)
    return m.group("model"), dataset_base, variant_hint, m.group("run_ts")


def _group_label(dataset_base: str, variant_hint: str) -> Optional[str]:
    if variant_hint == "original":
        return "original"
    m = _ZIPF_RE.search(dataset_base)
    if not m:
        return None
    low = m.group("low").replace("_", ".")
    high = m.group("high").replace("_", ".")
    window = f"{low}-{high}"
    return _WINDOW_TO_GROUP.get(window)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _mean_delta_nll(path: Path) -> Tuple[float, int]:
    total = 0.0
    n = 0
    for rec in _iter_jsonl(path):
        good = rec.get("good_total_nll")
        bad = rec.get("bad_total_nll")
        if not isinstance(good, (int, float)) or not isinstance(bad, (int, float)):
            continue
        total += float(bad - good)
        n += 1
    if n == 0:
        raise ValueError(f"No valid rows in {path}")
    return total / n, n


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot mean delta NLL across original/head/tail/xtail regimes."
    )
    ap.add_argument(
        "--pattern",
        default="results/blimp_pair_scores/*.jsonl",
        help="Glob for pair-score JSONL files.",
    )
    ap.add_argument(
        "--out",
        default="results/analysis_plots/delta_nll_regimes.png",
        help="Output PNG path.",
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional ordered list of model slugs to include.",
    )
    ap.add_argument(
        "--dataset-contains",
        default=None,
        help="Optional substring that dataset name must contain.",
    )
    args = ap.parse_args()

    DEFAULT_MPL_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path().glob(args.pattern))
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise SystemExit(f"No pair-score files found for {args.pattern}")

    best: Dict[Tuple[str, str], Tuple[str, Path]] = {}
    for p in paths:
        try:
            model, dataset_base, variant_hint, run_ts = _parse_pair_scores_path(p)
        except ValueError:
            continue
        if args.dataset_contains and args.dataset_contains not in dataset_base:
            continue
        group = _group_label(dataset_base, variant_hint)
        if group is None or group not in _GROUP_ORDER:
            continue
        key = (model, group)
        prev = best.get(key)
        if prev is None or prev[0] < run_ts:
            best[key] = (run_ts, p)

    models = args.models if args.models else sorted({k[0] for k in best})
    if not models:
        raise SystemExit("No model entries found to plot.")

    x_positions = [0, 2, 3, 4]
    x_labels = ["original", "head", "tail", "xtail"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model in models:
        ys: List[float] = []
        xs: List[float] = []
        for group, x in zip(_GROUP_ORDER, x_positions):
            entry = best.get((model, group))
            if not entry:
                continue
            _, path = entry
            mean_delta, _n = _mean_delta_nll(path)
            xs.append(x)
            ys.append(mean_delta)
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", label=model.replace("_", "."))

    ax.set_xticks(x_positions, x_labels)
    ax.axvline(1.0, color="0.7", linestyle="--", linewidth=1)
    ax.set_ylabel("Mean Δ NLL (bad - good)")
    ax.set_title("Mean delta NLL across regimes (original vs frequency windows)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
