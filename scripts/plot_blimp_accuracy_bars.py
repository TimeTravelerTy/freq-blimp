import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _parse_timestamp(text: str) -> Optional[str]:
    m = re.search(r"\d{8}-\d{6}", text)
    return m.group(0) if m else None


def _parse_scores_name(path: Path) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    stem = path.stem
    if stem.endswith("_pair-scores"):
        stem = stem[: -len("_pair-scores")]

    variant = None
    for cand in ("rare", "original", "auto", "both"):
        if stem.endswith(f"_{cand}"):
            variant = cand
            stem = stem[: -(len(cand) + 1)]
            break

    ts = _parse_timestamp(stem)
    if ts and stem.startswith(f"{ts}_"):
        rest = stem[len(ts) + 1 :]
    else:
        rest = stem

    model = None
    data = None
    match = re.search(r"_(rare_blimp|blimp)", rest)
    if match:
        model = rest[: match.start()]
        data = rest[match.start() + 1 :]
    else:
        parts = rest.split("_")
        if len(parts) >= 2:
            model = "_".join(parts[:-1])
            data = parts[-1]

    return model, data, variant, ts


def _group_label(data_slug: Optional[str]) -> Optional[str]:
    if not data_slug:
        return None
    if "zipf" not in data_slug:
        return "original dataset"
    match = re.search(r"zipf(?P<low>\d+_\d+)-(?P<high>\d+_\d+)", data_slug)
    if not match:
        return None
    low = match.group("low").replace("_", ".")
    high = match.group("high").replace("_", ".")
    window = f"{low}-{high}"
    mapping = {
        "4.0-5.2": "4.0-5.2: head",
        "2.2-3.0": "2.2-3.0: tail",
        "1.2-2.0": "1.2-2.0: xtail",
    }
    return mapping.get(window, window)


def _load_runs(runs_dir: Path, pattern: str) -> List[dict]:
    runs = []
    for path in sorted(runs_dir.glob(pattern)):
        if not path.is_file():
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        obj["_path"] = path
        runs.append(obj)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot grouped bar charts from blimp_accuracy_runs JSON outputs."
    )
    ap.add_argument(
        "--runs-dir",
        default="results/blimp_accuracy_runs",
        help="Directory containing blimp_accuracy_runs JSON outputs.",
    )
    ap.add_argument("--pattern", default="*.json", help="Glob pattern for runs.")
    ap.add_argument(
        "--output",
        default="results/blimp_accuracy_runs/blimp_accuracy_bar.png",
        help="Output path for the chart image.",
    )
    ap.add_argument(
        "--title",
        default=None,
        help="Optional chart title.",
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional ordered list of model slugs to include.",
    )
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    runs = _load_runs(runs_dir, args.pattern)
    if not runs:
        raise SystemExit(f"No runs found in {runs_dir} with pattern {args.pattern}")

    wanted_groups = [
        "original dataset",
        "4.0-5.2: head",
        "2.2-3.0: tail",
        "1.2-2.0: xtail",
    ]

    buckets: Dict[Tuple[str, str], dict] = {}
    models_found = set()
    for obj in runs:
        acc = obj.get("overall_accuracy")
        if not isinstance(acc, (int, float)):
            continue
        scores_path = obj.get("scores_path")
        if scores_path:
            model, data, _variant, ts = _parse_scores_name(Path(scores_path))
        else:
            model, data, _variant, ts = _parse_scores_name(obj["_path"])
        group = _group_label(data)
        if not model or not group:
            continue
        if group not in wanted_groups:
            continue
        models_found.add(model)
        key = (group, model)
        prev = buckets.get(key)
        if prev is None or (ts and (prev.get("_ts") or "") < ts):
            buckets[key] = {"accuracy": acc, "_ts": ts}

    if args.models:
        model_order = args.models
    else:
        model_order = sorted(models_found)

    if not model_order:
        raise SystemExit("No model entries found to plot.")

    import matplotlib.pyplot as plt

    group_positions = list(range(len(wanted_groups)))
    bar_width = 0.8 / max(1, len(model_order))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(model_order):
        offsets = [pos - 0.4 + bar_width / 2 + i * bar_width for pos in group_positions]
        heights = []
        for group in wanted_groups:
            bucket = buckets.get((group, model))
            heights.append(bucket["accuracy"] if bucket else 0.0)
        ax.bar(offsets, heights, width=bar_width, label=model)

    ax.set_ylabel("Accuracy")
    ax.set_xticks(group_positions)
    ax.set_xticklabels(wanted_groups, rotation=0, ha="center")
    ax.set_ylim(0.0, 1.0)
    if args.title:
        ax.set_title(args.title)
    ax.legend(title="Model", fontsize=9)
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved chart to {out_path}")


if __name__ == "__main__":
    main()
