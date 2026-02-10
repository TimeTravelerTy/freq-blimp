import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_TS_RE = re.compile(r"\d{8}-\d{6}")


def _parse_timestamp(text: str) -> Optional[str]:
    m = _TS_RE.search(text)
    return m.group(0) if m else None


def _strip_ts_prefix(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    m = _TS_RE.match(text)
    if m and text.startswith(f"{m.group(0)}_"):
        return text[len(m.group(0)) + 1 :]
    return text


def _strip_ts_suffix(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    m = _TS_RE.search(text)
    if m and text.endswith(f"_{m.group(0)}"):
        return text[: -(len(m.group(0)) + 1)]
    return text


def _strip_leading_timestamps(text: str) -> str:
    # Drop one or more leading timestamps separated by underscores.
    out = text
    while True:
        m = _TS_RE.match(out)
        if m and out.startswith(f"{m.group(0)}_"):
            out = out[len(m.group(0)) + 1 :]
        else:
            break
    return out


def _strip_pair_suffixes(stem: str) -> str:
    suffixes = [
        "_pair-scores_accuracy",
        "_pair-scores-accuracy",
        "_pair-scores_pair-scores-accuracy",
        "_pair-scores",
    ]
    out = stem
    for suf in suffixes:
        if out.endswith(suf):
            out = out[: -len(suf)]
    return out


def _parse_scores_name(path: Path) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    stem = _strip_pair_suffixes(path.stem)

    variant = None
    m_variant = re.search(r"_(rare|original|auto|both)$", stem)
    if m_variant:
        variant = m_variant.group(1)
        stem = stem[: m_variant.start()]

    ts = _parse_timestamp(stem)
    if ts and stem.startswith(f"{ts}_"):
        rest = stem[len(ts) + 1 :]
    else:
        rest = stem

    rest = _strip_leading_timestamps(rest)

    model = None
    data = None
    match = re.search(r"_(freq_blimp|rare_blimp|blimp)", rest)
    if match:
        model = rest[: match.start()]
        data = rest[match.start() + 1 :]
    else:
        parts = rest.split("_")
        if len(parts) >= 2:
            model = "_".join(parts[:-1])
            data = parts[-1]

    model = _strip_ts_suffix(model)
    data = _strip_ts_prefix(data)
    return model, data, variant, ts


def _group_label(data_slug: Optional[str]) -> Optional[str]:
    if not data_slug:
        return None
    if "original_pair" in data_slug or data_slug.endswith("original"):
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


def _display_model(model: str) -> str:
    return model.replace("_", ".")


def _infer_group_from_stem(stem: str) -> Optional[str]:
    if "original_pair" in stem or stem.endswith("original"):
        return "original dataset"
    # Look for any zipf window anywhere in the stem.
    m = re.search(r"zipf(?P<low>\d+_\d+)-(?P<high>\d+_\d+)", stem)
    if m:
        low = m.group("low").replace("_", ".")
        high = m.group("high").replace("_", ".")
        window = f"{low}-{high}"
        mapping = {
            "4.0-5.2": "4.0-5.2: head",
            "2.2-3.0": "2.2-3.0: tail",
            "1.2-2.0": "1.2-2.0: xtail",
        }
        return mapping.get(window, window)
    return "original dataset"


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
            parsed_path = Path(scores_path)
            model, data, _variant, ts = _parse_scores_name(parsed_path)
        else:
            parsed_path = obj["_path"]
            model, data, _variant, ts = _parse_scores_name(parsed_path)

        group = _group_label(data) if data is not None else None
        if group is None:
            group = _infer_group_from_stem(parsed_path.stem)

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
    label_map = {model: _display_model(model) for model in model_order}
    for i, model in enumerate(model_order):
        offsets = [pos - 0.4 + bar_width / 2 + i * bar_width for pos in group_positions]
        heights = []
        for group in wanted_groups:
            bucket = buckets.get((group, model))
            heights.append(bucket["accuracy"] if bucket else 0.0)
        ax.bar(offsets, heights, width=bar_width, label=label_map.get(model, model))

    ax.set_ylabel("Accuracy")
    ax.set_xticks(group_positions)
    ax.set_xticklabels(wanted_groups, rotation=0, ha="center")
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
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
