import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_TS_RE = re.compile(r"\d{8}-\d{6}")


def _parse_timestamp(text: str) -> Optional[str]:
    m = _TS_RE.search(text)
    return m.group(0) if m else None


def _strip_leading_timestamps(text: str) -> str:
    out = text
    while True:
        m = _TS_RE.match(out)
        if m and out.startswith(f"{m.group(0)}_"):
            out = out[len(m.group(0)) + 1 :]
        else:
            break
    return out


def _strip_ts_suffix(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    m = _TS_RE.search(text)
    if m and text.endswith(f"_{m.group(0)}"):
        return text[: -(len(m.group(0)) + 1)]
    return text


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
    match = re.search(r"_(rare_blimp|blimp)", rest)
    if match:
        model = rest[: match.start()]
        data = rest[match.start() + 1 :]
    else:
        parts = rest.split("_")
        if len(parts) >= 2:
            model = "_".join(parts[:-1])
            data = parts[-1]

    model = _strip_ts_suffix(model)
    return model, data, variant, ts


def _group_label(data_slug: Optional[str], stem: str) -> Optional[str]:
    if data_slug:
        if "original_pair" in data_slug or data_slug.endswith("original"):
            return "original dataset"
        m = re.search(r"zipf(?P<low>\d+_\d+)-(?P<high>\d+_\d+)", data_slug)
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
    # Fallback to stem sniffing.
    if "original_pair" in stem or stem.endswith("original"):
        return "original dataset"
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


def _display_model(model: str) -> str:
    return model.replace("_", ".")


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
        description="Print a table of accuracies per model and dataset window from blimp_accuracy_runs."
    )
    ap.add_argument("--runs-dir", default="results/blimp_accuracy_runs")
    ap.add_argument("--pattern", default="*.json")
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional ordered list of model slugs to include.",
    )
    args = ap.parse_args()

    runs = _load_runs(Path(args.runs_dir), args.pattern)
    if not runs:
        raise SystemExit(f"No runs found in {args.runs_dir} matching {args.pattern}")

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
        if obj.get("scores_path"):
            parsed_path = Path(obj["scores_path"])
        else:
            parsed_path = obj["_path"]
        model, data, variant, ts = _parse_scores_name(parsed_path)
        # Use variant if present; otherwise try to infer from data_slug
        if variant == "original":
            group = "original dataset"
        else:
            group = _group_label(data, parsed_path.stem)
        if not model or group not in wanted_groups:
            continue
        models_found.add(model)
        key = (group, model)
        prev = buckets.get(key)
        if prev is None or (ts and (prev.get("_ts") or "") < ts):
            buckets[key] = {"accuracy": acc, "_ts": ts}

    model_order = args.models if args.models else sorted(models_found)
    if not model_order:
        raise SystemExit("No model entries found to print.")

    # Build table rows.
    header = ["Dataset window"] + [_display_model(m) for m in model_order]
    rows: List[List[str]] = []
    for group in wanted_groups:
        row = [group]
        for model in model_order:
            bucket = buckets.get((group, model))
            if not bucket:
                row.append("-")
            else:
                row.append(f"{bucket['accuracy']:.3f}")
        rows.append(row)

    # Determine column widths.
    col_widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]

    def _fmt_row(row: List[str]) -> str:
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))

    separator = "-+-".join("-" * w for w in col_widths)
    print(_fmt_row(header))
    print(separator)
    for row in rows:
        print(_fmt_row(row))


if __name__ == "__main__":
    main()
