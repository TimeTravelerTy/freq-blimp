import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_TS_RE = re.compile(r"\d{8}-\d{6}")
_ZIPF_RE = re.compile(r"zipf(?P<low>\d+_\d+)-(?P<high>\d+_\d+)")

_WINDOW_TO_GROUP = {
    "4.0-5.2": "head",
    "3.6-5.0": "head",
    "2.2-3.0": "tail",
    "1.2-2.0": "xtail",
}
_GROUP_ORDER = ["head", "tail", "xtail"]


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
    return model, data, variant, ts


def _group_label(data_slug: Optional[str], stem: str) -> Optional[str]:
    text = data_slug or stem
    m = _ZIPF_RE.search(text)
    if not m:
        return None
    low = m.group("low").replace("_", ".")
    high = m.group("high").replace("_", ".")
    window = f"{low}-{high}"
    return _WINDOW_TO_GROUP.get(window)


def _infer_model_and_group(obj: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    scores_path = Path(obj["scores_path"]) if obj.get("scores_path") else obj["_path"]
    model, data, parsed_variant, ts = _parse_scores_name(scores_path)
    variant = str(obj.get("variant") or parsed_variant or "").lower()
    if variant == "original":
        group = "original"
    else:
        group = _group_label(data, scores_path.stem)
    return model, group, ts


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


def _rows_by_phenomenon(obj: dict) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for row in obj.get("by_phenomenon", []) or []:
        key = row.get("key")
        if key is None:
            continue
        acc = row.get("accuracy")
        total = row.get("total")
        if not isinstance(acc, (int, float)) or not isinstance(total, int):
            continue
        out[str(key)] = {
            "accuracy": acc,
            "total": total,
            "correct": row.get("correct"),
        }
    return out


def _display_model(model: str) -> str:
    return model.replace("_", ".")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Print per-phenomenon accuracy tables across head/tail/xtail windows."
    )
    ap.add_argument("--runs-dir", default="results/blimp_accuracy_runs")
    ap.add_argument("--pattern", default="*.json")
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional ordered list of model slugs to include.",
    )
    ap.add_argument(
        "--variant",
        default="rare",
        help="Variant to include (rare/original/auto/both/any). Default: rare.",
    )
    ap.add_argument(
        "--include-original",
        action="store_true",
        help="Include an original column alongside head/tail/xtail when available.",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Optional limit on number of phenomena to print (by total count).",
    )
    ap.add_argument(
        "--min-total",
        type=int,
        default=None,
        help="Optional minimum total count to include a phenomenon.",
    )
    ap.add_argument(
        "--show-total",
        action="store_true",
        help="Include total count per cell as acc (n).",
    )
    args = ap.parse_args()

    runs = _load_runs(Path(args.runs_dir), args.pattern)
    if not runs:
        raise SystemExit(f"No runs found in {args.runs_dir} matching {args.pattern}")

    wanted_variant = args.variant.lower() if args.variant else None

    best_runs: Dict[Tuple[str, str], dict] = {}
    models_found = set()
    for obj in runs:
        model, group, ts = _infer_model_and_group(obj)
        if not model:
            continue
        variant = str(obj.get("variant") or "").lower()
        if group == "original":
            if not args.include_original:
                if wanted_variant and wanted_variant != "any" and variant != wanted_variant:
                    continue
            else:
                if wanted_variant and wanted_variant not in {"any", "original"}:
                    if variant != "original":
                        continue
        else:
            if wanted_variant and wanted_variant not in {"any", "rare"}:
                continue
            if group is None or group not in _GROUP_ORDER:
                continue
        models_found.add(model)
        key = (model, group)
        prev = best_runs.get(key)
        if prev is None or (ts and (prev.get("_ts") or "") < ts):
            obj["_ts"] = ts
            best_runs[key] = obj

    model_order = args.models if args.models else sorted(models_found)
    if not model_order:
        raise SystemExit("No model entries found to print.")

    for model in model_order:
        rows_by_group: Dict[str, Dict[str, dict]] = {}
        totals_by_phen = {}
        for group in _GROUP_ORDER:
            obj = best_runs.get((model, group))
            if not obj:
                continue
            rows = _rows_by_phenomenon(obj)
            rows_by_group[group] = rows
            for phen, row in rows.items():
                totals_by_phen[phen] = totals_by_phen.get(phen, 0) + int(
                    row.get("total") or 0
                )
        if args.include_original:
            obj = best_runs.get((model, "original"))
            if obj:
                rows = _rows_by_phenomenon(obj)
                rows_by_group["original"] = rows
                for phen, row in rows.items():
                    totals_by_phen[phen] = totals_by_phen.get(phen, 0) + int(
                        row.get("total") or 0
                    )

        if not rows_by_group:
            continue

        phenomena = sorted(
            totals_by_phen.keys(),
            key=lambda k: (-totals_by_phen.get(k, 0), k),
        )
        if args.min_total is not None:
            phenomena = [p for p in phenomena if totals_by_phen.get(p, 0) >= args.min_total]
        if args.top_n is not None:
            phenomena = phenomena[: args.top_n]

        columns = list(_GROUP_ORDER)
        if args.include_original:
            columns = ["original"] + columns
        header = ["Phenomenon"] + [g for g in columns]
        table: List[List[str]] = []
        for phen in phenomena:
            row = [phen]
            for group in columns:
                cell = rows_by_group.get(group, {}).get(phen)
                if not cell:
                    row.append("-")
                else:
                    if args.show_total:
                        row.append(f"{cell['accuracy']:.3f} ({cell['total']})")
                    else:
                        row.append(f"{cell['accuracy']:.3f}")
            table.append(row)

        if not table:
            continue

        print(f"\nModel: {_display_model(model)}")
        col_widths = [max(len(row[i]) for row in [header] + table) for i in range(len(header))]

        def _fmt_row(row: List[str]) -> str:
            return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))

        separator = "-+-".join("-" * w for w in col_widths)
        print(_fmt_row(header))
        print(separator)
        for row in table:
            print(_fmt_row(row))


if __name__ == "__main__":
    main()
