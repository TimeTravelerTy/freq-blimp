import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow.ipc as ipc


def _default_cache_root() -> Path:
    env = os.environ.get("HF_DATASETS_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "huggingface" / "datasets"


def _normalize_field(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    val = str(value).strip()
    if not val:
        return None
    if val == "syntax_semantics":
        return "syntax/semantics"
    return val


def _field_from_arrow(path: Path) -> Optional[str]:
    with path.open("rb") as f:
        table = ipc.open_stream(f).read_all()
    if "field" not in table.schema.names:
        return None
    vals = {v for v in table.column("field").to_pylist() if v}
    if not vals:
        return None
    if len(vals) > 1:
        print(f"[Warn] Multiple field values in {path}: {sorted(vals)}")
    return _normalize_field(sorted(vals)[0])


def _build_field_map(cache_root: Path) -> Dict[str, str]:
    blimp_root = cache_root / "nyu-mll___blimp"
    if not blimp_root.exists():
        raise SystemExit(f"No BLiMP cache found at {blimp_root}")
    mapping: Dict[str, str] = {}
    for cfg_dir in sorted(p for p in blimp_root.iterdir() if p.is_dir()):
        arrow_paths = list(cfg_dir.glob("*/*/blimp-train.arrow"))
        if not arrow_paths:
            continue
        field = _field_from_arrow(arrow_paths[0])
        if field:
            mapping[cfg_dir.name] = field
    if not mapping:
        raise SystemExit(f"No BLiMP field metadata found under {blimp_root}")
    return mapping


def _iter_paths(pattern: str) -> List[Path]:
    paths = sorted(Path(p) for p in Path().glob(pattern) if Path(p).is_file())
    if not paths:
        raise SystemExit(f"No files found for pattern: {pattern}")
    return paths


def _process_file(
    path: Path,
    field_map: Dict[str, str],
    overwrite: bool,
) -> Tuple[int, int, int, Path]:
    total = 0
    changed = 0
    missing = 0
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with path.open("r", encoding="utf-8") as r, tmp_path.open("w", encoding="utf-8") as w:
        for line in r:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            before = rec.get("field")
            field = before
            needs_lookup = overwrite or before is None
            if needs_lookup:
                key = rec.get("subtask") or rec.get("UID")
                if isinstance(key, str) and key in field_map:
                    field = field_map[key]
                else:
                    missing += 1
            field = _normalize_field(field)
            if field is not None:
                rec["field"] = field
            if rec.get("field") != before:
                changed += 1
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return total, changed, missing, tmp_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add BLiMP task field metadata (syntax/semantics/etc) to JSONL files."
    )
    ap.add_argument(
        "--pattern",
        default="results/blimp_pair_scores/*.jsonl",
        help="Glob pattern for JSONL files to update.",
    )
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="HF datasets cache root (defaults to HF_DATASETS_CACHE or ~/.cache/huggingface/datasets).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing field values when a mapping is found.",
    )
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Replace files in-place (writes via a temporary file then renames).",
    )
    ap.add_argument(
        "--backup-suffix",
        default=".bak",
        help="If --inplace, write a backup copy with this suffix (set to empty to disable).",
    )
    args = ap.parse_args()

    cache_root = Path(args.cache_dir) if args.cache_dir else _default_cache_root()
    field_map = _build_field_map(cache_root)
    paths = _iter_paths(args.pattern)
    for path in paths:
        total, changed, missing, tmp_path = _process_file(path, field_map, args.overwrite)
        if args.inplace:
            if args.backup_suffix:
                backup_path = path.with_suffix(path.suffix + args.backup_suffix)
                backup_path.write_bytes(path.read_bytes())
            tmp_path.replace(path)
            out_path = path
        else:
            out_path = path.with_name(path.stem + ".with_field.jsonl")
            tmp_path.replace(out_path)
        print(f"{path} -> {out_path} ({changed}/{total} updated, {missing} missing mappings)")


if __name__ == "__main__":
    main()
