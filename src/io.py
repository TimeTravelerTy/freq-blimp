import json
import os
import pathlib
from typing import Dict, Optional, Sequence

from datasets import load_dataset

def load_blimp(config_name):
    # BLiMP requires a config like 'regular_plural_subject_verb_agreement_1'
    return load_dataset("nyu-mll/blimp", config_name)["train"]

def write_jsonl(path, records):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _default_hf_cache_root() -> pathlib.Path:
    env = os.environ.get("HF_DATASETS_CACHE")
    if env:
        return pathlib.Path(env)
    return pathlib.Path.home() / ".cache" / "huggingface" / "datasets"


def _normalize_blimp_field(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    val = str(value).strip()
    if not val:
        return None
    if val == "syntax_semantics":
        return "syntax/semantics"
    return val


def _field_from_arrow(path: pathlib.Path):
    import pyarrow.ipc as ipc

    with path.open("rb") as f:
        table = ipc.open_stream(f).read_all()
    if "field" not in table.schema.names:
        return None
    vals = {v for v in table.column("field").to_pylist() if v}
    if not vals:
        return None
    return _normalize_blimp_field(sorted(vals)[0])


def _load_static_field_map(configs: Optional[Sequence[str]] = None) -> Dict[str, str]:
    static_path = pathlib.Path("configs/blimp_field_map.json")
    if not static_path.is_file():
        return {}
    try:
        payload = json.loads(static_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    wanted = set(configs or ())
    out: Dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        if wanted and key not in wanted:
            continue
        norm = _normalize_blimp_field(value)
        if norm:
            out[key] = norm
    return out


def load_blimp_field_map(
    configs: Optional[Sequence[str]] = None,
    cache_root: Optional[pathlib.Path] = None,
) -> Dict[str, str]:
    """
    Best-effort lookup: map BLiMP config name -> field ("syntax", "semantics", etc.)
    from the local HF datasets cache.
    """
    wanted = set(configs or ())
    mapping: Dict[str, str] = _load_static_field_map(configs=configs)

    root = cache_root or _default_hf_cache_root()
    blimp_root = root / "nyu-mll___blimp"
    if not blimp_root.exists():
        return mapping

    for cfg_dir in sorted(p for p in blimp_root.iterdir() if p.is_dir()):
        cfg_name = cfg_dir.name
        if wanted and cfg_name not in wanted:
            continue
        arrow_paths = sorted(cfg_dir.glob("*/*/blimp-train.arrow"))
        if not arrow_paths:
            continue
        try:
            field = _field_from_arrow(arrow_paths[0])
        except Exception:
            continue
        if field:
            mapping[cfg_name] = field
    return mapping
