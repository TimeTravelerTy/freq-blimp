"""
Augment an existing verb inventory with phrasal-verb (particle) frames from VerbNet.

This keeps lemmas tokenized (e.g., stores lemma "wake" with frame "intr_particle"
and particle "up") so the swapper can inflect the verb and optionally replace the
particle token in the sentence.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.verb_inventory import build_inventory_from_verbnet


def _norm(text: str) -> str:
    return (text or "").strip().lower()


def _load_inventory(path: Path):
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _save_inventory(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Augment verb inventory with VerbNet particle frames.")
    ap.add_argument("--inventory", default="data/processed/verb_inventory_pruned_particles.json")
    ap.add_argument("--out", default="data/processed/verb_inventory_pruned_particles.json")
    ap.add_argument(
        "--verbnet_dir",
        default=str(Path.home() / "nltk_data" / "corpora" / "verbnet3"),
        help="Path to VerbNet corpus root (e.g., ~/nltk_data/corpora/verbnet3).",
    )
    args = ap.parse_args()

    inv_path = Path(args.inventory)
    out_path = Path(args.out)
    verbnet_dir = Path(args.verbnet_dir)

    base = _load_inventory(inv_path)
    base_by_lemma = {_norm(item.get("lemma", "")): item for item in base if isinstance(item, dict)}

    vn = build_inventory_from_verbnet(verbnet_dir)
    added_frames = 0
    added_verbs = 0

    for entry in vn.entries:
        particle_frames = [
            frame
            for frame in entry.frames
            if frame.kind and frame.kind.endswith("_particle") and frame.particle
        ]
        if not particle_frames:
            continue
        lemma = _norm(entry.lemma)
        if not lemma:
            continue

        target = base_by_lemma.get(lemma)
        if target is None:
            target = {"lemma": lemma, "frames": []}
            base_by_lemma[lemma] = target
            added_verbs += 1

        existing = target.get("frames") or []
        if not isinstance(existing, list):
            existing = []

        seen = set()
        for fr in existing:
            if not isinstance(fr, dict):
                continue
            seen.add((_norm(fr.get("type", "")), _norm(fr.get("prep", "")), _norm(fr.get("particle", ""))))

        for fr in particle_frames:
            key = (_norm(fr.kind), _norm(fr.prep or ""), _norm(fr.particle or ""))
            if key in seen:
                continue
            seen.add(key)
            existing.append({"type": fr.kind, "particle": fr.particle})
            added_frames += 1

        target["frames"] = existing

    merged = list(base_by_lemma.values())
    merged.sort(key=lambda x: _norm(x.get("lemma", "")))
    _save_inventory(out_path, merged)
    print(f"Wrote {out_path} | added verbs: {added_verbs} | added particle frames: {added_frames}")


if __name__ == "__main__":
    main()
