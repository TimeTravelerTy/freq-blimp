import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import wn
from wordfreq import zipf_frequency

from .lemma_bank import sample_rare_verbs_from_oewn
from .rarity import is_rare_lemma
import xml.etree.ElementTree as ET


@lru_cache(maxsize=8192)
def _zipf_freq_cached(lemma: str) -> float:
    try:
        return zipf_frequency(lemma or "", "en")
    except Exception:
        return 0.0


_SUPPORTED_FRAME_TYPES = {
    "intr",
    "trans",
    "ditrans",
    "intr_pp",
    "trans_pp",
    "ditrans_pp",
    "intr_particle",
    "trans_particle",
}

_PREP_KEYWORDS = (
    "about",
    "across",
    "after",
    "against",
    "along",
    "around",
    "at",
    "before",
    "behind",
    "by",
    "down",
    "for",
    "from",
    "in",
    "into",
    "near",
    "of",
    "off",
    "on",
    "onto",
    "over",
    "through",
    "to",
    "toward",
    "under",
    "up",
    "with",
)


def _safe_lower(text: Optional[str]) -> Optional[str]:
    return text.lower() if isinstance(text, str) else None


@dataclass(frozen=True)
class VerbFrameSpec:
    """
    Normalized representation of the structures a verb lemma can license.

    kind:
        Frame identifier, e.g., ``intr`` or ``trans_particle``.
    prep:
        Optional preposition string for *_pp frames. When ``None`` the swapper
        keeps the original preposition text found in the sentence.
    particle:
        Optional particle string for *_particle frames. When ``None`` the
        existing particle is preserved.
    separable:
        Flag for particle verbs whose particles can move after a direct object.
        The current pipeline does not rearrange tokens, but the flag is kept so
        downstream heuristics can make alignment decisions.
    """

    kind: str
    prep: Optional[str] = None
    particle: Optional[str] = None
    separable: bool = False


@dataclass(frozen=True)
class VerbEntry:
    lemma: str
    frames: Tuple[VerbFrameSpec, ...]
    metadata: Dict[str, str]


class VerbInventory:
    """
    Helper around the serialized verb frame metadata.
    """

    def __init__(self, entries: Sequence[VerbEntry]):
        self._entries: Tuple[VerbEntry, ...] = tuple(entries)
        frame_index: Dict[str, List[Tuple[VerbEntry, VerbFrameSpec]]] = {}
        lookup_index: Dict[Tuple[str, str], Tuple[VerbEntry, VerbFrameSpec]] = {}
        for entry in self._entries:
            for frame in entry.frames:
                frame_index.setdefault(frame.kind, []).append((entry, frame))
                lookup_index[(entry.lemma.lower(), frame.kind)] = (entry, frame)
        self._frame_index = frame_index
        self._lookup_index = lookup_index

    def __len__(self) -> int:
        return len(self._entries)

    def is_empty(self) -> bool:
        return not self._entries

    @property
    def entries(self) -> Tuple[VerbEntry, ...]:
        return self._entries

    def choices_for_frame(self, kind: str) -> Sequence[Tuple[VerbEntry, VerbFrameSpec]]:
        return self._frame_index.get(kind, ())

    def _filter_choices(
        self,
        choices: Sequence[Tuple[VerbEntry, VerbFrameSpec]],
        *,
        desired_prep: Optional[str] = None,
        desired_particle: Optional[str] = None,
    ) -> Optional[Sequence[Tuple[VerbEntry, VerbFrameSpec]]]:
        filtered = [
            (entry, frame)
            for entry, frame in choices
            # Avoid multiword/underscore lemmas; current pipeline does not handle particles/spacing.
            if "_" not in entry.lemma and " " not in entry.lemma and "-" not in entry.lemma
        ]
        if desired_prep:
            prep_lower = desired_prep.lower()
            with_prep = [
                (entry, frame)
                for entry, frame in filtered
                if frame.prep is None or frame.prep.lower() == prep_lower
            ]
            if not with_prep:
                return None
            filtered = with_prep
        if desired_particle:
            particle_lower = desired_particle.lower()
            with_particle = [
                (entry, frame)
                for entry, frame in filtered
                if frame.particle is None or frame.particle.lower() == particle_lower
            ]
            if not with_particle:
                return None
            filtered = with_particle
        return filtered

    def sample(
        self,
        kind: str,
        rng,
        *,
        desired_prep: Optional[str] = None,
        desired_particle: Optional[str] = None,
        zipf_weighted: bool = False,
        zipf_temp: float = 1.0,
    ) -> Optional[Tuple[VerbEntry, VerbFrameSpec]]:
        choices = self.choices_for_frame(kind)
        if not choices:
            return None
        filtered = self._filter_choices(
            choices,
            desired_prep=desired_prep,
            desired_particle=desired_particle,
        )
        if filtered is None:
            return None
        if filtered:
            choices = filtered
        if not zipf_weighted or len(choices) == 1:
            return rng.choice(choices)
        weights = []
        for entry, _frame in choices:
            freq = 10 ** _zipf_freq_cached(entry.lemma)
            adj = freq ** (1.0 / zipf_temp) if zipf_temp and zipf_temp > 0 else freq
            weights.append(adj if adj > 0 else 0.001)
        return rng.choices(choices, weights=weights, k=1)[0]

    def lookup(self, lemma: str, kind: str) -> Optional[Tuple[VerbEntry, VerbFrameSpec]]:
        lemma_norm = (lemma or "").strip().lower()
        if not lemma_norm:
            return None
        return self._lookup_index.get((lemma_norm, kind))

    def filter_by_zipf(self, zipf_thr: Optional[float]) -> "VerbInventory":
        if zipf_thr is None:
            return self
        filtered = [
            entry
            for entry in self._entries
            if is_rare_lemma(entry.lemma, zipf_thr)
        ]
        return VerbInventory(filtered)

    def restrict_to(self, lemmas: Iterable[str]) -> "VerbInventory":
        lemma_set = {lemma.strip().lower() for lemma in lemmas if lemma}
        if not lemma_set:
            return VerbInventory(tuple())
        filtered = [entry for entry in self._entries if entry.lemma.lower() in lemma_set]
        return VerbInventory(filtered)


def _parse_frame(entry: Dict) -> Optional[VerbFrameSpec]:
    raw_kind = entry.get("type")
    if raw_kind not in _SUPPORTED_FRAME_TYPES:
        return None
    prep = entry.get("prep")
    particle = entry.get("particle")
    separable = bool(entry.get("separable"))
    if raw_kind.endswith("_pp") and prep is None:
        prep = entry.get("preposition")
    if raw_kind.endswith("_particle") and particle is None:
        particle = entry.get("part")
    return VerbFrameSpec(kind=raw_kind, prep=prep, particle=particle, separable=separable)


def load_verb_inventory(path: Union[str, Path]) -> VerbInventory:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Verb inventory '{path}' not found.")
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    entries: List[VerbEntry] = []
    for item in data:
        lemma = item.get("lemma")
        frames_raw = item.get("frames") or ()
        frames = []
        for frame in frames_raw:
            parsed = _parse_frame(frame)
            if parsed:
                frames.append(parsed)
        if not lemma or not frames:
            continue
        metadata = {}
        if item.get("notes"):
            metadata["notes"] = str(item["notes"])
        entries.append(VerbEntry(lemma=lemma.strip().lower(), frames=tuple(frames), metadata=metadata))
    return VerbInventory(entries)


def write_verb_inventory(path: Union[str, Path], inventory: VerbInventory) -> None:
    """
    Serialize an inventory to JSON so future runs can reuse it.
    """
    data = []
    for entry in inventory.entries:
        frames = []
        for frame in entry.frames:
            frame_data = {"type": frame.kind}
            if frame.prep:
                frame_data["prep"] = frame.prep
            if frame.particle:
                frame_data["particle"] = frame.particle
            if frame.separable:
                frame_data["separable"] = True
            frames.append(frame_data)
        data.append({
            "lemma": entry.lemma,
            "frames": frames,
        })
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _frame_entry_from_text(frame: str) -> Optional[Dict]:
    """
    Map a WordNet frame template into one of our supported frame labels.
    """
    text = frame.strip().lower()
    if not text or "----" not in text:
        return None
    base = None
    if "somebody ----s somebody something" in text:
        base = "ditrans"
    elif "somebody ----s somebody" in text or "somebody ----s something" in text:
        base = "trans"
    elif "something ----s somebody" in text or "something ----s something" in text:
        base = "trans"
    elif "somebody ----s" in text or "something ----s" in text:
        base = "intr"
    if base is None:
        return None
    entry: Dict[str, str] = {"type": base}
    for prep in _PREP_KEYWORDS:
        needle = f" {prep} "
        if needle in text:
            entry["type"] = f"{base}_pp"
            entry["prep"] = prep
            break
    if "---- up" in text or "---- out" in text or "---- off" in text:
        # Heuristic: treat frames with explicit particles as particle verbs.
        for particle in ("up", "out", "off"):
            if f"---- {particle}" in text:
                entry["type"] = f"{base}_particle"
                entry["particle"] = particle
                break
    return entry


def _vn_syntax_to_frame(syntax_elem) -> Optional[VerbFrameSpec]:
    """
    Map a VerbNet <SYNTAX> sequence to a VerbFrameSpec.
    """
    if syntax_elem is None:
        return None
    seen_verb = False
    obj_nps = 0
    preps: List[str] = []
    for child in syntax_elem:
        tag = child.tag.upper()
        if tag == "VERB":
            seen_verb = True
            continue
        if tag == "NP":
            if seen_verb:
                obj_nps += 1
        elif tag == "PREP":
            if seen_verb:
                preps.append(child.attrib.get("value"))
        elif tag == "PP":
            if seen_verb:
                prep_child = child.find("PREP")
                if prep_child is not None:
                    preps.append(prep_child.attrib.get("value"))

    base = "intr"
    if obj_nps >= 1:
        base = "trans"
    if obj_nps >= 2:
        base = "ditrans"

    prep_val: Optional[str] = None
    for raw in preps:
        if not raw:
            continue
        # Values can be space or '|' separated alternatives; take the first non-empty token.
        tokens = [tok for tok in raw.replace("|", " ").split() if tok]
        if not tokens:
            continue
        cand = tokens[0].lower()
        if cand == "none":
            continue
        prep_val = cand
        break

    if prep_val:
        return VerbFrameSpec(kind=f"{base}_pp", prep=prep_val, particle=None, separable=False)
    return VerbFrameSpec(kind=base, prep=None, particle=None, separable=False)


def _collect_verbnet_frames(verbnet_root: Union[str, Path]) -> Dict[str, List[VerbFrameSpec]]:
    root_path = Path(verbnet_root)
    if not root_path.exists():
        raise RuntimeError(
            f"VerbNet path '{verbnet_root}' does not exist. Ensure verbnet3 is installed or pass --verbnet_dir."
        )
    lemma_frames: Dict[str, List[VerbFrameSpec]] = {}
    xml_files = sorted(root_path.glob("*.xml"))
    if not xml_files:
        raise RuntimeError(
            f"No VerbNet XML files found in '{verbnet_root}'. Expected files like absorb-39.8.xml."
        )
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
        except Exception:
            continue
        class_root = tree.getroot()
        members = class_root.findall(".//MEMBER")
        if not members:
            continue
        frames = class_root.findall(".//FRAME")
        specs: List[VerbFrameSpec] = []
        for frame in frames:
            syntax = frame.find("SYNTAX")
            spec = _vn_syntax_to_frame(syntax)
            if spec:
                specs.append(spec)
        if not specs:
            continue
        for member in members:
            name = member.attrib.get("name") or member.attrib.get("lemma")
            if not name:
                continue
            norm = name.strip().lower()
            if not norm:
                continue
            # Skip multiword / hyphenated lemmas; current pipeline doesn't handle them cleanly.
            if " " in norm or "_" in norm or "-" in norm:
                continue
            lst = lemma_frames.setdefault(norm, [])
            lst.extend(specs)

    deduped: Dict[str, List[VerbFrameSpec]] = {}
    for lemma, frames in lemma_frames.items():
        seen = set()
        merged: List[VerbFrameSpec] = []
        for frame in frames:
            key = (frame.kind, frame.prep, frame.particle)
            if key in seen:
                continue
            seen.add(key)
            merged.append(frame)
        deduped[lemma] = merged
    return deduped


@lru_cache(maxsize=8192)
def _lemma_frames(lemma: str, lexicon: str) -> Tuple[VerbFrameSpec, ...]:
    try:
        synsets = wn.synsets(lemma, pos="v", lexicon=lexicon)
    except Exception:
        return tuple()
    specs: List[VerbFrameSpec] = []
    seen = set()
    for syn in synsets:
        senses = ()
        try:
            senses = syn.senses()
        except Exception:
            senses = ()
        for sense in senses:
            getter = getattr(sense, "frames", None)
            if getter is None:
                continue
            try:
                items = getter()
            except TypeError:
                items = getter
            for frame in items or []:
                parsed_entry = _frame_entry_from_text(frame)
                if not parsed_entry:
                    continue
                spec = _parse_frame(parsed_entry)
                if not spec:
                    continue
                key = (spec.kind, spec.prep, spec.particle)
                if key in seen:
                    continue
                seen.add(key)
                specs.append(spec)
    return tuple(specs)


def build_inventory_from_lemmas(
    lemmas: Iterable[str],
    *,
    lexicon: str = "oewn:2021",
) -> VerbInventory:
    entries: List[VerbEntry] = []
    seen = set()
    for lemma in lemmas:
        if not lemma:
            continue
        norm = lemma.strip().lower()
        if not norm or norm in seen:
            continue
        frames = _lemma_frames(norm, lexicon)
        if not frames:
            continue
        entries.append(VerbEntry(lemma=norm, frames=frames, metadata={}))
        seen.add(norm)
    return VerbInventory(entries)


def build_inventory_from_oewn(
    zipf_max: float = 3.4,
    *,
    zipf_min: Optional[float] = None,
    min_length: int = 3,
    lexicon: str = "oewn:2021",
    limit: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 0,
) -> VerbInventory:
    lemmas = sample_rare_verbs_from_oewn(
        zipf_max=zipf_max,
        zipf_min=zipf_min,
        min_length=min_length,
        lexicon=lexicon,
        limit=limit,
        shuffle=shuffle,
        seed=seed,
    )
    return build_inventory_from_lemmas(lemmas, lexicon=lexicon)


def build_inventory_from_verbnet(
    verbnet_root: Union[str, Path],
) -> VerbInventory:
    """
    Build a verb inventory from VerbNet frames (preferred for structured subcat info).
    """
    frames_by_lemma = _collect_verbnet_frames(verbnet_root)
    entries: List[VerbEntry] = []
    for lemma, frames in frames_by_lemma.items():
        if not frames:
            continue
        # Skip multiword/underscore lemmas unless we can support particles (current parser does not).
        if "_" in lemma or " " in lemma:
            continue
        entries.append(VerbEntry(lemma=lemma, frames=tuple(frames), metadata={"source": "verbnet"}))
    if not entries:
        raise RuntimeError(
            f"VerbNet inventory is empty from '{verbnet_root}'. Check that verbnet3 is installed and the path is correct."
        )
    return VerbInventory(entries)
