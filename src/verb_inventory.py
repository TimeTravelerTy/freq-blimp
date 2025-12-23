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
    "ditrans_pp",
    # Phrasal-verb frames (particle is carried separately in VerbFrameSpec.particle).
    "intr_particle",
    "trans_particle",
    "ditrans_particle",
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


def _is_transitive_like(kind: Optional[str]) -> bool:
    if not kind:
        return False
    return kind.startswith("trans") or kind.startswith("ditrans")


def _entry_has_transitive_like(entry: "VerbEntry") -> bool:
    """
    Return True if this lemma should be treated as having transitive-like uses.

    In practice, ``ditrans_pp`` frames extracted from lexicons/corpora are noisier
    than core ``trans``/``ditrans`` frames, especially for verbs that are
    otherwise strongly intransitive. To avoid misclassifying such verbs as
    ambitransitive (e.g., ``fall``), treat ``ditrans_pp`` as evidence of
    transitivity only when the lemma is not otherwise intransitive.
    """
    has_core = False
    has_ditrans_pp = False
    for frame in entry.frames:
        if frame.kind in {"trans", "ditrans"}:
            has_core = True
        elif frame.kind == "ditrans_pp":
            has_ditrans_pp = True
        elif frame.kind and frame.kind.startswith("trans"):
            has_core = True
    if has_core:
        return True
    if has_ditrans_pp and not _entry_has_intr(entry):
        return True
    return False


def _entry_has_intr(entry: "VerbEntry") -> bool:
    for frame in entry.frames:
        if frame.kind and frame.kind.startswith("intr"):
            return True
    return False


def _entry_has_intr_core(entry: "VerbEntry") -> bool:
    """
    Return True if the entry has a core intransitive frame.

    We treat ``intr_pp`` as its own family because those frames can be noisy for
    otherwise transitive lemmas. Some callers only need to distinguish whether a
    lemma is explicitly ``intr`` vs. has a direct-object frame.
    """
    for frame in entry.frames:
        if frame.kind == "intr":
            return True
    return False


def _entry_has_trans(entry: "VerbEntry") -> bool:
    """
    Return True if the entry has a core transitive frame.

    Note: we intentionally treat ``ditrans*`` separately; for some uses we only
    want to avoid lemmas that are explicitly both intransitive and transitive.
    """
    for frame in entry.frames:
        if frame.kind == "trans" or (frame.kind and frame.kind.startswith("trans")):
            return True
    return False


@lru_cache(maxsize=8192)
def _wn_pos_counts_cached(lemma: str, lexicon: str) -> Tuple[int, int, int, int]:
    """
    Return (verb_synsets, noun_synsets, adj_synsets, adv_synsets) for the lemma.
    """
    norm = (lemma or "").strip().lower()
    if not norm:
        return (0, 0, 0)
    try:
        v = len(wn.synsets(norm, pos="v", lexicon=lexicon))
    except Exception:
        v = 0
    try:
        n = len(wn.synsets(norm, pos="n", lexicon=lexicon))
    except Exception:
        n = 0
    try:
        a = len(wn.synsets(norm, pos="a", lexicon=lexicon))
    except Exception:
        a = 0
    try:
        r = len(wn.synsets(norm, pos="r", lexicon=lexicon))
    except Exception:
        r = 0
    return (v, n, a, r)


_PARTICLE_KEYWORDS = {
    "about",
    "across",
    "ahead",
    "along",
    "around",
    "aside",
    "away",
    "back",
    "by",
    "down",
    "in",
    "off",
    "on",
    "out",
    "over",
    "round",
    "through",
    "together",
    "under",
    "up",
}


def _verb_share_wordnet(lemma: str, lexicon: str) -> float:
    v, n, a, r = _wn_pos_counts_cached(lemma, lexicon)
    total = v + n + a + r
    if total <= 0:
        return 1.0
    return v / total


def wordnet_pos_counts(lemma: str, *, lexicon: str = "oewn:2021") -> Tuple[int, int, int, int]:
    """
    Return (v, n, a, r) synset counts for the lemma in the given WordNet lexicon.
    """
    return _wn_pos_counts_cached((lemma or "").strip().lower(), lexicon)


@dataclass(frozen=True)
class VerbFrameSpec:
    """
    Normalized representation of the structures a verb lemma can license.

    kind:
        Frame identifier, e.g., ``intr`` or ``ditrans_pp``.
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
        lemma_index: Dict[str, VerbEntry] = {}
        for entry in self._entries:
            for frame in entry.frames:
                frame_index.setdefault(frame.kind, []).append((entry, frame))
                lookup_index[(entry.lemma.lower(), frame.kind)] = (entry, frame)
            lemma_index[entry.lemma.lower()] = entry
        self._frame_index = frame_index
        self._lookup_index = lookup_index
        self._lemma_index = lemma_index

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
        restrict_transitivity: Optional[str] = None,
    ) -> Optional[Sequence[Tuple[VerbEntry, VerbFrameSpec]]]:
        desired_prep_lower = desired_prep.lower() if desired_prep else None
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
                if (
                    (frame.kind and frame.kind.endswith("_particle") and frame.particle and frame.particle.lower() == particle_lower)
                    or (not (frame.kind and frame.kind.endswith("_particle")) and (frame.particle is None or frame.particle.lower() == particle_lower))
                )
            ]
            if not with_particle:
                return None
            filtered = with_particle

        # Transitivity restrictions are used by argument-structure subtasks to preserve
        # violations like "intransitive verb + direct object". For these modes, we only
        # enforce mutual exclusivity between core transitive ("trans*") and core
        # intransitive ("intr") frames; other frame families (e.g., intr_pp, ditrans*)
        # are allowed.
        if restrict_transitivity in {"intr_only", "trans_only"}:
            next_filtered = []
            for entry, frame in filtered:
                no_trans = not _entry_has_trans(entry)
                no_intr = not _entry_has_intr_core(entry)
                if restrict_transitivity == "intr_only":
                    if no_trans:
                        next_filtered.append((entry, frame))
                elif restrict_transitivity == "trans_only":
                    if no_intr:
                        next_filtered.append((entry, frame))
            filtered = next_filtered
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
        restrict_transitivity: Optional[str] = None,
        prefer_verb_lemmas: bool = True,
        min_verb_share: float = 0.75,
        verbiness_lexicon: str = "oewn:2021",
    ) -> Optional[Tuple[VerbEntry, VerbFrameSpec]]:
        choices = self.choices_for_frame(kind)
        if not choices and kind and kind.endswith("_particle"):
            # If the inventory lacks explicit phrasal-verb frames, avoid guessing
            # compatibility: failing is safer than producing "escape up"-type swaps.
            return None
        if not choices:
            return None
        filtered = self._filter_choices(
            choices,
            desired_prep=desired_prep,
            desired_particle=desired_particle,
            restrict_transitivity=restrict_transitivity,
        )
        if filtered is None:
            return None
        if filtered:
            choices = filtered

        if prefer_verb_lemmas and choices:
            # Wordfreq Zipf scores are not POS-specific, so for avoiding
            # denominal/deadjectival-looking "verbs" we rely on WordNet POS
            # ambiguity. In practice, many problematic replacements are cases
            # where a lemma has a weak verb entry but is strongly nominal or
            # adverbial (e.g., "winter", "weekend", "further").
            #
            # Heuristic: prefer lemmas that (a) have a verb entry, (b) are not
            # primarily nominal/adverbial, and (c) have at least 2 verb synsets
            # when they are also attested as nouns (to reduce denominal one-off
            # verb entries).
            try:
                noun_ratio_thr = float(min_verb_share)
            except Exception:
                noun_ratio_thr = 0.7
            noun_ratio_thr = max(0.0, noun_ratio_thr)

            def _passes(entry: VerbEntry, *, noun_ratio_thr: float) -> bool:
                v, n, a, r = _wn_pos_counts_cached(entry.lemma, verbiness_lexicon)
                if v <= 0:
                    return False
                # Suppress denominal one-off verb entries when the lemma is also
                # a noun: require multiple verb synsets.
                if n > 0 and v < 3:
                    return False
                if v / (n + 1) < noun_ratio_thr:
                    return False
                if v / (r + 1) < 1.0:
                    return False
                if v / (a + 1) < 1.0:
                    return False
                return True

            filtered_verby = [(entry, frame) for entry, frame in choices if _passes(entry, noun_ratio_thr=noun_ratio_thr)]
            if not filtered_verby and noun_ratio_thr > 0.6:
                # Relax slightly when the caller requested a very strict ratio.
                filtered_verby = [(entry, frame) for entry, frame in choices if _passes(entry, noun_ratio_thr=0.6)]
            if filtered_verby:
                choices = filtered_verby
            else:
                # If the ratio threshold eliminates everything, still apply the
                # non-denominal/adverbial guards so we avoid obvious non-verbs.
                fallback = [(entry, frame) for entry, frame in choices if _passes(entry, noun_ratio_thr=0.0)]
                if fallback:
                    choices = fallback

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

    def entry_for_lemma(self, lemma: str) -> Optional[VerbEntry]:
        lemma_norm = (lemma or "").strip().lower()
        if not lemma_norm:
            return None
        return self._lemma_index.get(lemma_norm)

    def lemma_transitivity(self, lemma: str) -> Tuple[bool, bool]:
        entry = self.entry_for_lemma(lemma)
        if entry is None:
            return (False, False)
        return (_entry_has_trans(entry), _entry_has_intr_core(entry))

    def filter_by_zipf(
        self,
        zipf_thr: Optional[float],
        *,
        zipf_min: Optional[float] = None,
    ) -> "VerbInventory":
        if zipf_thr is None and zipf_min is None:
            return self
        filtered = []
        for entry in self._entries:
            if zipf_thr is not None and not is_rare_lemma(entry.lemma, zipf_thr):
                continue
            if zipf_min is not None and _zipf_freq_cached(entry.lemma) < zipf_min:
                continue
            filtered.append(entry)
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
    if raw_kind.endswith("_pp") and prep is None:
        prep = entry.get("preposition")
    particle = entry.get("particle")
    separable = bool(entry.get("separable"))
    return VerbFrameSpec(
        kind=raw_kind,
        prep=_safe_lower(prep),
        particle=_safe_lower(particle),
        separable=separable,
    )


def _filter_transitivity_with_wordnet(
    lemma: str,
    frames: List[VerbFrameSpec],
    lexicon: Optional[str],
) -> List[VerbFrameSpec]:
    """
    Use WordNet frames as a veto to drop VerbNet-only transitive/ditransitive frames.

    If WordNet lacks any direct-object frames for the lemma, we remove trans*/ditrans*
    VerbNet frames but keep intr/PP frames. When VerbNet would be emptied by this
    pruning, we fall back to the WordNet frames so the lemma remains available for
    intransitive swaps.
    """
    if not lexicon:
        return frames

    wn_frames = _lemma_frames(lemma, lexicon)
    if not wn_frames:
        return frames

    has_transitive = any(_is_transitive_like(spec.kind) for spec in wn_frames)
    if has_transitive:
        return frames

    filtered = [frame for frame in frames if not _is_transitive_like(frame.kind)]
    if filtered:
        return filtered

    # If VerbNet only contributed transitive frames, keep the WordNet view instead.
    return list(wn_frames)


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

    tokens = text.replace(",", " ").replace("/", " ").split()
    try:
        verb_idx = next(i for i, tok in enumerate(tokens) if "----" in tok)
    except StopIteration:
        return None

    # Count NP placeholders between the verb and the first prep keyword.
    obj_tokens = {"somebody", "someone", "something"}
    prep_pos = None
    for i, tok in enumerate(tokens[verb_idx + 1 :], start=verb_idx + 1):
        if tok in _PREP_KEYWORDS:
            prep_pos = i
            break

    span = tokens[verb_idx + 1 : prep_pos] if prep_pos is not None else tokens[verb_idx + 1 :]
    obj_nps = sum(1 for tok in span if tok in obj_tokens)

    base = "intr"
    if obj_nps >= 2:
        base = "ditrans"
    elif obj_nps == 1:
        base = "trans"

    entry: Dict[str, str] = {"type": base}
    if prep_pos is not None:
        prep = tokens[prep_pos]
        if prep in _PREP_KEYWORDS:
            entry["type"] = "intr_pp" if obj_nps == 0 else "ditrans_pp"
            entry["prep"] = prep
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
        # Treat 'to' PPs as intransitive+PP to avoid inflating ditrans_pp counts.
        if prep_val == "to":
            return VerbFrameSpec(kind="intr_pp", prep=prep_val, particle=None, separable=False)

        # Treat other PP frames as intransitive+PP unless we see clear evidence of
        # two objects; this trims noisy NP+PP markings in some VerbNet classes.
        kind = "ditrans_pp" if obj_nps >= 2 else "intr_pp"
        return VerbFrameSpec(kind=kind, prep=prep_val, particle=None, separable=False)

    kind = "intr"
    # Treat true double-NP only when no PP is present.
    if obj_nps >= 2:
        kind = "ditrans"
    elif obj_nps == 1:
        kind = "trans"

    return VerbFrameSpec(kind=kind, prep=None, particle=None, separable=False)


def _collect_verbnet_frames(
    verbnet_root: Union[str, Path],
    *,
    transitivity_lexicon: Optional[str] = None,
) -> Dict[str, List[VerbFrameSpec]]:
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
            if " " in norm or "-" in norm:
                continue

            lemma_key = norm
            member_particle = None
            if "_" in norm:
                # VerbNet encodes many phrasal verbs as <base>_<particle> (e.g., "wake_up").
                # Convert those to explicit *_particle frames attached to the base lemma.
                parts = [p for p in norm.split("_") if p]
                if len(parts) == 2 and all(p.isalpha() for p in parts):
                    base, particle = parts
                    if particle in _PARTICLE_KEYWORDS:
                        lemma_key = base
                        member_particle = particle
                if member_particle is None:
                    continue

            lst = lemma_frames.setdefault(lemma_key, [])
            if member_particle is None:
                lst.extend(specs)
            else:
                for spec in specs:
                    if spec.prep:
                        continue
                    lst.append(
                        VerbFrameSpec(
                            kind=f"{spec.kind}_particle",
                            prep=None,
                            particle=member_particle,
                            separable=False,
                        )
                    )

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

        if merged:
            has_trans_like = any(_is_transitive_like(f.kind) for f in merged)
            has_ditrans = any(f.kind == "ditrans" for f in merged)
            # Drop ditrans_pp 'to' when there is no bare double-object support.
            if not has_ditrans:
                merged = [
                    f
                    for f in merged
                    if not (
                        f.kind == "ditrans_pp"
                        and f.prep
                        and f.prep.lower() == "to"
                    )
                ]
            merged = _filter_transitivity_with_wordnet(
                lemma,
                merged,
                transitivity_lexicon,
            )
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
    *,
    transitivity_lexicon: Optional[str] = "oewn:2021",
) -> VerbInventory:
    """
    Build a verb inventory from VerbNet frames (preferred for structured subcat info).

    WordNet augmentation is disabled by default to avoid noisy PP guesses, but a
    lexicon can still be provided so WordNet can veto implausible transitive frames.
    """
    frames_by_lemma = _collect_verbnet_frames(
        verbnet_root,
        transitivity_lexicon=transitivity_lexicon,
    )
    entries: List[VerbEntry] = []
    for lemma, frames in frames_by_lemma.items():
        if not frames:
            continue
        # Skip multiword/underscore lemmas; VerbNet phrasal verbs are normalized earlier.
        if "_" in lemma or " " in lemma:
            continue
        entries.append(VerbEntry(lemma=lemma, frames=tuple(frames), metadata={"source": "verbnet"}))
    if not entries:
        raise RuntimeError(
            f"VerbNet inventory is empty from '{verbnet_root}'. Check that verbnet3 is installed and the path is correct."
        )
    return VerbInventory(entries)
