"""
Semantic class constraint filtering for lexicalisation swaps.

Provides functions to filter candidate lemma pools so that replacements
share the same semantic class (WordNet hypernym or adjective supersense, and
optionally VerbNet class) as the original word being swapped.

Supports three modes (selected via the --semantic-match CLI flag):

  off    – no filtering (existing behaviour, default)
  soft   – apply filter; fall back to depth+1 or unconstrained if pool empties
  strict – apply filter; signal failure if no constrained candidates found
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, FrozenSet, Optional, Tuple

import wn

# ---------------------------------------------------------------------------
# Optional VerbNet integration via NLTK
# ---------------------------------------------------------------------------
try:
    from nltk.corpus import verbnet as _vn  # type: ignore
    # Probe to confirm data is downloaded; raises LookupError if missing.
    _vn.classids("run")
    _VERBNET_AVAILABLE = True
except Exception:
    _VERBNET_AVAILABLE = False

try:
    from nltk.corpus import wordnet as _nltk_wn  # type: ignore
    _nltk_wn.synsets("run")
    _NLTK_WN_AVAILABLE = True
except Exception:
    _NLTK_WN_AVAILABLE = False

_WN_LEXICON = "oewn:2021"

# ---------------------------------------------------------------------------
# Regime labels used in swap metadata and statistics
# ---------------------------------------------------------------------------
REGIME_OFF = "off"
REGIME_SATISFIED = "satisfied"
REGIME_FALLBACK_DEPTH = "fallback_depth"
REGIME_FALLBACK_UNCONSTRAINED = "fallback_unconstrained"
REGIME_DROPPED = "dropped"
# Anti-collision exhausted the semantic pool; widened to the full rare pool.
REGIME_ANTICOLLISION_FALLBACK = "anticollision_fallback"
# Tertiary fallback: NLTK WordNet lexname (broad semantic category) match.
REGIME_FALLBACK_LEXNAME = "fallback_lexname"

# Ordered for reporting
_REGIME_ORDER = (
    REGIME_SATISFIED,
    REGIME_FALLBACK_DEPTH,
    REGIME_FALLBACK_LEXNAME,
    REGIME_FALLBACK_UNCONSTRAINED,
    REGIME_ANTICOLLISION_FALLBACK,
    REGIME_DROPPED,
    REGIME_OFF,
)

# ---------------------------------------------------------------------------
# WordNet / VerbNet lookup helpers (all LRU-cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8192)
def noun_hypernym_ids(
    lemma: str, depth: int, lexicon: str = _WN_LEXICON
) -> FrozenSet[str]:
    """Return frozenset of noun hypernym synset IDs up to *depth* levels above."""
    try:
        syns = wn.synsets(lemma, pos="n", lexicon=lexicon) or ()
    except Exception:
        return frozenset()
    result: set = set()
    for syn in syns:
        current = [syn]
        for _ in range(depth):
            next_level = []
            for s in current:
                try:
                    hypers = s.hypernyms() or []
                except Exception:
                    hypers = []
                for h in hypers:
                    result.add(h.id)
                    next_level.append(h)
            current = next_level
    return frozenset(result)


@lru_cache(maxsize=8192)
def adj_attribute_ids(lemma: str, lexicon: str = _WN_LEXICON) -> FrozenSet[str]:
    """Return frozenset of noun synset IDs linked via the 'attribute' relation.

    In OEWN, head adjectives (pos='a') carry an ``attribute`` pointer to the
    noun concept they modify (e.g. both *large* and *small* point to the noun
    synset for *size*).  Satellite adjectives (pos='s') don't carry this
    pointer directly — they link back to their head via ``similar``.  This
    function resolves both cases so that satellite adjectives like *tiny* or
    *enormous* are treated the same as their head adjectives.
    """
    result: set = set()
    # Collect synsets for both head (pos='a') and satellite (pos='s') forms.
    synsets_to_check: list = []
    for pos in ("a", "s"):
        try:
            synsets_to_check.extend(wn.synsets(lemma, pos=pos, lexicon=lexicon) or ())
        except Exception:
            pass
    for syn in synsets_to_check:
        try:
            attrs = syn.get_related("attribute")
        except Exception:
            attrs = []
        for r in attrs:
            result.add(r.id)
        if not attrs:
            # Satellite: resolve to head adjective(s) via similar, then get attribute
            try:
                heads = syn.get_related("similar")
            except Exception:
                heads = []
            for head in heads:
                try:
                    for r in head.get_related("attribute"):
                        result.add(r.id)
                except Exception:
                    pass
    return frozenset(result)


@lru_cache(maxsize=8192)
def adj_also_ids(lemma: str, lexicon: str = _WN_LEXICON) -> FrozenSet[str]:
    """Return frozenset of adjective synset IDs reachable via the 'also' relation.

    The ``also`` relation groups near-synonym adjective clusters (e.g.
    *happy* → *cheerful*, *content*, *pleased*).  Two adjectives sharing an
    ``also``-cluster ID are semantically related, though less tightly than
    sharing the same attribute dimension.  Includes both pos='a' and pos='s'
    synsets so satellite adjectives are handled correctly.
    """
    result: set = set()
    for pos in ("a", "s"):
        try:
            syns = wn.synsets(lemma, pos=pos, lexicon=lexicon) or ()
        except Exception:
            syns = ()
        for syn in syns:
            result.add(syn.id)
            try:
                for r in syn.get_related("also"):
                    result.add(r.id)
            except Exception:
                pass
    return frozenset(result)


@lru_cache(maxsize=8192)
def verb_hypernym_ids(
    lemma: str, depth: int, lexicon: str = _WN_LEXICON
) -> FrozenSet[str]:
    """Return frozenset of verb hypernym synset IDs up to *depth* levels above."""
    try:
        syns = wn.synsets(lemma, pos="v", lexicon=lexicon) or ()
    except Exception:
        return frozenset()
    result: set = set()
    for syn in syns:
        current = [syn]
        for _ in range(depth):
            next_level = []
            for s in current:
                try:
                    hypers = s.hypernyms() or []
                except Exception:
                    hypers = []
                for h in hypers:
                    result.add(h.id)
                    next_level.append(h)
            current = next_level
    return frozenset(result)


@lru_cache(maxsize=8192)
def noun_lexnames(lemma: str) -> FrozenSet[str]:
    """Return frozenset of NLTK WordNet lexicographer file names for a noun lemma.

    Lexnames are coarse semantic categories (e.g. ``noun.animal``, ``noun.artifact``,
    ``noun.person``) used as a broad tertiary fallback when OEWN hypernym matching
    fails.  Returns an empty frozenset if NLTK WordNet is unavailable or the
    lemma has no coverage.
    """
    if not _NLTK_WN_AVAILABLE:
        return frozenset()
    try:
        syns = _nltk_wn.synsets(lemma, pos="n") or ()
        return frozenset(s.lexname() for s in syns if s.lexname())
    except Exception:
        return frozenset()


@lru_cache(maxsize=8192)
def verb_lexnames(lemma: str) -> FrozenSet[str]:
    """Return frozenset of NLTK WordNet lexicographer file names for a verb lemma."""
    if not _NLTK_WN_AVAILABLE:
        return frozenset()
    try:
        syns = _nltk_wn.synsets(lemma, pos="v") or ()
        return frozenset(s.lexname() for s in syns if s.lexname())
    except Exception:
        return frozenset()


@lru_cache(maxsize=8192)
def verbnet_classes_for_lemma(lemma: str) -> FrozenSet[str]:
    """Return frozenset of VerbNet class IDs (+ immediate parent) for a verb lemma."""
    if not _VERBNET_AVAILABLE:
        return frozenset()
    try:
        classes = _vn.classids(lemma) or []
        result: set = set()
        for cls in classes:
            result.add(cls)
            # Add parent class: "put-9.1-1" -> "put-9.1", "put-9.1" -> "put-9"
            parts = cls.rsplit("-", 1)
            if len(parts) == 2 and parts[1].replace(".", "").isdigit():
                result.add(parts[0])
        return frozenset(result)
    except Exception:
        return frozenset()


# ---------------------------------------------------------------------------
# Pool filter functions
# ---------------------------------------------------------------------------


def apply_noun_semantic_filter(
    pool: Tuple[str, ...],
    source_lemma: str,
    depth: int,
    lexicon: str = _WN_LEXICON,
) -> Tuple[Tuple[str, ...], str]:
    """
    Filter *pool* keeping only lemmas sharing a hypernym with *source_lemma*.

    Falls back to depth+1 if depth-N yields nothing, then to the unfiltered
    pool (REGIME_FALLBACK_UNCONSTRAINED).

    Returns ``(filtered_pool, regime)``.  Strict-mode dropping is the
    caller's responsibility.
    """
    if not source_lemma or not pool:
        return pool, REGIME_FALLBACK_UNCONSTRAINED

    src_ids = noun_hypernym_ids(source_lemma, depth, lexicon)
    if src_ids:
        filtered = tuple(
            w for w in pool
            if bool(src_ids & noun_hypernym_ids(w, depth, lexicon))
        )
        if filtered:
            return filtered, REGIME_SATISFIED

        # Try depth+1
        src_ids_plus = noun_hypernym_ids(source_lemma, depth + 1, lexicon)
        if src_ids_plus:
            filtered_plus = tuple(
                w for w in pool
                if bool(src_ids_plus & noun_hypernym_ids(w, depth + 1, lexicon))
            )
            if filtered_plus:
                return filtered_plus, REGIME_FALLBACK_DEPTH

    # Tertiary fallback: NLTK WordNet lexname (broad semantic category).
    # Runs whether or not OEWN had synsets for the source lemma.
    src_lexnames = noun_lexnames(source_lemma)
    if src_lexnames:
        filtered_lex = tuple(w for w in pool if bool(src_lexnames & noun_lexnames(w)))
        if filtered_lex:
            return filtered_lex, REGIME_FALLBACK_LEXNAME

    return pool, REGIME_FALLBACK_UNCONSTRAINED


def apply_adj_semantic_filter(
    pool: Tuple[str, ...],
    source_lemma: str,
    depth: int,
    lexicon: str = _WN_LEXICON,
) -> Tuple[Tuple[str, ...], str]:
    """
    Filter *pool* to adjectives in the same semantic dimension as *source_lemma*.

    Strategy (OEWN-specific):
    1. **Attribute match** (primary) — keep candidates sharing at least one
       ``attribute`` noun synset (e.g. both *large* and *small* map to *size*).
       Returns REGIME_SATISFIED.
    2. **Also-cluster match** (secondary) — keep candidates sharing an ``also``
       cluster synset (near-synonyms).  Returns REGIME_FALLBACK_DEPTH.
    3. **Unconstrained** fallback — return the original pool unchanged.
       Returns REGIME_FALLBACK_UNCONSTRAINED.

    The *depth* parameter is accepted for API uniformity but is not used,
    since OEWN adjectives do not have a meaningful hypernym hierarchy.
    """
    if not source_lemma or not pool:
        return pool, REGIME_FALLBACK_UNCONSTRAINED

    # Primary: attribute dimension match
    src_attr = adj_attribute_ids(source_lemma, lexicon)
    if src_attr:
        filtered = tuple(
            w for w in pool if bool(src_attr & adj_attribute_ids(w, lexicon))
        )
        if filtered:
            return filtered, REGIME_SATISFIED

    # Secondary: also-cluster (near-synonym) match
    src_also = adj_also_ids(source_lemma, lexicon)
    if src_also:
        filtered = tuple(
            w for w in pool if bool(src_also & adj_also_ids(w, lexicon))
        )
        if filtered:
            return filtered, REGIME_FALLBACK_DEPTH

    return pool, REGIME_FALLBACK_UNCONSTRAINED


def apply_verb_semantic_filter(
    lemma_pool: Tuple[str, ...],
    source_lemma: str,
    depth: int,
    lexicon: str = _WN_LEXICON,
) -> Tuple[Tuple[str, ...], str]:
    """
    Filter *lemma_pool* by VerbNet class (if NLTK VerbNet is available) or
    OEWN verb hypernyms at depth N / N+1.
    """
    if not source_lemma or not lemma_pool:
        return lemma_pool, REGIME_FALLBACK_UNCONSTRAINED

    # Try VerbNet first
    if _VERBNET_AVAILABLE:
        src_vn = verbnet_classes_for_lemma(source_lemma)
        if src_vn:
            filtered = tuple(
                w for w in lemma_pool
                if bool(src_vn & verbnet_classes_for_lemma(w))
            )
            if filtered:
                return filtered, REGIME_SATISFIED

    # Fall back to OEWN verb hypernyms at depth N
    src_hyp = verb_hypernym_ids(source_lemma, depth, lexicon)
    if src_hyp:
        filtered = tuple(
            w for w in lemma_pool
            if bool(src_hyp & verb_hypernym_ids(w, depth, lexicon))
        )
        if filtered:
            regime = REGIME_SATISFIED if not _VERBNET_AVAILABLE else REGIME_FALLBACK_DEPTH
            return filtered, regime

    # Depth N+1
    src_hyp_plus = verb_hypernym_ids(source_lemma, depth + 1, lexicon)
    if src_hyp_plus:
        filtered_plus = tuple(
            w for w in lemma_pool
            if bool(src_hyp_plus & verb_hypernym_ids(w, depth + 1, lexicon))
        )
        if filtered_plus:
            return filtered_plus, REGIME_FALLBACK_DEPTH

    # Tertiary fallback: NLTK WordNet lexname (broad semantic category).
    src_lexnames = verb_lexnames(source_lemma)
    if src_lexnames:
        filtered_lex = tuple(w for w in lemma_pool if bool(src_lexnames & verb_lexnames(w)))
        if filtered_lex:
            return filtered_lex, REGIME_FALLBACK_LEXNAME

    return lemma_pool, REGIME_FALLBACK_UNCONSTRAINED


# ---------------------------------------------------------------------------
# Statistics accumulator
# ---------------------------------------------------------------------------


@dataclass
class SemanticStats:
    """Accumulate per-POS semantic constraint statistics across a run."""

    # regime -> count
    noun: Dict[str, int] = field(default_factory=Counter)
    adjective: Dict[str, int] = field(default_factory=Counter)
    verb: Dict[str, int] = field(default_factory=Counter)

    # regime -> {lemma -> count}  (for top-N diversity reporting)
    _noun_by_regime: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter), repr=False
    )
    _adj_by_regime: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter), repr=False
    )
    _verb_by_regime: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter), repr=False
    )

    # Strict-mode item drops (no swap record generated, so tracked separately)
    noun_dropped: int = 0
    adj_dropped: int = 0
    verb_dropped: int = 0

    def record(
        self,
        pos: str,
        regime: str,
        new_lemma: Optional[str] = None,
    ) -> None:
        """Record one successful swap."""
        ctr_map = {
            "noun": self.noun,
            "adjective": self.adjective,
            "verb": self.verb,
        }
        by_regime_map = {
            "noun": self._noun_by_regime,
            "adjective": self._adj_by_regime,
            "verb": self._verb_by_regime,
        }
        ctr = ctr_map.get(pos)
        if ctr is not None:
            ctr[regime] = ctr.get(regime, 0) + 1
        if new_lemma:
            by_reg = by_regime_map.get(pos)
            if by_reg is not None:
                by_reg[regime][new_lemma] += 1

    def record_drop(self, pos: str) -> None:
        """Record one item drop due to strict semantic constraint failure."""
        if pos == "noun":
            self.noun_dropped += 1
        elif pos in ("adjective", "adj"):
            self.adj_dropped += 1
        elif pos == "verb":
            self.verb_dropped += 1

    def summary(self, top_n: int = 20) -> str:
        lines = ["", "=== Semantic Constraint Statistics ==="]
        pos_rows = [
            ("NOUN", self.noun, self._noun_by_regime, self.noun_dropped),
            ("ADJECTIVE", self.adjective, self._adj_by_regime, self.adj_dropped),
            ("VERB", self.verb, self._verb_by_regime, self.verb_dropped),
        ]
        for pos_name, ctr, by_regime, dropped in pos_rows:
            total_swaps = sum(ctr.values())
            total = total_swaps + dropped
            if not total:
                continue
            lines.append(f"\n[{pos_name}]  total attempts: {total}"
                         f"  (swapped: {total_swaps}, dropped: {dropped})")
            for regime in _REGIME_ORDER:
                n = ctr.get(regime, 0)
                if regime == REGIME_DROPPED:
                    n = dropped
                if not n:
                    continue
                pct = 100.0 * n / total
                lines.append(f"  {regime:32s}: {n:6d}  ({pct:.1f}%)")
            if by_regime:
                lines.append(f"  Top-{top_n} replacement lemmas per regime:")
                for regime in _REGIME_ORDER:
                    lemma_ctr = by_regime.get(regime)
                    if not lemma_ctr:
                        continue
                    regime_total = sum(lemma_ctr.values())
                    lines.append(f"    [{regime}]")
                    for lemma, cnt in lemma_ctr.most_common(top_n):
                        share = 100.0 * cnt / regime_total
                        lines.append(f"      {lemma:28s} {cnt:5d}  ({share:.1f}%)")
        lines.append("")
        return "\n".join(lines)
