import re
from functools import lru_cache
from typing import List, Optional, Sequence

import wn
from wordfreq import zipf_frequency


_DEFAULT_LEXICON = "oewn:2021"
_POS_ALIASES = {
    "n": "n",
    "noun": "n",
    "NOUN": "n",
    "v": "v",
    "verb": "v",
    "VERB": "v",
    "a": "a",
    "adj": "a",
    "adjective": "a",
    "ADJ": "a",
    "ADJECTIVE": "a",
    "r": "r",
    "adv": "r",
    "adverb": "r",
    "ADV": "r",
    "ADVERB": "r",
}

_WORD_RE = re.compile(r"^[a-z]+$")


class LemmaBankError(RuntimeError):
    """Raised when the lemma bank cannot be constructed."""


@lru_cache(maxsize=4)
def _load_lexicon(name: str) -> wn.Wordnet:
    try:
        return wn.Wordnet(name)
    except Exception as exc:  # pragma: no cover - defensive
        raise LemmaBankError(
            f"Failed to load OEWN lexicon '{name}'. Ensure the data is installed via `wn download {name}`."
        ) from exc


def _normalize_pos(pos: str) -> str:
    if pos not in _POS_ALIASES:
        raise ValueError(f"Unsupported POS '{pos}'. Expected one of: {sorted(set(_POS_ALIASES))}.")
    return _POS_ALIASES[pos]


def _extract_lemma(entry) -> Optional[str]:
    """
    Attempt to obtain a surface form from a wn.Word / lexical entry.
    Supports both the current `words()` API and the older
    `lexical_entries()` API.
    """
    for attr in ("lemma", "form", "word"):
        candidate = getattr(entry, attr, None)
        if callable(candidate):
            candidate = candidate()
        if candidate:
            return candidate
    return None


def _should_keep(lemma: str, min_len: int) -> bool:
    if len(lemma) < min_len:
        return False
    return bool(_WORD_RE.fullmatch(lemma))


def oewn_lemmas(
    pos: str = "n",
    *,
    lexicon: str = _DEFAULT_LEXICON,
) -> Sequence[str]:
    """
    Return a sorted sequence of OEWN lemmas for the given part-of-speech.
    Only simple ASCII tokens (a–z) are included.
    """
    pos_code = _normalize_pos(pos)
    lex = _load_lexicon(lexicon)

    if hasattr(lex, "words"):
        try:
            entries = lex.words(pos=pos_code)
        except TypeError:
            entries = lex.words()
    elif hasattr(lex, "lexical_entries"):
        entries = lex.lexical_entries(pos=pos_code)
    else:  # pragma: no cover - defensive
        raise LemmaBankError(
            "Unsupported Wordnet API: expected `words` or `lexical_entries` iterator."
        )

    seen = set()
    lemmas: List[str] = []
    for entry in entries:
        lemma = _extract_lemma(entry)
        if not lemma:
            continue
        norm = lemma.lower()
        if norm in seen:
            continue
        if not _should_keep(norm, min_len=1):
            continue
        seen.add(norm)
        lemmas.append(norm)
    lemmas.sort()
    return lemmas


def sample_rare_nouns_from_oewn(
    zipf_max: float = 3.4,
    *,
    zipf_min: Optional[float] = None,
    min_length: int = 3,
    lexicon: str = _DEFAULT_LEXICON,
    limit: Optional[int] = None,
) -> List[str]:
    """
    Collect rare noun lemmas from OEWN filtered by Zipf frequency.

    Parameters
    ----------
    zipf_max:
        Upper bound on `wordfreq.zipf_frequency` (exclusive).
    zipf_min:
        Optional lower bound on Zipf frequency (inclusive).
    min_length:
        Minimum number of characters in the lemma (after lowercasing).
    lexicon:
        OEWN lexicon identifier (e.g., ``oewn:2021``).
    limit:
        Optional maximum number of lemmas to return (deterministic prefix).
    """
    nouns = oewn_lemmas("n", lexicon=lexicon)
    rare: List[str] = []
    for lemma in nouns:
        if not _should_keep(lemma, min_len=min_length):
            continue
        score = zipf_frequency(lemma, "en")
        if score >= zipf_max:
            continue
        if zipf_min is not None and score < zipf_min:
            continue
        rare.append(lemma)
    if limit is not None:
        rare = rare[:max(0, limit)]
    return rare


@lru_cache(maxsize=4096)
def _is_person_noun_cached(lemma: str, lexicon: str) -> bool:
    if not lemma:
        return False
    # Ensure the lexicon is available; re-use the loader cache.
    _load_lexicon(lexicon)
    try:
        synsets = wn.synsets(lemma, pos="n", lexicon=lexicon)
    except Exception:
        return False
    for syn in synsets:
        lex_domain = None
        try:
            lex_domain = syn.lexdomain()
        except AttributeError:
            pass
        if lex_domain is not None and getattr(lex_domain, "id", None) == "noun.person":
            return True
        try:
            if syn.lexname() == "noun.person":
                return True
        except AttributeError:
            pass
        try:
            hypernyms = syn.hypernyms()
        except AttributeError:
            hypernyms = ()
        for hyper in hypernyms:
            hyper_domain = None
            try:
                hyper_domain = hyper.lexdomain()
            except AttributeError:
                pass
            if hyper_domain is not None and getattr(hyper_domain, "id", None) == "noun.person":
                return True
            try:
                if hyper.lexname() == "noun.person":
                    return True
            except AttributeError:
                continue
    return False


def is_person_noun(lemma: str, *, lexicon: str = _DEFAULT_LEXICON) -> bool:
    """
    Return True if WordNet marks the noun lemma as a person-denoting concept.
    """
    norm = (lemma or "").strip().lower()
    if not norm:
        return False
    return _is_person_noun_cached(norm, lexicon)
