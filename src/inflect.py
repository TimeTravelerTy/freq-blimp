from functools import lru_cache
from typing import Optional, Tuple

from lemminflect import getInflection, getLemma


_IRREGULAR_PLURALS = {
    "person": "people",
    "man": "men",
    "woman": "women",
    "child": "children",
    "tooth": "teeth",
    "foot": "feet",
    "goose": "geese",
    "mouse": "mice",
    "louse": "lice",
    "ox": "oxen",
    "leaf": "leaves",
    "wolf": "wolves",
    "calf": "calves",
    "half": "halves",
    "knife": "knives",
    "wife": "wives",
    "life": "lives",
    "loaf": "loaves",
    "thief": "thieves",
    "elf": "elves",
    "self": "selves",
    "sheaf": "sheaves",
    "belief": "beliefs",
    "chef": "chefs",
    "chief": "chiefs",
    "analysis": "analyses",
    "basis": "bases",
    "crisis": "crises",
    "diagnosis": "diagnoses",
    "parenthesis": "parentheses",
    "synopsis": "synopses",
    "thesis": "theses",
    "criterion": "criteria",
    "phenomenon": "phenomena",
    "datum": "data",
    "medium": "media",
    "index": "indices",
    "appendix": "appendices",
    "matrix": "matrices",
    "vertex": "vertices",
    "axis": "axes",
    "cactus": "cacti",
    "focus": "foci",
    "fungus": "fungi",
    "nucleus": "nuclei",
    "radius": "radii",
    "stimulus": "stimuli",
    "curriculum": "curricula",
    "formula": "formulae",
    "manservant": "menservants",
    "maidservant": "maidservants",
}
_IRREGULAR_SINGULARS = {v: k for k, v in _IRREGULAR_PLURALS.items()}


def _match_case(template: str, replacement: str) -> str:
    if not template:
        return replacement
    if template.isupper():
        return replacement.upper()
    if template.istitle():
        return replacement.title()
    if template.islower():
        return replacement.lower()
    return replacement


def _compound_person_plural(lower: str) -> Optional[str]:
    if lower.endswith("woman") and len(lower) > len("woman"):
        return lower[:-5] + "women"
    if lower.endswith("man") and len(lower) > len("man"):
        if lower.endswith("human"):
            return None
        return lower[:-3] + "men"
    return None


def _compound_person_singular(lower: str) -> Optional[str]:
    if lower.endswith("women") and len(lower) > len("women"):
        return lower[:-5] + "woman"
    if lower.endswith("men") and len(lower) > len("men"):
        candidate = lower[:-3] + "man"
        if candidate.endswith("human"):
            return None
        return candidate
    return None


def inflect_noun(lemma, tag):
    # tag is "NN" or "NNS"
    out = getInflection(lemma, tag=tag)
    form = out[0] if out else None
    if not lemma:
        return form
    lower = lemma.lower()
    if tag == "NNS":
        compound = _compound_person_plural(lower)
        if compound:
            if not form or form.lower() != compound:
                form = _match_case(lemma, compound)
    if tag == "NN":
        singular = _compound_person_singular(lower)
        if singular:
            if not form or form.lower() != singular:
                form = _match_case(lemma, singular)
    return form


def inflect_adjective(lemma: str, tag: str):
    """
    Inflect an adjective lemma for the desired POS tag (JJ/JJR/JJS).
    """
    if not lemma:
        return None
    out = getInflection(lemma, tag=tag)
    if not out:
        return None
    return out[0]


@lru_cache(maxsize=4096)
def pluralize_noun(lemma: str) -> Optional[str]:
    """Return a lower-cased plural form for the given lemma, if possible."""
    if not lemma:
        return None
    base = lemma.strip()
    if not base:
        return None
    lower = base.lower()
    if lower in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[lower]
    compound = _compound_person_plural(lower)
    if compound:
        return compound
    forms = getInflection(lower, tag="NNS")
    if not forms:
        return None
    return forms[0].lower()


@lru_cache(maxsize=4096)
def singularize_noun(form: str) -> Tuple[str, ...]:
    """Return possible singular lemmas for a plural form."""
    if not form:
        return tuple()
    word = form.strip()
    if not word:
        return tuple()
    lower = word.lower()
    if lower in _IRREGULAR_SINGULARS:
        return (_IRREGULAR_SINGULARS[lower],)
    compound = _compound_person_singular(lower)
    if compound:
        return (compound,)
    lemmas = getLemma(lower, upos="NOUN")
    if not lemmas:
        return tuple()
    return tuple(sorted({lemma.lower() for lemma in lemmas if lemma}))


@lru_cache(maxsize=4096)
def has_plural_form(lemma: str) -> bool:
    """
    Return True when we can derive a plausible plural noun form.
    """
    return pluralize_noun(lemma) is not None
