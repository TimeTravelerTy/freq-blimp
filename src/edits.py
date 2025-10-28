import random
from typing import Optional

from .inflect import inflect_noun
from .rarity import is_rare_lemma

_PARTITIVE_HEADS = {"lot", "lots", "bunch", "number", "couple", "plenty"}


def _is_partitive_quantifier(token):
    """
    Detect multiword quantifiers such as 'a lot of' so their heads stay intact.
    """
    if token.lower_ not in _PARTITIVE_HEADS:
        return False
    doc = token.doc
    if token.i + 1 >= len(doc):
        return False
    nxt = doc[token.i + 1]
    if nxt.lower_ != "of":
        return False
    if token.lower_ == "lots":
        return True
    if token.i == 0:
        return False
    prev = doc[token.i - 1]
    return prev.lower_ in {"a", "an", "the", "this", "that", "these", "those"}


def _is_properish(token):
    if "Prop" in token.morph.get("NounType"):
        return True
    if token.text and token.text[0].isupper() and token.lemma_ == token.text:
        return True
    return False


def _adjust_indefinite_articles(toks):
    vowels = set("aeiou")
    for i in range(len(toks) - 1):
        word = toks[i]
        nxt = toks[i + 1]
        if not word or not nxt:
            continue
        lower = word.lower()
        if lower not in {"a", "an"}:
            continue
        first = nxt[0].lower()
        if not first.isalpha():
            continue
        desired = "an" if first in vowels else "a"
        if lower != desired:
            toks[i] = desired.capitalize() if word[0].isupper() else desired


def candidate_nouns(doc):
    animate_reflexives = {"himself", "herself", "themselves", "myself", "yourself", "yourselves", "ourselves"}
    animate_reflexive_present = any(
        t.lower_ in animate_reflexives for t in doc if t.pos_ == "PRON"
    )
    noun_chunk_indices = set()
    for chunk in doc.noun_chunks:
        for token in chunk:
            noun_chunk_indices.add(token.i)
    # Only content nouns; skip PROPN, NE chunks, and ROOT (prevents verb mis-swaps like "reference")
    return [
        t for t in doc
        if t.pos_ == "NOUN"
        and t.tag_ in {"NN", "NNS"}
        and t.is_alpha
        and len(t.text) > 2
        and t.ent_type_ == ""
        and t.dep_ != "ROOT"  # skip main verb even if mis-tagged
        and t.dep_ != "relcl"
        and not (t.head == t and t.dep_ == "ROOT")  # extra guard: skip if token is its own head and ROOT
        and not _is_partitive_quantifier(t)
        and not _is_properish(t)
        and t.i in noun_chunk_indices
        and not (animate_reflexive_present and t.dep_ in {"nsubj", "nsubjpass"})
    ]

def noun_swap_all(
    doc,
    rare_lemmas,
    noun_mode="all",
    k=2,
    zipf_thr=3.4,
    becl_map=None,
    req=None,
    rng: Optional[random.Random]=None,
    forced_targets=None,
):
    """
    rare_lemmas: iterable[str] of candidate noun lemmas. If ``zipf_thr`` is None
        the lemmas are assumed to be pre-filtered for rarity.
    becl_map: lemma->CountClass (optional, used when req in {"COUNT","MASS"})
    req: "COUNT" | "MASS" | None
    rng: pass a pre-seeded Random to make Good/Bad use the same sequence
    forced_targets: optional iterable of (index, tag) pairs. When provided, the
        function swaps the tokens at the given indices (if in-range) and uses
        the supplied tag when inflecting, instead of detecting candidates from
        ``doc``.
    """
    if rng is None:
        rng = random

    toks = [t.text for t in doc]
    swaps = []

    if forced_targets is not None:
        seen = set()
        targets = []
        for idx, tag in forced_targets:
            if not isinstance(idx, int):
                continue
            if idx < 0 or idx >= len(doc):
                continue
            if idx in seen:
                continue
            token = doc[idx]
            forced_tag = tag or token.tag_
            targets.append((token, forced_tag))
            seen.add(idx)
    else:
        detected = candidate_nouns(doc)
        # Deterministic order across Good/Bad
        detected.sort(key=lambda t: t.i)
        if noun_mode == "k":
            limit = max(0, min(k, len(detected)))
            detected = detected[:limit]
        targets = [(t, t.tag_) for t in detected]

    if not targets:
        return None, swaps

    if not rare_lemmas:
        return None, swaps

    if noun_mode == "k":
        targets = targets[:max(0, min(k, len(targets)))]

    # Pre-filter pool by rarity (and countability if needed)
    if zipf_thr is None:
        pool = list(rare_lemmas)
    else:
        pool = [w for w in rare_lemmas if is_rare_lemma(w, zipf_thr)]

    def _becl_allows(lemma, suffixes, default=True):
        if not becl_map:
            return True
        cls = becl_map.get(lemma.lower())
        if cls is None:
            return default
        return str(cls).endswith(suffixes)

    if req == "COUNT":
        pool = [w for w in pool if _becl_allows(w, ("COUNT", "FLEX"), default=True)]
    elif req == "MASS":
        pool = [w for w in pool if _becl_allows(w, ("MASS", "FLEX"), default=False)]

    if not pool:
        return None, swaps

    # Choose in a deterministic sequence using rng
    for token, tag in targets:
        lemma = rng.choice(pool)
        form = inflect_noun(lemma, tag)
        if not form:
            continue
        toks[token.i] = form
        swaps.append({"i": token.i, "old": token.text, "new": form, "tag": tag})

    if not swaps:
        return None, swaps

    _adjust_indefinite_articles(toks)

    text = " ".join(toks)
    # light detokenization to clean spaces before punctuation
    text = (
        text.replace(" .", ".")
            .replace(" ,", ",")
            .replace(" !", "!")
            .replace(" ?", "?")
            .replace(" ;", ";")
            .replace(" :", ":")
            .replace(" n't", "n't")
    )
    # Capitalize the first character of the sentence
    if text:
        text = text[0].upper() + text[1:]
    return text, swaps
