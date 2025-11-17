import random
from functools import lru_cache
from typing import Optional

from .inflect import (
    inflect_adjective,
    inflect_noun,
    pluralize_noun,
    singularize_noun,
)
from .rarity import is_rare_lemma

_PARTITIVE_HEADS = {"lot", "lots", "bunch", "number", "couple", "plenty"}
_UPPER_SPECIAL = {"ii", "iii", "iv"}
_ANIMATE_REFLEXIVES = {"himself", "herself", "themselves"}
_REFLEXIVE_GENDER = {
    "herself": "female",
    "himself": "male",
}
_ARTICLE_DETERMINERS = {"a", "an", "the", "some"}
_ADJ_EXCLUDE = {"many", "most", "much", "few", "several"}
_NOUN_FALSE_FRIENDS = {"reference"}  # verbs commonly mistagged as nouns
_POOL_CACHE = {}

_ABSTRACT_SUFFIXES = ("ness", "hood", "ship", "ism", "ity", "ment", "ance", "ence", "ency", "age", "is", "ia")
_TAXON_SUFFIXES = ("idae", "ideae", "aceae", "ales")
_SPECIAL_PLURALS = {
    "men",
    "women",
    "people",
    "children",
    "teeth",
    "feet",
    "geese",
    "mice",
    "lice",
    "oxen",
    "indices",
    "matrices",
    "vertices",
    "criteria",
    "phenomena",
    "data",
    "media",
    "dice",
    "cacti",
    "octopi",
    "alumni",
}
_PLURAL_SUFFIXES = ("ches", "shes", "xes", "zes", "ses", "ies", "ves")


def reflexive_subject_indices(doc):
    """
    Return metadata for nominal subjects that bind an animate reflexive.

    The returned value acts like a mapping of ``token.i`` to a dictionary
    containing at least the reflexive pronoun (`pronoun`) and an optional
    gender label inferred from the pronoun (`gender`).
    """
    indices = {}
    for pron in doc:
        lower = pron.lower_
        if pron.pos_ != "PRON" or lower not in _ANIMATE_REFLEXIVES:
            continue
        heads = []
        head = pron.head
        if head is not None:
            heads.append(head)
            parent = head.head
            if parent is not None and parent is not head:
                heads.append(parent)
        for candidate_head in heads:
            for child in candidate_head.children:
                if child.dep_ in {"nsubj", "nsubjpass"} and child.pos_ == "NOUN":
                    indices[child.i] = {
                        "pronoun": pron.text,
                        "gender": _REFLEXIVE_GENDER.get(lower),
                    }
    return indices


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


def _has_preceding_article(token) -> bool:
    if token.i == 0:
        return False
    doc = token.doc
    if doc is None:
        return False
    prev = doc[token.i - 1]
    if prev.lower_ in _ARTICLE_DETERMINERS:
        return True
    return prev.pos_ == "DET"


def _is_properish(token):
    if "Prop" in token.morph.get("NounType"):
        return True
    text = token.text
    if not text:
        return False
    if text[0].isupper():
        if token.lemma_ == text:
            return True
        if not _has_preceding_article(token):
            return True
    return False


def _looks_like_common_plural(token) -> bool:
    """
    Treat capitalized irregular plurals (e.g., Oxen, Fungi) as common nouns
    when they have a retrievable singular form.
    """
    if token.tag_ not in {"NNS", "NNPS"}:
        return False
    singulars = singularize_noun(token.text)
    return bool(singulars)


def _force_pluralish(token) -> bool:
    """
    Fallback for badly tagged plurals (e.g., 'Analyses' tagged as ADJ).
    """
    lower = token.text.lower()
    if lower in _SPECIAL_PLURALS:
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


def _detokenize(toks):
    attach_no_space = {".", ",", "!", "?", ";", ":", "n't", "'s", "'re", "'ve", "'d", "'ll", "'m"}
    text = ""
    prev = ""
    for tok in toks:
        if not tok:
            continue
        if not text:
            text = tok
        else:
            if tok in attach_no_space or tok.startswith("'") or prev == "'":
                text += tok
            else:
                text += " " + tok
        prev = tok
    return text


def _match_casing(template: str, replacement: str) -> str:
    repl = replacement
    if template.isupper() or template in _UPPER_SPECIAL:
        return repl.upper()
    if template.istitle():
        return repl.title()
    if template.islower():
        return repl.lower()
    return repl.title() if template[:1].isupper() else repl.lower()


def _plural_form_ok(plural: str, lemma: str, singular_forms: tuple[str, ...]) -> bool:
    lower = plural.lower()
    if lower in _SPECIAL_PLURALS:
        return True
    if any(lower.endswith(sfx) for sfx in _PLURAL_SUFFIXES):
        return True
    if lower.endswith("men"):
        return True
    if lower.endswith("s") and len(lower) > 3 and not lower.endswith("ss"):
        return True
    # Permit irregulars where the singular round-trips to the original lemma.
    return lemma in singular_forms


@lru_cache(maxsize=8192)
def _is_countable_lemma(lemma: str, becl_cls: Optional[str]) -> bool:
    if not lemma:
        return False
    lower = lemma.strip().lower()
    if not lower or not lower.isalpha():
        return False
    if lemma != lower:
        return False
    if becl_cls == "MASS":
        return False
    if any(lower.endswith(sfx) for sfx in _TAXON_SUFFIXES):
        return False
    if any(lower.endswith(sfx) for sfx in _ABSTRACT_SUFFIXES):
        return False

    plural = pluralize_noun(lower)
    if not plural or plural == lower:
        return False
    singular_forms = singularize_noun(plural)
    if not singular_forms or lower not in singular_forms:
        return False
    if not _plural_form_ok(plural, lower, singular_forms):
        return False
    return True


def _prepare_pools(
    rare_lemmas,
    rare_person_lemmas,
    req,
    zipf_thr,
    becl_map,
):
    rare_tuple = rare_lemmas if isinstance(rare_lemmas, tuple) else tuple(rare_lemmas)
    person_tuple = ()
    if rare_person_lemmas:
        person_tuple = rare_person_lemmas if isinstance(rare_person_lemmas, tuple) else tuple(rare_person_lemmas)

    key = (
        req,
        rare_tuple,
        person_tuple,
        zipf_thr,
        id(becl_map) if becl_map is not None else None,
    )
    cached = _POOL_CACHE.get(key)
    if cached is not None:
        return cached

    if zipf_thr is None:
        pool = list(rare_tuple)
        pool_person = list(person_tuple)
    else:
        pool = [w for w in rare_tuple if is_rare_lemma(w, zipf_thr)]
        pool_person = [w for w in person_tuple if is_rare_lemma(w, zipf_thr)]

    def _becl_class(lemma: str) -> Optional[str]:
        if not becl_map:
            return None
        cls = becl_map.get(lemma.lower())
        if cls is None:
            return None
        return getattr(cls, "value", str(cls).split(".")[-1])

    if req == "COUNT":
        filtered = []
        for w in pool:
            cls = _becl_class(w)
            if _is_countable_lemma(w, cls):
                filtered.append(w)
        pool = filtered

        filtered_person = []
        for w in pool_person:
            cls = _becl_class(w)
            if _is_countable_lemma(w, cls):
                filtered_person.append(w)
        pool_person = filtered_person
    elif req == "MASS" and becl_map:
        allowed = {"MASS", "FLEX"}
        pool = [w for w in pool if _becl_class(w) in allowed]
        pool_person = [w for w in pool_person if _becl_class(w) in allowed]

    result = (tuple(pool), tuple(pool_person))
    _POOL_CACHE[key] = result
    return result


def candidate_nouns(doc):
    noun_chunk_indices = set()
    for chunk in doc.noun_chunks:
        for token in chunk:
            noun_chunk_indices.add(token.i)
    subject_indices = {t.i for t in doc if t.dep_ in {"nsubj", "nsubjpass"}}
    # Only content nouns; skip PROPN, NE chunks, and ROOT (prevents verb mis-swaps like "reference")
    return [
        t for t in doc
        if (
            (t.pos_ == "NOUN" and t.tag_ in {"NN", "NNS"})
            or (t.pos_ == "PROPN" and _looks_like_common_plural(t))
            or _force_pluralish(t)
        )
        and t.is_alpha
        and len(t.text) > 2
        and t.ent_type_ == ""
        and t.dep_ != "ROOT"  # skip main verb even if mis-tagged
        and t.dep_ != "relcl"
        and t.lemma_.lower() not in _NOUN_FALSE_FRIENDS
        and not any(child.dep_ in {"nsubj", "nsubjpass"} for child in t.children)  # avoid verb heads mis-tagged as nouns
        and not (t.head == t and t.dep_ == "ROOT")  # extra guard: skip if token is its own head and ROOT
        and not _is_partitive_quantifier(t)
        and not (_is_properish(t) and not _looks_like_common_plural(t))
        and (t.i in noun_chunk_indices or t.i in subject_indices)
    ]


def candidate_adjectives(doc):
    candidates = []
    for t in doc:
        if t.dep_ not in {"amod", "compound"}:
            continue
        if t.pos_ in {"DET", "PRON", "PROPN", "NUM"}:
            continue
        if not t.is_alpha or len(t.text) <= 2:
            continue
        if t.lower_ in _ADJ_EXCLUDE:
            continue
        if t.i == 0 and t.text and t.text[0].isupper() and t.lemma_ == t.text:
            continue
        if t.ent_type_:
            continue
        head = t.head
        if head is None or head.pos_ != "NOUN":
            continue
        if head.dep_ == "ROOT":
            # Skip when the whole clause is misparsed as a noun headed phrase.
            continue
        if head.tag_ not in {"NN", "NNS"}:
            continue
        if abs(head.i - t.i) > 2:
            continue
        candidates.append(t)
    return candidates


def _normalize_adj_tag(tag: str) -> str:
    if not tag:
        return "JJ"
    if tag.startswith("JJ"):
        return tag
    return "JJ"

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
    rare_person_lemmas=None,
    rare_gender_lemmas=None,
    override_lemmas=None,
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
    rare_person_lemmas: optional iterable[str] of human-denoting rare lemmas
        used when a swap position must stay animate (e.g., reflexive subjects).
    rare_gender_lemmas: optional mapping[str, Sequence[str]] of gender-specific
        person lemmas (e.g., {"female": [...], "male": [...]}).
    override_lemmas: optional iterable of lemma strings to use (in target order)
        instead of sampling from the rare pools. When provided, the function
        ignores ``rare_lemmas`` and ``rng`` for selection.
    """
    if rng is None:
        rng = random

    toks = [t.text for t in doc]
    swaps = []
    reflexive_subjects = reflexive_subject_indices(doc)

    if forced_targets is not None:
        seen = set()
        targets = []
        for entry in forced_targets:
            require_person = None
            require_gender = None
            if isinstance(entry, dict):
                idx = entry.get("i")
                tag = entry.get("tag")
                if "require_person" in entry:
                    require_person = bool(entry.get("require_person"))
                if "require_gender" in entry:
                    require_gender = entry.get("require_gender")
            elif isinstance(entry, (tuple, list)):
                if len(entry) < 2:
                    continue
                idx, tag = entry[0], entry[1]
            else:
                continue
            if not isinstance(idx, int):
                continue
            if idx < 0 or idx >= len(doc):
                continue
            if idx in seen:
                continue
            token = doc[idx]
            forced_tag = tag or token.tag_
            reflexive_info = reflexive_subjects.get(idx) if isinstance(reflexive_subjects, dict) else None
            if require_person is None:
                require_person = bool(reflexive_info)
            if require_gender is None and reflexive_info:
                require_gender = reflexive_info.get("gender")
            require_person = bool(require_person or require_gender)
            targets.append((token, forced_tag, bool(require_person), require_gender))
            seen.add(idx)
    else:
        detected = candidate_nouns(doc)
        # Deterministic order across Good/Bad
        detected.sort(key=lambda t: t.i)
        if noun_mode == "k":
            limit = max(0, min(k, len(detected)))
            detected = detected[:limit]
        targets = []
        for t in detected:
            reflexive_info = reflexive_subjects.get(t.i) if isinstance(reflexive_subjects, dict) else None
            require_gender = reflexive_info.get("gender") if reflexive_info else None
            require_person = bool(reflexive_info) or bool(require_gender)
            targets.append((t, t.tag_, require_person, require_gender))

    if not targets:
        return None, swaps

    override_list = None
    if override_lemmas is not None:
        override_list = list(override_lemmas)

    if override_list is None and not rare_lemmas:
        return None, swaps

    if noun_mode == "k":
        targets = targets[:max(0, min(k, len(targets)))]

    if override_list is not None:
        if len(override_list) != len(targets):
            return None, []
        if not override_list:
            return None, []
        for (token, tag, _, _), lemma in zip(targets, override_list):
            if not lemma or not isinstance(lemma, str):
                return None, []
            form = inflect_noun(lemma, tag)
            if not form:
                return None, []
            toks[token.i] = form
            swaps.append({
                "i": token.i,
                "old": token.text,
                "new": form,
                "tag": tag,
                "lemma": lemma,
            })
    else:
        rare_tuple = rare_lemmas if isinstance(rare_lemmas, tuple) else tuple(rare_lemmas)
        person_tuple = ()
        if rare_person_lemmas:
            person_tuple = rare_person_lemmas if isinstance(rare_person_lemmas, tuple) else tuple(rare_person_lemmas)

        pool, pool_person = _prepare_pools(
            rare_tuple,
            person_tuple,
            req,
            zipf_thr,
            becl_map,
        )

        pool = list(pool)
        pool_person = list(pool_person)

        if not pool and not pool_person:
            return None, swaps

        pool_set = set(pool)
        person_set = set(pool_person)
        pool_non_person = [w for w in pool if w not in person_set]

        gender_pools = {}
        if rare_gender_lemmas:
            for gender, items in rare_gender_lemmas.items():
                filtered = [w for w in items if w in person_set]
                if filtered:
                    gender_pools[gender] = filtered

        enforce_gender = bool(gender_pools)

        needs_person = any(require_person for _, _, require_person, _ in targets)
        needs_non_person = any(not require_person for _, _, require_person, _ in targets)
        required_genders = {gender for _, _, _, gender in targets if gender}

        if needs_person and not pool_person:
            return None, []
        if needs_non_person and not pool_non_person:
            return None, []
        if enforce_gender:
            for gender in required_genders:
                if gender and not gender_pools.get(gender):
                    return None, []

        # Choose in a deterministic sequence using rng
        for token, tag, require_person, require_gender in targets:
            if require_gender and enforce_gender:
                choice_pool = gender_pools.get(require_gender)
            elif require_person:
                choice_pool = pool_person
            else:
                choice_pool = pool_non_person
            if not choice_pool:
                return None, []
            lemma = rng.choice(choice_pool)
            form = inflect_noun(lemma, tag)
            if not form:
                return None, []
            toks[token.i] = form
            swaps.append({
                "i": token.i,
                "old": token.text,
                "new": form,
                "tag": tag,
                "lemma": lemma,
            })

    if not swaps:
        return None, swaps

    _adjust_indefinite_articles(toks)
    text = _detokenize(toks)
    # Capitalize the first character of the sentence
    if text:
        text = text[0].upper() + text[1:]
    return text, swaps


def _prepare_adj_pool(rare_lemmas, zipf_thr):
    rare_tuple = rare_lemmas if isinstance(rare_lemmas, tuple) else tuple(rare_lemmas)
    if zipf_thr is None:
        return rare_tuple
    return tuple(w for w in rare_tuple if is_rare_lemma(w, zipf_thr))


def adjective_swap_all(
    doc,
    rare_lemmas,
    adj_mode="all",
    k=2,
    zipf_thr=3.4,
    rng: Optional[random.Random]=None,
    forced_targets=None,
    override_lemmas=None,
):
    """
    Swap attributive adjectives with rare lemmas.
    """
    if rng is None:
        rng = random

    toks = [t.text for t in doc]
    swaps = []

    if forced_targets is not None:
        seen = set()
        targets = []
        for entry in forced_targets:
            if isinstance(entry, dict):
                idx = entry.get("i")
                tag = _normalize_adj_tag(entry.get("tag"))
            elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
                idx, tag = entry[0], entry[1]
                tag = _normalize_adj_tag(tag)
            else:
                continue
            if not isinstance(idx, int) or idx < 0 or idx >= len(doc):
                continue
            if idx in seen:
                continue
            token = doc[idx]
            targets.append((token, tag or token.tag_))
            seen.add(idx)
    else:
        detected = candidate_adjectives(doc)
        detected.sort(key=lambda t: t.i)
        if adj_mode == "k":
            limit = max(0, min(k, len(detected)))
            detected = detected[:limit]
        targets = [(t, _normalize_adj_tag(t.tag_)) for t in detected]

    if not targets:
        return None, swaps

    override_list = None
    if override_lemmas is not None:
        override_list = list(override_lemmas)

    if override_list is None and not rare_lemmas:
        return None, swaps

    if adj_mode == "k":
        targets = targets[:max(0, min(k, len(targets)))]

    if override_list is not None:
        if len(override_list) != len(targets):
            return None, []
        if not override_list:
            return None, []
        for (token, tag), lemma in zip(targets, override_list):
            if not lemma or not isinstance(lemma, str):
                return None, []
            form = inflect_adjective(lemma, tag)
            if not form:
                return None, []
            toks[token.i] = _match_casing(token.text, form)
            swaps.append({
                "i": token.i,
                "old": token.text,
                "new": toks[token.i],
                "tag": tag,
                "lemma": lemma,
            })
    else:
        pool = list(_prepare_adj_pool(rare_lemmas, zipf_thr))
        if not pool:
            return None, swaps
        for token, tag in targets:
            # Try a deterministic random order until we find an inflectable lemma.
            pool_order = list(pool)
            rng.shuffle(pool_order)
            chosen = None
            for lemma in pool_order:
                form = inflect_adjective(lemma, tag)
                if form:
                    chosen = (lemma, form)
                    break
            if not chosen:
                # Skip this target if no inflectable form; keep others.
                continue
            lemma, form = chosen
            toks[token.i] = _match_casing(token.text, form)
            swaps.append({
                "i": token.i,
                "old": token.text,
                "new": toks[token.i],
                "tag": tag,
                "lemma": lemma,
            })

    if not swaps:
        return None, swaps

    _adjust_indefinite_articles(toks)
    text = _detokenize(toks)
    if text:
        text = text[0].upper() + text[1:]
    return text, swaps
