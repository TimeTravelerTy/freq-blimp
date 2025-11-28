import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Optional
from wordfreq import zipf_frequency

from .inflect import (
    inflect_adjective,
    inflect_noun,
    inflect_verb,
    pluralize_noun,
    singularize_noun,
)
from .rarity import is_rare_lemma
from .verb_inventory import VerbInventory

_PARTITIVE_HEADS = {"lot", "lots", "bunch", "number", "couple", "plenty"}
_UPPER_SPECIAL = {"ii", "iii", "iv"}
_ANIMATE_REFLEXIVES = {"himself", "herself", "themselves"}
_INANIMATE_REFLEXIVES = {"itself"}
_REFLEXIVE_GENDER = {
    "herself": "female",
    "himself": "male",
}
_ARTICLE_DETERMINERS = {"a", "an", "the", "some"}
_ADJ_EXCLUDE = {"many", "most", "much", "few", "several"}
_POOL_CACHE = {}
_WEIGHT_CACHE = {}

_ABSTRACT_SUFFIXES = ("ness", "hood", "ship", "ism", "ity", "ment", "ance", "ence", "ency", "age", "is", "ia", "ry")
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
    "alumni",
}
_PLURAL_SUFFIXES = ("ches", "shes", "xes", "zes", "ses", "ies", "ves")
_VERB_EXCLUDE = {"be", "have", "do"}
_VERB_AUX_DEPS = {"aux", "auxpass", "cop"}
_OBJ_DEPS = {"dobj", "obj"}
_IOBJ_DEPS = {"iobj", "dative"}
_PREP_DEPS = {"prep"}
_PARTICLE_DEPS = {"prt"}
_THAT_VERBS = (
    "accept",
    "acknowledge",
    "add",
    "admit",
    "advert",
    "affirm",
    "agree",
    "allege",
    "allow",
    "announce",
    "answer",
    "anticipate",
    "apprehend",
    "argue",
    "ascertain",
    "assert",
    "assever",
    "asseverate",
    "assume",
    "attest",
    "aver",
    "avouch",
    "avow",
    "believe",
    "betray",
    "boast",
    "bruit",
    "calculate",
    "certify",
    "claim",
    "comment",
    "complain",
    "concede",
    "conceive",
    "conclude",
    "confess",
    "confirm",
    "conjecture",
    "consider",
    "contend",
    "decide",
    "declare",
    "deduce",
    "demonstrate",
    "deny",
    "depose",
    "determine",
    "discern",
    "disclose",
    "discover",
    "divine",
    "divulge",
    "doubt",
    "dream",
    "emphasize",
    "ensure",
    "establish",
    "estimate",
    "expect",
    "explain",
    "extrapolate",
    "fancy",
    "fear",
    "feel",
    "figure",
    "find",
    "foresee",
    "forget",
    "gainsay",
    "gather",
    "grant",
    "guarantee",
    "guess",
    "hear",
    "hold",
    "hope",
    "hypothesize",
    "imagine",
    "imply",
    "indicate",
    "infer",
    "insist",
    "intimate",
    "judge",
    "know",
    "learn",
    "maintain",
    "mean",
    "mention",
    "note",
    "notice",
    "observe",
    "opine",
    "perceive",
    "pledge",
    "portend",
    "posit",
    "postulate",
    "predict",
    "presage",
    "presume",
    "presuppose",
    "pretend",
    "proclaim",
    "profess",
    "prognosticate",
    "promise",
    "pronounce",
    "propose",
    "propound",
    "prove",
    "provide",
    "purport",
    "ratiocinate",
    "read",
    "realize",
    "reason",
    "recall",
    "reckon",
    "recognize",
    "record",
    "reflect",
    "remark",
    "remember",
    "repeat",
    "reply",
    "report",
    "request",
    "require",
    "resolve",
    "respond",
    "reveal",
    "rule",
    "say",
    "see",
    "sense",
    "show",
    "signify",
    "speculate",
    "state",
    "stipulate",
    "submit",
    "suggest",
    "suppose",
    "surmise",
    "suspect",
    "swear",
    "teach",
    "testify",
    "theorize",
    "think",
    "threaten",
    "trust",
    "understand",
    "urge",
    "verify",
    "vouch",
    "vouchsafe",
    "vow",
    "warn",
    "warrant",
    "whisper",
    "wish",
    "worry",
    "write",
)

# Allow compatible frame backoffs when exact matching fails.
_FRAME_FAMILY = {
    "intr": ["trans"],               # intransitive verbs can align with simple transitives
    "intr_pp": ["trans_pp"],         # PP-taking verbs with no object can align to transitive+PP frames
    "intr_particle": ["trans_particle"],
    "trans_pp": ["intr_pp"],         # sometimes inventory marks PP complements as intransitive
}

@lru_cache(maxsize=8192)
def _zipf_freq_cached(lemma: str) -> float:
    try:
        return zipf_frequency(lemma or "", "en")
    except Exception:
        return 0.0


def _zipf_weights(pool: List[str], temp: float = 1.0) -> List[float]:
    if not pool:
        return []
    lemmas = tuple(pool)
    key = (lemmas, float(temp))
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        return list(cached)
    weights = []
    for lemma in lemmas:
        freq = 10 ** _zipf_freq_cached(lemma)
        adj = freq ** (1.0 / temp) if temp and temp > 0 else freq
        weights.append(adj if adj > 0 else 0.001)
    _WEIGHT_CACHE[key] = tuple(weights)
    return weights


def _weighted_choice_by_zipf(pool: List[str], rng: random.Random, temp: float = 1.0) -> str:
    """
    Prefer lemmas closer to the Zipf ceiling while keeping randomness.
    """
    if len(pool) == 1:
        return pool[0]
    weights = _zipf_weights(pool, temp=temp)
    return rng.choices(pool, weights=weights, k=1)[0]


def _weighted_order_by_zipf(pool: List[str], rng: random.Random, max_draws: int = 64, temp: float = 1.0) -> List[str]:
    """
    Return a de-duplicated order biased toward higher Zipf frequencies.
    Draws are capped to avoid O(N) shuffles on huge pools.
    """
    if len(pool) <= 1:
        return list(pool)
    weights = _zipf_weights(pool, temp=temp)
    draws = min(len(pool), max_draws)
    sample = rng.choices(pool, weights=weights, k=draws)
    seen = set()
    ordered = []
    for lemma in sample:
        if lemma in seen:
            continue
        seen.add(lemma)
        ordered.append(lemma)
    if draws < len(pool):
        return ordered
    if len(ordered) < len(pool):
        for lemma in pool:
            if lemma not in seen:
                ordered.append(lemma)
    return ordered

def reflexive_subject_indices(doc):
    """
    Return metadata for nominal subjects that bind an animate reflexive.

    The returned value acts like a mapping of ``token.i`` to a dictionary
    containing at least the reflexive pronoun (`pronoun`) and an optional
    gender label inferred from the pronoun (`gender`), plus an `animate`
    boolean flag.
    """
    indices = {}
    for pron in doc:
        lower = pron.lower_
        if pron.pos_ != "PRON" or lower not in (_ANIMATE_REFLEXIVES | _INANIMATE_REFLEXIVES):
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
                if child.dep_ in {"nsubj", "nsubjpass"} and child.pos_ in {"NOUN", "PROPN"}:
                    indices[child.i] = {
                        "pronoun": pron.text,
                        "gender": _REFLEXIVE_GENDER.get(lower),
                        "animate": lower in _ANIMATE_REFLEXIVES,
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


def candidate_nouns(doc, reflexive_subjects=None):
    noun_chunk_indices = set()
    for chunk in doc.noun_chunks:
        for token in chunk:
            noun_chunk_indices.add(token.i)
    subject_indices = {t.i for t in doc if t.dep_ in {"nsubj", "nsubjpass"}}
    reflexive_subjects = reflexive_subjects if reflexive_subjects is not None else reflexive_subject_indices(doc)

    candidates = []
    for t in doc:
        is_reflexive_subject = t.i in reflexive_subjects
        if not (
            (t.pos_ == "NOUN" and t.tag_ in {"NN", "NNS"})
            or _force_pluralish(t)
        ):
            continue
        if not t.is_alpha or len(t.text) <= 2:
            continue
        if t.ent_type_:
            continue
        if t.dep_ == "ROOT" or t.dep_ == "relcl":
            continue
        # Avoid verb heads mis-tagged as nouns
        if any(child.dep_ in {"nsubj", "nsubjpass"} for child in t.children):
            continue
        if t.head == t and t.dep_ == "ROOT":
            continue
        if _is_partitive_quantifier(t):
            continue
        if _is_properish(t) and not (_looks_like_common_plural(t) or _force_pluralish(t)):
            continue
        if not (t.i in noun_chunk_indices or t.i in subject_indices or is_reflexive_subject):
            continue
        candidates.append(t)

    return candidates


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


@dataclass
class VerbTarget:
    token: Any
    tag: str
    frame_kind: str
    lemma: str
    prep_token: Optional[Any]
    particle_token: Optional[Any]
    has_that_clause: bool = False


def _frame_kind_for_verb(token):
    objs = [child for child in token.children if child.dep_ in _OBJ_DEPS]
    iobjs = [child for child in token.children if child.dep_ in _IOBJ_DEPS]
    preps = [child for child in token.children if child.dep_ in _PREP_DEPS]
    particles = [child for child in token.children if child.dep_ in _PARTICLE_DEPS]
    if len(objs) > 1 or len(iobjs) > 1:
        return None
    if len(preps) > 1 or len(particles) > 1:
        return None
    obj = objs[0] if objs else None
    iobj = iobjs[0] if iobjs else None
    prep = preps[0] if preps else None
    particle = particles[0] if particles else None
    base = "intr"
    if obj:
        base = "trans"
    if obj and iobj:
        base = "ditrans"
    elif iobj and not obj:
        return None
    suffix = ""
    if prep and particle:
        return None
    if particle:
        suffix = "_particle"
    elif prep:
        suffix = "_pp"
    return base + suffix, prep, particle


def candidate_verbs(doc):
    """
    Return verb heads that are eligible for swapping. The function focuses on
    lexical verbs (no auxiliaries) and simple argument structures: intransitive,
    transitive, ditransitive, verbs selecting a single PP complement, and
    phrasal verbs with a single particle.
    """
    candidates = []
    def _has_that_complement(token):
        for child in token.children:
            if child.dep_ == "ccomp":
                # Look for 'that' marker inside the clause.
                for cc_tok in child.subtree:
                    if cc_tok.text.lower() == "that" and cc_tok.dep_ == "mark":
                        return True
        return False

    for token in doc:
        is_verbish = token.pos_ == "VERB" or token.tag_.startswith("VB")
        if not is_verbish:
            continue
        if token.dep_ in _VERB_AUX_DEPS:
            continue
        if token.tag_ == "MD":
            continue
        lemma = token.lemma_.lower()
        if lemma in _VERB_EXCLUDE:
            continue
        if not token.is_alpha:
            continue
        frame_info = _frame_kind_for_verb(token)
        if not frame_info:
            continue
        kind, prep, particle = frame_info
        candidates.append(VerbTarget(
            token=token,
            tag=token.tag_,
            frame_kind=kind,
            lemma=lemma,
            prep_token=prep,
            particle_token=particle,
            has_that_clause=_has_that_complement(token),
        ))

    return candidates


def noun_swap_all(
    doc,
    rare_lemmas,
    noun_mode="all",
    k=2,
    zipf_thr=3.4,
    zipf_weighted: bool = False,
    zipf_temp: float = 1.0,
    becl_map=None,
    req=None,
    rng: Optional[random.Random]=None,
    forced_targets=None,
    rare_person_lemmas=None,
    rare_gender_lemmas=None,
    override_lemmas=None,
    reflexive_subjects=None,
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
    reflexive_subjects = reflexive_subjects if reflexive_subjects is not None else reflexive_subject_indices(doc)

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
                require_person = bool(reflexive_info.get("animate")) if reflexive_info else False
            if require_gender is None and reflexive_info:
                require_gender = reflexive_info.get("gender")
            require_person = bool(require_person or require_gender)
            targets.append((token, forced_tag, bool(require_person), require_gender))
            seen.add(idx)
    else:
        detected = candidate_nouns(doc, reflexive_subjects=reflexive_subjects)
        # Deterministic order across Good/Bad
        detected.sort(key=lambda t: t.i)
        if noun_mode == "k":
            limit = max(0, min(k, len(detected)))
            detected = detected[:limit]
        targets = []
        for t in detected:
            reflexive_info = reflexive_subjects.get(t.i) if isinstance(reflexive_subjects, dict) else None
            require_gender = reflexive_info.get("gender") if reflexive_info else None
            require_person = (reflexive_info.get("animate") if reflexive_info else False) or bool(require_gender)
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

        pool_person_checked = list(pool_person)
        pool_non_person_checked = list(pool_non_person)

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

        if needs_person and not pool_person_checked:
            return None, []
        if needs_non_person and not pool_non_person_checked:
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
                choice_pool = pool_person_checked
            else:
                choice_pool = pool_non_person_checked
            if not choice_pool:
                return None, []
            lemma = _weighted_choice_by_zipf(choice_pool, rng, temp=zipf_temp) if zipf_weighted else rng.choice(choice_pool)
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
    zipf_weighted: bool = False,
    zipf_temp: float = 1.0,
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
            # Try a deterministic order until we find an inflectable lemma.
            pool_order = (
                _weighted_order_by_zipf(pool, rng, temp=zipf_temp) if zipf_weighted else list(pool)
            )
            if not zipf_weighted:
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


def _frame_requires_prep(kind: str) -> bool:
    return bool(kind and kind.endswith("_pp"))


def _frame_requires_particle(kind: str) -> bool:
    return bool(kind and kind.endswith("_particle"))


def _normalize_forced_verb_targets(doc, forced_targets) -> List[VerbTarget]:
    seen = set()
    targets: List[VerbTarget] = []
    for entry in forced_targets or []:
        if isinstance(entry, dict):
            idx = entry.get("i")
            tag = entry.get("tag")
            frame = entry.get("frame")
            prep_idx = entry.get("prep_i")
            particle_idx = entry.get("particle_i")
            that_clause = bool(entry.get("that_clause"))
        elif isinstance(entry, (tuple, list)) and len(entry) >= 3:
            idx, tag, frame = entry[:3]
            prep_idx = entry[3] if len(entry) > 3 else None
            particle_idx = entry[4] if len(entry) > 4 else None
            that_clause = False
        else:
            continue
        if not isinstance(idx, int) or idx < 0 or idx >= len(doc):
            continue
        if idx in seen:
            continue
        token = doc[idx]
        actual_tag = tag or token.tag_
        frame_kind = frame or _frame_kind_for_verb(token)
        if isinstance(frame_kind, tuple):
            frame_kind = frame_kind[0]
        if not isinstance(frame_kind, str):
            continue
        prep_token = None
        particle_token = None
        if prep_idx is not None and isinstance(prep_idx, int) and 0 <= prep_idx < len(doc):
            prep_token = doc[prep_idx]
        if particle_idx is not None and isinstance(particle_idx, int) and 0 <= particle_idx < len(doc):
            particle_token = doc[particle_idx]
        if _frame_requires_prep(frame_kind) and prep_token is None:
            continue
        if _frame_requires_particle(frame_kind) and particle_token is None:
            continue
        targets.append(VerbTarget(
            token=token,
            tag=actual_tag,
            frame_kind=frame_kind,
            lemma=token.lemma_.lower(),
            prep_token=prep_token,
            particle_token=particle_token,
            has_that_clause=that_clause or False,
        ))
        seen.add(idx)
    return targets


def verb_swap_all(
    doc,
    inventory: Optional[VerbInventory],
    verb_mode: str = "k",
    k: int = 1,
    zipf_thr: Optional[float] = None,
    zipf_weighted: bool = False,
    zipf_temp: float = 1.0,
    rng: Optional[random.Random] = None,
    forced_targets=None,
    override_specs=None,
):
    """
    Swap lexical verbs using the precomputed verb inventory.

    ``zipf_thr`` filters optional that-clause replacements for rarity.
    When ``zipf_weighted`` is True, sampling prefers higher Zipf lemmas
    within the allowed pools.
    """
    if rng is None:
        rng = random

    if inventory is None or inventory.is_empty():
        return None, []

    toks = [t.text for t in doc]
    swaps = []

    if forced_targets is not None:
        targets = _normalize_forced_verb_targets(doc, forced_targets)
    else:
        detected = candidate_verbs(doc)
        detected.sort(key=lambda t: t.token.i)
        if verb_mode == "k":
            limit = max(0, min(k, len(detected)))
            detected = detected[:limit]
        targets = detected

    if not targets:
        return None, swaps

    override_list = None
    if override_specs is not None:
        override_list = list(override_specs)
        if len(override_list) != len(targets):
            return None, []

    for idx, target in enumerate(targets):
        if target.has_that_clause:
            spec = override_list[idx] if override_list is not None else None
            forced_lemma = spec.get("lemma") if isinstance(spec, dict) else None
            forced_frame = spec.get("frame") if isinstance(spec, dict) else None
            if forced_lemma:
                options = [forced_lemma]
            else:
                options = list(_THAT_VERBS)
                if zipf_thr is not None:
                    filtered = [lemma for lemma in options if is_rare_lemma(lemma, zipf_thr)]
                    if filtered:
                        options = filtered
            if zipf_weighted:
                options = _weighted_order_by_zipf(options, rng, temp=zipf_temp)
            else:
                rng.shuffle(options)
            chosen_lemma = None
            chosen_form = None
            for lemma in options:
                if not lemma:
                    continue
                form = inflect_verb(lemma, target.tag)
                if form:
                    chosen_lemma = lemma
                    chosen_form = form
                    break
            if not chosen_form:
                return None, []
            toks[target.token.i] = _match_casing(target.token.text, chosen_form)
            swaps.append({
                "i": target.token.i,
                "old": target.token.text,
                "new": toks[target.token.i],
                "tag": target.tag,
                "lemma": chosen_lemma,
                "frame": forced_frame or "that_clause",
                "prep_i": None,
                "prep_old": None,
                "prep_new": None,
                "particle_i": None,
                "particle_old": None,
                "particle_new": None,
            })
            continue

        if override_list is not None:
            spec = override_list[idx]
            lemma = spec.get("lemma") if isinstance(spec, dict) else None
            frame_name = spec.get("frame") if isinstance(spec, dict) else None
            lookup = inventory.lookup(lemma, frame_name or target.frame_kind)
            if not lookup:
                return None, []
            entry, frame = lookup
        else:
            prep_text = target.prep_token.text if target.prep_token is not None else None
            particle_text = target.particle_token.text if target.particle_token is not None else None
            frame_order = [target.frame_kind] + _FRAME_FAMILY.get(target.frame_kind, [])
            sample = None
            for fk in frame_order:
                sample = inventory.sample(
                    fk,
                    rng,
                    desired_prep=prep_text,
                    desired_particle=particle_text,
                    zipf_weighted=zipf_weighted,
                    zipf_temp=zipf_temp,
                )
                if not sample and (prep_text or particle_text):
                    sample = inventory.sample(fk, rng, zipf_weighted=zipf_weighted, zipf_temp=zipf_temp)
                if sample:
                    break
            if not sample:
                return None, []
            entry, frame = sample

        form = inflect_verb(entry.lemma, target.tag)
        if not form:
            return None, []
        toks[target.token.i] = _match_casing(target.token.text, form)

        prep_old = target.prep_token.text if target.prep_token is not None else None
        prep_new = None
        if _frame_requires_prep(frame.kind):
            prep_token = target.prep_token
            if prep_token is None:
                return None, []
            # If the sampled frame suggests a different preposition from the
            # original, prefer keeping the original to avoid ungrammatical
            # mixes like "collude to".
            if frame.prep and prep_old and frame.prep.lower() != prep_old.lower():
                replacement = prep_old
            else:
                replacement = frame.prep or prep_old
            if not replacement:
                return None, []
            toks[prep_token.i] = replacement
            prep_new = replacement

        particle_old = target.particle_token.text if target.particle_token is not None else None
        particle_new = None
        if _frame_requires_particle(frame.kind):
            particle_token = target.particle_token
            if particle_token is None:
                return None, []
            if frame.particle and particle_old and frame.particle.lower() != particle_old.lower():
                replacement = particle_old
            else:
                replacement = frame.particle or particle_old
            if not replacement:
                return None, []
            toks[particle_token.i] = replacement
            particle_new = replacement

        swaps.append({
            "i": target.token.i,
            "old": target.token.text,
            "new": toks[target.token.i],
            "tag": target.tag,
            "lemma": entry.lemma,
            "frame": frame.kind,
            "prep_i": target.prep_token.i if target.prep_token is not None else None,
            "prep_old": prep_old,
            "prep_new": prep_new,
            "particle_i": target.particle_token.i if target.particle_token is not None else None,
            "particle_old": particle_old,
            "particle_new": particle_new,
        })

    if not swaps:
        return None, swaps

    text = _detokenize(toks)
    if text:
        text = text[0].upper() + text[1:]
    return text, swaps
