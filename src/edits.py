import random
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional
from wordfreq import zipf_frequency
import wn

from .inflect import (
    inflect_adjective,
    inflect_noun,
    inflect_verb,
    pluralize_noun,
    singularize_noun,
)
from .rarity import is_rare_lemma
from .verb_inventory import VerbInventory, wordnet_pos_counts

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

# Common phrasal-verb particles that spaCy may tag as `prt` or `advmod`.
_PARTICLE_WORDS = {
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
    "open",
    "out",
    "over",
    "round",
    "through",
    "together",
    "under",
    "up",
}
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

# Optional COCA-derived whitelist for clausal complement verbs.
_CLAUSE_VERB_WHITELIST_PATH = Path("data") / "processed" / "clause_verb_whitelists.json"

# Allow compatible frame backoffs when exact matching fails.
_FRAME_FAMILY = {
    # Intransitive verbs can align with simple transitives.
    # PP frames should not back off to ditrans_pp; failing is safer than
    # mislabeling.
    "intr": ["trans"],
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
    attach_no_space = {".", ",", "!", "?", ";", ":", "n't", "'s", "'re", "'ve", "'d", "'ll", "'m", "-"}
    text = ""
    prev = ""
    for tok in toks:
        if not tok:
            continue
        if not text:
            text = tok
        else:
            if tok in attach_no_space or tok.startswith("'") or prev == "-":
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


def _is_ex_compound(token) -> bool:
    """
    Return True when the noun token is part of an 'ex-<noun>' compound
    tokenized as ['ex', '-', '<noun>'].
    """
    if token.i < 2:
        return False
    doc = token.doc
    try:
        prev_dash = doc[token.i - 1]
        prev_ex = doc[token.i - 2]
    except Exception:
        return False
    return prev_ex.text.lower() == "ex" and prev_dash.text == "-"


def _apply_noun_form(token, form: str, toks) -> str:
    """
    Place the inflected noun form into the token list, folding 'ex-<noun>' back
    into a single token when present. Returns the original surface string that
    was replaced.
    """
    if _is_ex_compound(token):
        doc = token.doc
        prev_dash = doc[token.i - 1]
        prev_ex = doc[token.i - 2]
        old_surface = f"{prev_ex.text}{prev_dash.text}{token.text}"
        toks[token.i - 2] = f"ex-{form}"
        toks[token.i - 1] = ""
        toks[token.i] = ""
        return old_surface
    toks[token.i] = form
    # Fix possessive marker for plural / s-final heads (e.g., "horrors's" -> "horrors'").
    # spaCy tokenizes possessives as a following token with tag_ == "POS" and surface "'s"/"’s".
    try:
        doc = token.doc
        if doc is not None and token.i + 1 < len(doc):
            nxt = doc[token.i + 1]
            if nxt is not None and getattr(nxt, "tag_", None) == "POS":
                raw = (nxt.text or "").strip()
                if raw in {"'s", "’s"}:
                    lower = (form or "").strip().lower()
                    if lower.endswith("s") and len(lower) > 1:
                        toks[nxt.i] = "'" if raw.startswith("'") else "’"
    except Exception:
        pass
    return token.text


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
def _noun_subjects_cached(lemma: str, lexicon: str = "oewn:2021") -> tuple[str, ...]:
    norm = (lemma or "").strip().lower()
    if not norm:
        return tuple()
    try:
        synsets = wn.synsets(norm, pos="n", lexicon=lexicon)
    except Exception:
        synsets = ()
    subjects = set()
    for syn in synsets or ():
        md = None
        try:
            md = syn.metadata() if callable(getattr(syn, "metadata", None)) else getattr(syn, "metadata", None)
        except Exception:
            md = None
        if isinstance(md, dict):
            subj = md.get("subject")
            if isinstance(subj, str) and subj:
                subjects.add(subj)
    return tuple(sorted(subjects))


@lru_cache(maxsize=8192)
def _noun_only_in_subjects(lemma: str, subjects: tuple[str, ...], lexicon: str = "oewn:2021") -> bool:
    subs = _noun_subjects_cached(lemma, lexicon=lexicon)
    if not subs:
        return False
    allowed = set(subjects or ())
    return all(s in allowed for s in subs)


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


@lru_cache(maxsize=8192)
def _normalize_noun_lemma(lemma: str) -> str:
    """
    Normalize noun lemmas so we don't double-inflect plurals.

    Some sources (e.g., OEWN) can include plural-looking noun lemmas like
    "leaders" or "details". If we treat those as base lemmas and then inflect
    to NNS, we can produce forms like "leaderses".
    """
    if not lemma or not isinstance(lemma, str):
        return ""
    lower = lemma.strip().lower()
    if not lower:
        return ""

    # If the lemma already looks plural-like (e.g., "humans"), try to recover a
    # singular that round-trips back to the original surface form. This prevents
    # double-inflection like "humanses" when later inflecting to NNS.
    if lower.endswith("s") and len(lower) > 3 and not lower.endswith("ss"):
        candidates = singularize_noun(lower)
        for cand in candidates:
            if cand and cand != lower and pluralize_noun(cand) == lower:
                return cand

    # If this lemma already looks like its own plural (e.g., "sheep"), keep it.
    plural = pluralize_noun(lower)
    if plural is None or plural != lower:
        return lower
    return lower


def _prepare_pools(
    rare_lemmas,
    rare_person_lemmas,
    req,
    zipf_thr,
    becl_map,
):
    rare_tuple_raw = rare_lemmas if isinstance(rare_lemmas, tuple) else tuple(rare_lemmas)
    person_tuple = ()
    if rare_person_lemmas:
        person_tuple = rare_person_lemmas if isinstance(rare_person_lemmas, tuple) else tuple(rare_person_lemmas)

    # Normalize noun lemmas (avoid plural bases like "leaders") and de-dupe.
    def _norm_dedupe(items):
        out = []
        seen = set()
        for item in items:
            norm = _normalize_noun_lemma(item)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            out.append(norm)
        return tuple(out)

    rare_tuple = _norm_dedupe(rare_tuple_raw)
    if person_tuple:
        person_tuple = _norm_dedupe(person_tuple)

    # Avoid noun replacements that are primarily verbs (e.g., "led" as a noun
    # replacement even though it's far more common as a verb form).
    def _noun_dominant(lemma: str) -> bool:
        v, n, a, r = wordnet_pos_counts(lemma, lexicon="oewn:2021")
        if n <= 0:
            return False
        # Filter out lexicographer-domain nouns that are rarely used as concrete
        # count nouns in BLiMP contexts (e.g., "spiritual" as a noun).
        # We only reject lemmas whose noun senses are exclusively in these domains.
        if _noun_only_in_subjects(lemma, ("noun.communication",), lexicon="oewn:2021"):
            return False
        # Keep nouns that are clearly nouns rather than adjective/verb/adverb
        # items used nominally (e.g., "a wild", "an independent").
        if v >= n or a >= n or r >= n:
            return False
        # Suppress gerund-like nouns with weak noun support (e.g., "studying",
        # "talking") while keeping established -ing nouns ("building",
        # "meeting", "painting"). We only apply this when the base verb exists.
        if lemma.endswith("ing") and len(lemma) > 5:
            base = lemma[:-3]
            try:
                base_verbs = wn.synsets(base, pos="v", lexicon="oewn:2021")
            except Exception:
                base_verbs = ()
            if base_verbs and n < 3:
                return False
        return n / (v + 1) >= 1.0

    noun_filtered = tuple(w for w in rare_tuple if _noun_dominant(w))
    if noun_filtered:
        rare_tuple = noun_filtered
    if person_tuple:
        person_filtered = tuple(w for w in person_tuple if _noun_dominant(w))
        if person_filtered:
            person_tuple = person_filtered

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
    has_clausal_complement: bool = False
    wh_marker: Optional[str] = None
    enforce_transitivity: bool = False


def _frame_kind_for_verb(token):
    objs = [child for child in token.children if child.dep_ in _OBJ_DEPS]
    iobjs = [child for child in token.children if child.dep_ in _IOBJ_DEPS]
    preps = [child for child in token.children if child.dep_ in _PREP_DEPS]
    particles = [child for child in token.children if child.dep_ in _PARTICLE_DEPS]
    # Heuristic: some particles are tagged as `advmod` (e.g., "hide away", "stand up").
    particles += [
        child
        for child in token.children
        if (
            child.dep_ == "advmod"
            and child.pos_ in {"ADV", "PART", "ADP", "ADJ"}
            and (child.text or "").strip().lower() in _PARTICLE_WORDS
        )
    ]
    # spaCy sometimes tags clause markers like "that" as dobj; ignore those so
    # we do not inflate ditrans labels.
    objs = [
        child
        for child in objs
        if not (child.dep_ == "dobj" and child.text.lower() == "that")
    ]
    if len(objs) > 1 or len(iobjs) > 1:
        return None
    if len(preps) > 1 or len(particles) > 1:
        return None
    obj = objs[0] if objs else None
    iobj = iobjs[0] if iobjs else None
    prep = preps[0] if preps else None
    particle = particles[0] if particles else None
    # Phrasal verbs with both a particle and a PP complement are rare and tricky
    # to handle robustly (particle movement, PP attachment ambiguity).
    if particle is not None and prep is not None:
        return None
    if prep is None:
        # Heuristic: spaCy sometimes fails to attach a PP complement as a `prep`
        # child under ellipsis/coordination; fall back to a short lookahead.
        doc = token.doc
        if doc is not None:
            for offset in (1, 2):
                idx = token.i + offset
                if idx >= len(doc):
                    break
                cand = doc[idx]
                if cand is None:
                    continue
                if cand.pos_ == "PUNCT":
                    break
                if cand.pos_ in {"ADV", "PART"}:
                    continue
                if cand.pos_ == "ADP" and (cand.text or "").strip().lower() == "about":
                    prep = cand
                break
    base = "intr"
    if obj:
        base = "trans"
    if obj and iobj:
        base = "ditrans"
    elif iobj and not obj:
        return None
    if prep:
        kind = "intr_pp" if base == "intr" else "ditrans_pp"
    else:
        kind = base
    if particle is not None:
        kind = f"{kind}_particle"
    return kind, prep, particle


def candidate_verbs(doc):
    """
    Return verb heads that are eligible for swapping. The function focuses on
    lexical verbs (no auxiliaries) and simple argument structures: intransitive,
    transitive, ditransitive, and verbs selecting a single PP complement.
    """
    candidates = []
    def _ccomp_child(token):
        for child in token.children:
            if child.dep_ == "ccomp":
                return child
        return None

    def _has_that_complement(ccomp_token):
        if ccomp_token is None:
            return False
        # Look for 'that' marker inside the clause.
        for cc_tok in ccomp_token.subtree:
            if cc_tok.text.lower() == "that" and cc_tok.dep_ == "mark":
                return True
        return False

    def _wh_marker(ccomp_token) -> Optional[str]:
        if ccomp_token is None:
            return None
        for cc_tok in ccomp_token.subtree:
            lower = cc_tok.text.lower()
            if lower in {"what", "who"}:
                return lower
        return None

    def _maybe_participle_amod(tok) -> Optional[VerbTarget]:
        """
        Heuristic for cases where spaCy mis-tags participial relative clauses as ADJ/amod.

        Example (desired): "sketches stunning Rhonda" where "stunning" should behave
        like a transitive VBG verb with object "Rhonda", but spaCy can label it
        ADJ/JJ with dep=amod and head=Rhonda (PROPN).
        """
        if tok.pos_ != "ADJ":
            return None
        if tok.dep_ != "amod":
            return None
        text = (tok.text or "").strip().lower()
        if not text or not text.isalpha():
            return None
        if not text.endswith("ing") or len(text) < 5:
            return None
        if tok.i + 1 >= len(doc):
            return None
        nxt = doc[tok.i + 1]
        if nxt is None or nxt.pos_ not in {"PROPN", "NOUN", "PRON"}:
            return None
        if tok.head != nxt:
            return None
        return VerbTarget(
            token=tok,
            tag="VBG",
            frame_kind="trans",
            lemma=text,
            prep_token=None,
            particle_token=None,
            has_that_clause=False,
            has_clausal_complement=False,
            wh_marker=None,
        )

    def _maybe_participle_attr_noun(tok) -> Optional[VerbTarget]:
        """
        Heuristic for cases where a participle is mislabeled as a noun attribute
        under `be` in existential-there infinitivals.

        Example: "... there to be a unicycle dropping." where "dropping" should
        behave like an intransitive VBG verb.
        """
        if tok.pos_ != "NOUN":
            return None
        if tok.dep_ != "attr":
            return None
        head = tok.head
        if head is None or head.lemma_.lower() != "be":
            return None
        text = (tok.text or "").strip().lower()
        if not text or not text.isalpha():
            return None
        if not text.endswith("ing") or len(text) < 5:
            return None
        if tok.i == 0:
            return None
        prev = tok.doc[tok.i - 1]
        # spaCy can mis-tag the noun as ADJ in "a <noun> <participle>" sequences.
        if prev is None or prev.pos_ not in {"NOUN", "PROPN", "ADJ"}:
            return None
        # Typically sentence-final (before punctuation).
        if tok.i + 1 < len(tok.doc):
            nxt = tok.doc[tok.i + 1]
            if nxt is not None and nxt.pos_ != "PUNCT":
                return None
        return VerbTarget(
            token=tok,
            tag="VBG",
            frame_kind="intr",
            lemma=text,
            prep_token=None,
            particle_token=None,
            has_that_clause=False,
            has_clausal_complement=False,
            wh_marker=None,
        )

    for token in doc:
        # Primary path: true verbs.
        if token.pos_ != "VERB":
            # Fallback: misparsed participial modifiers.
            fallback = _maybe_participle_amod(token)
            if fallback is not None:
                candidates.append(fallback)
            fallback2 = _maybe_participle_attr_noun(token)
            if fallback2 is not None:
                candidates.append(fallback2)
            continue
        if token.dep_ in _VERB_AUX_DEPS:
            continue
        if token.tag_ == "MD":
            continue
        lemma = token.lemma_.lower()
        if lemma in _VERB_EXCLUDE:
            continue
        if _is_ex_compound(token):
            continue
        if not token.is_alpha:
            continue
        frame_info = _frame_kind_for_verb(token)
        if not frame_info:
            continue
        kind, prep, particle = frame_info
        ccomp = _ccomp_child(token)
        has_ccomp = ccomp is not None
        if prep is not None:
            has_ccomp = False
            ccomp = None
        candidates.append(
            VerbTarget(
                token=token,
                tag=token.tag_,
                frame_kind=kind,
                lemma=lemma,
                prep_token=prep,
                particle_token=particle,
                has_that_clause=_has_that_complement(ccomp),
                has_clausal_complement=has_ccomp,
                wh_marker=_wh_marker(ccomp),
            )
        )

    return candidates


@lru_cache(maxsize=1)
def _load_clause_verb_whitelists() -> dict:
    path = _CLAUSE_VERB_WHITELIST_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _clause_verb_options(target: VerbTarget, *, zipf_thr: Optional[float]) -> List[str]:
    """
    Return a list of candidate lemmas for verbs that embed clausal complements.
    Falls back to the hardcoded list when no whitelist is available.
    """
    whitelists = _load_clause_verb_whitelists()
    that_list = whitelists.get("that") if isinstance(whitelists.get("that"), list) else []
    both_list = whitelists.get("both") if isinstance(whitelists.get("both"), list) else []
    wh_map = whitelists.get("wh") if isinstance(whitelists.get("wh"), dict) else {}
    wh_list = []
    if target.wh_marker and isinstance(target.wh_marker, str) and isinstance(wh_map, dict):
        raw = wh_map.get(target.wh_marker.strip().lower())
        if isinstance(raw, list):
            wh_list = raw

    # Wh adjacency in COCA is noisy (often wh-objects like "buy what"), so for
    # BLiMP filler-gap tasks we rely on clause-embedding verbs (that/ccomp) for
    # both that- and wh-complements.
    # Prefer the COCA-derived ``both`` set when available; it tends to exclude
    # false positives that co-occur with "that" but rarely embed wh-clauses.
    options = list(wh_list) + list(both_list) + list(that_list) + list(_THAT_VERBS)

    # Normalize + dedupe.
    normalized = []
    seen = set()
    for lemma in (options or ()):
        if not lemma or not isinstance(lemma, str):
            continue
        norm = lemma.strip().lower()
        if not norm or " " in norm or "_" in norm or "-" in norm:
            continue
        if norm in _VERB_EXCLUDE:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        normalized.append(norm)

    if zipf_thr is None:
        return normalized

    filtered = [lemma for lemma in normalized if is_rare_lemma(lemma, zipf_thr)]
    return filtered if filtered else normalized


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
            if _is_ex_compound(token):
                require_person = True
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
            if _is_ex_compound(t):
                require_person = True
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
            old_surface = _apply_noun_form(token, form, toks)
            swaps.append({
                "i": token.i,
                "old": old_surface,
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

        # Tie identical original lemmas to the same replacement lemma.
        # Keep animate/gender requirements in the key so reflexive constraints remain safe.
        def _noun_key(tok, require_person: bool, require_gender: Optional[str]):
            lemma_key = (tok.lemma_ or tok.text or "").strip().lower()
            return (lemma_key, bool(require_person), (require_gender or None))

        groups = {}
        for token, tag, require_person, require_gender in targets:
            key = _noun_key(token, require_person, require_gender)
            groups.setdefault(key, []).append((token, tag, require_person, require_gender))

        chosen_by_key = {}
        for key, group in groups.items():
            require_person = any(req_person for _, _, req_person, _ in group)
            require_gender = group[0][3]
            if require_gender and enforce_gender:
                choice_pool = gender_pools.get(require_gender)
            elif require_person:
                choice_pool = pool_person_checked
            else:
                choice_pool = pool_non_person_checked
            if not choice_pool:
                return None, []
            needed_tags = {tag for _, tag, _, _ in group}
            pool_order = (
                _weighted_order_by_zipf(choice_pool, rng, temp=zipf_temp)
                if zipf_weighted
                else list(choice_pool)
            )
            if not zipf_weighted:
                rng.shuffle(pool_order)
            chosen_lemma = None
            for lemma in pool_order:
                if all(inflect_noun(lemma, tag) for tag in needed_tags):
                    chosen_lemma = lemma
                    break
            if not chosen_lemma:
                return None, []
            chosen_by_key[key] = chosen_lemma

        for token, tag, require_person, require_gender in targets:
            key = _noun_key(token, require_person, require_gender)
            lemma = chosen_by_key.get(key)
            if not lemma:
                return None, []
            form = inflect_noun(lemma, tag)
            if not form:
                return None, []
            old_surface = _apply_noun_form(token, form, toks)
            swaps.append({"i": token.i, "old": old_surface, "new": form, "tag": tag, "lemma": lemma})

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

    def _degree_prefix(target_token, target_tag: str) -> Optional[str]:
        if target_tag not in {"JJR", "JJS"}:
            return None
        lemma = (target_token.lemma_ or "").strip().lower()
        if lemma not in {"good", "bad"}:
            return None
        return "more" if target_tag == "JJR" else "most"

    def _inflect_adj_surface(target_token, target_tag: str, lemma: str) -> Optional[str]:
        prefix = _degree_prefix(target_token, target_tag)
        if prefix:
            return f"{prefix} {lemma}"
        form = inflect_adjective(lemma, target_tag)
        if form:
            return form
        if target_tag in {"JJR", "JJS"}:
            prefix = "more" if target_tag == "JJR" else "most"
            return f"{prefix} {lemma}"
        return None

    if override_list is not None:
        if len(override_list) != len(targets):
            return None, []
        if not override_list:
            return None, []
        for (token, tag), lemma in zip(targets, override_list):
            if not lemma or not isinstance(lemma, str):
                return None, []
            surface = _inflect_adj_surface(token, tag, lemma)
            if not surface:
                return None, []
            toks[token.i] = _match_casing(token.text, surface)
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
        # Tie identical original lemmas to the same replacement lemma.
        groups = {}
        for token, tag in targets:
            key = (token.lemma_ or token.text or "").strip().lower()
            groups.setdefault(key, []).append((token, tag))

        chosen_by_key = {}
        for key, group in groups.items():
            pool_order = _weighted_order_by_zipf(pool, rng, temp=zipf_temp) if zipf_weighted else list(pool)
            if not zipf_weighted:
                rng.shuffle(pool_order)
            chosen_lemma = None
            for lemma in pool_order:
                if all(_inflect_adj_surface(token, tag, lemma) for token, tag in group):
                    chosen_lemma = lemma
                    break
            if chosen_lemma:
                chosen_by_key[key] = chosen_lemma

        for token, tag in targets:
            key = (token.lemma_ or token.text or "").strip().lower()
            lemma = chosen_by_key.get(key)
            if not lemma:
                continue
            surface = _inflect_adj_surface(token, tag, lemma)
            if not surface:
                continue
            toks[token.i] = _match_casing(token.text, surface)
            swaps.append({"i": token.i, "old": token.text, "new": toks[token.i], "tag": tag, "lemma": lemma})

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

    def _ccomp_child(tok):
        for child in tok.children:
            if child.dep_ == "ccomp":
                return child
        return None

    def _has_that(ccomp_tok) -> bool:
        if ccomp_tok is None:
            return False
        for cc_tok in ccomp_tok.subtree:
            if cc_tok.text.lower() == "that" and cc_tok.dep_ == "mark":
                return True
        return False

    def _wh_marker(ccomp_tok) -> Optional[str]:
        if ccomp_tok is None:
            return None
        for cc_tok in ccomp_tok.subtree:
            lower = cc_tok.text.lower()
            if lower in {"what", "who"}:
                return lower
        return None

    for entry in forced_targets or []:
        if isinstance(entry, dict):
            idx = entry.get("i")
            tag = entry.get("tag")
            frame = entry.get("frame")
            prep_idx = entry.get("prep_i")
            particle_idx = entry.get("particle_i")
            that_clause = bool(entry.get("that_clause"))
            forced_wh = entry.get("wh_marker")
            enforce_transitivity = bool(entry.get("enforce_transitivity"))
        elif isinstance(entry, (tuple, list)) and len(entry) >= 3:
            idx, tag, frame = entry[:3]
            prep_idx = entry[3] if len(entry) > 3 else None
            particle_idx = entry[4] if len(entry) > 4 else None
            that_clause = False
            forced_wh = None
            enforce_transitivity = False
        else:
            continue
        if not isinstance(idx, int) or idx < 0 or idx >= len(doc):
            continue
        if idx in seen:
            continue
        token = doc[idx]
        ccomp = _ccomp_child(token)
        has_ccomp = ccomp is not None
        inferred_that = _has_that(ccomp)
        inferred_wh = _wh_marker(ccomp)
        wh_marker = forced_wh if isinstance(forced_wh, str) and forced_wh.strip() else inferred_wh
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
            has_that_clause=bool(that_clause or inferred_that),
            has_clausal_complement=bool(has_ccomp or that_clause or inferred_that),
            wh_marker=wh_marker,
            enforce_transitivity=enforce_transitivity,
        ))
        seen.add(idx)
    return targets


def verb_swap_all(
    doc,
    inventory: Optional[VerbInventory],
    *,
    transitivity_inventory: Optional[VerbInventory] = None,
    verb_mode: str = "k",
    k: int = 1,
    zipf_thr: Optional[float] = None,
    zipf_weighted: bool = False,
    zipf_temp: float = 1.0,
    prefer_verb_lemmas: bool = False,
    min_verb_share: float = 0.75,
    verbiness_lexicon: str = "oewn:2021",
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

    transitivity_inv = transitivity_inventory if transitivity_inventory is not None else inventory

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

    def _verb_key(t: VerbTarget):
        return (t.lemma, bool(t.has_clausal_complement), t.frame_kind)

    chosen_by_key = {}

    for idx, target in enumerate(targets):
        verb_key = _verb_key(target)
        tied_lemma = chosen_by_key.get(verb_key) if override_list is None else None

        if target.has_clausal_complement:
            spec = override_list[idx] if override_list is not None else None
            forced_lemma = spec.get("lemma") if isinstance(spec, dict) else None
            forced_frame = spec.get("frame") if isinstance(spec, dict) else None
            if tied_lemma:
                options = [tied_lemma]
            elif forced_lemma:
                options = [forced_lemma]
            else:
                options = _clause_verb_options(target, zipf_thr=zipf_thr)
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
            if override_list is None and chosen_lemma:
                chosen_by_key[verb_key] = chosen_lemma
            toks[target.token.i] = _match_casing(target.token.text, chosen_form)
            swaps.append({
                "i": target.token.i,
                "old": target.token.text,
                "new": toks[target.token.i],
                "tag": target.tag,
                "lemma": chosen_lemma,
                "frame": forced_frame or ("that_clause" if target.has_that_clause else "clausal_complement"),
                "prep_i": None,
                "prep_old": None,
                "prep_new": None,
            })
            continue

        if override_list is not None:
            spec = override_list[idx]
            if isinstance(spec, dict):
                lemma = spec.get("lemma")
                frame_name = spec.get("frame")
                lookup = inventory.lookup(lemma, frame_name or target.frame_kind)
                if not lookup:
                    return None, []
                entry, frame = lookup
            else:
                spec = None
                entry = frame = None

        forced_kind = None
        forced_restrict = None
        has_trans = has_intr = False
        if transitivity_inv is not None:
            has_trans, has_intr = transitivity_inv.lemma_transitivity(target.lemma)
            target_transitive_like = target.frame_kind.startswith("trans") or target.frame_kind.startswith("ditrans")
            target_intransitive_like = target.frame_kind.startswith("intr")
            wants_intr_only = has_intr and not has_trans
            wants_trans_only = has_trans and not has_intr
            suffix_pp = target.frame_kind.endswith("_pp")
            if target_transitive_like and wants_intr_only:
                forced_kind = "intr_pp" if suffix_pp else "intr"
                forced_restrict = "intr_only"
            elif target_intransitive_like and wants_trans_only:
                forced_kind = "ditrans_pp" if suffix_pp else "trans"
                forced_restrict = "trans_only"

        if override_list is None or entry is None:
            prep_text = target.prep_token.text if target.prep_token is not None else None
            particle_text = target.particle_token.text if target.particle_token is not None else None
            frame_kind = forced_kind or target.frame_kind
            frame_order = [frame_kind]
            # When we need transitivity exclusivity to preserve an argument-structure
            # violation, do not back off to alternate frame families (e.g., intr -> trans).
            if forced_kind is None and not target.enforce_transitivity:
                frame_order += _FRAME_FAMILY.get(frame_kind, [])
            sample = None

            observed = _frame_kind_for_verb(target.token)
            observed_kind = None
            if isinstance(observed, tuple) and observed:
                observed_kind = observed[0]

            if tied_lemma:
                for fk in frame_order:
                    lookup = inventory.lookup(tied_lemma, fk)
                    if lookup:
                        sample = lookup
                        break

            for fk in frame_order:
                if sample:
                    break
                restrict = forced_restrict
                if restrict is None:
                    if target.enforce_transitivity:
                        if fk.startswith("intr"):
                            restrict = "intr_only"
                        elif fk.startswith("trans") or fk.startswith("ditrans"):
                            restrict = "trans_only"
                    else:
                        # Only enforce transitivity exclusivity when we are preserving an
                        # argument-structure violation (i.e., the requested frame does not
                        # match the surface arguments in the sentence).
                        if fk.startswith("intr"):
                            if observed_kind and (observed_kind.startswith("trans") or observed_kind.startswith("ditrans")):
                                restrict = "intr_only"
                        elif fk.startswith("trans") or fk.startswith("ditrans"):
                            if observed_kind and observed_kind.startswith("intr"):
                                restrict = "trans_only"
                sample = inventory.sample(
                    fk,
                    rng,
                    desired_prep=prep_text,
                    desired_particle=particle_text,
                    zipf_weighted=zipf_weighted,
                    zipf_temp=zipf_temp,
                    restrict_transitivity=restrict,
                    prefer_verb_lemmas=prefer_verb_lemmas,
                    min_verb_share=min_verb_share,
                    verbiness_lexicon=verbiness_lexicon,
                )
                if not sample and prep_text is not None:
                    # If the inventory lacks an exact preposition match, fall back to
                    # a frame match without constraining the preposition. The swapper
                    # keeps the original preposition text when it differs anyway.
                    sample = inventory.sample(
                        fk,
                        rng,
                        desired_prep=None,
                        desired_particle=particle_text,
                        zipf_weighted=zipf_weighted,
                        zipf_temp=zipf_temp,
                        restrict_transitivity=restrict,
                        prefer_verb_lemmas=prefer_verb_lemmas,
                        min_verb_share=min_verb_share,
                        verbiness_lexicon=verbiness_lexicon,
                    )
                if sample:
                    break
            if not sample:
                # Backwards-compat: if the inventory lacks explicit particle frames,
                # treat phrasal verbs as ineligible rather than failing the entire swap.
                if (
                    override_list is None
                    and forced_targets is None
                    and target.frame_kind
                    and target.frame_kind.endswith("_particle")
                ):
                    continue
                return None, []
            entry, frame = sample
            if override_list is None and entry and entry.lemma:
                chosen_by_key[verb_key] = entry.lemma

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


def verb_swap_from_pool(
    doc,
    lemma_pool,
    *,
    forced_targets=None,
    rng: Optional[random.Random] = None,
):
    """
    Swap verbs using a provided lemma pool, preserving the detected frames and
    keeping any existing prepositions/particles in place.
    """
    if rng is None:
        rng = random

    targets = _normalize_forced_verb_targets(doc, forced_targets or [])
    if not targets:
        return None, []

    pool = [
        (lemma or "").strip().lower()
        for lemma in (lemma_pool or [])
        if isinstance(lemma, str)
    ]
    pool = [lemma for lemma in pool if lemma and " " not in lemma and "_" not in lemma]
    if not pool:
        return None, []

    toks = [t.text for t in doc]
    swaps = []

    chosen_by_key = {}
    for target in targets:
        if _frame_requires_prep(target.frame_kind) and target.prep_token is None:
            return None, []
        if _frame_requires_particle(target.frame_kind) and target.particle_token is None:
            return None, []

        key = (target.lemma, target.frame_kind)
        tied = chosen_by_key.get(key)
        chosen = None
        if tied:
            form = inflect_verb(tied, target.tag)
            if form:
                chosen = (tied, form)

        if not chosen:
            pool_order = list(pool)
            rng.shuffle(pool_order)
            for lemma in pool_order:
                form = inflect_verb(lemma, target.tag)
                if form:
                    chosen = (lemma, form)
                    break
        if not chosen:
            return None, []
        lemma, form = chosen
        chosen_by_key[key] = lemma
        toks[target.token.i] = _match_casing(target.token.text, form)

        prep_old = target.prep_token.text if target.prep_token is not None else None
        prep_new = prep_old
        if _frame_requires_prep(target.frame_kind):
            toks[target.prep_token.i] = prep_old

        particle_old = target.particle_token.text if target.particle_token is not None else None
        particle_new = particle_old
        if _frame_requires_particle(target.frame_kind):
            toks[target.particle_token.i] = particle_old

        swaps.append({
            "i": target.token.i,
            "old": target.token.text,
            "new": toks[target.token.i],
            "tag": target.tag,
            "lemma": lemma,
            "frame": target.frame_kind,
            "prep_i": target.prep_token.i if target.prep_token is not None else None,
            "prep_old": prep_old,
            "prep_new": prep_new,
        })

    text = _detokenize(toks)
    if text:
        text = text[0].upper() + text[1:]
    return text, swaps
