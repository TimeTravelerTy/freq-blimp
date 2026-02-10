import random, spacy, yaml, time
from typing import Optional, Sequence
from pathlib import Path
from spacy.lang.en.stop_words import STOP_WORDS as _STOP_WORDS
from wordfreq import zipf_frequency
from .io import load_blimp, load_blimp_field_map, write_jsonl
from .becl import load_becl_tsv
from .quantifier import load_quant_rules, requirement
from .edits import (
    adjective_swap_all,
    noun_swap_all,
    candidate_adjectives,
    candidate_nouns,
    candidate_verbs,
    reflexive_subject_indices,
    verb_swap_all,
    verb_swap_from_pool,
)
from .rarity import is_rare_lemma
from .lemma_bank import is_person_noun, is_location_noun, is_time_noun, is_proper_noun
from .gender_lexicon import load_gender_lexicon
from .verb_inventory import VerbInventory, load_verb_inventory, wordnet_pos_counts
from .zipf_aggregates import add_zipf_aggregates


def _format_duration(seconds):
    if seconds is None or seconds == float("inf"):
        return "?"
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _print_progress(done, total, start_time, last_update, width=30):
    if not total:
        return time.time()
    now = time.time()
    if done < total and now - last_update < 0.5:
        return last_update
    ratio = min(1.0, done / total)
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = now - start_time
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else float("inf")
    msg = f"\r[{bar}] {done}/{total} ({ratio*100:5.1f}%)  elapsed { _format_duration(elapsed) }  eta { _format_duration(eta) }"
    print(msg, end="", flush=True)
    if done == total:
        print()
    return now


def _normalize_blimp_field(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    val = str(value).strip()
    if not val:
        return None
    if val == "syntax_semantics":
        return "syntax/semantics"
    return val


_SWAP_KEEP_KEYS = ("i", "old", "new", "tag", "lemma")


def _slim_swap_entry(entry, pos_label: str):
    if not isinstance(entry, dict):
        return None
    out = {}
    for key in _SWAP_KEEP_KEYS:
        val = entry.get(key)
        if val is not None:
            out[key] = val
    if "old" not in out or "new" not in out:
        return None
    out["pos"] = pos_label
    return out


def _merge_and_slim_swaps(noun_swaps, adj_swaps, verb_swaps):
    merged = []
    for pos_label, swaps in (
        ("noun", noun_swaps),
        ("adjective", adj_swaps),
        ("verb", verb_swaps),
    ):
        for entry in swaps or ():
            slim = _slim_swap_entry(entry, pos_label)
            if slim is not None:
                merged.append(slim)
    merged.sort(key=lambda it: (it.get("i", -1), it.get("pos", "")))
    return merged


_ACRONYM_VOWELS = set("aeiou")
_VULGAR_NOUNS = {
    "asshole",
    "bitch",
    "cock",
    "cunt",
    "dick",
    "dildo",
    "fag",
    "faggot",
    "fuck",
    "motherfucker",
    "penis",
    "pussy",
    "slut",
    "twat",
    "vagina",
    "whore",
}

_SUBJECT_RAISING_VERBS = [
    "seem", "appear", "happen", "tend", "look", "prove", "remain", "continue",
    "begin", "start", "stop", "fail", "threaten", "promise", "cease", "come",
    "grow", "commence", "chance", "transpire", "hap",
    "bid_fair", "loom", "deserve",
]

_OBJECT_RAISING_VERBS = [
    "want",
    "need",
    "like",
    "expect",
    "hate",
    "prefer",
    "mean",
    "wish",
    "imagine",
    "guess",
    "suppose",
    "reckon",
    "believe",
    "find",
    "feel",
    "know",
    "consider",
    "allow",
    "cause",
    "understand",
    "assume",
    "presume",
    "suspect",
    "claim",
    "maintain",
    "declare",
    "state",
    "report",
    "judge",
    "hold",
    "deem",
    "view",
    "perceive",
    "observe",
    "note",
    "reveal",
    "prove",
    "show",
    "require",
    "permit",
    "order",
    "command",
    "forbid",
    "authorize",
    "intend",
    "discover",
    "estimate",
    "calculate",
    "acknowledge",
    "admit",
    "affirm",
    "allege",
    "anticipate",
    "apprehend",
    "ascertain",
    "assert",
    "avow",
    "certify",
    "conceive",
    "conjecture",
    "deduce",
    "demonstrate",
    "determine",
    "discern",
    "enable",
    "envisage",
    "envision",
    "foresee",
    "grant",
    "guarantee",
    "hypothesize",
    "infer",
    "posit",
    "postulate",
    "predict",
    "proclaim",
    "pronounce",
    "recognize",
    "recollect",
    "remember",
    "stipulate",
    "surmise",
    "verify",
    "warrant",
]

_RAISING_ADJECTIVES = [
    "likely", "unlikely", "sure", "certain", "supposed", "going", "bound",
    "apt", "liable", "set", "due", "prone", "fated", "destined", "doomed",
    "slated", "scheduled", "wont", "calculated", "rumored", "reputed",
    "alleged", "poised", "expected", "anticipated", "predicted", "estimated",
    "projected", "forecast", "reported", "believed", "thought", "assumed",
    "known", "understood", "claimed", "presumed", "suspected", "deemed",
    "said", "acknowledged", "recognized", "accepted", "conceded", "noted",
    "asserted", "inferred", "deduced", "established", "demonstrated",
    "posited", "postulated", "hypothesized", "proposed", "required", "needed",
    "mandated", "stipulated", "specified", "feared", "hoped",
]

_CONTROL_ADJECTIVES = [
    "eager", "ready", "willing", "unable", "able", "happy", "sad", "afraid",
    "anxious", "excited", "careful", "lucky", "reluctant", "hesitant", "keen",
    "loath", "quick", "slow", "prepared", "determined", "motivated",
    "impatient", "intent", "disinclined", "indisposed", "solicitous",
    "studious", "importunate", "powerless", "impotent", "desperate",
    "resolved", "fain", "minded", "chary",
]

_TOUGH_ADJECTIVES = [
    "easy",
    "hard",
    "fun",
    "nice",
    "good",
    "bad",
    "cool",
    "great",
    "safe",
    "boring",
    "weird",
    "tough",
    "simple",
    "scary",
    "sweet",
    "pleasant",
    "unpleasant",
    "lovely",
    "wonderful",
    "terrible",
    "horrible",
    "awful",
    "difficult",
    "dangerous",
    "impossible",
    "interesting",
    "exciting",
    "amazing",
    "relaxing",
    "stressful",
    "awkward",
    "strange",
    "tiring",
    "painful",
    "annoying",
    "tricky",
    "risky",
    "useful",
    "pointless",
    "delightful",
    "enjoyable",
    "fascinating",
    "engaging",
    "entertaining",
    "challenging",
    "complicated",
    "confusing",
    "frustrating",
    "exhausting",
    "draining",
    "demanding",
    "intimidating",
    "charming",
    "refreshing",
    "comforting",
    "soothing",
    "calming",
    "satisfying",
    "disappointing",
    "disturbing",
    "shocking",
    "embarrassing",
    "frightening",
    "tedious",
    "tiresome",
    "dreadful",
    "costly",
    "wasteful",
    "stimulating",
    "enlightening",
    "educational",
    "informative",
    "illuminating",
    "effortless",
    "breezy",
    "laborious",
    "arduous",
    "burdensome",
    "onerous",
    "strenuous",
    "exasperating",
    "infuriating",
    "maddening",
    "monotonous",
    "unbearable",
    "intolerable",
    "insufferable",
    "excruciating",
    "hazardous",
    "treacherous",
    "perilous",
    "formidable",
    "loathsome",
    "repulsive",
    "irksome",
    "wearisome",
    "cumbersome",
    "unsettling",
    "unnerving",
    "disarming",
    "invigorating",
    "uplifting",
    "edifying",
    "fruitless",
    "futile",
]

def _control_raising_adj_targets(doc):
    """
    Detect predicate modifiers (typically ADJ, sometimes ADV) in control/raising
    constructions so we can swap them even when they are not attributive (amod).

    Handles:
    - expletive-there + be + ADJ + to ...
    - it + to + be + ADJ + (that ...)
    - copular be + ADJ/ADV + to + VERB (tough/raising-like surface forms)
    """
    there_indices = {
        t.i
        for t in doc
        if t.text.lower() == "there" and t.dep_ in {"expl", "nsubj", "nsubjpass", "csubj", "csubjpass"}
    }

    def _has_to_infinitive(be_tok) -> bool:
        if be_tok is None:
            return False
        if any(child.text.lower() == "to" and child.dep_ in {"aux", "mark"} for child in be_tok.children):
            return True
        for child in be_tok.children:
            if child.dep_ != "xcomp":
                continue
            if any(gc.text.lower() == "to" and gc.dep_ in {"aux", "mark"} for gc in child.children):
                return True
        return False

    def _has_to_infinitive_via_predicate(pred_tok) -> bool:
        if pred_tok is None:
            return False
        for child in pred_tok.children:
            if child.dep_ != "xcomp":
                continue
            if any(gc.text.lower() == "to" and gc.dep_ in {"aux", "mark"} for gc in child.children):
                return True
        return False

    targets = []
    seen = set()

    for tok in doc:
        if tok.pos_ not in {"ADJ", "ADV"}:
            continue
        if tok.dep_ not in {"acomp", "advmod"}:
            continue
        head = tok.head
        if head is None or head.lemma_.lower() != "be":
            continue
        if not (_has_to_infinitive(head) or _has_to_infinitive_via_predicate(tok)):
            continue
        if tok.i in seen:
            continue
        # Preserve the original expletive-there constraint for those cases.
        if there_indices and not any(idx < tok.i for idx in there_indices):
            continue
        seen.add(tok.i)
        targets.append(tok)

    targets.sort(key=lambda t: t.i)
    return targets


def _unique(seq):
    seen = set()
    out = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _control_verb_pool(inventory):
    if inventory is None:
        return tuple()
    lemmas = []
    seen = set()
    for entry in inventory.entries:
        if any(frame.kind == "trans" for frame in entry.frames):
            if entry.lemma not in seen:
                lemmas.append(entry.lemma)
                seen.add(entry.lemma)
    return tuple(lemmas)


def _raising_verb_list_for_cfg(cfg) -> Sequence[str]:
    cfg_key = str(cfg or "").strip().lower()
    if "object_raising" in cfg_key:
        return _OBJECT_RAISING_VERBS
    return _SUBJECT_RAISING_VERBS


def _has_to_infinitive_xcomp(verb_tok) -> bool:
    if verb_tok is None:
        return False
    for child in verb_tok.children:
        if child.dep_ != "xcomp":
            continue
        if any(gc.text.lower() == "to" and gc.dep_ in {"aux", "mark"} for gc in child.children):
            return True
    return False


def _mark_turn_out_as_raising(spec: dict, doc) -> None:
    if not isinstance(spec, dict) or doc is None:
        return
    idx = spec.get("i")
    particle_idx = spec.get("particle_i")
    if not isinstance(idx, int) or not isinstance(particle_idx, int):
        return
    if idx < 0 or idx >= len(doc) or particle_idx < 0 or particle_idx >= len(doc):
        return
    verb_tok = doc[idx]
    particle_tok = doc[particle_idx]
    if verb_tok is None or particle_tok is None:
        return
    if verb_tok.lemma_.lower() != "turn":
        return
    if (particle_tok.text or "").strip().lower() != "out":
        return
    if not _has_to_infinitive_xcomp(verb_tok):
        return
    frame = str(spec.get("frame") or "")
    if frame.endswith("_particle"):
        spec["frame"] = frame[: -len("_particle")]
    spec["drop_particle"] = True


def _looks_like_acronym(token: str) -> bool:
    if not token:
        return False
    if any(ch.isdigit() for ch in token):
        return True
    letters = [ch for ch in token if ch.isalpha()]
    if not letters:
        return False
    if not any(ch in _ACRONYM_VOWELS for ch in letters):
        return True
    return False


def _noun_target_tag(token):
    """
    Return an NN/NNS tag that reflects the token's number features.
    """
    number = token.morph.get("Number")
    if "Plur" in number:
        return "NNS"
    if "Sing" in number:
        return "NN"
    if token.tag_ in {"NN", "NNS"}:
        return token.tag_
    return "NNS" if token.text.lower().endswith("s") else "NN"


def _prepare_lemma_pool(candidates, zipf_thr=None, *, zipf_min=None, skip_acronyms=True):
    """
    Deduplicate and optionally rarity-filter a list of string lemmas.
    """
    if not candidates:
        return tuple()
    seen = set()
    filtered = []
    for lemma in candidates:
        if not lemma:
            continue
        norm = lemma.strip().lower()
        if not norm or norm in seen:
            continue
        if skip_acronyms and _looks_like_acronym(norm):
            continue
        if zipf_thr is not None and not is_rare_lemma(norm, zipf_thr, zipf_min):
            continue
        seen.add(norm)
        filtered.append(norm)
    return tuple(filtered)


def _noun_dominant(lemma: str, *, lexicon: str = "oewn:2021") -> bool:
    v, n, a, r = wordnet_pos_counts(lemma, lexicon=lexicon)
    if n <= 0:
        return False
    if n < max(v, a, r):
        return False
    if (v + a + r) == 0:
        if n == 1:
            return False
        if n <= 2 and zipf_frequency(lemma, "en") >= 4.8:
            return False
    return True


def _required_gender(token, reflexive_meta, gender_lex):
    info = reflexive_meta.get(token.i) if isinstance(reflexive_meta, dict) else None
    if info:
        pronoun_gender = info.get("gender")
        if pronoun_gender:
            return pronoun_gender
    if gender_lex is None:
        return None
    return gender_lex.lemma_gender(token.lemma_.lower())


def _normalize_swap_targets(targets):
    """
    Normalize a user-provided target spec into a set of labels like
    {"nouns", "adjectives"}.
    """
    if targets is None:
        return set()
    if isinstance(targets, str):
        targets_list = [targets]
    else:
        targets_list = list(targets)
    out = set()
    for entry in targets_list:
        if not entry:
            continue
        lower = str(entry).strip().lower()
        if lower in {"noun", "nouns"}:
            out.add("nouns")
        elif lower in {"adj", "adjective", "adjectives"}:
            out.add("adjectives")
        elif lower in {"verb", "verbs"}:
            out.add("verbs")
        elif lower in {"both", "all"}:
            out.update({"nouns", "adjectives", "verbs"})
    return out


def _filter_pool_with_fallback(items, zipf_thr, zipf_min=None):
    """
    Keep only lemmas that pass the zipf rarity check; if the pool is too small,
    relax the threshold incrementally instead of falling back to the full list.
    """
    pool = tuple(items or ())
    if not pool or zipf_thr is None:
        return pool
    min_size = min(8, len(pool))
    step = 0.3
    max_steps = 6

    best = tuple(lemma for lemma in pool if is_rare_lemma(lemma, zipf_thr, zipf_min))
    if len(best) >= min_size:
        return best

    thr = zipf_thr
    for _ in range(max_steps):
        thr += step
        cand = tuple(lemma for lemma in pool if is_rare_lemma(lemma, thr, zipf_min))
        if len(cand) > len(best):
            best = cand
        if len(best) >= min_size:
            break
    return best


def build_pilot(tier_cfg_path, becl_path, quant_cfg_path, out_path,
                noun_mode="all", k=2, zipf_thr=3.4, rare_lemmas=None,
                adj_mode="all", adj_zipf_thr=3.4, rare_adj_lemmas=None,
                verb_mode="k", verb_zipf_thr=3.4, rare_verb_lemmas=None,
                verb_zipf_min=None,
                noun_zipf_min=None,
                adj_zipf_min=None,
                verb_min_verb_share: float = 1.0,
                verbiness_lexicon: str = "oewn:2021",
                swap_targets=("nouns",), seed=0, show_progress=True,
                record_limit: Optional[int] = None,
                phenomenon_filter: Optional[Sequence[str]] = None,
                exclude_proper_nouns: bool = True,
                rare_name_path="data/external/rare_names.tsv",
                name_lookup_path="data/external/name_gender_lookup.tsv",
                name_conf=0.75,
                gender_lexicon_path="data/processed/wiktionary_gender_lemmas.json",
                verb_inventory_path=None,
                verb_inventory=None,
                zipf_weighted_sampling: bool = False,
                zipf_temp: float = 1.0,
                spacy_n_process: int = 1,
                spacy_batch_size: int = 128,
                match_token_count: bool = False,
                swap_tokenizer: Optional[str] = None,
                original_out_path: Optional[Path] = None):
    if record_limit is not None:
        try:
            record_limit = max(0, int(record_limit))
        except Exception:
            record_limit = 0
        # For small capped runs, multi-process spaCy startup dominates runtime.
        if record_limit and spacy_n_process and spacy_n_process > 1:
            spacy_n_process = 1

    nlp = spacy.load("en_core_web_sm")
    with open(tier_cfg_path, encoding="utf-8") as f:
        tasks_cfg = yaml.safe_load(f)
    becl_map = load_becl_tsv(becl_path)
    qrules = load_quant_rules(quant_cfg_path)
    records = []
    gender_lex = load_gender_lexicon(gender_lexicon_path)

    swap_targets_set = _normalize_swap_targets(swap_targets)
    if not swap_targets_set:
        swap_targets_set = {"nouns"}
    do_noun_swaps = "nouns" in swap_targets_set
    do_adj_swaps = "adjectives" in swap_targets_set
    do_verb_swaps = "verbs" in swap_targets_set

    verb_inventory_obj = verb_inventory
    if verb_inventory_obj is None and verb_inventory_path:
        try:
            verb_inventory_obj = load_verb_inventory(verb_inventory_path)
        except FileNotFoundError:
            verb_inventory_obj = VerbInventory(tuple())

    verb_inventory_transitivity = verb_inventory_obj

    if do_verb_swaps:
        if verb_inventory_obj is None:
            do_verb_swaps = False
        else:
            sampling_inventory = verb_inventory_obj
            if rare_verb_lemmas:
                sampling_inventory = sampling_inventory.restrict_to(rare_verb_lemmas)
            sampling_inventory = sampling_inventory.filter_by_zipf(
                verb_zipf_thr,
                zipf_min=verb_zipf_min,
            )
            if sampling_inventory.is_empty():
                do_verb_swaps = False
            else:
                verb_inventory_obj = sampling_inventory
    else:
        verb_inventory_obj = None
        verb_inventory_transitivity = None

    verb_sampling_kwargs = {
        "prefer_verb_lemmas": True,
        "min_verb_share": float(verb_min_verb_share),
        "verbiness_lexicon": verbiness_lexicon,
    }
    token_match_kwargs = {
        "match_token_count": match_token_count,
        "tokenizer_name": swap_tokenizer,
    }

    def _take_first(dataset, n: int):
        if n is None:
            return dataset
        n = max(0, int(n))
        if hasattr(dataset, "select"):
            # HuggingFace Datasets: `ds[:n]` returns a column dict, not a Dataset.
            return dataset.select(range(n))
        return dataset[:n]

    if do_noun_swaps:
        pool = _prepare_lemma_pool(rare_lemmas, zipf_thr, zipf_min=noun_zipf_min)
        if pool:
            pool_no_stop = tuple(lemma for lemma in pool if lemma not in _STOP_WORDS)
            if pool_no_stop:
                pool = pool_no_stop
        if pool:
            pool_no_vulgar = tuple(lemma for lemma in pool if lemma not in _VULGAR_NOUNS)
            if pool_no_vulgar:
                pool = pool_no_vulgar
        if pool:
            noun_dominant_pool = tuple(lemma for lemma in pool if _noun_dominant(lemma))
            if noun_dominant_pool:
                pool = noun_dominant_pool
        rare_pool = tuple(
            lemma
            for lemma in pool
            if (not exclude_proper_nouns or not is_proper_noun(lemma))
            if not is_location_noun(lemma)
            if not is_time_noun(lemma)
        )
    else:
        rare_pool = tuple()

    rare_gender_map = {}
    if do_noun_swaps and gender_lex and gender_lex.has_data():
        for gender in ("female", "male"):
            lemmas = []
            seen_gender = set()
            for lemma in gender_lex.iter_gender(gender):
                if not lemma:
                    continue
                norm = lemma.strip().lower()
                if not norm or norm in seen_gender:
                    continue
                if zipf_thr is not None and not is_rare_lemma(norm, zipf_thr):
                    continue
                seen_gender.add(norm)
                lemmas.append(norm)
            if lemmas:
                rare_gender_map[gender] = tuple(lemmas)

    lexicon_person = []
    if do_noun_swaps:
        for gender in ("female", "male"):
            lexicon_person.extend(rare_gender_map.get(gender, ()))
        lexicon_person = _unique(lexicon_person)

        rare_person_pool_list = list(lexicon_person)
        seen_person = set(rare_person_pool_list)
        # Add person nouns from the full pool even if we excluded them from the
        # general rare_pool so reflexive constraints can still be satisfied.
        for lemma in pool:
            if lemma in seen_person:
                continue
            if lemma in _VULGAR_NOUNS:
                continue
            if (
                (not exclude_proper_nouns or not is_proper_noun(lemma))
                and not is_location_noun(lemma)
                and not is_time_noun(lemma)
                and is_person_noun(lemma)
            ):
                rare_person_pool_list.append(lemma)
                seen_person.add(lemma)
        rare_person_pool = tuple(rare_person_pool_list)
    else:
        rare_person_pool = tuple()

    if do_adj_swaps:
        rare_adj_pool = _prepare_lemma_pool(rare_adj_lemmas, adj_zipf_thr, zipf_min=adj_zipf_min)
        # Prefer lemmas that are primarily adjectives (avoid adjective-noun
        # ambiguities like "independent" / "wild" being treated as adjectives).
        if rare_adj_pool:
            def _adj_dominant(lemma: str) -> bool:
                v, n, a, r = wordnet_pos_counts(lemma, lexicon="oewn:2021")
                if a <= 0:
                    return False
                if n >= a or v >= a or r >= a:
                    return False
                # If the lemma is also a noun, require a stronger adjective signal
                # (reduces adjective-noun ambiguities like "criminal" as an adjective).
                if n > 0 and a < 2:
                    return False
                return True

            filtered_adj = tuple(w for w in rare_adj_pool if _adj_dominant(w))
            if filtered_adj:
                rare_adj_pool = filtered_adj
        if not rare_adj_pool:
            do_adj_swaps = False
    else:
        rare_adj_pool = tuple()

    # Build a lightweight config list (don't load every BLiMP subdataset up
    # front; that can be slow, especially when record_limit is small).
    allowed_phenomena = None
    if phenomenon_filter:
        allowed_phenomena = {
            str(entry).strip().lower()
            for entry in phenomenon_filter
            if entry and str(entry).strip()
        }
    cfg_list = []
    for group_name, meta in (tasks_cfg or {}).items():
        phenomenon = meta["phenomenon"]
        if allowed_phenomena:
            group_key = str(group_name).strip().lower()
            phen_key = str(phenomenon).strip().lower()
            if group_key not in allowed_phenomena and phen_key not in allowed_phenomena:
                continue
        configs = meta.get("configs", [])
        if not configs:
            continue
        for cfg in configs:
            cfg_list.append((group_name, phenomenon, cfg))

    field_map = load_blimp_field_map(configs=[cfg for _, _, cfg in cfg_list])

    total_items = 0
    if record_limit:
        total_items = record_limit
    else:
        for _, _, cfg in cfg_list:
            ds = load_blimp(cfg)
            total_items += len(ds)

    processed = 0
    start_time = time.time()
    last_update = start_time
    stop_early = False
    if show_progress:
        last_update = _print_progress(processed, total_items, start_time, last_update)

    for group_name, phenomenon, cfg in cfg_list:
        if record_limit and len(records) >= record_limit:
            break
        ds = load_blimp(cfg)
        if record_limit:
            remaining = record_limit - len(records)
            if remaining <= 0:
                break
            ds = _take_first(ds, remaining)
        # Parse all good/bad sentences for this subtask in batches to reduce
        # spaCy overhead and allow multi-process parsing.
        def _iter_good_bad(dataset):
            for record in dataset:
                yield record["sentence_good"]
                yield record["sentence_bad"]

        docs_iter = nlp.pipe(_iter_good_bad(ds), batch_size=spacy_batch_size, n_process=spacy_n_process)

        is_control_raising = phenomenon == "control_raising"
        control_verb_pool = _control_verb_pool(verb_inventory_obj) if is_control_raising else tuple()

        for i, r in enumerate(ds):
                g, b = r["sentence_good"], r["sentence_bad"]
                gdoc_orig = next(docs_iter)
                bdoc_orig = next(docs_iter)
                gdoc_working = gdoc_orig
                bdoc_working = bdoc_orig
                g_reflexive_subjects = reflexive_subject_indices(gdoc_orig)
                b_reflexive_subjects = reflexive_subject_indices(bdoc_orig)

                # Quantifier requirement per sentence
                req_g = requirement(gdoc_orig, qrules)  # None/COUNT/MASS
                req_b = requirement(bdoc_orig, qrules)
                if phenomenon == "quantifiers":
                    if req_g is None and req_b in {"COUNT", "MASS"}:
                        req_g = req_b
                    elif req_b is None and req_g in {"COUNT", "MASS"}:
                        req_b = req_g

                # Per-pair RNG so Good/Bad use the same sequence of choices
                pair_seed = (hash((group_name, cfg, i)) + seed) & 0xFFFFFFFF

                g_swaps, b_swaps = [], []
                g_adj_swaps, b_adj_swaps = [], []
                g_verb_swaps, b_verb_swaps = [], []
                g_variant = g
                b_variant = b
                verb_changed = False
                noun_changed = False
                adj_changed = False
                swap_failures = []

                def _note_swap_failure(reason: str):
                    if reason and reason not in swap_failures:
                        swap_failures.append(reason)

                def _note_swap_reason(reason: Optional[str]) -> bool:
                    if reason:
                        _note_swap_failure(reason)
                        return True
                    return False

                # Stage 0: verb swaps
                # Skip verb swaps for irregular-forms tasks (where the target
                # contrast is morphological).
                if do_verb_swaps and phenomenon != "irregular_forms":
                    # Ensure these are initialized each iteration; otherwise they can
                    # leak across loop iterations when a branch doesn't assign them.
                    g_verb_candidates = []
                    b_verb_candidates = []
                    verb_attempted = False
                    verb_candidates_present = False
                    verb_matches_present = False
                    verb_failure_reported = False

                    def _find_passive_head(doc):
                        for tok in doc:
                            if tok.pos_ != "VERB" or not tok.tag_.startswith("VB"):
                                continue
                            if any(child.dep_ == "auxpass" for child in tok.children):
                                return tok
                        for tok in doc:
                            if tok.pos_ != "VERB" or not tok.tag_.startswith("VB"):
                                continue
                            if tok.dep_ == "ROOT" and tok.tag_ == "VBN":
                                return tok
                        for tok in doc:
                            if tok.pos_ != "VERB" or not tok.tag_.startswith("VB"):
                                continue
                            if tok.tag_ == "VBN":
                                return tok
                        return None

                    # Special-case passive tasks: spaCy doesn't expose the underlying
                    # transitivity after passivization, so force a transitive frame
                    # for the good sentence and an intransitive frame for the bad
                    # sentence to preserve the intended violation.
                    if cfg in {"passive_1", "passive_2"}:
                        g_head = _find_passive_head(gdoc_working)
                        b_head = _find_passive_head(bdoc_working)
                        if g_head is not None and b_head is not None:
                            rng_good = random.Random(pair_seed - 1)
                            rng_bad = random.Random(pair_seed - 2)
                            g_forced = [{
                                "i": g_head.i,
                                "tag": g_head.tag_,
                                "frame": "trans",
                                "lemma": g_head.lemma_.lower(),
                                "prep_i": None,
                                "particle_i": None,
                                "that_clause": False,
                            }]
                            b_forced = [{
                                "i": b_head.i,
                                "tag": b_head.tag_,
                                "frame": "intr",
                                "lemma": b_head.lemma_.lower(),
                                "prep_i": None,
                                "particle_i": None,
                                "that_clause": False,
                            }]
                            g_verb_variant, g_verb_swaps, g_verb_reason = verb_swap_all(
                                gdoc_working,
                                verb_inventory_obj,
                                transitivity_inventory=verb_inventory_transitivity,
                                verb_mode="all",
                                k=1,
                                zipf_thr=verb_zipf_thr,
                                zipf_min=verb_zipf_min,
                                zipf_weighted=zipf_weighted_sampling,
                                zipf_temp=zipf_temp,
                                **verb_sampling_kwargs,
                                rng=rng_good,
                                forced_targets=g_forced,
                                **token_match_kwargs,
                            )
                            verb_attempted = True
                            if _note_swap_reason(g_verb_reason):
                                verb_failure_reported = True
                            b_verb_variant, b_verb_swaps, b_verb_reason = verb_swap_all(
                                bdoc_working,
                                verb_inventory_obj,
                                transitivity_inventory=verb_inventory_transitivity,
                                verb_mode="all",
                                k=1,
                                zipf_thr=verb_zipf_thr,
                                zipf_min=verb_zipf_min,
                                zipf_weighted=zipf_weighted_sampling,
                                zipf_temp=zipf_temp,
                                **verb_sampling_kwargs,
                                rng=rng_bad,
                                forced_targets=b_forced,
                                **token_match_kwargs,
                            )
                            verb_attempted = True
                            if _note_swap_reason(b_verb_reason):
                                verb_failure_reported = True
                            if not (
                                g_verb_variant
                                and b_verb_variant
                                and g_verb_swaps
                                and b_verb_swaps
                                and len(g_verb_swaps) == len(b_verb_swaps)
                            ):
                                g_verb_swaps = []
                                b_verb_swaps = []
                            else:
                                g_variant = g_verb_variant
                                b_variant = b_verb_variant
                                verb_changed = True
                                gdoc_working = nlp(g_variant)
                                bdoc_working = nlp(b_variant)
                            # Skip the generic verb-swap logic for passives when
                            # the forced-swap path succeeds.
                            g_verb_candidates = []
                            b_verb_candidates = []
                        else:
                            # Fallback when we can't locate a passive head.
                            g_verb_candidates = candidate_verbs(gdoc_working)
                            g_verb_candidates.sort(key=lambda t: t.token.i)
                            b_verb_candidates = candidate_verbs(bdoc_working)
                            b_verb_candidates.sort(key=lambda t: t.token.i)
                    else:
                        g_verb_candidates = candidate_verbs(gdoc_working)
                        g_verb_candidates.sort(key=lambda t: t.token.i)
                        b_verb_candidates = candidate_verbs(bdoc_working)
                        b_verb_candidates.sort(key=lambda t: t.token.i)

                    verb_candidates_present = bool(g_verb_candidates or b_verb_candidates)

                    verb_matches = []
                    used_bad_verbs = set()
                    matched_by_identity = True
                    unmatched_good = []
                    unmatched_bad = []

                    def _match_verb(g_target):
                        for b_target in b_verb_candidates:
                            if b_target.token.i in used_bad_verbs:
                                continue
                            if b_target.lemma == g_target.lemma:
                                return b_target
                        for b_target in b_verb_candidates:
                            if b_target.token.i in used_bad_verbs:
                                continue
                            if b_target.token.text.lower() == g_target.token.text.lower():
                                return b_target
                        return None

                    for g_target in g_verb_candidates:
                        b_match = _match_verb(g_target)
                        if b_match is None:
                            unmatched_good.append(g_target)
                            continue
                        verb_matches.append((
                            {
                                "i": g_target.token.i,
                                "tag": g_target.tag,
                                "frame": g_target.frame_kind,
                                "lemma": g_target.lemma,
                                "prep_i": g_target.prep_token.i if g_target.prep_token is not None else None,
                                "particle_i": g_target.particle_token.i if g_target.particle_token is not None else None,
                                "that_clause": g_target.has_that_clause,
                                "wh_marker": g_target.wh_marker,
                            },
                            {
                                "i": b_match.token.i,
                                "tag": b_match.tag,
                                "frame": b_match.frame_kind,
                                "lemma": b_match.lemma,
                                "prep_i": b_match.prep_token.i if b_match.prep_token is not None else None,
                                "particle_i": b_match.particle_token.i if b_match.particle_token is not None else None,
                                "that_clause": b_match.has_that_clause,
                                "wh_marker": b_match.wh_marker,
                            }
                        ))
                        used_bad_verbs.add(b_match.token.i)

                    for b_target in b_verb_candidates:
                        if b_target.token.i not in used_bad_verbs:
                            unmatched_bad.append(b_target)

                    if unmatched_good and unmatched_bad:
                        limit = min(len(unmatched_good), len(unmatched_bad))
                        for g_target, b_target in zip(unmatched_good[:limit], unmatched_bad[:limit]):
                            verb_matches.append((
                                {
                                    "i": g_target.token.i,
                                    "tag": g_target.tag,
                                    "frame": g_target.frame_kind,
                                    "lemma": g_target.lemma,
                                    "prep_i": g_target.prep_token.i if g_target.prep_token is not None else None,
                                    "particle_i": g_target.particle_token.i if g_target.particle_token is not None else None,
                                    "that_clause": g_target.has_that_clause,
                                },
                                {
                                    "i": b_target.token.i,
                                    "tag": b_target.tag,
                                    "frame": b_target.frame_kind,
                                    "lemma": b_target.lemma,
                                    "prep_i": b_target.prep_token.i if b_target.prep_token is not None else None,
                                    "particle_i": b_target.particle_token.i if b_target.particle_token is not None else None,
                                    "that_clause": b_target.has_that_clause,
                                }
                            ))
                        matched_by_identity = False

                    if verb_mode == "k" and verb_matches:
                        limit = max(0, min(k, len(verb_matches)))
                        verb_matches = verb_matches[:limit]

                    verb_matches_present = bool(verb_matches)

                    if is_control_raising and verb_matches:
                        for g_spec, b_spec in verb_matches:
                            _mark_turn_out_as_raising(g_spec, gdoc_working)
                            _mark_turn_out_as_raising(b_spec, bdoc_working)

                    g_target_specs = []
                    b_target_specs = []
                    if verb_matches:
                        if is_control_raising and matched_by_identity and not unmatched_good and not unmatched_bad:
                            pass
                        else:
                            for g_spec, b_spec in verb_matches:
                                g_entry = dict(g_spec)
                                b_entry = dict(b_spec)
                                g_entry["that_clause"] = bool(g_spec.get("that_clause"))
                                b_entry["that_clause"] = bool(b_spec.get("that_clause"))
                                # For filler-gap tasks, the bad sentence may contain the wh-word
                                # ("what"/"who") even when the good sentence uses "that". Use the
                                # union marker so both sides sample from wh-compatible verbs.
                                shared_wh = g_spec.get("wh_marker") or b_spec.get("wh_marker")
                                if shared_wh:
                                    g_entry["wh_marker"] = shared_wh
                                    b_entry["wh_marker"] = shared_wh

                                # Argument-structure subtasks often encode the grammaticality
                                # contrast via (mis)matching a verb's transitivity with its
                                # surface arguments. spaCy frame detection can be noisy, so
                                # for these subtasks we force the intended frame labels.
                                if phenomenon == "argument_structure":
                                    def _force_frame(spec: dict, base: str) -> str:
                                        frame = str(spec.get("frame") or "")
                                        suffix = "_particle" if frame.endswith("_particle") else ""
                                        return f"{base}{suffix}"

                                    if cfg in {"causative", "transitive"}:
                                        g_entry["frame"] = _force_frame(g_entry, "trans")
                                        b_entry["frame"] = _force_frame(b_entry, "intr")
                                        # The bad side needs exclusivity so the violation (intr + dobj)
                                        # stays ungrammatical after swapping.
                                        b_entry["enforce_transitivity"] = True
                                    elif cfg in {"inchoative", "intransitive"}:
                                        g_entry["frame"] = _force_frame(g_entry, "intr")
                                        b_entry["frame"] = _force_frame(b_entry, "trans")
                                        # The bad side needs exclusivity so the violation (trans w/o dobj)
                                        # stays ungrammatical after swapping.
                                        b_entry["enforce_transitivity"] = True
                                    elif cfg == "drop_argument":
                                        g_entry["frame"] = _force_frame(g_entry, "intr")
                                        # If the bad sentence strands a preposition (e.g., "talk to ."),
                                        # keep intr_pp; otherwise force a transitive verb with its object dropped.
                                        b_frame = str(b_entry.get("frame") or "")
                                        if not b_frame.startswith("intr_pp"):
                                            b_entry["frame"] = _force_frame(b_entry, "trans")
                                            b_entry["enforce_transitivity"] = True

                                g_target_specs.append(g_entry)
                                b_target_specs.append(b_entry)
                        rng_verbs = random.Random(pair_seed - 1)
                        if is_control_raising:
                            g_verb_variant = None
                            b_verb_variant = None
                            raising_verbs = _raising_verb_list_for_cfg(cfg)
                            raising_pool = _filter_pool_with_fallback(
                                raising_verbs, verb_zipf_thr, zipf_min=verb_zipf_min
                            )
                            control_pool = _filter_pool_with_fallback(
                                control_verb_pool or raising_verbs, verb_zipf_thr, zipf_min=verb_zipf_min
                            )
                            same_g_specs, same_b_specs = [], []
                            diff_g_specs, diff_b_specs = [], []
                            for g_spec, b_spec in verb_matches:
                                g_lemma = (g_spec.get("lemma") or "").lower()
                                b_lemma = (b_spec.get("lemma") or "").lower()
                                if g_lemma and g_lemma == b_lemma:
                                    same_g_specs.append(g_spec)
                                    same_b_specs.append(b_spec)
                                else:
                                    diff_g_specs.append(g_spec)
                                    diff_b_specs.append(b_spec)

                            g_curr_text = g
                            b_curr_text = b
                            g_curr_doc = gdoc_working
                            b_curr_doc = bdoc_working
                            g_swaps_total, b_swaps_total = [], []
                            swap_failed = False

                            # Swap verbs that should stay identical across good/bad using the inventory.
                            if same_g_specs:
                                g_same_text, g_same_swaps, g_same_reason = verb_swap_all(
                                    g_curr_doc,
                                    verb_inventory_obj,
                                    transitivity_inventory=verb_inventory_transitivity,
                                    verb_mode=verb_mode,
                                    k=k,
                                    zipf_thr=verb_zipf_thr,
                                    zipf_min=verb_zipf_min,
                                    zipf_weighted=zipf_weighted_sampling,
                                    zipf_temp=zipf_temp,
                                    **verb_sampling_kwargs,
                                    rng=rng_verbs,
                                    forced_targets=same_g_specs,
                                    **token_match_kwargs,
                                )
                                verb_attempted = True
                                if _note_swap_reason(g_same_reason):
                                    verb_failure_reported = True
                                if not (g_same_text and g_same_swaps and len(g_same_swaps) == len(same_g_specs)):
                                    swap_failed = True
                                else:
                                    override_specs = []
                                    for entry in g_same_swaps:
                                        lemma = entry.get("lemma")
                                        frame_name = entry.get("frame")
                                        if not lemma or not frame_name:
                                            override_specs = []
                                            break
                                        override_specs.append({"lemma": lemma, "frame": frame_name})
                                    if len(override_specs) != len(same_b_specs):
                                        swap_failed = True
                                    else:
                                        b_same_text, b_same_swaps, b_same_reason = verb_swap_all(
                                            b_curr_doc,
                                            verb_inventory_obj,
                                            transitivity_inventory=verb_inventory_transitivity,
                                            verb_mode=verb_mode,
                                            k=k,
                                            zipf_thr=verb_zipf_thr,
                                            zipf_min=verb_zipf_min,
                                            zipf_weighted=zipf_weighted_sampling,
                                            zipf_temp=zipf_temp,
                                            **verb_sampling_kwargs,
                                            rng=random.Random(pair_seed - 1),
                                            forced_targets=same_b_specs,
                                            override_specs=override_specs,
                                            **token_match_kwargs,
                                        )
                                        verb_attempted = True
                                        if _note_swap_reason(b_same_reason):
                                            verb_failure_reported = True
                                        if not (
                                            b_same_text
                                            and b_same_swaps
                                            and len(b_same_swaps) == len(override_specs)
                                        ):
                                            swap_failed = True
                                        else:
                                            g_curr_text = g_same_text
                                            b_curr_text = b_same_text
                                            g_swaps_total.extend(g_same_swaps)
                                            b_swaps_total.extend(b_same_swaps)
                                            g_curr_doc = nlp(g_curr_text)
                                            b_curr_doc = nlp(b_curr_text)

                            # Swap the differing raising/control verb only.
                            if not swap_failed and diff_g_specs:
                                g_verb_variant, g_verb_swaps_diff, g_verb_reason = verb_swap_from_pool(
                                    g_curr_doc,
                                    raising_pool,
                                    forced_targets=diff_g_specs,
                                    rng=rng_verbs,
                                    **token_match_kwargs,
                                )
                                verb_attempted = True
                                if _note_swap_reason(g_verb_reason):
                                    verb_failure_reported = True
                                b_verb_variant = None
                                if g_verb_variant and g_verb_swaps_diff and len(diff_b_specs) == len(g_verb_swaps_diff):
                                    b_verb_variant, b_verb_swaps_diff, b_verb_reason = verb_swap_from_pool(
                                        b_curr_doc,
                                        control_pool,
                                        forced_targets=diff_b_specs,
                                        rng=random.Random(pair_seed - 2 if not matched_by_identity else pair_seed - 1),
                                        **token_match_kwargs,
                                    )
                                    verb_attempted = True
                                    if _note_swap_reason(b_verb_reason):
                                        verb_failure_reported = True
                                    if b_verb_variant and b_verb_swaps_diff:
                                        g_curr_text = g_verb_variant
                                        b_curr_text = b_verb_variant
                                        g_swaps_total.extend(g_verb_swaps_diff)
                                        b_swaps_total.extend(b_verb_swaps_diff)
                                else:
                                    swap_failed = True

                            if not swap_failed and g_swaps_total and b_swaps_total and len(g_swaps_total) == len(b_swaps_total):
                                g_verb_variant = g_curr_text
                                b_verb_variant = b_curr_text
                                g_verb_swaps = g_swaps_total
                                b_verb_swaps = b_swaps_total
                            else:
                                g_verb_variant = None
                                b_verb_variant = None
                            # If the control/raising-specific path fails (often due to
                            # misparses or pool/constraint mismatches), fall back to the
                            # generic inventory-driven verb swap so we still get a rare
                            # variant when possible.
                            if g_verb_variant is None or b_verb_variant is None:
                                g_verb_variant, g_verb_swaps, g_verb_reason = verb_swap_all(
                                    gdoc_working,
                                    verb_inventory_obj,
                                    transitivity_inventory=verb_inventory_transitivity,
                                    verb_mode=verb_mode,
                                    k=k,
                                    zipf_thr=verb_zipf_thr,
                                    zipf_min=verb_zipf_min,
                                    zipf_weighted=zipf_weighted_sampling,
                                    zipf_temp=zipf_temp,
                                    **verb_sampling_kwargs,
                                    rng=rng_verbs,
                                    forced_targets=g_target_specs,
                                    **token_match_kwargs,
                                )
                                verb_attempted = True
                                if _note_swap_reason(g_verb_reason):
                                    verb_failure_reported = True
                                b_verb_variant = None
                                if g_verb_variant and g_verb_swaps and len(b_target_specs) == len(g_verb_swaps):
                                    b_seed = pair_seed - 1 if matched_by_identity else pair_seed - 2
                                    b_verb_variant, b_verb_swaps, b_verb_reason = verb_swap_all(
                                        bdoc_working,
                                        verb_inventory_obj,
                                        transitivity_inventory=verb_inventory_transitivity,
                                        verb_mode=verb_mode,
                                        k=k,
                                        zipf_thr=verb_zipf_thr,
                                        zipf_min=verb_zipf_min,
                                        zipf_weighted=zipf_weighted_sampling,
                                        zipf_temp=zipf_temp,
                                        **verb_sampling_kwargs,
                                        rng=random.Random(b_seed),
                                        forced_targets=b_target_specs,
                                        **token_match_kwargs,
                                    )
                                    verb_attempted = True
                                    if _note_swap_reason(b_verb_reason):
                                        verb_failure_reported = True

                            if not (
                                g_verb_variant
                                and b_verb_variant
                                and g_verb_swaps
                                and b_verb_swaps
                                and len(g_verb_swaps) == len(b_verb_swaps)
                            ):
                                g_verb_swaps = []
                                b_verb_swaps = []
                            else:
                                g_variant = g_verb_variant
                                b_variant = b_verb_variant
                                verb_changed = True
                                gdoc_working = nlp(g_variant)
                                bdoc_working = nlp(b_variant)
                        else:
                            g_verb_variant, g_verb_swaps, g_verb_reason = verb_swap_all(
                                gdoc_working,
                                verb_inventory_obj,
                                transitivity_inventory=verb_inventory_transitivity,
                                verb_mode=verb_mode,
                                k=k,
                                zipf_thr=verb_zipf_thr,
                                zipf_min=verb_zipf_min,
                                zipf_weighted=zipf_weighted_sampling,
                                zipf_temp=zipf_temp,
                                **verb_sampling_kwargs,
                                rng=rng_verbs,
                                forced_targets=g_target_specs,
                                **token_match_kwargs,
                            )
                            verb_attempted = True
                            if _note_swap_reason(g_verb_reason):
                                verb_failure_reported = True
                            b_verb_variant = None
                            if g_verb_variant and g_verb_swaps:
                                override_specs = []
                                if verb_matches and len(verb_matches) == len(g_verb_swaps):
                                    for idx_pair, entry in enumerate(g_verb_swaps):
                                        g_spec = verb_matches[idx_pair][0]
                                        b_spec = verb_matches[idx_pair][1]
                                        same_orig = g_spec.get("lemma") == b_spec.get("lemma")
                                        if same_orig:
                                            lemma = entry.get("lemma")
                                            frame_name = entry.get("frame")
                                            if not lemma or not frame_name:
                                                override_specs = []
                                                break
                                            override_specs.append({"lemma": lemma, "frame": frame_name})
                                        else:
                                            override_specs.append(None)
                                if len(b_target_specs) == len(g_verb_swaps):
                                    b_seed = pair_seed - 1 if matched_by_identity else pair_seed - 2
                                    b_verb_variant, b_verb_swaps, b_verb_reason = verb_swap_all(
                                        bdoc_working,
                                        verb_inventory_obj,
                                        transitivity_inventory=verb_inventory_transitivity,
                                        verb_mode=verb_mode,
                                        k=k,
                                        zipf_thr=verb_zipf_thr,
                                        zipf_min=verb_zipf_min,
                                        zipf_weighted=zipf_weighted_sampling,
                                        zipf_temp=zipf_temp,
                                        **verb_sampling_kwargs,
                                        rng=random.Random(b_seed),
                                        forced_targets=b_target_specs,
                                        override_specs=override_specs or None,
                                        **token_match_kwargs,
                                    )
                                    verb_attempted = True
                                    if _note_swap_reason(b_verb_reason):
                                        verb_failure_reported = True
                            if not (
                                g_verb_variant
                                and b_verb_variant
                                and g_verb_swaps
                                and b_verb_swaps
                                and len(g_verb_swaps) == len(b_verb_swaps)
                            ):
                                g_verb_swaps = []
                                b_verb_swaps = []
                            else:
                                g_variant = g_verb_variant
                                b_variant = b_verb_variant
                                verb_changed = True
                                gdoc_working = nlp(g_variant)
                                bdoc_working = nlp(b_variant)
                    if not verb_changed:
                        if verb_failure_reported:
                            pass
                        elif verb_attempted:
                            _note_swap_failure("verb_swap_failed")
                        elif verb_candidates_present:
                            _note_swap_failure("verb_no_matches")
                        else:
                            _note_swap_failure("verb_no_candidates")

                # Stage 1: noun swaps
                if do_noun_swaps:
                    noun_attempted = False
                    noun_failure_reported = False
                    # Detect noun targets on the original parse so earlier swaps
                    # (e.g., verbs) don't change POS tags and get re-swapped.
                    g_verb_indices = {entry.get("i") for entry in g_verb_swaps or () if isinstance(entry, dict)}
                    b_verb_indices = {entry.get("i") for entry in b_verb_swaps or () if isinstance(entry, dict)}

                    g_candidates = [t for t in candidate_nouns(gdoc_orig, reflexive_subjects=g_reflexive_subjects) if t.i not in g_verb_indices]
                    g_candidates.sort(key=lambda t: t.i)
                    b_candidates = [t for t in candidate_nouns(bdoc_orig, reflexive_subjects=b_reflexive_subjects) if t.i not in b_verb_indices]
                    b_candidates.sort(key=lambda t: t.i)
                    noun_candidates_present = bool(g_candidates or b_candidates)

                    noun_matches = []
                    used_bad_indices = set()

                    def _find_match(g_tok):
                        g_lemma = g_tok.lemma_.lower()
                        g_text = g_tok.text.lower()
                        for b_tok in b_candidates:
                            if b_tok.i in used_bad_indices:
                                continue
                            if b_tok.lemma_.lower() == g_lemma:
                                return b_tok
                        for b_tok in b_candidates:
                            if b_tok.i in used_bad_indices:
                                continue
                            if b_tok.text.lower() == g_text:
                                return b_tok
                        return None

                    for g_tok in g_candidates:
                        b_match = _find_match(g_tok)
                        if b_match is None:
                            continue
                        reflexive_info_g = g_reflexive_subjects.get(g_tok.i) if isinstance(g_reflexive_subjects, dict) else None
                        reflexive_info_b = b_reflexive_subjects.get(b_match.i) if isinstance(b_reflexive_subjects, dict) else None
                        require_gender = _required_gender(g_tok, g_reflexive_subjects, gender_lex) or _required_gender(b_match, b_reflexive_subjects, gender_lex)
                        require_person = bool(require_gender) or (reflexive_info_g.get("animate") if reflexive_info_g else False) or (reflexive_info_b.get("animate") if reflexive_info_b else False)
                        if not require_person and is_person_noun(g_tok.lemma_.lower()):
                            require_person = True
                        noun_matches.append(
                            (
                                (g_tok.i, _noun_target_tag(g_tok)),
                                (b_match.i, _noun_target_tag(b_match)),
                                require_person,
                                require_gender,
                            )
                        )
                        used_bad_indices.add(b_match.i)

                    if noun_mode == "k" and noun_matches:
                        limit = max(0, min(k, len(noun_matches)))
                        noun_matches = noun_matches[:limit]

                    if noun_matches:
                        allow_extra_good_nouns = phenomenon == "ellipsis" and cfg == "ellipsis_n_bar_2"
                        needs_plural = any(
                            g_spec[1] == "NNS" or b_spec[1] == "NNS"
                            for g_spec, b_spec, _, _ in noun_matches
                        )
                        shared_req = req_g if req_g else req_b
                        if needs_plural and shared_req != "MASS":
                            shared_req = "COUNT"
                        g_target_specs = []
                        b_target_specs = []
                        for g_spec, b_spec, require_person, require_gender in noun_matches:
                            g_idx, g_tag = g_spec
                            b_idx, b_tag = b_spec
                            g_target_specs.append({
                                "i": g_idx,
                                "tag": g_tag,
                                "require_person": require_person,
                                "require_gender": require_gender,
                            })
                            b_target_specs.append({
                                "i": b_idx,
                                "tag": b_tag,
                                "require_person": require_person,
                                "require_gender": require_gender,
                            })
                        matched_count = len(g_target_specs)
                        if allow_extra_good_nouns:
                            matched_indices = {spec["i"] for spec in g_target_specs}
                            matched_lemmas = {}
                            for (g_idx, _), _, require_person, require_gender in noun_matches:
                                g_tok = gdoc_orig[g_idx]
                                g_lemma = g_tok.lemma_.lower()
                                matched_lemmas.setdefault(g_lemma, (require_person, require_gender))
                            for g_tok in g_candidates:
                                if g_tok.i in matched_indices:
                                    continue
                                g_lemma = g_tok.lemma_.lower()
                                if g_lemma in matched_lemmas:
                                    require_person, require_gender = matched_lemmas[g_lemma]
                                    g_target_specs.append({
                                        "i": g_tok.i,
                                        "tag": _noun_target_tag(g_tok),
                                        "require_person": require_person,
                                        "require_gender": require_gender,
                                    })
                        rng = random.Random(pair_seed)
                        g_rare, g_swaps, g_noun_reason = noun_swap_all(
                            gdoc_working, rare_pool,
                            noun_mode=noun_mode, k=k, zipf_thr=None, zipf_min=noun_zipf_min,
                            zipf_weighted=zipf_weighted_sampling,
                            zipf_temp=zipf_temp,
                            becl_map=becl_map, req=shared_req, rng=rng,
                            forced_targets=g_target_specs,
                            rare_person_lemmas=rare_person_pool,
                            rare_gender_lemmas=rare_gender_map,
                            reflexive_subjects=g_reflexive_subjects,
                            **token_match_kwargs,
                        )
                        noun_attempted = True
                        if _note_swap_reason(g_noun_reason):
                            noun_failure_reported = True
                        b_rare = None
                        if g_rare and g_swaps:
                            lemmas = [entry.get("lemma") for entry in g_swaps[:matched_count]]
                            if lemmas and all(lemma for lemma in lemmas):
                                b_rare, b_swaps, b_noun_reason = noun_swap_all(
                                    bdoc_working, rare_pool,
                                    noun_mode=noun_mode, k=k, zipf_thr=None, zipf_min=noun_zipf_min,
                                    zipf_weighted=zipf_weighted_sampling,
                                    zipf_temp=zipf_temp,
                                    becl_map=becl_map, req=shared_req,
                                    rng=random.Random(pair_seed),
                                    forced_targets=b_target_specs,
                                    rare_person_lemmas=rare_person_pool,
                                    rare_gender_lemmas=rare_gender_map,
                                    override_lemmas=lemmas,
                                    reflexive_subjects=b_reflexive_subjects,
                                    **token_match_kwargs,
                                )
                                noun_attempted = True
                                if _note_swap_reason(b_noun_reason):
                                    noun_failure_reported = True
                        if not (
                            g_rare
                            and b_rare
                            and g_swaps
                            and b_swaps
                            and len(g_swaps) == len(g_target_specs)
                            and len(b_swaps) == len(b_target_specs)
                        ):
                            g_rare = None
                            b_rare = None
                            g_swaps = []
                            b_swaps = []
                        else:
                            g_variant = g_rare
                            b_variant = b_rare
                            noun_changed = True
                            gdoc_working = nlp(g_variant)
                            bdoc_working = nlp(b_variant)
                    elif phenomenon == "s-selection" and g_candidates and b_candidates:
                        if noun_mode == "k":
                            g_candidates = g_candidates[:max(0, min(k, len(g_candidates)))]
                            b_candidates = b_candidates[:max(0, min(k, len(b_candidates)))]

                        def _build_unmatched_targets(candidates, reflexive_meta):
                            targets = []
                            for tok in candidates:
                                reflexive_info = reflexive_meta.get(tok.i) if isinstance(reflexive_meta, dict) else None
                                require_gender = _required_gender(tok, reflexive_meta, gender_lex)
                                require_person = bool(require_gender) or (reflexive_info.get("animate") if reflexive_info else False)
                                if not require_person and is_person_noun(tok.lemma_.lower()):
                                    require_person = True
                                targets.append({
                                    "i": tok.i,
                                    "tag": _noun_target_tag(tok),
                                    "require_person": require_person,
                                    "require_gender": require_gender,
                                })
                            return targets

                        g_target_specs = _build_unmatched_targets(g_candidates, g_reflexive_subjects)
                        b_target_specs = _build_unmatched_targets(b_candidates, b_reflexive_subjects)
                        g_rare = None
                        b_rare = None

                        if g_target_specs and b_target_specs:
                            g_rare, g_swaps, g_noun_reason = noun_swap_all(
                                gdoc_working, rare_pool,
                                noun_mode=noun_mode, k=k, zipf_thr=None, zipf_min=noun_zipf_min,
                                zipf_weighted=zipf_weighted_sampling,
                                zipf_temp=zipf_temp,
                                becl_map=becl_map, req=req_g, rng=random.Random(pair_seed),
                                forced_targets=g_target_specs,
                                rare_person_lemmas=rare_person_pool,
                                rare_gender_lemmas=rare_gender_map,
                                reflexive_subjects=g_reflexive_subjects,
                                **token_match_kwargs,
                            )
                            noun_attempted = True
                            if _note_swap_reason(g_noun_reason):
                                noun_failure_reported = True
                            b_rare = None
                            if g_rare and g_swaps:
                                b_rare, b_swaps, b_noun_reason = noun_swap_all(
                                    bdoc_working, rare_pool,
                                    noun_mode=noun_mode, k=k, zipf_thr=None, zipf_min=noun_zipf_min,
                                    zipf_weighted=zipf_weighted_sampling,
                                    zipf_temp=zipf_temp,
                                    becl_map=becl_map, req=req_b,
                                    rng=random.Random(pair_seed - 1),
                                    forced_targets=b_target_specs,
                                    rare_person_lemmas=rare_person_pool,
                                    rare_gender_lemmas=rare_gender_map,
                                    reflexive_subjects=b_reflexive_subjects,
                                    **token_match_kwargs,
                                )
                                noun_attempted = True
                                if _note_swap_reason(b_noun_reason):
                                    noun_failure_reported = True
                        if not (
                            g_rare
                            and b_rare
                            and g_swaps
                            and b_swaps
                            and len(g_swaps) == len(b_swaps)
                        ):
                            g_rare = None
                            b_rare = None
                            g_swaps = []
                            b_swaps = []
                        else:
                            g_variant = g_rare
                            b_variant = b_rare
                            noun_changed = True
                            gdoc_working = nlp(g_variant)
                            bdoc_working = nlp(b_variant)
                    if not noun_changed:
                        if noun_failure_reported:
                            pass
                        elif noun_attempted:
                            _note_swap_failure("noun_swap_failed")
                        elif noun_candidates_present:
                            _note_swap_failure("noun_no_matches")
                        else:
                            _note_swap_failure("noun_no_candidates")

                # Stage 2: adjective swaps (operates on the noun-swapped text)
                if do_adj_swaps and phenomenon != "irregular_forms":
                    adj_attempted = False
                    adj_failure_reported = False
                    adj_good_pool = rare_adj_pool
                    adj_bad_pool = rare_adj_pool
                    allow_ellipsis_adj = phenomenon == "ellipsis" and cfg == "ellipsis_n_bar_2"
                    if is_control_raising:
                        raising_adj_pool = _filter_pool_with_fallback(_RAISING_ADJECTIVES, adj_zipf_thr)
                        control_adj_pool = _filter_pool_with_fallback(_CONTROL_ADJECTIVES, adj_zipf_thr)
                        tough_adj_pool = _filter_pool_with_fallback(_TOUGH_ADJECTIVES, adj_zipf_thr)
                        cfg_key = str(cfg).strip().lower()
                        if cfg_key == "tough_vs_raising_1":
                            adj_good_pool = tough_adj_pool
                            adj_bad_pool = raising_adj_pool
                        elif cfg_key == "tough_vs_raising_2":
                            adj_good_pool = raising_adj_pool
                            adj_bad_pool = tough_adj_pool
                        elif cfg_key == "expletive_it_object_raising":
                            adj_good_pool = tough_adj_pool
                            adj_bad_pool = tough_adj_pool
                        else:
                            adj_good_pool = raising_adj_pool
                            adj_bad_pool = control_adj_pool
                    gdoc_adj = gdoc_working
                    bdoc_adj = bdoc_working
                    # Detect adjective targets on the original parse so noun swaps
                    # that confuse POS tagging (e.g., rare nouns tagged as PROPN)
                    # don't block adjective swaps.
                    g_adj_candidates = candidate_adjectives(gdoc_orig)
                    g_adj_candidates.sort(key=lambda t: t.i)
                    b_adj_candidates = candidate_adjectives(bdoc_orig)
                    b_adj_candidates.sort(key=lambda t: t.i)
                    if allow_ellipsis_adj:
                        adj_pool_set = set(adj_good_pool or ())

                        def _adj_dominant(lemma: str) -> bool:
                            v, n, a, r = wordnet_pos_counts(lemma, lexicon="oewn:2021")
                            if a <= 0:
                                return False
                            if n >= a or v >= a or r >= a:
                                return False
                            return True

                        def _extend_ellipsis_candidates(doc, base):
                            by_i = {t.i: t for t in base}
                            for tok in doc:
                                if tok.i in by_i:
                                    continue
                                if tok.pos_ not in {"ADJ", "NOUN", "ADV"}:
                                    continue
                                if tok.dep_ not in {"dobj", "attr", "acomp", "oprd", "advmod"}:
                                    continue
                                if not tok.is_alpha or len(tok.text) <= 2:
                                    continue
                                if tok.ent_type_:
                                    continue
                                lemma = (tok.lemma_ or "").strip().lower()
                                if not lemma:
                                    continue
                                if adj_pool_set and lemma not in adj_pool_set and not _adj_dominant(lemma):
                                    continue
                                by_i[tok.i] = tok
                            return sorted(by_i.values(), key=lambda t: t.i)

                        g_adj_candidates = _extend_ellipsis_candidates(gdoc_orig, g_adj_candidates)
                        b_adj_candidates = _extend_ellipsis_candidates(bdoc_orig, b_adj_candidates)
                    adj_candidates_present = bool(g_adj_candidates or b_adj_candidates)
                    if is_control_raising:
                        g_pred = _control_raising_adj_targets(gdoc_orig)
                        b_pred = _control_raising_adj_targets(bdoc_orig)
                        if g_pred and b_pred:
                            g_adj_candidates = _unique(sorted(g_pred, key=lambda t: t.i))
                            b_adj_candidates = _unique(sorted(b_pred, key=lambda t: t.i))
                        else:
                            g_adj_candidates.extend(g_pred)
                            g_adj_candidates = _unique(sorted(g_adj_candidates, key=lambda t: t.i))
                            b_adj_candidates.extend(b_pred)
                            b_adj_candidates = _unique(sorted(b_adj_candidates, key=lambda t: t.i))

                    adj_matches = []
                    used_bad_adjs = set()

                    def _find_adj_match(g_tok):
                        g_lemma = g_tok.lemma_.lower()
                        g_text = g_tok.text.lower()
                        for b_tok in b_adj_candidates:
                            if b_tok.i in used_bad_adjs:
                                continue
                            if b_tok.lemma_.lower() == g_lemma:
                                return b_tok
                        for b_tok in b_adj_candidates:
                            if b_tok.i in used_bad_adjs:
                                continue
                            if b_tok.text.lower() == g_text:
                                return b_tok
                        return None

                    for g_tok in g_adj_candidates:
                        b_match = _find_adj_match(g_tok)
                        if b_match is None:
                            continue
                        adj_matches.append(
                            (
                                (g_tok.i, g_tok.tag_),
                                (b_match.i, b_match.tag_),
                            )
                        )
                        used_bad_adjs.add(b_match.i)

                    if is_control_raising and not adj_matches:
                        limit = min(len(g_adj_candidates), len(b_adj_candidates))
                        for g_tok, b_tok in zip(g_adj_candidates[:limit], b_adj_candidates[:limit]):
                            adj_matches.append(
                                (
                                    (g_tok.i, g_tok.tag_),
                                    (b_tok.i, b_tok.tag_),
                                )
                            )

                    if adj_mode == "k" and adj_matches:
                        limit = max(0, min(k, len(adj_matches)))
                        adj_matches = adj_matches[:limit]

                    if adj_matches:
                        rng_adj = random.Random(pair_seed + 1)
                        if is_control_raising:
                            same_pairs, diff_pairs = [], []
                            for (g_idx, g_tag), (b_idx, b_tag) in adj_matches:
                                g_tok = gdoc_orig[g_idx]
                                b_tok = bdoc_orig[b_idx]
                                if g_tok.lemma_.lower() == b_tok.lemma_.lower():
                                    same_pairs.append(((g_idx, g_tag), (b_idx, b_tag)))
                                else:
                                    diff_pairs.append(((g_idx, g_tag), (b_idx, b_tag)))

                            g_adj_text = g_variant
                            b_adj_text = b_variant
                            g_adj_doc_curr = gdoc_adj
                            b_adj_doc_curr = bdoc_adj
                            g_adj_swaps_all, b_adj_swaps_all = [], []
                            swap_failed = False

                            if same_pairs:
                                g_specs_same = [{"i": gi, "tag": gt} for (gi, gt), _ in same_pairs]
                                b_specs_same = [{"i": bi, "tag": bt} for _, (bi, bt) in same_pairs]
                                g_same_text, g_same_swaps, g_adj_reason = adjective_swap_all(
                                    g_adj_doc_curr,
                                    adj_good_pool,
                                    adj_mode=adj_mode,
                                    k=k,
                                    zipf_thr=adj_zipf_thr,
                                    zipf_min=adj_zipf_min,
                                    zipf_weighted=zipf_weighted_sampling,
                                    zipf_temp=zipf_temp,
                                    rng=rng_adj,
                                    forced_targets=g_specs_same,
                                    **token_match_kwargs,
                                )
                                adj_attempted = True
                                if _note_swap_reason(g_adj_reason):
                                    adj_failure_reported = True
                                if not (g_same_text and g_same_swaps and len(g_same_swaps) == len(g_specs_same)):
                                    swap_failed = True
                                else:
                                    override_lemmas = []
                                    for entry in g_same_swaps:
                                        lemma = entry.get("lemma")
                                        if not lemma:
                                            override_lemmas = []
                                            break
                                        override_lemmas.append(lemma)
                                    if len(override_lemmas) != len(b_specs_same):
                                        swap_failed = True
                                    else:
                                        b_same_text, b_same_swaps, b_adj_reason = adjective_swap_all(
                                            b_adj_doc_curr,
                                            adj_bad_pool,
                                            adj_mode=adj_mode,
                                            k=k,
                                            zipf_thr=adj_zipf_thr,
                                            zipf_min=adj_zipf_min,
                                            zipf_weighted=zipf_weighted_sampling,
                                            zipf_temp=zipf_temp,
                                            rng=random.Random(pair_seed + 1),
                                            forced_targets=b_specs_same,
                                            override_lemmas=override_lemmas,
                                            **token_match_kwargs,
                                        )
                                        adj_attempted = True
                                        if _note_swap_reason(b_adj_reason):
                                            adj_failure_reported = True
                                        if not (
                                            b_same_text
                                            and b_same_swaps
                                            and len(b_same_swaps) == len(b_specs_same)
                                        ):
                                            swap_failed = True
                                        else:
                                            g_adj_text = g_same_text
                                            b_adj_text = b_same_text
                                            g_adj_swaps_all.extend(g_same_swaps)
                                            b_adj_swaps_all.extend(b_same_swaps)
                                            g_adj_doc_curr = nlp(g_adj_text)
                                            b_adj_doc_curr = nlp(b_adj_text)

                            if not swap_failed and diff_pairs:
                                g_specs_diff = [{"i": gi, "tag": gt} for (gi, gt), _ in diff_pairs]
                                b_specs_diff = [{"i": bi, "tag": bt} for _, (bi, bt) in diff_pairs]
                                g_diff_text, g_diff_swaps, g_adj_reason = adjective_swap_all(
                                    g_adj_doc_curr,
                                    adj_good_pool,
                                    adj_mode=adj_mode,
                                    k=k,
                                    zipf_thr=adj_zipf_thr,
                                    zipf_min=adj_zipf_min,
                                    zipf_weighted=zipf_weighted_sampling,
                                    zipf_temp=zipf_temp,
                                    rng=rng_adj,
                                    forced_targets=g_specs_diff,
                                    **token_match_kwargs,
                                )
                                adj_attempted = True
                                if _note_swap_reason(g_adj_reason):
                                    adj_failure_reported = True
                                b_diff_text, b_diff_swaps = None, None
                                if g_diff_text and g_diff_swaps and len(g_diff_swaps) == len(g_specs_diff):
                                    b_diff_text, b_diff_swaps, b_adj_reason = adjective_swap_all(
                                        b_adj_doc_curr,
                                        adj_bad_pool,
                                        adj_mode=adj_mode,
                                        k=k,
                                        zipf_thr=adj_zipf_thr,
                                        zipf_min=adj_zipf_min,
                                        zipf_weighted=zipf_weighted_sampling,
                                        zipf_temp=zipf_temp,
                                        rng=random.Random(pair_seed + 2),
                                        forced_targets=b_specs_diff,
                                        **token_match_kwargs,
                                    )
                                    adj_attempted = True
                                    if _note_swap_reason(b_adj_reason):
                                        adj_failure_reported = True
                                if not (
                                    g_diff_text
                                    and b_diff_text
                                    and g_diff_swaps
                                    and b_diff_swaps
                                    and len(g_diff_swaps) == len(b_diff_swaps)
                                ):
                                    swap_failed = True
                                else:
                                    g_adj_text = g_diff_text
                                    b_adj_text = b_diff_text
                                    g_adj_swaps_all.extend(g_diff_swaps)
                                    b_adj_swaps_all.extend(b_diff_swaps)

                            if not swap_failed and g_adj_swaps_all and b_adj_swaps_all and len(g_adj_swaps_all) == len(b_adj_swaps_all):
                                g_variant = g_adj_text
                                b_variant = b_adj_text
                                g_adj_swaps = g_adj_swaps_all
                                b_adj_swaps = b_adj_swaps_all
                                adj_changed = True
                        else:
                            g_target_specs = [{"i": g_idx, "tag": g_tag} for (g_idx, g_tag), _ in adj_matches]
                            b_target_specs = [{"i": b_idx, "tag": b_tag} for _, (b_idx, b_tag) in adj_matches]
                            g_adj_variant, g_adj_swaps, g_adj_reason = adjective_swap_all(
                                gdoc_adj,
                                adj_good_pool,
                                adj_mode=adj_mode,
                                k=k,
                                zipf_thr=adj_zipf_thr,
                                zipf_min=adj_zipf_min,
                                zipf_weighted=zipf_weighted_sampling,
                                zipf_temp=zipf_temp,
                                rng=rng_adj,
                                forced_targets=g_target_specs,
                                **token_match_kwargs,
                            )
                            adj_attempted = True
                            if _note_swap_reason(g_adj_reason):
                                adj_failure_reported = True
                            b_adj_variant = None
                            if g_adj_variant and g_adj_swaps:
                                adj_lemmas = [entry.get("lemma") for entry in g_adj_swaps]
                                if adj_lemmas and all(lemma for lemma in adj_lemmas):
                                    b_adj_variant, b_adj_swaps, b_adj_reason = adjective_swap_all(
                                        bdoc_adj,
                                        adj_bad_pool,
                                        adj_mode=adj_mode,
                                        k=k,
                                        zipf_thr=adj_zipf_thr,
                                        zipf_min=adj_zipf_min,
                                        zipf_weighted=zipf_weighted_sampling,
                                        zipf_temp=zipf_temp,
                                        rng=random.Random(pair_seed + 1),
                                        forced_targets=b_target_specs,
                                        override_lemmas=adj_lemmas,
                                        **token_match_kwargs,
                                    )
                                    adj_attempted = True
                                    if _note_swap_reason(b_adj_reason):
                                        adj_failure_reported = True
                            if not (
                                g_adj_variant
                                and b_adj_variant
                                and g_adj_swaps
                                and b_adj_swaps
                                and len(g_adj_swaps) == len(b_adj_swaps)
                            ):
                                g_adj_swaps = []
                                b_adj_swaps = []
                            else:
                                g_variant = g_adj_variant
                                b_variant = b_adj_variant
                                adj_changed = True
                    if not adj_changed:
                        if adj_failure_reported:
                            pass
                        elif adj_attempted:
                            _note_swap_failure("adj_swap_failed")
                        elif adj_candidates_present:
                            _note_swap_failure("adj_no_matches")
                        else:
                            _note_swap_failure("adj_no_candidates")

                if not (verb_changed or noun_changed or adj_changed):
                    good_rare_val = None
                    bad_rare_val = None
                else:
                    good_rare_val = g_variant
                    bad_rare_val = b_variant

                swap_failed_reason = None
                if not (verb_changed or noun_changed or adj_changed):
                    if swap_failures:
                        swap_failed_reason = "|".join(swap_failures)
                    else:
                        swap_failed_reason = "no_swap"
                if good_rare_val and bad_rare_val and good_rare_val == bad_rare_val:
                    # Rare variants must preserve the good/bad contrast. When swapping
                    # collapses the pair, drop the rare variants and record the reason.
                    good_rare_val = None
                    bad_rare_val = None
                    g_swaps, b_swaps = [], []
                    g_adj_swaps, b_adj_swaps = [], []
                    g_verb_swaps, b_verb_swaps = [], []
                    verb_changed = False
                    noun_changed = False
                    adj_changed = False
                    swap_failed_reason = "pair_collapsed"

                blimp_field = _normalize_blimp_field(r.get("field"))
                if blimp_field is None:
                    blimp_field = field_map.get(cfg)

                g_merged_swaps = _merge_and_slim_swaps(g_swaps, g_adj_swaps, g_verb_swaps)
                b_merged_swaps = _merge_and_slim_swaps(b_swaps, b_adj_swaps, b_verb_swaps)
                record = {
                    "phenomenon": phenomenon,
                    "subtask": cfg,
                    "field": blimp_field,
                    "idx": i,
                    "good_original": g,
                    "bad_original": b,
                    "good_rare": good_rare_val,
                    "bad_rare": bad_rare_val,
                    "meta": {
                        "g_swaps": g_merged_swaps,
                        "b_swaps": b_merged_swaps,
                        "swap_failed_reason": swap_failed_reason
                    }
                }
                records.append(add_zipf_aggregates(record))
                processed += 1
                if show_progress:
                    last_update = _print_progress(processed, total_items, start_time, last_update)
        # Exhaust the generator to avoid leaving pending docs when breaking early.
        if record_limit and len(records) >= record_limit:
            stop_early = True
        if stop_early:
            break

    generated_records = []
    original_records = []
    for rec in records:
        generated = dict(rec)
        generated.pop("good_original", None)
        generated.pop("bad_original", None)
        generated_records.append(generated)

        if original_out_path is not None:
            original = dict(rec)
            original.pop("good_rare", None)
            original.pop("bad_rare", None)
            original_records.append(original)

    write_jsonl(out_path, generated_records)
    if original_out_path is not None:
        write_jsonl(original_out_path, original_records)
