import random, spacy, yaml, time
from pathlib import Path
from .io import load_blimp, write_jsonl
from .becl import load_becl_tsv
from .quantifier import load_quant_rules, requirement
from .edits import (
    noun_swap_all,
    candidate_nouns,
    reflexive_subject_indices,
)
from .rarity import is_rare_lemma
from .lemma_bank import is_person_noun


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


def build_pilot(tier_cfg_path, becl_path, quant_cfg_path, out_path,
                noun_mode="all", k=2, zipf_thr=3.4, rare_lemmas=None, seed=0,
                show_progress=True,
                rare_name_path="data/external/rare_names.tsv",
                name_lookup_path="data/external/name_gender_lookup.tsv",
                name_conf=0.75):
    base_rng = random.Random(seed)
    nlp = spacy.load("en_core_web_sm")
    with open(tier_cfg_path, encoding="utf-8") as f:
        tasks_cfg = yaml.safe_load(f)
    becl_map = load_becl_tsv(becl_path)
    qrules = load_quant_rules(quant_cfg_path)
    records = []


    if rare_lemmas:
        seen = set()
        filtered = []
        for lemma in rare_lemmas:
            if not lemma:
                continue
            norm = lemma.strip().lower()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            if zipf_thr is None or is_rare_lemma(norm, zipf_thr):
                filtered.append(norm)
        rare_pool = tuple(filtered)
    else:
        rare_pool = tuple()

    if rare_pool:
        rare_person_pool = tuple(lemma for lemma in rare_pool if is_person_noun(lemma))
    else:
        rare_person_pool = tuple()

    worklist = []
    total_items = 0

    for group_name, meta in tasks_cfg.items():
        phenomenon = meta["phenomenon"]
        configs = meta.get("configs", [])
        if not configs:
            continue
        for cfg in configs:
            ds = load_blimp(cfg)
            count = len(ds)
            total_items += count
            worklist.append((group_name, phenomenon, cfg, ds))

    processed = 0
    start_time = time.time()
    last_update = start_time

    for group_name, phenomenon, cfg, ds in worklist:
        for i, r in enumerate(ds):
                g, b = r["sentence_good"], r["sentence_bad"]
                gdoc, bdoc = nlp(g), nlp(b)
                g_reflexive_subjects = reflexive_subject_indices(gdoc)
                b_reflexive_subjects = reflexive_subject_indices(bdoc)

                # Quantifier requirement per sentence
                req_g = requirement(gdoc, qrules)  # None/COUNT/MASS
                req_b = requirement(bdoc, qrules)
                if phenomenon == "quantifiers":
                    if req_g is None and req_b in {"COUNT", "MASS"}:
                        req_g = req_b
                    elif req_b is None and req_g in {"COUNT", "MASS"}:
                        req_b = req_g

                # Per-pair RNG so Good/Bad use the same sequence of choices
                pair_seed = hash((group_name, cfg, i)) & 0xFFFFFFFF

                # Stage 1: noun swaps
                g_candidates = candidate_nouns(gdoc)
                g_candidates.sort(key=lambda t: t.i)
                b_candidates = candidate_nouns(bdoc)
                b_candidates.sort(key=lambda t: t.i)

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
                    require_person = (
                        g_tok.i in g_reflexive_subjects
                        and b_match.i in b_reflexive_subjects
                    )
                    noun_matches.append(
                        (
                            (g_tok.i, _noun_target_tag(g_tok)),
                            (b_match.i, _noun_target_tag(b_match)),
                            require_person,
                        )
                    )
                    used_bad_indices.add(b_match.i)

                if noun_mode == "k" and noun_matches:
                    limit = max(0, min(k, len(noun_matches)))
                    noun_matches = noun_matches[:limit]

                g_swaps, b_swaps = [], []
                g_variant = g
                b_variant = b
                noun_changed = False

                if noun_matches:
                    needs_plural = any(
                        g_spec[1] == "NNS" or b_spec[1] == "NNS"
                        for g_spec, b_spec, _ in noun_matches
                    )
                    shared_req = req_g if req_g else req_b
                    if needs_plural and shared_req != "MASS":
                        shared_req = "COUNT"
                    g_target_specs = []
                    b_target_specs = []
                    for g_spec, b_spec, require_person in noun_matches:
                        g_idx, g_tag = g_spec
                        b_idx, b_tag = b_spec
                        g_target_specs.append({
                            "i": g_idx,
                            "tag": g_tag,
                            "require_person": require_person,
                        })
                        b_target_specs.append({
                            "i": b_idx,
                            "tag": b_tag,
                            "require_person": require_person,
                        })
                    rng = random.Random(pair_seed)
                    g_rare, g_swaps = noun_swap_all(
                        gdoc, rare_pool,
                        noun_mode=noun_mode, k=k, zipf_thr=None,
                        becl_map=becl_map, req=shared_req, rng=rng,
                        forced_targets=g_target_specs,
                        rare_person_lemmas=rare_person_pool
                    )
                    b_rare = None
                    if g_rare and g_swaps:
                        lemmas = [entry.get("lemma") for entry in g_swaps]
                        if lemmas and all(lemma for lemma in lemmas):
                            b_rare, b_swaps = noun_swap_all(
                                bdoc, rare_pool,
                                noun_mode=noun_mode, k=k, zipf_thr=None,
                                becl_map=becl_map, req=shared_req,
                                rng=random.Random(pair_seed),
                                forced_targets=b_target_specs,
                                rare_person_lemmas=rare_person_pool,
                                override_lemmas=lemmas,
                            )
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

                g_name_swaps, b_name_swaps = [], []
                name_changed = False

                if not (noun_changed or name_changed):
                    good_rare_val = None
                    bad_rare_val = None
                else:
                    good_rare_val = g_variant
                    bad_rare_val = b_variant

                records.append({
                    "group": group_name,
                    "phenomenon": phenomenon,
                    "subtask": cfg,
                    "idx": i,
                    "good_typical": g,
                    "bad_typical": b,
                    "good_rare": good_rare_val,
                    "bad_rare":  bad_rare_val,
                    "meta": {
                        "g_swaps": g_swaps,
                        "b_swaps": b_swaps,
                        "g_name_swaps": g_name_swaps,
                        "b_name_swaps": b_name_swaps,
                        "noun_mode": noun_mode,
                        "k": k,
                        "zipf_thr": zipf_thr,
                        "name_swapped": name_changed,
                        "noun_swapped": noun_changed,
                        "req_good": req_g,
                        "req_bad": req_b
                    }
                })
                processed += 1
                if show_progress:
                    last_update = _print_progress(processed, total_items, start_time, last_update)

    write_jsonl(out_path, records)
