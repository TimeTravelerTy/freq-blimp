import random, spacy, yaml, time
from pathlib import Path
from .io import load_blimp, write_jsonl
from .becl import load_becl_tsv
from .quantifier import load_quant_rules, requirement
from .edits import (
    noun_swap_all,
    candidate_nouns,
    person_name_candidates,
    person_name_swap,
)
from .rarity import is_rare_lemma
from .names import NameBank


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

    name_bank = None
    rare_name_path = Path(rare_name_path) if rare_name_path else None
    name_lookup_path = Path(name_lookup_path) if name_lookup_path else None
    if rare_name_path and name_lookup_path:
        try:
            name_bank = NameBank(rare_name_path, name_lookup_path, min_lookup_confidence=name_conf)
        except FileNotFoundError:
            print(f"[NameBank] Skipping name swaps; missing files {rare_name_path} or {name_lookup_path}.")
            name_bank = None
        except ValueError as exc:
            print(f"[NameBank] Skipping name swaps: {exc}")
            name_bank = None

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

                # Quantifier requirement per sentence
                req_g = requirement(gdoc, qrules)  # None/COUNT/MASS
                req_b = requirement(bdoc, qrules)

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
                    noun_matches.append(
                        ((g_tok.i, g_tok.tag_), (b_match.i, b_match.tag_))
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
                    g_target_specs = [spec for spec, _ in noun_matches]
                    b_target_specs = [spec for _, spec in noun_matches]
                    rng = random.Random(pair_seed)
                    g_rare, g_swaps = noun_swap_all(
                        gdoc, rare_pool,
                        noun_mode=noun_mode, k=k, zipf_thr=None,
                        becl_map=becl_map, req=req_g, rng=rng,
                        forced_targets=g_target_specs
                    )
                    rng = random.Random(pair_seed)
                    b_rare, b_swaps = noun_swap_all(
                        bdoc, rare_pool,
                        noun_mode=noun_mode, k=k, zipf_thr=None,
                        becl_map=becl_map, req=req_b, rng=rng,
                        forced_targets=b_target_specs
                    )
                    if (
                        g_rare
                        and b_rare
                        and g_swaps
                        and b_swaps
                        and len(g_swaps) == len(b_swaps)
                    ):
                        g_variant = g_rare
                        b_variant = b_rare
                        noun_changed = True

                # Stage 2: proper-name swaps (after noun swaps)
                g_name_swaps, b_name_swaps = [], []
                name_changed = False
                if name_bank:
                    gdoc_after = nlp(g_variant)
                    bdoc_after = nlp(b_variant)
                    g_name_candidates = person_name_candidates(gdoc_after, name_bank)
                    b_name_candidates = person_name_candidates(bdoc_after, name_bank)

                    if g_name_candidates or b_name_candidates:
                        if len(g_name_candidates) == len(b_name_candidates) and g_name_candidates:
                            used_bad = set()
                            name_matches = []
                            success = True
                            for g_tok, g_gender in g_name_candidates:
                                match = None
                                for b_tok, b_gender in b_name_candidates:
                                    if b_tok.i in used_bad:
                                        continue
                                    if b_tok.text.lower() == g_tok.text.lower() and b_gender == g_gender:
                                        match = (b_tok.i, b_gender)
                                        break
                                if match is None:
                                    success = False
                                    break
                                used_bad.add(match[0])
                                name_matches.append(((g_tok.i, g_gender), match))
                            if success and len(name_matches) == len(b_name_candidates):
                                name_seed = (pair_seed + 0x9E3779B9) & 0xFFFFFFFF
                                rng_names = random.Random(name_seed)
                                g_targets = [spec for spec, _ in name_matches]
                                b_targets = [spec for _, spec in name_matches]

                                g_name_text, g_name_swaps = person_name_swap(
                                    gdoc_after, name_bank, rng=rng_names, forced_targets=g_targets
                                )
                                rng_names = random.Random(name_seed)
                                b_name_text, b_name_swaps = person_name_swap(
                                    bdoc_after, name_bank, rng=rng_names, forced_targets=b_targets
                                )
                                if (
                                    g_name_text
                                    and b_name_text
                                    and g_name_swaps
                                    and b_name_swaps
                                    and len(g_name_swaps) == len(b_name_swaps)
                                ):
                                    g_variant = g_name_text
                                    b_variant = b_name_text
                                    name_changed = True
                        # else: mismatch in number of candidates; skip name swap

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
