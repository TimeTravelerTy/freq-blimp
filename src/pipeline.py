import random, spacy, yaml, time
from .io import load_blimp, write_jsonl
from .becl import load_becl_tsv
from .quantifier import load_quant_rules, requirement
from .edits import noun_swap_all, candidate_nouns
from .rarity import is_rare_lemma


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
                show_progress=True):
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
                rng = random.Random(pair_seed)

                g_candidates = candidate_nouns(gdoc)
                g_candidates.sort(key=lambda t: t.i)
                b_candidates = candidate_nouns(bdoc)
                shared_indices = {t.i for t in b_candidates}
                target_specs = [
                    (tok.i, tok.tag_)
                    for tok in g_candidates
                    if tok.i in shared_indices
                ]
                if noun_mode == "k":
                    limit = max(0, min(k, len(target_specs)))
                    target_specs = target_specs[:limit]

                if target_specs:
                    rng = random.Random(pair_seed)
                    g_rare, g_swaps = noun_swap_all(
                        gdoc, rare_pool,
                        noun_mode=noun_mode, k=k, zipf_thr=None,
                        becl_map=becl_map, req=req_g, rng=rng,
                        forced_targets=target_specs
                    )
                    # Reuse the SAME rng sequence for Bad by re-seeding with the same seed
                    rng = random.Random(pair_seed)
                    b_rare, b_swaps = noun_swap_all(
                        bdoc, rare_pool,
                        noun_mode=noun_mode, k=k, zipf_thr=None,
                        becl_map=becl_map, req=req_b, rng=rng,
                        forced_targets=target_specs
                    )
                else:
                    g_rare, g_swaps = None, []
                    b_rare, b_swaps = None, []

                # Only set good_rare and bad_rare if both variants received aligned swaps
                if (
                    g_rare
                    and b_rare
                    and g_swaps
                    and b_swaps
                    and len(g_swaps) == len(b_swaps)
                ):
                    good_rare_val = g_rare
                    bad_rare_val = b_rare
                else:
                    good_rare_val = None
                    bad_rare_val = None

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
                        "noun_mode": noun_mode,
                        "k": k,
                        "zipf_thr": zipf_thr,
                        "req_good": req_g,
                        "req_bad": req_b
                    }
                })
                processed += 1
                if show_progress:
                    last_update = _print_progress(processed, total_items, start_time, last_update)

    write_jsonl(out_path, records)
