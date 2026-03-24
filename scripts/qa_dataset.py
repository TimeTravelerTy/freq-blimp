#!/usr/bin/env python3
"""
QA script for FreqBLiMP generated datasets.

Usage:
    python scripts/qa_dataset.py data/processed/freq_blimp_*.jsonl
    python scripts/qa_dataset.py --consistency data/processed/freq_blimp_xtail_cond3_*.jsonl

Always reports:
  - Record count and null rate
  - Non-alpha swap check (period-bug diagnostic)
  - Wearisome % (adjective diagnostic)
  - Occurrence-level semantic regime breakdown (if semantic_regime field present)

With --consistency (cond3 / --consistent-lemma-map datasets only):
  - Noun / adjective / verb lemma-map consistency
  - Top inconsistent keys per POS
"""

import argparse
import collections
import json
import sys

REGIME_OFF = "off"  # lemma_map cache hit — semantic filter not run


def analyze(path: str, check_consistency: bool = False, verbose: bool = False):
    total = null = 0
    noun_c = collections.defaultdict(lambda: collections.Counter())
    adj_c  = collections.defaultdict(lambda: collections.Counter())
    verb_c = collections.defaultdict(lambda: collections.Counter())
    non_alpha = collections.Counter()
    wearisome = adj_total = 0

    # Occurrence-level semantic regime counts per POS
    sem_regime = {"noun": collections.Counter(), "adjective": collections.Counter(), "verb": collections.Counter()}

    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            total += 1
            meta = rec.get("meta", {})
            if meta.get("swap_failed_reason"):
                null += 1
                continue
            for side in ("g_swaps", "b_swaps"):
                for sw in meta.get(side, []):
                    if not isinstance(sw, dict):
                        continue
                    pos  = sw.get("pos", "")
                    old  = sw.get("old", "")
                    repl = (sw.get("lemma") or sw.get("new") or "").lower()
                    regime = sw.get("semantic_regime")

                    # Period/non-alpha check
                    if old and not old.isalpha() and "-" not in old:
                        non_alpha[old] += 1

                    if pos == "adjective":
                        adj_total += 1
                        if repl == "wearisome":
                            wearisome += 1

                    if regime is not None and pos in sem_regime:
                        sem_regime[pos][regime] += 1

                    if check_consistency:
                        if pos == "noun":
                            noun_c[old.lower()][repl] += 1
                        elif pos == "adjective":
                            adj_c[old.lower()][repl] += 1
                        elif pos == "verb":
                            key = (sw.get("src_lemma", ""), sw.get("base_frame", ""))
                            verb_c[key][repl] += 1

    print(f"\n{'='*60}")
    print(f"Dataset: {path}")
    print(f"{'='*60}")
    print(f"Records : {total:,}   Null: {null:,} ({100*null/total:.2f}%)")

    if non_alpha:
        print(f"\n⚠  NON-ALPHA SWAPS: {sum(non_alpha.values())} ({len(non_alpha)} unique tokens)")
        for tok, cnt in non_alpha.most_common(5):
            print(f"   '{tok}': {cnt}")
    else:
        print(f"\n✓  No non-alpha swaps (period bug absent)")

    if adj_total:
        print(f"   Wearisome adj: {wearisome}/{adj_total} ({100*wearisome/adj_total:.1f}%)")

    # Semantic regime breakdown (occurrence-level, only if field present in data)
    any_regime = any(sum(v.values()) > 0 for v in sem_regime.values())
    if any_regime:
        print(f"\nSemantic regime (occurrence-level, excludes lemma_map cache hits):")
        for pos_name in ("noun", "adjective", "verb"):
            c = sem_regime[pos_name]
            total_sem = sum(v for k, v in c.items() if k != REGIME_OFF)
            cache_hits = c.get(REGIME_OFF, 0)
            all_occ = sum(c.values())
            if not all_occ:
                continue
            print(f"  {pos_name.upper()}: {all_occ:,} occ total, {cache_hits:,} cache hits ({100*cache_hits/all_occ:.1f}%)")
            if total_sem:
                satisfied = sum(v for k, v in c.items() if k not in (REGIME_OFF,) and "fallback" not in k)
                fallback_depth = c.get("fallback_depth", 0)
                fallback_unc = c.get("fallback_unconstrained", 0)
                anticoll = c.get("anticollision_fallback", 0)
                print(f"    Fresh samples : {total_sem:,}")
                print(f"    satisfied     : {satisfied:,} ({100*satisfied/total_sem:.1f}%)")
                if fallback_depth:
                    print(f"    fallback_depth: {fallback_depth:,} ({100*fallback_depth/total_sem:.1f}%)")
                if fallback_unc:
                    print(f"    fallback_unc  : {fallback_unc:,} ({100*fallback_unc/total_sem:.1f}%)")
                if anticoll:
                    print(f"    anticollision : {anticoll:,} ({100*anticoll/total_sem:.1f}%)")
                # Occurrence-level: what % of non-cache-hit occurrences are satisfied?
                # (for consistent mapping, cache hits use the lemma_map's stored regime)

    if check_consistency:
        def _report(name, d):
            if not d:
                return
            keys      = len(d)
            total_occ = sum(sum(c.values()) for c in d.values())
            consist_k = sum(1 for c in d.values() if len(c) == 1)
            dom_occ   = sum(c.most_common(1)[0][1] for c in d.values())
            max_repls = max(len(c) for c in d.values())
            print(f"\n{name}  ({keys} unique keys, {total_occ:,} swap occurrences)")
            print(f"  Key-level consistency  : {consist_k}/{keys} ({100*consist_k/keys:.1f}%)")
            print(f"  Occurrence consistency : {dom_occ}/{total_occ} ({100*dom_occ/total_occ:.2f}%)")
            print(f"  Max unique repls/key   : {max_repls}")
            bad = sorted(
                [(k, c) for k, c in d.items() if len(c) > 1],
                key=lambda x: -sum(x[1].values()),
            )
            n_show = 5 if verbose else 3
            for k, c in bad[:n_show]:
                t = sum(c.values())
                dom = c.most_common(1)[0]
                print(f"  ↳ {k!r}  {len(c)} repls  {t} occ  dom={dom[0]!r} ({100*dom[1]/t:.1f}%)")

        print("\n[Consistency — only meaningful for --consistent-lemma-map datasets]")
        _report("NOUN", noun_c)
        _report("ADJECTIVE", adj_c)
        _report("VERB", verb_c)

    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="JSONL dataset files to check")
    ap.add_argument(
        "--consistency", action="store_true",
        help="Check lemma-map consistency (cond3 / --consistent-lemma-map datasets only)",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="Show top 5 inconsistent keys (default: 3)")
    args = ap.parse_args()

    for p in args.paths:
        try:
            analyze(p, check_consistency=args.consistency, verbose=args.verbose)
        except Exception as e:
            print(f"ERROR: {p}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
