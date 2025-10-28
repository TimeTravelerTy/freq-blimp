# Grammar-Features Pilot — Project Context & Instructions

**High-level goal:** Discover, stress-test, and causally validate **grammar-sensitive internal features** in open-weights LMs (e.g., Llama/Mistral) using **closed data**. We (1) build controlled evals (G/U × Typical/Rare), (2) learn **sparse autoencoders (SAEs)** to expose features, (3) link features to **linguistic phenomena** via probes, and (4) **intervene** (zero/clamp) to test causal responsibility.

---

## Train/Eval protocol (pilot)

- **Train on Typical; evaluate on Rare**  
  - **SAE training:** Unsupervised, on **Typical-only** activations; keep a small **Typical hold-out** shard for reconstruction/sparsity tuning.  
  - **Probes:** Supervised probes trained on **Typical** labels (BLiMP-style minimal pairs / Tier-A).  
  - **Evaluation:** **Rare/Nonce = test-only** (never seen by SAEs or probes). Report Typical→Rare deltas.  
  - **Optional (pilot+):** A **small GEC slice** as an *additional* OOD test to sanity-check natural errors (longer contexts, noisier labels).

**Why:** This isolates **structure-generalization**: features that truly encode grammar should retain selectivity on Rare/Nonce even when trained on Typical distributions.

---

## Timeline (pilot sprint = 4 weeks)

- **Week 1 — Data prep & stressors**
  - Build **Tier-A** minimal-pair datasets: **G/U × Typical/Rare**.
  - Rare = **multi-swap nouns** with **rare, real lemmas**; preserve tags (NN/NNS) and **COUNT/MASS** when quantifiers require it.
  - Aim for **nonce-ish but grammatical** G-Rare and **same violation** U-Rare.
  - **Split discipline:** Mark **Typical** items for train/dev; **Rare** held out as **test-only**.

- **Week 2 — Features (SAEs)**
  - Choose target model + layers. Train **SAEs** on **Typical-only** hidden states (small shards).
  - Lock SAE hyperparams (sparsity, code size). Save dictionaries & activations.
  - Validate on **Typical hold-out** (loss-recovered / reconstruction; sparsity).

- **Week 3 — Probing & analysis**
  - Train **simple probes** (e.g., logistic/linear) from SAE codes to **Typical** Tier-A labels/phenomena.
  - Derive axes:  
    - **Polarity (G vs U)** = sign of Δ-activation on minimal pairs.  
    - **Robustness** = sign consistency and ≥ α effect retention on **Rare** (e.g., α=0.6).
  - Run **Rare stress tests** (evaluation only): Typical→Rare deltas; identify **ConceptOnly** features.
  - Curate a short list of candidate features per phenomenon.

- **Week 4 — Causality & report**
  - **Causal zeroing/clamping** on candidate features; measure accuracy/logit-margin shifts on **Typical dev** and **Rare test**.
  - Optional: **GEC slice eval** (pilot+): small set of natural error/fix pairs; measure pre→post likelihood and intervention effects.
  - Draft report: methodology, feature cards, causal findings.

---

## Design principles

- **Closed-data, open-weights.** No train-data matching; focus on black-box-compatible methods and open activations.
- **Label integrity first.** All edits preserve the grammatical phenomenon; QC is structural (tags/COUNT/MASS), not fluency.
- **Determinism.** Seed per pair; Good/Bad share the same sampled replacements.
- **Lean first, deepen later.** Start with noun swaps; add verbs/names/resources incrementally.
- **Strict OOD discipline.** **Rare** (and **GEC**, if used) are never seen in SAE or probe training.

---

## Week-1 data generation (what exists now)

- Targets: **SV-Agreement, Det–N Agreement, Reflexive/Anaphor, Quantifiers**.
- Edits: **Noun-only multi-swap** (or top-k), skip `ROOT`/entities, preserve **NN/NNS** via `lemminflect`.
- Rarity: `wordfreq` **Zipf < τ**.
- Countability: filter swaps with a **countability lexicon** (BECL lemma→COUNT/MASS/FLEX) **only** when quantifiers require it.
- Output: JSONL per item with **G-Typical, U-Typical, G-Rare, U-Rare** and swap metadata.
- **Split file map:** `typical_train.jsonl`, `typical_dev.jsonl`, `rare_test.jsonl` (+ optional `gec_test.jsonl`).

---

## Metrics & checks (minimal)

- **Accuracy & AUROC:** Typical dev vs Rare test per phenomenon; report Δ.
- **Margin retention:** Δ(logit) between grammatical vs ungrammatical; Typical→Rare drop.
- **Selectivity:** Feature→phenomenon MI / top-k overlap Typical vs Rare.
- **Causal sign stability:** Interventions move predictions in the **same direction** on Typical and Rare.
- **(Optional GEC)**: pre→post edit likelihood; local intervention effect near edited span.

---

## Week-2/3/4 interfaces (so code stays future-proof)

- **SAE training I/O:** given `dataset.jsonl` + model + layer ids → saves `codes.pt`, `dict.pt`, `cfg.json`.
- **Probing I/O:** given `codes.pt` + labels → saves metrics, ROC, per-feature weights; emits **feature cards** (top examples, activation stats).
- **Causal I/O:** given feature indices + clamp/zero spec → runs eval on **typical_dev** and **rare_test**; saves deltas.

---

## Immediate next steps (short)

1. **Verb randomization (opt-in module)**  
   - Swap **main lexical verb** only (exclude AUX).  
   - Preserve morph tag (`VB/VBD/VBG/VBN/VBP/VBZ`) with `lemminflect`.  
   - **Transitivity guard:** prefer candidates matching presence/absence of direct object (light verb-frames list).  
   - Same per-pair RNG so Good/Bad align.  
   - QC stays structural (only VERB diffs; tag identical).

2. **Uncommon names**
   - Replace PERSON entities with **rare real names** (cap-preserving, coref-consistent).  
   - Skip if gendered pronouns present to avoid agreement drift.  
   - Reduces BLiMP frequent-name bias.

3. **Replace hardcoded lemma lists with datasets**  
   - **Rare nouns/verbs:** build from WordNet/other lexicons and filter by Zipf.  
   - **Countability:** public **countability lexicon** (collapse sense→lemma majority; FLEX if mixed).  
   - **Verb frames:** lightweight transitive/intransitive flags (VerbNet-style) or curated TSV.  
   - **Names:** uncommon given/surnames from public lists; rarity-filtered.


## Nonce Lemma Bank — Datasets & Sampling (concise)

**Purpose:** Build an open, constraint-aware lemma bank to sample **rare-but-real** replacements for **NOUN/VERB/ADJ/ADV** while preserving phenomenon labels.

### Data sources (open + lightweight)
- **Lemma inventory:** Open English WordNet (OEWN) → lemma + POS.
- **Rarity score:** `wordfreq` Zipf (single threshold knob, e.g., Zipf < 2.8–3.0).
- **Noun countability:** Wiktextract (Wiktionary) → `countable/uncountable/plural-only`; BECL optional.
- **Verb frames:** VerbNet → derive `has_object` (transitive) / intransitive flags.
- **Inflection:** Lemminflect (match NN/NNS, VBZ/VBD/…).
- **Names:** SSA given names + US Census surnames (sample tail).

### Build (one-time; cache as `lemma_bank.parquet`)
1) From **OEWN**, collect `{lemma, POS}` for NOUN/VERB/ADJ/ADV.  
2) Annotate **Zipf** with `wordfreq`.  
3) Join **Wiktextract** countability (nouns) → collapse to majority label per lemma.  
4) Tag **VerbNet** frames → `verb_has_object: True/False`.  
5) Store minimal schema:
```json
{"lemma":"bequeath","pos":"VERB","zipf":2.4,"verb_has_object":true,"sources":["OEWN","wordfreq","VerbNet"]}

---

## Minimal repo (current → expandable)