# freq-BLiMP

A pipeline for generating and evaluating **frequency-controlled BLiMP-style minimal pairs**.

This repository focuses on:
- Creating frequency-swapped variants of BLiMP items (`good_freq` / `bad_freq`)
- Scoring pairs with causal LMs via sentence NLL
- Producing accuracy summaries and analysis plots

## What This Repo Does

At a high level:
1. Load BLiMP phenomena from a tier config
2. Swap nouns/adjectives/verbs under Zipf-based constraints
3. Write generated pairs with metadata (`good_freq`, `bad_freq`, swap traces, Zipf aggregates)
4. Score generated/original pairs with one or more LMs
5. Compute BLiMP accuracy and produce diagnostic plots

## Repository Layout

- `scripts/make_freq_blimp.py`: Main dataset generator (single-run and batch)
- `scripts/blimp_pair_scores_timestamp_batch.py`: Batch NLL scoring over datasets
- `scripts/blimp_accuracy.py`: Accuracy computation from pair-score JSONL
- `scripts/evaluate_freq_blimp.py`: End-to-end evaluate pipeline (score -> accuracy -> plots)
- `scripts/plot_zipf_vs_nll.py`: Zipf vs NLL analysis
- `scripts/plot_zipf_vs_token_len.py`: Zipf vs token-length analysis
- `scripts/regime_diagnostics.py`: Window/regime diagnostics
- `src/`: Core generation and scoring modules
- `configs/`: Tier/config mapping files
- `data/processed/`: Generated datasets and derived artifacts
- `scripts/legacy/`: Older scripts preserved for reference

## Setup

### 1. Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Install spaCy English model

```bash
python3 -m spacy download en_core_web_sm
```

### 3. (Optional but common) Hugging Face auth

For gated models (e.g., Llama variants), authenticate first:

```bash
huggingface-cli login
```

## Data Format

Generated dataset records contain at least:
- `phenomenon`, `subtask`, `field`, `idx`
- `good_freq`, `bad_freq`
- `meta.g_swaps`, `meta.b_swaps`
- `meta.zipf_swapped_position_aggregates`

Note:
- Some scripts remain backward-compatible with older `good_rare` / `bad_rare` datasets.

## Quick Start

### Generate one dataset

```bash
python3 scripts/make_freq_blimp.py \
  --zipf_max_all 3.0 \
  --zipf_min_all 2.2
```

### Generate a batch with explicit windows

```bash
python3 scripts/make_freq_blimp.py \
  --zipf-windows 1.2-2.0 2.2-3.0 3.6-5.0
```

### Dry-run batch commands

```bash
python3 scripts/make_freq_blimp.py \
  --zipf-windows 1.2-2.0 2.2-3.0 \
  --dry-run
```

### Evaluate datasets end-to-end

```bash
python3 scripts/evaluate_freq_blimp.py \
  --data-pattern "data/processed/*freq_blimp*.jsonl" \
  --variant auto
```

This produces:
- Pair scores in `results/blimp_pair_scores/`
- Accuracy JSON in `results/blimp_accuracy_runs/`
- CSV summaries in `results/eval_runs/`
- Plots/tables in `results/analysis_plots/`

## Core Workflows

### A) Generation only

Use `scripts/make_freq_blimp.py` with either:
- `--zipf-values` for fixed-point batches
- `--zipf-windows` for lower/upper window batches

Useful knobs:
- `--swap_target` (`nouns`, `adjectives`, `verbs`, `all`)
- `--limit`
- `--match-token-count`
- `--zipf_weighted_sampling --zipf_temp ...`

### B) Scoring only

```bash
python3 scripts/blimp_pair_scores_timestamp_batch.py \
  --pattern "data/processed/*freq_blimp*.jsonl" \
  --models meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-3B \
  --variant auto
```

### C) Accuracy only

```bash
python3 scripts/blimp_accuracy.py \
  --scores results/blimp_pair_scores/<your_pair_scores_file>.jsonl
```

## Important CLI Notes

- `--variant freq` is the canonical swapped-variant name.
- `--variant rare` is accepted in some scripts for compatibility and mapped to `freq` behavior.
- `scripts/legacy/` contains older utilities that may still use historical naming conventions.

## Reproducibility Tips

- Set `--seed` during generation.
- Keep generated datasets in `data/processed/` and pair-scores in `results/blimp_pair_scores/`.
- Use `--run-timestamp` in evaluation scripts when you want stable output file prefixes.

## Legacy Scripts

Older scripts live in `scripts/legacy/`. They are kept for reference and prior analyses, but current workflows should use:
- `scripts/make_freq_blimp.py`
- `scripts/evaluate_freq_blimp.py`
- `scripts/blimp_pair_scores_timestamp_batch.py`
