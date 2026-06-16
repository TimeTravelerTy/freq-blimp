# FreqBLiMP Eval

This repository contains evaluation, QC, and paper-analysis code for FreqBLiMP.
The generator and final datasets live in the companion
[`freqblimp-generation`](https://github.com/TimeTravelerTy/freqblimp-generation)
repository.

Current scripts assume generator-produced records with FreqBLiMP fields such as
`good_freq`, `bad_freq`, `sentence_good`, `sentence_bad`, regime names `head`,
`tail`, and `xtail`, and per-paradigm manifests.

## What Is Included

- `scripts/score_acceptability_methods.py`: score minimal pairs with supported
  acceptability methods.
- `scripts/blimp_pair_scores_timestamp_batch.py`: batch pair scoring.
- `scripts/blimp_accuracy.py`: accuracy summaries from pair-score JSONL files.
- `scripts/qc_freq_blimp.py`: dataset QC checks.
- `scripts/make_frequency_figures_current.py` and related `analyze_*` scripts:
  paper analysis and figure/table generation.
- `src/`: scoring, data-loading, lexical support, and analysis helpers.
- `configs/`: BLiMP field and analysis configuration files.
- `data/processed/blimp_original.jsonl`: original BLiMP comparison data.
- `data/external/becl_lemma.tsv` and small processed support inventories used by
  QC/analysis scripts.

Large model scores, paper result bundles, logs, and local outputs are ignored by
Git. Publish or fetch those as external artifacts when needed.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

For gated Hugging Face models, authenticate outside this repo before scoring.

## Expected Data

Point scripts at the final dataset in the generation repo:

```text
../freq-blimp/data/freqblimp/
  head/
  tail/
  xtail/
```

Each regime contains 67 paper-scope paradigm files and matching manifests.
Evaluation defaults should use the 67-subtask paper-scope set unless you
explicitly include the two generator-only extra paradigms.

## Score Acceptability

Score one or more dataset files with a causal LM:

```bash
python3 scripts/score_acceptability_methods.py \
  --pattern "../freq-blimp/data/freqblimp/head/*.jsonl" \
  --models meta-llama/Llama-3.1-8B \
  --methods lp_readout \
  --variant freq \
  --output-dir results/acceptability_pair_scores
```

`lp_readout` is the paper-facing name for the former plain-LM NLL scoring path.
Raw score fields may still include `*_total_nll` because they store underlying
negative log-likelihood values.

## Summaries And Analysis

Compute pair-score accuracy:

```bash
python3 scripts/blimp_accuracy.py \
  --scores results/acceptability_pair_scores/<pair-score-file>.jsonl
```

Run diagnostics or paper assets from scored outputs:

```bash
python3 scripts/regime_diagnostics.py --help
python3 scripts/make_frequency_figures_current.py --help
python3 scripts/analyze_linguistic_frequency_effects.py --help
```

The analysis scripts write to `results/` by default. Keep generated result
bundles outside Git and attach final bundles as release/artifact files.

## Repository Hygiene

- Keep source changes in Git, but keep model outputs, logs, local caches, and
  bulky result bundles out of Git.
- The public repository should be named `freq-blimp-eval`; update the GitHub
  repository name/remote when publishing.
