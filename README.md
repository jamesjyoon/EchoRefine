# EchoRefine: Fine-Tuned LLM Auditing for Low-Resource Translation

EchoRefine is a hybrid machine translation refinement pipeline. It uses mBART-50
to produce an anchored draft, a QLoRA-adapted Llama-3.3-70B-Instruct refiner to
propose edits, MBR-style candidate selection to choose a local consensus output,
and a CometKiwi quality-estimation gate to decide whether to keep the LLM edit or
fall back to the mBART anchor.

## Reviewer Revision Note

EchoRefine should be described as producing selective gains, not as strictly
dominating every baseline on every language and metric. The hybrid gatekeeper is
risk-averse: it improves stability in several settings, but it can preserve weak
anchors and can reduce potential gains when the anchor is poor. The scripts in
this repository support the requested statistical tests, ablations, stronger
baselines, synthetic-data diagnostics, and qualitative examples.

## Current Outputs

The completed multilingual FLORES-200 devtest runs are stored as:

- `progress_<lang>.jsonl`: segment-level outputs for each language.
- `results_<lang>.json`: aggregate BLEU, chrF, and COMET scores.
- `chart_<lang>.png`: per-language comparison plots.

Supported language IDs are `npi`, `ben`, `sin`, `mya`, `kor`, `tam`, `hin`, and
`fra`.

## Common Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Aggregate the existing result JSON files into paper tables:

```bash
python analysis/aggregate_results.py
```

Run paired bootstrap significance tests from the completed progress logs:

```bash
python analysis/significance.py --lang all --metrics bleu chrf
python analysis/significance.py --lang fra --metrics comet --output results/tables/significance_fra_comet.csv
```

Generate synthetic dataset diagnostics:

```bash
python data/validate_synthetic.py --dataset train_data_multilingual.csv --flores-overlap
```

Extract qualitative examples for paper error analysis:

```bash
python examples/generate_paper_examples.py
```

Run an ablation without overwriting the completed default run:

```bash
python evaluate_multilingual_resumable.py --lang fra --decoding-method greedy --gate-mode always_llm --run-name fra_greedy_raw
python evaluate_multilingual_resumable.py --lang fra --decoding-method mbr --num-candidates 10 --gate-mode qe --qe-margin 0.2 --run-name fra_mbr_k10_qe_m02
```

Generate the reviewer-requested missing experiments:

```bash
# Inspect the exact commands first.
python generate_missing_experiments.py --groups high-priority --dry-run

# Run decoding, threshold, and strong-baseline experiments, skipping completed outputs.
python generate_missing_experiments.py --groups high-priority --skip-existing

# Aggregate completed ablation outputs into paper tables.
python analysis/aggregate_missing_experiments.py

# Cluster version; override groups/languages with env vars if needed.
sbatch run_missing_experiments.sbatch
```

Component ablation is available separately:

```bash
python generate_missing_experiments.py --groups components --skip-existing
python analysis/aggregate_missing_experiments.py
```

Run stronger baselines:

```bash
python run_strong_baselines.py --baseline nllb-200 --lang all
python run_strong_baselines.py --baseline aya-23 --lang fra --batch-size 2
```

## Repository Structure

```text
EchoRefine/
  analysis/
    aggregate_results.py       # CSV tables from existing result files
    aggregate_missing_experiments.py # Tables from ablation/sensitivity runs
    significance.py            # Paired bootstrap CIs and p-values
  baselines/
    nllb_200.py                # NLLB-200 baseline wrapper
    aya_23.py                  # Aya-23 prompting baseline wrapper
  configs/
    ablations.yaml             # Candidate, decoding, and QE ablation grid
  data/
    validate_synthetic.py      # Synthetic dataset diagnostics
  docs/
    paper_revision_notes.md    # Manuscript text and reviewer-response notes
  examples/
    generate_paper_examples.py # Qualitative example extraction
  generate_missing_experiments.py # Reviewer ablation/baseline runner
  run_missing_experiments.sbatch  # Slurm wrapper for missing experiments
```

## Reproducibility Notes

- Default `evaluate_multilingual_resumable.py` settings preserve the original
  output names: `progress_<lang>.jsonl`, `results_<lang>.json`, and
  `chart_<lang>.png`.
- Non-default ablation settings add a suffix or `--run-name`, preventing mixed
  progress logs.
- The recommended model identifier used in the experiments is
  `meta-llama/Llama-3.3-70B-Instruct`; the paper should cite the exact model card
  and access date used for the submitted run.
