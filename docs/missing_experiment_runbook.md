# Missing Reviewer Experiments Runbook

This runbook covers the experiments still required by the reviewers after the
completed significance, gatekeeper-isolation, synthetic-data diagnostics, and
qualitative-example passes.

## 1. Inspect Commands

```bash
python generate_missing_experiments.py --groups high-priority --dry-run
```

`high-priority` expands to:

- decoding ablation: greedy, beam-4, beam-8, MBR k=3/5/10/20.
- CometKiwi threshold sensitivity: always-LLM endpoint, QE margins 0.1/0.2/0.3,
  always-anchor endpoint.
- stronger baselines: NLLB-200 distilled and Aya-23-35B.

## 2. Run High-Priority Experiments

Interactive or existing allocation:

```bash
python generate_missing_experiments.py --groups high-priority --skip-existing
```

Slurm:

```bash
sbatch run_missing_experiments.sbatch
```

The Slurm wrapper requests 8 hours with 2 A100 GPUs. This fits the queue's
GPU-minute policy; a 16-hour request with 2 A100s is held with
`QOSMaxGRESMinutesPerJob`. It is safe to resubmit the same job: the runner uses
`--skip-existing`, each evaluator run writes `progress_<lang>_<run>.jsonl`, and
incomplete evaluator runs resume from the existing progress line count.

The default Python is `/storage/ice1/6/3/jyoon370/miniconda3/bin/python`. Override
with `PYTHON_BIN=/path/to/python sbatch run_missing_experiments.sbatch` if you
recreate a dedicated environment.

Do not commit a real Hugging Face token. If gated model access is needed, copy
`.env.example` to `.env`, set `HF_TOKEN=...` or `HUGGING_FACE_HUB_TOKEN=...`,
and submit normally. The Slurm wrapper sources `.env` at runtime, and `.env` is
ignored by git.

Useful Slurm overrides:

```bash
MISSING_EXPERIMENT_GROUPS="decoding thresholds" sbatch run_missing_experiments.sbatch
MISSING_EXPERIMENT_LANGUAGES="fra npi tam" sbatch run_missing_experiments.sbatch
MISSING_EXPERIMENT_LIMIT=128 sbatch run_missing_experiments.sbatch
```

Prefer separate submissions if the all-in-one job does not finish:

```bash
MISSING_EXPERIMENT_GROUPS="decoding" sbatch run_missing_experiments.sbatch
MISSING_EXPERIMENT_GROUPS="thresholds" sbatch run_missing_experiments.sbatch
MISSING_EXPERIMENT_GROUPS="components" sbatch run_missing_experiments.sbatch
MISSING_EXPERIMENT_GROUPS="baselines" sbatch run_missing_experiments.sbatch
```

## 3. Run Component Ablation

```bash
python generate_missing_experiments.py --groups components --skip-existing
```

This produces:

- `abl_component_no_bt`: LLM refinement without back-translation.
- `abl_component_bt_no_mbr`: back-translation with greedy decoding, no MBR.
- `abl_component_mbr_no_qe`: back-translation plus MBR, no QE gate.
- `abl_component_full_qe`: full system with QE margin 0.2.

The mBART anchor row comes from the existing `results_<lang>.json` files.

## 4. Aggregate Tables

```bash
python analysis/aggregate_missing_experiments.py
```

Expected outputs:

- `results/tables/decoding_ablation_by_language.csv`
- `results/tables/decoding_ablation_macro.csv`
- `results/tables/threshold_sensitivity_by_language.csv`
- `results/tables/threshold_sensitivity_macro.csv`
- `results/tables/component_ablation_by_language.csv`
- `results/tables/component_ablation_macro.csv`
- `results/tables/strong_baselines_by_language.csv`
- `results/tables/strong_baselines_macro.csv`
- `results/tables/missing_experiment_outputs.json`

The missing-output manifest lists any configurations that have not finished yet.

## 5. Baseline Wrappers

The attachment's command names are supported:

```bash
python baselines/run_nllb.py --lang all --output-dir results/strong_baselines --progress-dir results/strong_baselines
python baselines/run_aya.py --lang all --batch-size 2 --output-dir results/strong_baselines --progress-dir results/strong_baselines
```

Both wrappers call `run_strong_baselines.py` and use local FLORES-200 files when
available.
