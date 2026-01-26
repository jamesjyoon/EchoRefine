# EchoRefine

EchoRefine is a research-oriented codebase for training and evaluating language models on multilingual “echo” or refinement-style tasks, with support for direct and multilingual training, resumable evaluation, and data generation pipelines.

## Features

- **Multilingual data generation** via `generate_multilingual_data.py`, with explicit handling of low-resource languages (including Tamil).[page:1]  
- Resumable multilingual evaluation through `evaluate_multilingual_resumable.py` so long-running benchmarks can be safely restarted.[page:1]  
- Separate training scripts for:
  - Direct training (`train_direct.py`)
  - Multilingual training (`train_multilingual.py`, designed to be resumable)[page:1]  
- Reproducible environments managed through `requirements.txt`.

## Repository structure

- `generate_multilingual_data.py`: Script to construct multilingual datasets, including low-resource languages such as Tamil. Adjust language lists, sampling strategies, and output paths inside this script.
- `evaluate_multilingual_resumable.py`: Evaluation entry point for multilingual experiments with checkpointing/resume support (e.g., to continue after a crash or pre-emption).
- `train_direct.py`: Training script for “direct” (typically monolingual or single-configuration) experiments.  
- `train_multilingual.py`: Training script for multilingual experiments, with logic to resume from previous checkpoints.
- `requirements.txt`: Python dependencies for running experiments end-to-end.


