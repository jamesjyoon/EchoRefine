# EchoRefine

EchoRefine is a research-oriented codebase for training and evaluating language models on multilingual “echo” or refinement-style tasks, with support for direct and multilingual training, resumable evaluation, and data generation pipelines.[page:1][page:2]

## Features

- **Multilingual data generation** via `generate_multilingual_data.py`, with explicit handling of low-resource languages (including Tamil).[page:1]  
- Resumable multilingual evaluation through `evaluate_multilingual_resumable.py` so long-running benchmarks can be safely restarted.[page:1]  
- Separate training scripts for:
  - Direct training (`train_direct.py`)
  - Multilingual training (`train_multilingual.py`, designed to be resumable)[page:1]  
- Reproducible environments managed through `requirements.txt`.[page:1]  

## Repository structure

- `generate_multilingual_data.py`: Script to construct multilingual datasets, including low-resource languages such as Tamil. Adjust language lists, sampling strategies, and output paths inside this script.[page:1]  
- `evaluate_multilingual_resumable.py`: Evaluation entry point for multilingual experiments with checkpointing/resume support (e.g., to continue after a crash or pre-emption).[page:1]  
- `train_direct.py`: Training script for “direct” (typically monolingual or single-configuration) experiments.[page:1]  
- `train_multilingual.py`: Training script for multilingual experiments, with logic to resume from previous checkpoints.[page:1]  
- `requirements.txt`: Python dependencies for running experiments end-to-end.[page:1]  


