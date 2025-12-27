#!/bin/bash
#SBATCH -J EchoRefine_Full_Task
#SBATCH --account=coc
#SBATCH -N 1
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=192G
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=8
#SBATCH -q coc-ice
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# --- 1. Environment Setup ---
# Load CUDA so the GPU is visible to torch
module load cuda/12.1

# Define the absolute path to your environment's python
# This ensures we use the version that has 'torch' installed
PYTHON_PATH="/storage/ice1/6/3/jyoon370/conda_envs/llm_bti/bin/python"

# Ensure we are in the correct working directory
cd /storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine

export HF_HOME="/storage/ice1/6/3/jyoon370/hf_cache"
export TOKENIZERS_PARALLELISM=false
# This helps 'bitsandbytes' find the CUDA library on PACE
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# --- 2. Run Python using the absolute path ---
echo "Running job on $(hostname)"
echo "Using Python from: $PYTHON_PATH"

# Run the training script
$PYTHON_PATH train_echorefine.py

# Run the evaluation script
$PYTHON_PATH evaluate_final.py
