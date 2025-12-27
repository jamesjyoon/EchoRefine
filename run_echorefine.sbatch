#!/bin/bash
#SBATCH -J EchoRefine_Full_Task           # Job name
#SBATCH --account=coc                    # Your account (based on your logs)
#SBATCH -N 1                             # Number of nodes
#SBATCH --gres=gpu:A100:2                # Request 2x A100 GPUs
#SBATCH --mem=192G                       # CPU RAM (70B model loading needs high RAM)
#SBATCH -t 08:00:00                      # 8 hours limit (Train ~6h, Eval ~2h)
#SBATCH -q coc-ice                       # QOS (based on your logs)
#SBATCH -o logs/%j.out                   # Standard output log
#SBATCH -e logs/%j.err                   # Standard error log
#SBATCH --cpus-per-task=8

# --- 1. Environment Setup ---
module load anaconda3
module load cuda/12.1                    # Match your torch/bitsandbytes version
conda activate /home/hice1/jyoon370/scratch/conda_envs/llm_bti

# Safety: Set Hugging Face cache to storage (avoid filling home directory)
export HF_HOME="/storage/ice1/6/3/jyoon370/hf_cache"
export TOKENIZERS_PARALLELISM=false

# --- 2. Check GPU Status ---
echo "Starting job on node: $(hostname)"
nvidia-smi

# --- 3. Step 1: Train the Adapter (Fine-Tuning) ---
echo ">>> STARTING STAGE 1: QLoRA Fine-Tuning"
# We use 'time' to track how long training takes for your paper
time python train_echorefine.py

# Check if training succeeded before starting evaluation
if [ $? -eq 0 ]; then
    echo "Fine-tuning completed successfully."
else
    echo "Fine-tuning failed. Exiting before evaluation."
    exit 1
fi

# --- 4. Step 2: Evaluation (Benchmarking) ---
echo ">>> STARTING STAGE 2: Evaluation & Plotting"
# This script will use the saved adapter from Stage 1
time python evaluate_final.py

echo ">>> ALL TASKS COMPLETE."
