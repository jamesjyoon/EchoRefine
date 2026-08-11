#!/bin/bash
#SBATCH -J EchoRefine_Research_v2
#SBATCH --account=coc
#SBATCH -N 1
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=256G
#SBATCH -t 8:00:00
#SBATCH --cpus-per-task=16
#SBATCH -q coc-ice
#SBATCH -o /storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine/logs/%j.out
#SBATCH -e /storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine/logs/%j.err

# --- 1. Environment Setup ---
module load anaconda3
module load cuda/12.1

# Define absolute paths
PYTHON_BIN="/storage/ice1/6/3/jyoon370/conda_envs/llm_bti/bin/python"
PROJECT_DIR="/storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine"

# Export Environment Variables
export HF_HOME="/storage/ice1/6/3/jyoon370/hf_cache"
export TMPDIR="/storage/ice1/6/3/jyoon370/tmp"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export TOKENIZERS_PARALLELISM=false

# Load .env manually to ensure variables exist
cd "$PROJECT_DIR"
if [ -f .env ]; then
    export $(grep -v '^#' .env | sed 's/ *= */=/g' | xargs)
fi

# Ensure folders exist
mkdir -p "$PROJECT_DIR/logs" "$HF_HOME" "$TMPDIR"

# --- 2. Run Pipeline Stages ---
echo "Starting job on $(hostname) at $(date)"

# --- STEP 1: GENERATE DATA ---
# echo ">>> STAGE 1: Generating 10k Synthetic Samples..."
# time $PYTHON_BIN generate_multilingual_data.py

# if [ $? -ne 0 ]; then
#     echo "CRITICAL: Data generation failed. Exiting."
#     exit 1
# fi

# --- STEP 2a: TRAIN REFINEMENT ADAPTER ---
# echo ">>> STAGE 2a: Starting QLoRA Fine-Tuning (Refinement)..."
# time $PYTHON_BIN train_multilingual.py

# if [ $? -ne 0 ]; then
#     echo "CRITICAL: Refinement training failed. Exiting."
#     exit 1
# fi

# --- STEP 2b: TRAIN DIRECT ADAPTER ---
# echo ">>> STAGE 2b: Starting QLoRA Fine-Tuning (Direct Translation)..."
# # Note: VRAM is cleared automatically between python calls
# time $PYTHON_BIN train_direct.py

# if [ $? -ne 0 ]; then
#     echo "CRITICAL: Direct training failed. Exiting."
#     exit 1
# fi

# --- STEP 3: EVALUATION ---
echo ">>> STAGE 3: Running Multilingual Benchmarks..."

LANGUAGES="npi ben sin mya kor tam hin fra"

for lang in $LANGUAGES; do
    echo "------------------------------------------------"
    echo ">>> Evaluating Language: $lang"
    echo "------------------------------------------------"
    
    time $PYTHON_BIN evaluate_multilingual_resumable.py --lang $lang --limit 1012
    
    if [ $? -ne 0 ]; then
        echo "Warning: Evaluation for $lang failed or timed out."
    fi
done

echo ">>> ALL RESEARCH STAGES COMPLETE at $(date)"
