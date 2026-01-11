#!/bin/bash
#SBATCH -J EchoRefine_Research_v2       # Job name
#SBATCH --account=coc                    # Your account
#SBATCH -N 1                             # One node
#SBATCH --gres=gpu:a100:2                # 2x A100 GPUs (80GB total)
#SBATCH --mem=256G                       # Increased RAM to handle 70B loading spikes
#SBATCH -t 8:00:00                      # 8 hours time limit
#SBATCH --cpus-per-task=16               # Higher CPU count for faster data processing
#SBATCH -q coc-ice                       # QOS
#SBATCH -o /storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine/logs/%j.out
#SBATCH -e /storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine/logs/%j.err

# --- 1. Environment Setup ---
module load anaconda3
module load cuda/12.1

conda activate llm_bti

# Load .env variables if file exists (handling spaces around =)
if [ -f .env ]; then
    export $(grep -v '^#' .env | sed 's/ *= */=/g' | xargs)
fi

# Path to your specific environment python executable
PYTHON_BIN="/storage/ice1/6/3/jyoon370/conda_envs/llm_bti/bin/python"
PROJECT_DIR="/storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine"

# Safety: Redirect ALL cache to storage to avoid 30GB Home limit
export HF_HOME="/storage/ice1/6/3/jyoon370/hf_cache"
export TMPDIR="/storage/ice1/6/3/jyoon370/tmp"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export TOKENIZERS_PARALLELISM=false

# Ensure folders exist
mkdir -p "$PROJECT_DIR/logs" "$HF_HOME" "$TMPDIR"

cd "$PROJECT_DIR"

# --- 2. Check Hardware ---
echo "Starting job on $(hostname) at $(date)"
nvidia-smi

# --- 3. Run Pipeline Stages ---

echo ">>> STAGE 1: Generating 10k Synthetic Refinement Samples..."
time $PYTHON_BIN generate_multilingual_data.py

if [ $? -eq 0 ]; then
    echo ">>> STAGE 2: Starting QLoRA Fine-Tuning (Rank 32)..."
    time $PYTHON_BIN train_multilingual.py
else
    echo "Data generation failed. Exiting."
    exit 1
fi

if [ $? -eq 0 ]; then
    echo ">>> STAGE 2: Starting QLoRA Fine-Tuning (Rank 32)..."
    time $PYTHON_BIN train_direct.py
else
    echo "Data generation failed. Exiting."
    exit 1
fi

if [ $? -eq 0 ]; then
    echo ">>> STAGE 3: Running Multilingual Benchmarks..."
    
    LANGUAGES="npi ben sin mya kor tam hin fra"
    
    for lang in $LANGUAGES; do
        echo "------------------------------------------------"
        echo ">>> Evaluating Language: $lang"
        echo "------------------------------------------------"
        
        # Run the python script with the required argument
        time $PYTHON_BIN evaluate_multilingual_resumable.py --lang $lang
        
        # Check if it failed
        if [ $? -ne 0 ]; then
            echo "Warning: Evaluation for $lang failed or timed out."
        fi
    done
else
    echo "Fine-tuning failed. Exiting."
    exit 1
fi

echo ">>> ALL RESEARCH STAGES COMPLETE at $(date)"
