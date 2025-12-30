#!/bin/bash
#SBATCH -J EchoRefine_Final
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
module load cuda/12.1
PYTHON_PATH="/storage/ice1/6/3/jyoon370/conda_envs/llm_bti/bin/python"

# Force ALL caches and temp files to the LARGE storage area
export HF_HOME="/storage/ice1/6/3/jyoon370/hf_cache"
export TMPDIR="/storage/ice1/6/3/jyoon370/tmp"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export TOKENIZERS_PARALLELISM=false

mkdir -p $HF_HOME $TMPDIR logs

cd /storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine

# --- 2. Run Pipeline ---
echo "Starting Stage 1: Fine-Tuning..."
$PYTHON_PATH train_echorefine.py

if [ $? -eq 0 ]; then
    echo "Fine-tuning successful. Starting Stage 2: Evaluation..."
    $PYTHON_PATH evaluate_final.py
else
    echo "Training failed. Check logs/%j.err"
    exit 1
fi

