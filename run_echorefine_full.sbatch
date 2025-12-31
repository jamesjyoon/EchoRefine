#!/bin/bash
#SBATCH -J EchoRefine_Final
#SBATCH --account=coc
#SBATCH -N 1
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=192G
#SBATCH -t 08:00:00
#SBATCH --cpus-per-task=8
#SBATCH -q coc-ice
#SBATCH -o /storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine/logs/%j.out
#SBATCH -e /storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine/logs/%j.err

# --- 1. Define Absolute Paths ---
PROJECT_DIR="/storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine"
PYTHON_BIN="/storage/ice1/6/3/jyoon370/conda_envs/llm_bti/bin/python"
CACHE_DIR="/storage/ice1/6/3/jyoon370/hf_cache"

# --- 2. Environment Setup ---
module load cuda/12.1
export HF_HOME="$CACHE_DIR"
export TMPDIR="/storage/ice1/6/3/jyoon370/tmp"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(grep HUGGING_FACE_HUB_TOKEN .env | cut -d '=' -f2)
export HF_HUB_TOKEN=$HF_TOKEN

# Ensure directories exist before starting
mkdir -p "$PROJECT_DIR/logs" "$TMPDIR" "$HF_HOME"

# Move into the project directory
cd "$PROJECT_DIR"

# --- 3. Run Pipeline ---
echo "Job started on $(hostname) at $(date)"
echo "Working directory: $(pwd)"

echo ">>> Starting Stage 1: Fine-Tuning..."
$PYTHON_BIN train_echorefine.py

if [ $? -eq 0 ]; then
    echo ">>> Training successful. Starting Stage 2: Evaluation..."
    $PYTHON_BIN evaluate_final.py
else
    echo ">>> Training failed! check the .err file."
    exit 1
fi

echo "Job finished at $(date)"
