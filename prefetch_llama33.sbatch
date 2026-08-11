#!/bin/bash
#SBATCH -J EchoRefine_Prefetch_Llama33
#SBATCH --account=coc
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH --cpus-per-task=4
#SBATCH -q coc-ice
#SBATCH -o /storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine/logs/prefetch_llama33_%j.out
#SBATCH -e /storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine/logs/prefetch_llama33_%j.err

module load anaconda3

PYTHON_BIN="${PYTHON_BIN:-/storage/ice1/6/3/jyoon370/miniconda3/bin/python}"
PROJECT_DIR="${PROJECT_DIR:-/storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine}"

export HF_HOME="${HF_HOME:-/storage/ice1/6/3/jyoon370/hf_cache}"
export TMPDIR="${TMPDIR:-/storage/ice1/6/3/jyoon370/tmp}"
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_DIR"
mkdir -p logs "$HF_HOME" "$TMPDIR"

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

echo "Prefetching Llama-3.3-70B-Instruct safetensors at $(date)"
"$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download

token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
snapshot_download(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    token=token,
    allow_patterns=[
        "*.json",
        "*.safetensors",
        "*.model",
        "*.txt",
        "*.md",
        "*.py",
        "*.jinja",
    ],
    ignore_patterns=[
        "original/*",
        "*.pth",
        "*.bin",
    ],
)
PY

echo "Finished Llama prefetch at $(date)"
