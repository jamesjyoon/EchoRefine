#!/bin/bash
#SBATCH -J EchoRefine_Missing_Ablations
#SBATCH --account=coc
#SBATCH -N 1
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=256G
#SBATCH -t 8:00:00
#SBATCH --cpus-per-task=16
#SBATCH -q coc-ice
#SBATCH -o /storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine/logs/missing_%j.out
#SBATCH -e /storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine/logs/missing_%j.err

module load anaconda3
module load cuda/12.1

PYTHON_BIN="${PYTHON_BIN:-/storage/ice1/6/3/jyoon370/miniconda3/bin/python}"
PROJECT_DIR="${PROJECT_DIR:-/storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine}"
EXPERIMENT_GROUPS="${MISSING_EXPERIMENT_GROUPS:-high-priority}"
EXPERIMENT_NAMES="${MISSING_EXPERIMENT_NAMES:-}"
LANGUAGES="${MISSING_EXPERIMENT_LANGUAGES:-npi ben sin mya kor tam hin fra}"
LIMIT="${MISSING_EXPERIMENT_LIMIT:-1012}"
BATCH_SIZE="${MISSING_EXPERIMENT_BATCH_SIZE:-8}"
INCLUDE_AUXILIARY="${MISSING_EXPERIMENT_INCLUDE_AUXILIARY:-0}"
MATERIALIZE_ABLATIONS="${MISSING_EXPERIMENT_MATERIALIZE:-1}"

export HF_HOME="${HF_HOME:-/storage/ice1/6/3/jyoon370/hf_cache}"
export TMPDIR="${TMPDIR:-/storage/ice1/6/3/jyoon370/tmp}"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJECT_DIR"
mkdir -p logs "$HF_HOME" "$TMPDIR" results/tables results/strong_baselines

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

echo "Starting missing EchoRefine experiments on $(hostname) at $(date)"
echo "Groups: $EXPERIMENT_GROUPS"
echo "Experiments: ${EXPERIMENT_NAMES:-all selected by group}"
echo "Languages: $LANGUAGES"
echo "Limit: $LIMIT"
echo "Batch size: $BATCH_SIZE"
echo "Include auxiliary generations: $INCLUDE_AUXILIARY"
echo "Materialize ablation aliases: $MATERIALIZE_ABLATIONS"
echo "This job is resumable: completed result JSON files are skipped, and incomplete progress JSONL files resume per language/configuration."
echo "The 2xA100 request uses an 8-hour wall time to satisfy the queue's GPU-minute policy."

runner_cmd=(
    "$PYTHON_BIN" generate_missing_experiments.py
    --groups $EXPERIMENT_GROUPS
    --languages $LANGUAGES
    --limit "$LIMIT"
    --batch-size "$BATCH_SIZE"
    --python-bin "$PYTHON_BIN"
    --skip-existing
)

if [ -n "$EXPERIMENT_NAMES" ]; then
    runner_cmd+=(--experiments $EXPERIMENT_NAMES)
fi

if [ "$INCLUDE_AUXILIARY" = "1" ]; then
    runner_cmd+=(--include-auxiliary-generations)
fi

time "${runner_cmd[@]}"

status=$?
if [ $status -ne 0 ]; then
    echo "Missing experiment runner failed with exit code $status"
    exit $status
fi

if [ "$MATERIALIZE_ABLATIONS" = "1" ]; then
    time "$PYTHON_BIN" analysis/materialize_ablation_aliases.py
fi

time "$PYTHON_BIN" analysis/aggregate_missing_experiments.py

echo "Finished missing EchoRefine experiments at $(date)"
