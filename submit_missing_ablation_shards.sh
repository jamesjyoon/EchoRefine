#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine}"
LANGUAGES="${MISSING_EXPERIMENT_LANGUAGES:-npi ben sin mya kor tam hin fra}"
LIMIT="${MISSING_EXPERIMENT_LIMIT:-1012}"
MAX_SUBMITS="${MAX_SUBMITS:-0}"
DRY_RUN="${DRY_RUN:-0}"

cd "$PROJECT_DIR"

source_runs=(
    abl_component_no_bt
    abl_decoding_greedy
    abl_decoding_beam4
    abl_decoding_beam8
    abl_decoding_mbr3
    abl_decoding_mbr5
    abl_decoding_mbr10
    abl_decoding_mbr20
)

group_for_run() {
    case "$1" in
        abl_component_no_bt) echo components ;;
        *) echo decoding ;;
    esac
}

batch_for_run() {
    case "$1" in
        abl_decoding_mbr10|abl_decoding_mbr20) echo 1 ;;
        abl_decoding_mbr3|abl_decoding_mbr5|abl_decoding_beam4|abl_decoding_beam8) echo 2 ;;
        *) echo 4 ;;
    esac
}

submitted=0
skipped=0

for run_name in "${source_runs[@]}"; do
    group_name="$(group_for_run "$run_name")"
    batch_size="$(batch_for_run "$run_name")"

    for lang in $LANGUAGES; do
        result_path="results_${lang}_${run_name}.json"
        if [ -f "$result_path" ]; then
            skipped=$((skipped + 1))
            continue
        fi

        job_name="ER_${lang}_${run_name#abl_}"
        cmd=(
            sbatch
            --export=ALL,MISSING_EXPERIMENT_GROUPS="$group_name",MISSING_EXPERIMENT_NAMES="$run_name",MISSING_EXPERIMENT_LANGUAGES="$lang",MISSING_EXPERIMENT_LIMIT="$LIMIT",MISSING_EXPERIMENT_BATCH_SIZE="$batch_size"
            --job-name="$job_name"
            run_missing_experiments.sbatch
        )

        if [ "$DRY_RUN" = "1" ]; then
            printf '%q ' "${cmd[@]}"
            printf '\n'
        else
            "${cmd[@]}"
        fi

        submitted=$((submitted + 1))
        if [ "$MAX_SUBMITS" -gt 0 ] && [ "$submitted" -ge "$MAX_SUBMITS" ]; then
            echo "Reached MAX_SUBMITS=$MAX_SUBMITS; skipped_existing=$skipped submitted=$submitted"
            exit 0
        fi
    done
done

echo "Finished submit scan; skipped_existing=$skipped submitted=$submitted"
