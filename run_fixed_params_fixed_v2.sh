#!/usr/bin/env bash
# Fixed-parameter Linux runner for the coffee-bean CNN experiments.
# This version matches train_fixed.py's run directory and checkpoint naming.
set -euo pipefail

MODELS=(squeezenet shufflenet mobilenetv1 mobilenetv2 mobilenetv3 efficientnet)
VARIETIES=(arabica robusta)
OPTIMIZERS=(adam adamw sgd rmsprop adan)
LR=1e-4
BATCH_SIZE=16
DROPOUT=0.2
MAX_EPOCHS=100
RUNS_DIR="linuxruns_v2"
TRAIN_SCRIPT="train_fixed_v2.py"
DRY_RUN=0
STAGE="1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --stage) STAGE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "$STAGE" != "all" && "$STAGE" != "1" && "$STAGE" != "2" ]]; then
    echo "Invalid --stage value: $STAGE (expected all|1|2)" >&2; exit 1
fi
mkdir -p "$RUNS_DIR"
LR_TAG="${LR}"
RUN_TAG="NORLR"

run_exists() { [[ -f "$RUNS_DIR/$1/results.json" ]]; }

run_cmd() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '  [DRY]'; printf ' %q' "$@"; printf '\n'
    else
        echo ""; printf '  >>>'; printf ' %q' "$@"; printf '\n'
        "$@"
    fi
}

run_if_needed() {
    local name="$1"; shift
    if run_exists "$name"; then
        echo "  [SKIP] Already completed: $name"
    else
        run_cmd "$@"
    fi
}

base_name() { printf 'fixed_%s_%s_%s_%s_lr%s_bs%s_do%s' "$RUN_TAG" "$1" "$2" "$3" "$LR_TAG" "$BATCH_SIZE" "$DROPOUT"; }
ft_name() { printf 'fixed_ft_%s_%s_to_%s_%s_%s_lr%s_bs%s_do%s' "$RUN_TAG" "$1" "$2" "$3" "$4" "$LR_TAG" "$BATCH_SIZE" "$DROPOUT"; }

get_checkpoint_path() {
    local checkpoint="$RUNS_DIR/$1/best_model.pth"
    if [[ -f "$checkpoint" ]]; then
        printf '%s\n' "$checkpoint"
    fi
}

echo "============================================================="
echo " Fixed CNN experiment runner"
echo " runs=$RUNS_DIR  lr=$LR  batch=$BATCH_SIZE  dropout=$DROPOUT"
echo "============================================================="

if [[ "$STAGE" == "all" || "$STAGE" == "1" ]]; then
    echo "STAGE 1 - Fixed-config training"
    for variety in "${VARIETIES[@]}"; do
        for model in "${MODELS[@]}"; do
            for opt in "${OPTIMIZERS[@]}"; do
                name=$(base_name "$variety" "$model" "$opt")
                run_if_needed "$name" python "$TRAIN_SCRIPT" \
                    --variety "$variety" --model "$model" --optimizer "$opt" \
                    --lr "$LR" --batch_size "$BATCH_SIZE" --dropout "$DROPOUT" \
                    --max_epochs "$MAX_EPOCHS" --runs_dir "$RUNS_DIR" --run_name "$name"
            done
        done
    done
fi

if [[ "$STAGE" == "all" || "$STAGE" == "2" ]]; then
    echo "STAGE 2 - Cross-variety fine-tuning"
    for model in "${MODELS[@]}"; do
        for src_variety in "${VARIETIES[@]}"; do
            if [[ "$src_variety" == "arabica" ]]; then dst_variety=robusta; else dst_variety=arabica; fi
            for opt in "${OPTIMIZERS[@]}"; do
                src_name=$(base_name "$src_variety" "$model" "$opt")
                ckpt=$(get_checkpoint_path "$src_name")
                name=$(ft_name "$src_variety" "$dst_variety" "$model" "$opt")

                # Dry-run shows every intended Stage 2 experiment even when
                # the Stage 1 checkpoints have not been trained yet.
                if [[ "$DRY_RUN" -eq 1 ]]; then
                    expected_ckpt="$RUNS_DIR/$src_name/best_model.pth"
                    run_cmd python "$TRAIN_SCRIPT" \
                        --variety "$dst_variety" --model "$model" --optimizer "$opt" \
                        --lr "$LR" --batch_size "$BATCH_SIZE" --dropout "$DROPOUT" \
                        --max_epochs "$MAX_EPOCHS" --pretrained_from "$expected_ckpt" \
                        --runs_dir "$RUNS_DIR" --run_name "$name"
                elif [[ -z "$ckpt" ]]; then
                    echo "  [SKIP] No Stage 1 checkpoint for: $src_name"
                    echo "          Checked path: $RUNS_DIR/$src_name/best_model.pth"
                    echo "          Expected Stage 1 folder naming uses LR=$LR."
                    echo "          Matching folders for model=$model optimizer=$opt:"
                    find "$RUNS_DIR" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' \
                        | grep -E "^fixed_.*_${model}_${opt}_lr${LR}" \
                        || echo "            (none found)"
                    continue
                else
                    run_if_needed "$name" python "$TRAIN_SCRIPT" \
                        --variety "$dst_variety" --model "$model" --optimizer "$opt" \
                        --lr "$LR" --batch_size "$BATCH_SIZE" --dropout "$DROPOUT" \
                        --max_epochs "$MAX_EPOCHS" --pretrained_from "$ckpt" \
                        --runs_dir "$RUNS_DIR" --run_name "$name"
                fi
            done
        done
    done
fi

echo "============================================================="
echo " Selected stages complete"
echo "============================================================="
