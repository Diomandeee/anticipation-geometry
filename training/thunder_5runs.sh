#!/bin/bash
# 5 Controlled Runs on Thunder-Train (Mac4+Mac5, TB5, Qwen2.5-3B)
# Replaces the Vast.ai A100 plan with local distributed training.
#
# Thunder-Train pools Mac4+Mac5 via TB5 ring backend.
# Data parallel: full model on each machine, gradients averaged → 2x throughput.
#
# Run from Mac4: bash thunder_5runs.sh

set -e

MODEL="mlx-community/Qwen2.5-3B-Instruct-4bit"
THUNDER="$HOME/projects/thunder-train"
DATA_DIR="$HOME/projects/atx-data"
ITERS=2000
LORA_RANK=16
LR="2e-4"
BATCH=1
MAX_SEQ=1024
LOG_EVERY=50
EVAL_EVERY=200
SAVE_EVERY=500

echo "========================================"
echo "Anticipatory Transformer — 5 Thunder-Train Runs"
echo "Model: $MODEL (3B, 4-bit)"
echo "Cluster: Mac4 (10.0.5.1) + Mac5 (10.0.5.2)"
echo "Strategy: data parallel (2x throughput)"
echo "Iters: $ITERS per run"
echo "========================================"
echo ""

run_thunder() {
    local NAME="$1"
    local SEED="$2"
    local TRAIN_DATA="$3"
    local ADAPTER_DIR="$DATA_DIR/adapters/$NAME"
    local START_TIME=$(date +%s)

    echo ""
    echo "========== [$NAME] seed=$SEED =========="
    echo "  Data: $TRAIN_DATA"
    echo "  Adapter: $ADAPTER_DIR"
    mkdir -p "$ADAPTER_DIR"

    # Use distributed_launch.sh for 2-node TB5 training
    bash "$THUNDER/distributed_launch.sh" \
        --model "$MODEL" \
        --train-data "$TRAIN_DATA" \
        --valid-data "$DATA_DIR/real_conv_eval.jsonl" \
        --strategy data \
        --num-iters "$ITERS" \
        --batch-size "$BATCH" \
        --learning-rate "$LR" \
        --lora-rank "$LORA_RANK" \
        --max-seq-len "$MAX_SEQ" \
        --adapter-path "$ADAPTER_DIR" \
        --log-every "$LOG_EVERY" \
        --eval-every "$EVAL_EVERY" \
        --save-every "$SAVE_EVERY" \
        --seed "$SEED"

    # Wait for both ranks to finish (check rank 0 log for completion)
    echo "  Waiting for training to complete..."
    while ssh mac4 "ps aux | grep thunder_train | grep -v grep | wc -l" 2>/dev/null | grep -q '[1-9]'; do
        sleep 30
    done

    local END_TIME=$(date +%s)
    local ELAPSED=$(( (END_TIME - START_TIME) / 60 ))
    echo "  [$NAME] DONE in ${ELAPSED}min → $ADAPTER_DIR"
    echo ""

    # Capture final val loss from rank 0 log
    ssh mac4 "grep 'val_loss\|Final' ~/projects/karl/thunder-rank0.log | tail -3" 2>/dev/null
    echo ""
}

# 1. Control (standard SFT data, no inscription prefix)
run_thunder "control" 41 "$DATA_DIR/sft_standard.jsonl"

# 2. Treatment (full inscription + scalars)
run_thunder "treatment" 42 "$DATA_DIR/sft_inscription.jsonl"

# 3. Ablation: inscription only (no gate/commitment scalars)
run_thunder "no-gate" 44 "$DATA_DIR/sft_inscription_nogate.jsonl"

# 4. Ablation: gate scalars only (no inscription label)
run_thunder "no-inscription" 43 "$DATA_DIR/sft_noinscrip_gate.jsonl"

# 5. Ablation: same as treatment (for LSE, re-use inscription data with different seed)
run_thunder "no-lse" 45 "$DATA_DIR/sft_inscription.jsonl"

echo ""
echo "========================================"
echo "All 5 Thunder-Train runs complete."
echo "Adapters at $DATA_DIR/adapters/"
echo "========================================"
echo ""
echo "Run summary:"
for RUN in control treatment no-gate no-inscription no-lse; do
    echo "  $RUN: $(ssh mac4 "grep 'Final validation' ~/projects/karl/thunder-rank0.log 2>/dev/null | tail -1" 2>/dev/null)"
done
