#!/bin/bash
# Run all 5 experiments on Vast.ai A100.
# Budget: ~$6.40 total (~$1/hr A100 80GB)
# Estimated: 5 runs × ~1hr each = ~5hrs = ~$5

set -e

CONFIG="configs/qwen3_4b_qlora.yaml"

echo "========================================"
echo "Anticipatory Transformer — 5 Runs"
echo "========================================"
echo ""

# 1. Control (no anticipation)
echo "[1/5] Control (no anticipation)"
python train.py --config $CONFIG \
  --override anticipation.use_inscription=false \
  --override anticipation.use_gate=false \
  --override anticipation.use_lse_loop=false \
  --override training.output_dir=runs/qwen3-4b-control \
  --override training.seed=41

# 2. Treatment (full)
echo "[2/5] Treatment (full anticipation)"
python train.py --config $CONFIG \
  --override training.output_dir=runs/qwen3-4b-treatment \
  --override training.seed=42

# 3. Ablation: no inscription
echo "[3/5] Ablation: no inscription"
python train.py --config $CONFIG \
  --override anticipation.use_inscription=false \
  --override training.output_dir=runs/qwen3-4b-no-inscription \
  --override training.seed=43

# 4. Ablation: no gate
echo "[4/5] Ablation: no gate"
python train.py --config $CONFIG \
  --override anticipation.use_gate=false \
  --override training.output_dir=runs/qwen3-4b-no-gate \
  --override training.seed=44

# 5. Ablation: no LSE loop
echo "[5/5] Ablation: no LSE loop"
python train.py --config $CONFIG \
  --override anticipation.use_lse_loop=false \
  --override training.output_dir=runs/qwen3-4b-no-lse \
  --override training.seed=45

echo ""
echo "========================================"
echo "All 5 training runs complete."
echo "Running evaluation with bootstrap CIs..."
echo "========================================"

EVAL_FILE="/workspace/data/real_conv_eval.jsonl"

for RUN in control treatment no-inscription no-gate no-lse; do
  CKPT="runs/qwen3-4b-${RUN}/final"
  if [ -d "$CKPT" ]; then
    echo ""
    echo "--- Evaluating: $RUN ---"
    python evaluate.py --ckpt "$CKPT" \
      --eval_file "$EVAL_FILE" \
      --bootstrap 1000 \
      --metrics nll,commitment_acc,scalar_mse,ece \
      --output "runs/qwen3-4b-${RUN}/eval_results.json"
  fi
done

echo ""
echo "========================================"
echo "All evaluations complete."
echo "Results in runs/qwen3-4b-*/eval_results.json"
echo "========================================"
