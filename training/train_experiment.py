#!/usr/bin/env python3
"""Autoresearch experiment runner for inscription conditioning.

This is the file the agent modifies. 100-step training runs, ~2 min each.
Reports val_loss on standard eval data (no prefix).

Usage: python3 train_experiment.py [--experiment-id exp001] [--description "baseline"]
"""

import argparse
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx_lm import load

# ============================================================
# CONFIGURATION — Agent modifies these
# ============================================================

MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
TRAIN_DATA = Path.home() / "projects/atx-data/sft_noinscrip_gate.jsonl"
EVAL_DATA = Path.home() / "projects/atx-data/real_conv_eval.jsonl"

# LoRA config
LORA_RANK = 16
LORA_LAYERS = 8
LEARNING_RATE = 2e-4

# Training config
NUM_STEPS = 100
BATCH_SIZE = 1
MAX_SEQ_LEN = 512
SEED = 42

# Prefix config — the main thing to experiment with
USE_PREFIX = True  # Set True to add inscription prefix
PREFIX_FORMAT = "full"  # "full", "motif_only", "scalars_only", "none"
MASK_PREFIX_LOSS = False  # If True, don't compute loss on prefix tokens

# ============================================================
# DATA LOADING
# ============================================================

def load_data(path, tokenizer, max_records=5000):
    """Load ChatML JSONL and tokenize."""
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_records:
                break
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                messages = d["messages"]
                try:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False)
                except Exception:
                    text = "\n".join(
                        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
                        for m in messages)

                tokens = tokenizer.encode(text)
                if len(tokens) > MAX_SEQ_LEN:
                    tokens = tokens[:MAX_SEQ_LEN]
                if len(tokens) >= 10:
                    # Find where assistant response starts for loss masking
                    asst_start = 0
                    if MASK_PREFIX_LOSS:
                        # Find the last <|im_start|>assistant token position
                        asst_marker = tokenizer.encode("<|im_start|>assistant\n")
                        for j in range(len(tokens) - len(asst_marker)):
                            if tokens[j:j+len(asst_marker)] == asst_marker:
                                asst_start = j + len(asst_marker)
                    records.append((mx.array(tokens), asst_start))
            except Exception:
                continue
    return records


def compute_loss(model, record):
    """Compute causal LM loss, optionally masking prefix tokens."""
    if isinstance(record, tuple):
        tokens, asst_start = record
    else:
        tokens, asst_start = record, 0

    x = tokens[:-1]
    y = tokens[1:]
    logits = model(x[None])[0]  # (seq_len, vocab)

    if MASK_PREFIX_LOSS and asst_start > 0:
        # Only compute loss on tokens after assistant start
        logits = logits[asst_start:]
        y = y[asst_start:]
        if len(y) == 0:
            return mx.array(0.0)

    loss = nn.losses.cross_entropy(logits, y, reduction="mean")
    return loss


def eval_loss_all_tokens(model, tokens):
    """Compute loss on ALL tokens (no masking). Used for eval consistency."""
    if isinstance(tokens, tuple):
        tokens = tokens[0]  # Strip the asst_start, eval on everything
    x = tokens[:-1]
    y = tokens[1:]
    logits = model(x[None])[0]
    return nn.losses.cross_entropy(logits, y, reduction="mean")


def evaluate(model, eval_data, max_batches=50):
    """Evaluate on held-out data. Always computes loss on ALL tokens for consistency."""
    total_loss = 0.0
    n = 0
    for record in eval_data[:max_batches]:
        loss = eval_loss_all_tokens(model, record)
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


# ============================================================
# TRAINING LOOP
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default="exp000")
    parser.add_argument("--description", default="baseline")
    args = parser.parse_args()

    mx.random.seed(SEED)

    # Load model
    print(f"Loading {MODEL}...")
    model, tokenizer = load(MODEL)

    # Apply LoRA
    from mlx_lm.tuner.utils import linear_to_lora_layers
    model.freeze()
    linear_to_lora_layers(model, LORA_LAYERS, {"rank": LORA_RANK, "alpha": LORA_RANK * 2, "dropout": 0.05, "scale": 2.0})

    trainable = sum(p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    total = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"LoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.3f}%)")

    # Load data
    print(f"Loading train data: {TRAIN_DATA}")
    train_data = load_data(TRAIN_DATA, tokenizer, max_records=5000)
    print(f"Loading eval data: {EVAL_DATA}")
    eval_data = load_data(EVAL_DATA, tokenizer, max_records=500)
    print(f"Train: {len(train_data)} samples, Eval: {len(eval_data)} samples")

    # Optimizer
    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    # Loss + grad function
    loss_and_grad = nn.value_and_grad(model, lambda m, t: compute_loss(m, t))

    # Initial eval
    val_loss_0 = evaluate(model, eval_data)
    print(f"Initial val_loss: {val_loss_0:.4f}")

    # Training
    print(f"\nTraining for {NUM_STEPS} steps...")
    t0 = time.time()
    import random
    random.seed(SEED)

    for step in range(1, NUM_STEPS + 1):
        tokens = random.choice(train_data)
        loss, grads = loss_and_grad(model, tokens)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [step {step:4d}] loss={loss.item():.4f} elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0

    # Final eval
    val_loss = evaluate(model, eval_data)
    print(f"\nFinal val_loss: {val_loss:.4f} (Δ={val_loss - val_loss_0:+.4f})")

    # Report
    print(f"\n---")
    print(f"val_loss:     {val_loss:.6f}")
    print(f"steps:        {NUM_STEPS}")
    print(f"time_seconds: {elapsed:.1f}")
    print(f"experiment:   {args.experiment_id}")
    print(f"description:  {args.description}")

    # Append to results.tsv
    results_path = Path(__file__).parent / "results.tsv"
    header = "experiment_id\tdescription\tval_loss\tsteps\ttime_s\tchanges\n"
    if not results_path.exists():
        results_path.write_text(header)
    with open(results_path, "a") as f:
        f.write(f"{args.experiment_id}\t{args.description}\t{val_loss:.6f}\t{NUM_STEPS}\t{elapsed:.1f}\t-\n")

    print(f"\nResults appended to {results_path}")


if __name__ == "__main__":
    main()
