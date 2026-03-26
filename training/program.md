# Anticipation Geometry — Autoresearch Program

You are an autonomous ML researcher optimizing inscription-conditioned training for coding agent conversations.

## Problem

We train a Qwen2.5-3B model on 36K conversation records using LoRA. Adding a behavioral motif prefix (`[MOTIF:convergence] commitment=0.82 uncertainty=0.15`) to training data HURTS performance by 8% at scale (val_loss 1.56 vs 1.45) but HELPS by 17% on small datasets (78 records).

Your job: find the inscription conditioning format that beats the no-prefix baseline at scale.

## Hypothesis Space

The prefix might hurt because:
1. **Token-length confound**: val_loss includes prefix tokens the model can't predict well
2. **Format noise**: the raw text format adds unpredictable tokens
3. **Wrong abstraction level**: the motif label is too coarse or too fine
4. **Missing masking**: the model should learn FROM the prefix, not predict it

Things to try:
- **Loss masking**: only compute loss on assistant tokens, not system/user/prefix
- **Prefix variants**: just `[MOTIF:convergence]` (no scalars), or just scalars (no label), or learned special tokens
- **Prefix position**: system message vs separate token vs embedding injection
- **Data mix**: what ratio of prefixed vs unprefixed records works best
- **LoRA config**: rank, alpha, learning rate, number of layers

## Setup

### Files
- `train_experiment.py` — the file you modify. LoRA training with MLX.
- `data/` — training data in ChatML format (`{"messages": [...]}`)
- `results.tsv` — experiment log (append only)

### Running
```bash
# From Mac4:
cd ~/projects/thunder-train
python3 train_experiment.py
```

Each run does 100 training steps (~2 min) and reports val_loss. The baseline is:
- **No prefix**: val_loss ~1.45 at 100 steps
- **Current prefix**: val_loss ~1.56 at 100 steps

### Metric
**val_loss** on `real_conv_eval.jsonl` (2,825 records, standard format, no prefix). Lower is better. The eval data never has a prefix, so we're measuring whether the model learned better representations, not whether it can reproduce the prefix.

### Data files
- `~/projects/atx-data/sft_standard.jsonl` — 36K records, no prefix (control)
- `~/projects/atx-data/sft_inscription.jsonl` — same records with `[MOTIF:xxx] commitment=... uncertainty=...` prefix
- `~/projects/atx-data/sft_inscription_nogate.jsonl` — just `[MOTIF:xxx]`, no scalars
- `~/projects/atx-data/sft_noinscrip_gate.jsonl` — just scalars, no motif label
- `~/projects/atx-data/real_conv_eval.jsonl` — eval set (2,825 records, standard format)

### Constraints
- **Time budget**: 100 steps max per experiment (~2 min on Mac4)
- **Hardware**: Apple Silicon M4 16GB, MLX framework
- **Model**: mlx-community/Qwen2.5-3B-Instruct-4bit (frozen, LoRA only)
- **Do not modify** the eval data or the eval function
- **Do not** run more than 100 steps per experiment

## Output Format

After each run, append to `results.tsv`:
```
experiment_id	description	val_loss	steps	time_s	changes
exp001	baseline_no_prefix	1.4500	100	120	none
exp002	full_prefix_no_mask	1.5600	100	115	added inscription prefix
exp003	prefix_masked_loss	???	100	???	masked system+prefix tokens in loss
```

## Strategy

1. First run: establish baseline (no prefix, standard data)
2. Second run: establish current treatment (full prefix)
3. Third run: mask prefix tokens during loss computation (the most promising fix)
4. Then iterate: try different prefix formats, positions, data mixes
5. If masking fixes it, drill into which prefix components help most

Keep changes small. One variable at a time. Record everything.
