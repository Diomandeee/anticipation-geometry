# Data Formats

Anticipation Geometry supports multiple data formats across different training backends. This document specifies each format, when to use it, and migration paths between them.

## Format Overview

| Format | Backend | File | `mask_prompt` | Notes |
|--------|---------|------|---------------|-------|
| ChatML messages | Thunder-Train, HuggingFace | `.jsonl` | N/A | Recommended default |
| Text (pre-formatted) | MLX `mlx_lm.lora` | `.jsonl` | No | Fastest for MLX, no masking |
| Prompt/completion | MLX `mlx_lm.lora` | `.jsonl` | Yes | Use when masking prompts |
| Annotated messages | This library's conditioner | `.jsonl` | N/A | Adds inscription + scalars |

## 1. ChatML Messages Format

**Used by:** Thunder-Train, HuggingFace `transformers`, evaluation scripts

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Fix the broken test suite"},
    {"role": "assistant", "content": "Looking at the test files, I found..."}
  ]
}
```

**Requirements:**
- At least 2 messages (user + assistant)
- System message is optional but recommended
- Messages array is the **only** required field

**Optional fields** (used by anticipation conditioning):
```json
{
  "messages": [...],
  "scalars": {
    "commitment": 0.82,
    "uncertainty": 0.15,
    "transition_pressure": 0.03,
    "recovery_margin": 0.91
  },
  "inscription": "convergence",
  "position": 0.75
}
```

## 2. Text Format (Pre-formatted ChatML)

**Used by:** MLX `mlx_lm.lora` without prompt masking

```json
{"text": "<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n<|im_start|>user\nFix the broken test suite<|im_end|>\n<|im_start|>assistant\nLooking at the test files...<|im_end|>"}
```

**When to use:** Single-machine MLX training where you don't need prompt masking. Fastest loading since no tokenizer template processing is needed.

**Gotcha:** Setting `mask_prompt: true` in the MLX config with this format throws:
```
ValueError: Prompt masking not supported for text dataset.
```
Either switch to prompt/completion format or set `mask_prompt: false`.

## 3. Prompt/Completion Format

**Used by:** MLX `mlx_lm.lora` with prompt masking enabled

```json
{
  "prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nFix the test<|im_end|>\n<|im_start|>assistant\n",
  "completion": "Looking at the test files, I found the issue...<|im_end|>"
}
```

**When to use:** When you want the model to only learn from the assistant's response (masked prompt tokens don't contribute to loss).

**Gotcha on older MLX (< 0.30):** The tokenizer's `apply_chat_template` may fail with `jinja2.exceptions.UndefinedError: dict object has no element 0` if the prompt string triggers template processing. Pre-format the ChatML tokens manually (as shown above) rather than passing structured messages.

## 4. Annotated Messages Format

**Used by:** This library's `InscriptionConditioner` for experiment data preparation

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Fix the broken test suite"},
    {"role": "assistant", "content": "Found the issue..."}
  ],
  "scalars": {
    "commitment": 0.82,
    "uncertainty": 0.15,
    "transition_pressure": 0.03,
    "recovery_margin": 0.91,
    "position": 0.75,
    "focus": 0.67,
    "stability_derived": 0.97
  },
  "inscription": "convergence",
  "position": 0.75
}
```

The conditioner transforms this into one of 4 modes:

| Mode | System message becomes |
|------|----------------------|
| `standard` | Original system message (no modification) |
| `inscription` | `[MOTIF:convergence] commitment=0.82 uncertainty=0.15 transition=0.03\n{original}` |
| `inscription_nogate` | `[MOTIF:convergence]\n{original}` |
| `noinscrip_gate` | `commitment=0.82 uncertainty=0.15\n{original}` |

## Converting Between Formats

```python
from anticipation_geometry.training import convert_format

# ChatML → Text
convert_format("data.jsonl", output="data_text.jsonl", target="text")

# ChatML → Prompt/Completion
convert_format("data.jsonl", output="data_pc.jsonl", target="prompt_completion")

# Text → ChatML (requires parsing ChatML tokens)
convert_format("data_text.jsonl", output="data_chatml.jsonl", target="messages")
```

## Inscription Vocabulary

10 behavioral motif categories, derived from coding agent session dynamics:

| Inscription | Pattern | Scalar Signature |
|------------|---------|------------------|
| stabilization | Consistent tool success | High commitment, low uncertainty |
| transition | Shift in approach | High transition_pressure |
| oscillation | Back-and-forth | High uncertainty, high transition_pressure |
| correction | Error recovery | Low recovery_margin |
| exploration | Trying new approaches | High novelty, high uncertainty |
| convergence | Narrowing to solution | High commitment, decreasing uncertainty |
| expansion | Scope increase | Moderate commitment, high novelty |
| regression | Previously working things break | Falling commitment, low recovery_margin |
| stagnation | No progress | Low transition_pressure, low novelty |
| completion | Task finished | Very high commitment, very low uncertainty |

## Validation

```python
from anticipation_geometry.training import validate_dataset

report = validate_dataset("data.jsonl")
print(report)
# DatasetReport(
#   format="messages",
#   n_records=36485,
#   n_with_scalars=25417,
#   n_with_inscription=25417,
#   inscription_distribution={"stabilization": 8234, "convergence": 5123, ...},
#   avg_tokens=412,
#   max_tokens=2048,
#   warnings=["3 records exceed max_seq_len=1024"]
# )
```
