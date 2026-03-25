# Gotchas

Hard-won lessons from running anticipation-conditioned training across MLX, HuggingFace, and distributed setups. Every item here cost real debugging time.

## Data Format

### MLX `mask_prompt` requires prompt/completion format
```
ValueError: Prompt masking not supported for text dataset.
```
**Fix:** Use `{"prompt": "...", "completion": "..."}` format, OR set `mask_prompt: false` in config. The `{"text": "..."}` format does not support prompt masking.

### ChatML format required for Thunder-Train
Thunder-Train's `ChatMLDataset` calls `tokenizer.apply_chat_template(messages)`. If you pass `{"text": "..."}` instead of `{"messages": [...]}`, it silently loads **0 samples**.
```
Loaded 0 samples from /path/to/data.jsonl
```
**Fix:** Always use `{"messages": [{"role": ..., "content": ...}, ...]}` for Thunder-Train.

### Jinja template error on prompt/completion with older MLX
```
jinja2.exceptions.UndefinedError: dict object has no element 0
```
Older `mlx_lm` (Python 3.9, MLX <0.30) tries to run `apply_chat_template` on prompt/completion strings. The Qwen template expects a list of dicts, not a string.

**Fix:** Pre-format ChatML tokens manually in the text field. Don't rely on the tokenizer template for structured formatting.

## Memory

### MLX `val_batches=25` OOMs on 16GB Apple Silicon
The fine-tune daemon defaults to `val_batches: 25`. On M4 16GB, this crashes with:
```
[METAL] Command buffer execution failed: Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)
```
**Fix:** Set `val_batches: 1` or `val_batches: 2`. The validation still gives useful signal.

### Qwen2.5-7B OOMs on 16GB solo
Even 4-bit quantized, 7B needs ~6GB for weights + KV cache + gradients.

**Fix:** Use Thunder-Train data-parallel (2x 16GB = 32GB pool over TB5), OR use Qwen2.5-3B which peaks at 3.7GB.

### MLX server + training = OOM
If the MLX inference server (`mlx_lm.server`) is running on the same machine, training will OOM. Metal GPU memory is shared.

**Fix:** Kill the MLX server before training: `pkill -f mlx_lm.server`.

## Dependencies

### torch 2.11 + CUDA 12.1 driver = no CUDA
```
CUDA initialization: The NVIDIA driver on your system is too old (found version 12010)
```
Vast.ai instances with CUDA 12.1 drivers need torch 2.2.x, not 2.11.

**Fix:** `pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121`

### transformers 5.x breaks peft imports
```
ModuleNotFoundError: Could not import module 'PreTrainedModel'
```
`transformers>=5.0` changed the import structure. peft 0.18 hasn't caught up.

**Fix:** `pip install "transformers>=4.44,<5" "peft>=0.12,<1"`

### Old torchvision conflicts with new torch
```
RuntimeError: operator torchvision::nms does not exist
```
The pre-installed torchvision (compiled for torch 2.2) has incompatible operator registrations with torch 2.11.

**Fix:** `pip uninstall -y torchvision torchaudio` (not needed for LLM training).

### bitsandbytes 0.49 requires torch >= 2.3
If you pinned torch 2.2.1, bitsandbytes 0.49 will error.

**Fix:** `pip install "bitsandbytes>=0.42,<0.44"`

## MLX / Apple Silicon

### MLX 0.31+ required for `mlx.launch`
Mac5 on Python 3.9 (Xcode bundled) caps at MLX 0.29. There are no MLX 0.31 wheels for Python 3.9.

**Fix:** Use `distributed_launch.sh` (SSH + env vars) instead of `python3 -m mlx.launch`.

### Thunder-Train: Python version mismatch across machines
Mac4 may have Python 3.14, Mac5 may have Python 3.9. The launch script uses `/opt/homebrew/bin/python3` which resolves differently on each machine.

**Fix:** The `distributed_launch.sh` script hardcodes `PYTHON="/opt/homebrew/bin/python3"`. If your machines differ, edit this per-host.

### MLX training: sequences longer than max_seq_length
```
[WARNING] Some sequences are longer than 1024 tokens. The longest sentence 1618 will be truncated to 1024.
```
This is a warning, not an error. Training continues. But very long sequences cause memory spikes.

**Fix:** Pre-filter data to max_seq_length, or increase max_seq_length if memory allows.

## Training Behavior

### Small datasets (< 100 records) overfit fast
With 78-82 records, validation loss bottoms at iter 100-200 then rises. This is expected.

**Observation:** Treatment (inscription-conditioned) overfits slower than control, suggesting the conditioning signal provides regularization.

### Distributed training: both ranks must show identical loss
If rank 0 and rank 1 show different losses at the same step, gradient averaging is broken. Check:
1. TB5 link: `ping -c 1 10.0.5.2`
2. Ring backend env vars: `MLX_COMM_RING_ADDR`, `MLX_COMM_RING_PORT`, `MLX_COMM_NUM_RANKS`
3. Both processes started simultaneously (within ~1s)

### Vast.ai instances can be preempted
The instance may exit without warning. Always run in `tmux` and save checkpoints every 200 steps.

## Evaluation

### Regex-based specificity doesn't discriminate between adapter variants
Both LoRA-32 and LoRA-8 showed identical specificity scores (0.330) despite a 0.53 val loss gap. The surface-level regex check (file paths, project names, action verbs) doesn't capture the deeper differences.

**Fix:** Use validation loss as the primary metric. Specificity scoring is useful for filtering training data, not for comparing trained models.
