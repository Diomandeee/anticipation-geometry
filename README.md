# Anticipation Geometry

**7 numbers that describe where a trajectory is going.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)

Give it a sequence of vectors, any kind, and it returns 7 scalars that tell you what the trajectory is doing: committing, wandering, about to change direction, stuck, etc. Works on motion capture data, conversation embeddings, knowledge graph paths, whatever.

The key number is **transition pressure** (`d(commitment)/dt - d(uncertainty)/dt`). When it spikes, the trajectory is about to shift. It predicts conversation convergence at 71.8% accuracy and picks valid knowledge graph paths at 81.0%, with no training at all.

## What's here

```
anticipation_geometry/     Python library (pip install)
crates/                    Rust library (real-time, <2ms per frame)
training/                  Fine-tuning scripts (MLX + HuggingFace + distributed)
transformer/               Custom model with trajectory-biased attention
paper/                     Research paper
```

## Install

```bash
pip install anticipation-geometry

# If you want to train models with it
pip install anticipation-geometry[train]

# From source
git clone https://github.com/Diomandeee/anticipation-geometry.git
pip install -e ".[train]"
```

## The 7 scalars

| Name | What it tells you |
|------|------------------|
| **commitment** | How locked-in the trajectory is. High = deep into a sustained phase. |
| **uncertainty** | How many directions it could still go. High = lots of options open. |
| **transition_pressure** | Speed of futures collapsing. Positive spike = about to change. |
| **recovery_margin** | How easy to reverse course. Low = past the point of no return. |
| **phase_stiffness** | How rhythmic/consistent the motion is. |
| **novelty** | How different this is from recent history. |
| **stability** | How predictable the local dynamics are. |

## Use it

### Compute scalars from anything

```python
from anticipation_geometry import AnticipationGeometry
import numpy as np

geometry = AnticipationGeometry(k_neighbors=5)

# Works on any sequence of vectors
trajectory = [np.random.randn(384) for _ in range(50)]
packet = geometry.compute(trajectory)

for t in range(len(trajectory)):
    print(f"Step {t}: commitment={packet.commitment[t]:.3f}, "
          f"uncertainty={packet.uncertainty[t]:.3f}")
```

### Label conversations with behavioral motifs

The library classifies each conversation turn into one of 10 patterns (stabilization, transition, oscillation, correction, exploration, convergence, expansion, regression, stagnation, completion) based on the scalar values. We call these "inscriptions."

```python
from anticipation_geometry.conversation import annotate_conversation

turns = [
    {"role": "user", "content": "Fix the broken test suite"},
    {"role": "assistant", "content": "Looking at the test files..."},
]
annotated = annotate_conversation(turns)
print(annotated.inscription)  # "convergence"
```

### Add inscriptions to training data

The main finding: if you prepend the inscription label and scalar values to training data, the model learns better. 17% lower val loss compared to the same data without the prefix.

```python
# Treatment: add motif prefix to system message
# "[MOTIF:convergence] commitment=0.82 uncertainty=0.15\nYou are a coding assistant."

# Control: leave it as is
# "You are a coding assistant."
```

## Training

Three ways to train:

**1. Distributed across two Macs (Thunder-Train, free)**
```bash
cd training && bash thunder_5runs.sh
```
Uses Thunderbolt 5 to pool Mac4+Mac5 into 32GB. Handles 3B-14B models.

**2. Single Mac with MLX**
```bash
python3 -m mlx_lm lora --config training/configs/qwen25_3b_mlx.yaml
```

**3. Cloud GPU (Vast.ai, Lambda, etc.)**
```bash
python training/train.py --config training/configs/qwen3_4b_qlora.yaml
```

See [DATA_FORMATS.md](DATA_FORMATS.md) for data format details.

## Data formats

The training pipeline accepts three data formats. The most common mistake is using the wrong one for your backend.

| Format | When to use | Example |
|--------|------------|---------|
| `{"messages": [...]}` | Thunder-Train, HuggingFace | Recommended default |
| `{"text": "..."}` | MLX without prompt masking | Pre-formatted ChatML string |
| `{"prompt": "...", "completion": "..."}` | MLX with prompt masking | Splits input/output |

Full spec with gotchas: [DATA_FORMATS.md](DATA_FORMATS.md)

## Results

### Does the geometry actually work?

| Test | Accuracy | Details |
|------|----------|---------|
| Predict if a conversation converges | 71.8% | 5,000 turns, p<0.007 |
| Pick valid vs fake knowledge graph paths | 81.0% | 199 paths, Cohen's d=2.23 |
| High-dimensional embeddings (1024D) | 69.8% | 86 conversations |

### Does the inscription prefix help training?

| | Val loss (iter 100) | Val loss (iter 200) |
|---|---|---|
| With inscription prefix | **0.402** | **0.576** |
| Without prefix | 0.416 | 0.694 |
| Difference | -3.4% | **-17.0%** |

The model with the inscription prefix fits the held-out data 17% better by iter 200.

## Common problems

See [GOTCHAS.md](GOTCHAS.md) for the full list. The ones that bite most:

- **MLX `val_batches=25` crashes 16GB Macs.** Set it to 1.
- **Thunder-Train ignores `{"text": ...}` format.** Use `{"messages": [...]}`.
- **torch 2.11 on old CUDA drivers breaks everything.** Pin torch 2.2.1.
- **transformers 5.x breaks peft.** Pin transformers <5.

## Rust

If you need real-time (motion capture, live performance):

```bash
cargo build --release
cargo test  # 340+ tests
```

The Rust library computes all 7 scalars in <2ms per frame at 50Hz. Zero heap allocation on the hot path.

## Citation

```bibtex
@article{diomande2026anticipation,
  title={Compact Behavioral Annotation for Coding Agent Training},
  author={Diomande, Mohamed},
  year={2026}
}
```

## License

MIT
