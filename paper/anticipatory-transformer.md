# The Anticipatory Transformer: Geometry-Steered Attention for Trajectory-Aware Reasoning

**Mohamed Diomande**
Independent Researcher

---

## Abstract

Standard transformers attend based on learned position encodings (sinusoidal, RoPE, ALiBi) that encode *where* tokens are in a sequence but not *what the sequence is doing* as a geometric process. I introduce the Anticipatory Transformer, a modified transformer architecture where seven geometric scalars derived from Anticipation Geometry (commitment, uncertainty, transition pressure, recovery margin, phase stiffness, novelty, stability) steer the multi-head attention mechanism via additive bias. The trajectory bias is computed by a learned network that maps the seven scalars at each position to per-head, position-dependent attention biases, enabling different heads to specialize to different geometric dimensions of the reasoning trajectory. I also introduce the CommitmentGate, a threshold-based mechanism that determines *when* to emit tokens: when the model's predicted commitment is below a learned threshold, it buffers hidden states and defers emission, enabling variable-rate generation that mirrors the deliberative pauses of human reasoning. The architecture further incorporates a dual-pathway design: a fast pathway with local windowed attention (128-token window, updated every token) for high-frequency pattern capture, and a slow pathway with global attention (full context) for long-range dependency modeling. In smoke tests on a 678,206-parameter model trained for 50 steps on synthetic data, the commitment gate achieves +0.93 correlation with the commitment scalar, attention heads specialize to 3 out of 4 unique dominant scalars, scalar prediction MSE drops from 0.15 to 0.07, and the orthogonality penalty converges to 0.005. I present this as a complete, implemented architecture with preliminary validation, not as a benchmark-breaking result. I argue that the trajectory-bias mechanism is suited for three application domains where standard position encodings are insufficient: agent reasoning over multi-step plans, multi-hop knowledge graph traversal, and real-time motion-to-audio synthesis.

---

## 1. Introduction

The attention mechanism in transformers (Vaswani et al., 2017) computes relevance scores between all pairs of positions in a sequence. The fundamental equation is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Position encodings inject information about *where* each token sits in the sequence. Sinusoidal encodings (Vaswani et al., 2017) use fixed frequency patterns. Rotary position embeddings (Su et al., 2024) encode relative distances via rotation matrices. ALiBi (Press et al., 2022) adds a linear bias proportional to token distance, penalizing long-range attention without learned parameters.

All of these approaches encode the same fundamental quantity: *sequential position*. Token 5 is 3 positions away from token 8. But this tells us nothing about what is *happening* in the sequence at those positions. Is the model in the middle of committing to a conclusion (high commitment, low uncertainty)? Is it exploring multiple hypotheses (low commitment, high uncertainty)? Is it at a phase transition where futures are collapsing rapidly (high transition pressure)?

I propose that attention should be steered not only by positional information but by the *geometric state of the reasoning trajectory* at each position. The trajectory through embedding space is itself a geometric object with measurable properties: curvature, velocity autocorrelation, neighborhood density, jerk. These properties, formalized as the seven scalars of Anticipation Geometry (Diomande, 2026), carry information about the *process* of reasoning that position encodings cannot capture.

The key equation of the Anticipatory Transformer replaces the standard attention computation with:

$$\text{scores} = \frac{QK^\top}{\sqrt{d_k}} + \text{TrajectoryBias}(\mathbf{s})$$

where $\mathbf{s} \in \mathbb{R}^{B \times L \times 7}$ is the tensor of seven anticipation scalars at each position, and $\text{TrajectoryBias}$ is a learned network that maps these scalars to per-head, pairwise attention biases. The bias is *additive*, not multiplicative, for numerical stability: it shifts the attention logits before softmax, steering the attention pattern without risking gradient explosion.

This paper presents the architecture in full detail, grounds every component in a working PyTorch implementation (2,867 lines across 6 source files), and reports preliminary results from smoke tests that demonstrate the mechanism works as intended. We do not claim state-of-the-art performance on any benchmark. We claim that this is a viable and novel approach to attention steering that merits further investigation at scale.

### 1.1 Contributions

1. **Trajectory-aware attention**: A mechanism that injects geometric trajectory information into the attention computation via additive bias, with per-head specialization (Section 4).

2. **The CommitmentGate**: A variable-rate generation mechanism where the model can choose to *wait* before emitting tokens, controlled by the commitment scalar (Section 5).

3. **Dual-pathway architecture**: A design combining fast (local, high-frequency) and slow (global, low-frequency) attention pathways, mixed by a trajectory-aware gate (Section 6).

4. **Self-referential anticipation prediction**: The model learns to predict its own geometric state from hidden representations, enabling inference without external scalar computation (Section 4.4).

5. **Complete implementation**: A working PyTorch codebase with training, evaluation, and generation, demonstrating feasibility at 678K parameters (Section 7).

---

## 2. Related Work

### 2.1 Position Encodings in Transformers

The original transformer (Vaswani et al., 2017) uses sinusoidal positional encodings that inject a fixed frequency pattern into the embedding at each position. This approach is simple and requires no learned parameters, but it encodes only absolute position and has limited generalization to sequence lengths unseen during training.

Rotary Position Embeddings (RoPE; Su et al., 2024) encode relative position by rotating the query and key vectors by angles proportional to their positions. RoPE has become the dominant position encoding in modern large language models (LLaMA, Qwen, Mistral) because it naturally captures relative distances and allows extrapolation to longer contexts via NTK-aware scaling.

ALiBi (Press et al., 2022) takes a different approach: instead of modifying embeddings, it adds a linear bias directly to the attention logits. The bias penalizes attention to distant positions, with the penalty slope varying per head. This is conceptually the closest precursor to the work, as it demonstrates that additive bias to attention logits is a viable and effective mechanism for steering attention. However, ALiBi's bias encodes only distance. The trajectory bias encodes the *geometric state* of the trajectory, which is a richer signal.

Relative position encodings (Shaw et al., 2018) and T5-style learned biases (Raffel et al., 2020) round out the landscape of position-aware attention modifications. All of these approaches answer the question "how far apart are these tokens?" The approach answers a different question: "what is the geometric character of the reasoning process at each token?"

### 2.2 Geometric Deep Learning

Bronstein et al. (2021) establish the theoretical foundations for applying geometric principles to deep learning, identifying four geometric domains (grids, graphs, groups, geodesics) and defining equivariance as the key inductive bias. The work extends this perspective in a specific direction: I treat the *trajectory* through a representation space as a geometric object and extract scalar invariants from it.

Graph Attention Networks (GATs; Velickovic et al., 2018) use attention over graph neighborhoods, where the attention weights are computed from node features. When applied to knowledge graphs, GATs attend based on entity and relation embeddings. The trajectory bias could be composed with GAT-style attention to provide both structural (graph topology) and process (trajectory geometry) information.

### 2.3 Anticipatory Computing and Predictive Processing

Pezzulo et al. (2008) formalize anticipatory behavior as processes that depend on predictions about the future, not only past and present. Anticipation Geometry (Diomande, 2026) operationalizes this by defining seven computable scalars that characterize how constrained the future is at any point in a trajectory. The Anticipatory Transformer closes the loop: instead of merely computing these scalars as external analysis, it uses them to *steer the process that generates the trajectory*.

The predictive processing framework in cognitive science (Clark, 2013; Friston, 2010) proposes that the brain constantly generates predictions and updates them based on prediction errors. The AnticipationHead in the architecture serves a formally similar function: the model predicts its own geometric state, and the discrepancy between predicted and actual scalars provides a training signal that encourages the model to build accurate internal models of its own reasoning process.

### 2.4 Variable-Rate Generation

Standard autoregressive transformers emit one token per forward pass, unconditionally. Several works have explored deviations from this paradigm. Speculative decoding (Leviathan et al., 2023) uses a draft model to propose multiple tokens, which a larger model verifies in parallel. Medusa (Cai et al., 2024) adds multiple prediction heads to generate several tokens simultaneously. Pause tokens (Goyal et al., 2024) insert special tokens that allow the model to "think" before answering.

The CommitmentGate differs from all of these in that the decision to emit or defer is grounded in a *geometric* criterion (the commitment scalar), not a learned heuristic or a fixed token. The gate does not insert dummy tokens; it genuinely defers computation, buffering hidden states until the geometric state indicates sufficient commitment to generate.

### 2.5 Knowledge Graph-Grounded Reasoning

Belova et al. (2025) propose domain-specific superintelligence (DSS) as an alternative to monolithic scaling, with knowledge graph paths providing training curricula and reward signals. Their three-signal reward (axiomatic validity, chain continuity, terminal grounding) scores reasoning outputs based on KG structure. We extend this by observing that trajectory bias could replace or augment position encoding for graph traversal: instead of encoding hop distance, the model would encode the geometric state of the traversal (e.g., how constrained the remaining path options are), enabling attention patterns that naturally adapt to graph topology.

---

## 3. Anticipation Geometry Recap

Below is a brief summary of the seven scalars of Anticipation Geometry. Full definitions, proofs of domain-generality, and empirical validation across three domains are provided in the companion paper (Diomande, 2026).

### 3.1 The Seven Scalars

Let $\hat{\tau} = \{(\phi(s_1), t_1), \ldots, (\phi(s_T), t_T)\}$ be a latent trajectory, where $\phi$ is an embedding function mapping domain observations to vectors in a metric space $(X, d)$.

| Scalar | Symbol | Range | Definition |
|--------|--------|-------|------------|
| **Commitment** | $\mathcal{C}(t)$ | $[0, 1]$ | Weighted combination of $(1 - \mathcal{U})$, mean constraint proximity, and directional persistence. Measures irreversibility. |
| **Uncertainty** | $\mathcal{U}(t)$ | $[0, 1]$ | Directional entropy of $k$-nearest-neighbor continuation vectors. Measures how many futures remain. |
| **Transition Pressure** | $\mathcal{T}(t)$ | $(-\infty, +\infty)$ | $d\mathcal{C}/dt - d\mathcal{U}/dt$. Rate at which futures are collapsing. Positive = converging, negative = diverging. |
| **Recovery Margin** | $\mathcal{R}(t)$ | $[0, 1]$ | Complement of mean constraint proximity. How far from constraint boundaries. |
| **Phase Stiffness** | $\mathcal{P}(t)$ | $[0, 1]$ | Cosine similarity between consecutive velocity vectors. High = ballistic (locked direction). Low = gestural (direction changing). |
| **Novelty** | $\mathcal{N}(t)$ | $[0, 1]$ | Normalized minimum distance to all previous states. High = unexplored territory. Low = revisiting known regions. |
| **Stability** | $\sigma(t)$ | $[0, 1]$ | Inverse of normalized jerk magnitude (third derivative of position). High = smooth trajectory. Low = abrupt acceleration changes. |

### 3.2 Key Results from Paper 1

The companion paper establishes that these scalars are domain-general and carry statistically significant predictive signal:

- **Conversation convergence**: Transition pressure sign predicts conversation convergence at 71.8% accuracy ($z = 2.72$, $p < 0.007$) across 164 conversations.
- **KG path discrimination**: Anticipation-augmented rewards discriminate valid from hard-negative KG paths with 81.0% pairwise accuracy (Cohen's $d = 2.23$) across 199 multi-hop paths.
- **Cross-domain consistency**: The same seven scalars, with no domain-specific tuning, produce valid distributions on kinematic, conversational, and graph traversal data.

These results motivate the present work: if the scalars carry predictive signal about reasoning quality, then injecting them into the attention mechanism should improve the model's ability to attend to relevant context during reasoning.

### 3.3 The GeometryBridge

The connection between the numpy-based Anticipation Geometry engine and the PyTorch transformer is implemented by the `GeometryBridge` class (410 lines, `geometry_bridge.py`). The bridge performs two functions:

**Forward direction**: Given a batch of embedding tensors `(batch, seq, d_model)`, the bridge detaches them from the computation graph, converts to numpy, computes all seven scalars using the `AnticipationGeometry` engine, and returns a `(batch, seq, 7)` tensor suitable for the trajectory bias network. The original four scalars (commitment, uncertainty, transition pressure, recovery margin) come from the core engine. The three extended scalars (phase stiffness, novelty, stability) are computed directly in the bridge:

- **Phase stiffness**: Cosine similarity between consecutive velocity vectors $\mathbf{v}_{t} = \mathbf{z}_t - \mathbf{z}_{t-1}$, mapped from $[-1, 1]$ to $[0, 1]$.
- **Novelty**: For each position $t$, the minimum Euclidean distance to all previous states $\{\mathbf{z}_1, \ldots, \mathbf{z}_{t-1}\}$, normalized by twice the maximum distance from the centroid.
- **Stability**: For each position $t \geq 3$, the jerk $\mathbf{j}_t = \mathbf{a}_t - \mathbf{a}_{t-1}$ (third derivative), normalized and inverted: $\sigma(t) = 1 - \|\mathbf{j}_t\| / \max_t \|\mathbf{j}_t\|$.

**Inverse direction**: The bridge converts model-predicted scalar tensors back to `AnticipationPacket` objects for analysis, de-normalizing transition pressure from $[0, 1]$ back to its native $[-2, 2]$ range.

---

## 4. Architecture

The Anticipatory Transformer consists of seven components, assembled in a specific pipeline. We describe each component, its role, its parameter count, and its connections to the others.

### 4.1 TrajectoryBiasNetwork

**Purpose**: Map seven anticipation scalars to per-head, pairwise attention biases.

**Architecture**: A three-layer MLP with GELU activations:

$$\text{scalars} \in \mathbb{R}^7 \xrightarrow{\text{Linear}(7, 56)} \text{GELU} \xrightarrow{\text{Linear}(56, 56)} \text{GELU} \xrightarrow{\text{Linear}(56, n_{\text{heads}})}$$

The hidden dimension is $7 \times 8 = 56$, chosen to be a small multiple of the input dimensionality. The network is initialized with small weights (Xavier uniform, gain 0.1) so that the trajectory bias starts near zero, allowing the model to learn from a standard-attention baseline and gradually incorporate geometric information.

**Output construction**: The network produces a per-position, per-head bias magnitude $\mathbf{b} \in \mathbb{R}^{B \times L \times H}$. This is combined with a learned distance kernel to produce the full $(B, H, L, L)$ bias tensor:

$$\text{bias}[b, h, i, j] = b_{b,i,h} \cdot \exp\left(-\alpha_h \cdot |i - j| \cdot 0.01 + \beta_h\right)$$

where $\alpha_h$ and $\beta_h$ are learned per-head distance scale and offset parameters. The distance kernel introduces an inductive bias toward local attention when $\alpha_h$ is large, or toward uniform attention when $\alpha_h$ is near zero. The per-head parameterization allows different heads to have different "attention radii" modulated by the geometric state.

**Orthogonality penalty**: To encourage each head to use a different aspect of the anticipation geometry, I regularize the first layer's weight matrix. I reshape $W \in \mathbb{R}^{56 \times 7}$ into per-head groups $(n_{\text{heads}}, 56/n_{\text{heads}}, 7)$, compute a mean vector per head, L2-normalize these vectors, and penalize the off-diagonal entries of the resulting Gram matrix:

$$\mathcal{L}_{\text{ortho}} = \sum_{i \neq j} \left(\hat{\mathbf{w}}_i^\top \hat{\mathbf{w}}_j\right)^2$$

where $\hat{\mathbf{w}}_i$ is the normalized mean weight vector for head $i$. This penalty is added to the total loss with weight $\lambda_{\text{ortho}} = 0.1$.

### 4.2 TrajectoryAttention

**Purpose**: Standard multi-head attention with additive trajectory bias.

**Architecture**: Fused QKV projection $(d_{\text{model}} \to 3 \cdot d_{\text{model}})$, followed by split into heads, scaled dot-product attention with additive bias, and output projection.

The core computation at inference is:

```
qkv = W_qkv @ x                          # (B, L, 3*d_model)
Q, K, V = split_heads(qkv)               # each (B, H, L, d_head)
scores = (Q @ K^T) / sqrt(d_head)        # (B, H, L, L)
scores = scores + trajectory_bias         # ADDITIVE BIAS (core innovation)
scores = scores + causal_mask             # -inf for future positions
attn = softmax(scores) @ V               # (B, H, L, d_head)
output = W_out @ merge_heads(attn)        # (B, L, d_model)
```

**Local windowed attention** (fast pathway): When `is_local=True`, a window mask is applied that sets attention scores to $-\infty$ for positions outside a window of size `fast_window` centered on each query position. This restricts the fast pathway to local context while the slow pathway attends globally.

### 4.3 DualPathwayBlock

**Purpose**: A transformer block that can operate as either a fast (local) or slow (global) pathway element.

**Architecture**: Pre-norm residual connections with LayerNorm, following the GPT-2 / LLaMA convention:

$$\mathbf{x} \leftarrow \mathbf{x} + \text{Dropout}(\text{TrajectoryAttn}(\text{LN}(\mathbf{x}), \text{bias}, \text{mask}))$$
$$\mathbf{x} \leftarrow \mathbf{x} + \text{FFN}(\text{LN}(\mathbf{x}))$$

The FFN uses GELU activation: $\text{FFN}(\mathbf{x}) = W_2 \cdot \text{GELU}(W_1 \cdot \mathbf{x})$.

Both fast and slow pathway blocks receive the same trajectory bias, but the local windowed attention in fast blocks restricts its effect to nearby positions, while global attention in slow blocks allows trajectory-informed attention across the full sequence.

### 4.4 AnticipationHead

**Purpose**: Predict the seven anticipation scalars from the model's hidden states.

**Architecture**: LayerNorm followed by a two-layer MLP:

$$\text{LN}(\mathbf{h}) \xrightarrow{\text{Linear}(d_{\text{model}}, d_{\text{model}}/2)} \text{GELU} \xrightarrow{\text{Linear}(d_{\text{model}}/2, 7)} \text{activations}$$

Each output dimension has a scalar-appropriate activation:

| Scalar | Activation | Output Range |
|--------|-----------|-------------|
| Commitment | sigmoid | $[0, 1]$ |
| Uncertainty | sigmoid | $[0, 1]$ |
| Transition Pressure | $\tanh \times 2$ | $[-2, 2]$ |
| Recovery Margin | sigmoid | $[0, 1]$ |
| Phase Stiffness | sigmoid | $[0, 1]$ |
| Novelty | sigmoid | $[0, 1]$ |
| Stability | sigmoid | $[0, 1]$ |

**Self-referential training**: The model contains two AnticipationHead instances. The *initial* head predicts scalars from the post-embedding representations (before attention). The *refined* head predicts scalars from the post-attention hidden states. During training with external supervision, the refined head's predictions are compared against externally computed ground truth via MSE loss. During inference, the refined head's predictions are fed back into the trajectory bias network, closing the self-referential loop: the model steers its own attention based on its own predicted geometric state.

### 4.5 Full Model Pipeline

The `AnticipatoryTransformer` assembles all components into a 11-step pipeline:

1. **Token embedding + positional encoding**: Standard embedding lookup plus sinusoidal positional encoding.
2. **Initial anticipation prediction**: The initial AnticipationHead predicts scalars from embeddings.
3. **Scalar selection**: During training with supervision, external scalars (from GeometryBridge) are used. Otherwise, the initial predictions are used.
4. **Trajectory bias computation**: The TrajectoryBiasNetwork maps selected scalars to $(B, H, L, L)$ attention bias.
5. **Causal masking**: Standard upper-triangular $-\infty$ mask for autoregressive generation.
6. **Fast pathway**: $n_{\text{fast}}$ DualPathwayBlocks with local windowed attention, all receiving trajectory bias.
7. **Slow pathway**: $n_{\text{slow}}$ DualPathwayBlocks with global attention, all receiving trajectory bias.
8. **Pathway gate**: A learned gate combines fast and slow pathway outputs based on scalars and hidden states (described in Section 6).
9. **Refined anticipation prediction**: The refined AnticipationHead predicts scalars from the combined hidden states.
10. **Final LayerNorm + LM head**: Standard language model head with weight tying (shared embedding and output projection weights).
11. **Commitment gate**: The CommitmentGate produces emission probabilities (described in Section 5).

**Weight tying**: The output projection matrix $W_{\text{out}} \in \mathbb{R}^{V \times d_{\text{model}}}$ shares weights with the token embedding matrix, following standard practice (Press & Wolf, 2017). This reduces parameters and improves sample efficiency.

**Initialization**: All linear layers are initialized with $\mathcal{N}(0, 0.02)$, embeddings with $\mathcal{N}(0, 0.02)$, LayerNorm with weight 1 and bias 0, following GPT-2 conventions.

---

## 5. The Commitment Gate

### 5.1 Motivation

Standard autoregressive generation emits one token per forward pass, unconditionally. Every position in the sequence produces output regardless of the model's internal confidence. This is computationally efficient but cognitively unrealistic: human language production involves deliberation, pauses, and internal rehearsal before committing to speech.

The Anticipation Geometry framework provides a principled basis for variable-rate generation. The commitment scalar $\mathcal{C}(t)$ directly measures how irreversible the trajectory has become. When commitment is low, the model is still exploring, and forcing token emission risks premature commitment to a suboptimal continuation. When commitment is high, the model has converged on a direction and can safely produce output.

### 5.2 Architecture

The CommitmentGate combines two signals: a learned projection from hidden states and the predicted commitment scalar.

$$g_{\text{learned}} = \sigma(W_g \mathbf{h}_t + b_g)$$
$$g_{\text{combined}} = \sigma\left(W_c \begin{bmatrix} g_{\text{learned}} \\ \mathcal{C}(t) \end{bmatrix} + b_c\right)$$

where $W_g \in \mathbb{R}^{1 \times d_{\text{model}}}$ projects hidden states to a scalar, $\sigma$ is the sigmoid function, and $W_c \in \mathbb{R}^{1 \times 2}$ combines the learned gate with the commitment scalar.

### 5.3 Training vs. Inference

**Training (soft mode)**: The gate produces differentiable values in $(0, 1)$. The logits are multiplied by the gate value:

$$\hat{\mathbf{y}}_t = g_t \cdot W_{\text{LM}} \mathbf{h}_t$$

This soft gating allows gradient flow through the gate while modulating the influence of low-commitment positions on the loss.

**Inference (hard mode)**: The gate applies a hard threshold:

$$\text{emit}(t) = \begin{cases} 1 & \text{if } g_t > \theta \\ 0 & \text{otherwise} \end{cases}$$

where $\theta$ is the commitment threshold (default 0.8). When the gate blocks emission, the hidden state is buffered and the model performs another forward pass with additional context before re-evaluating.

### 5.4 Buffer Dynamics and Threshold Decay

To prevent infinite stalling, two safety mechanisms are implemented:

1. **Max buffer**: After `max_buffer` (default 5) consecutive non-emission steps, the gate forces emission regardless of commitment.

2. **Threshold decay**: The effective threshold decreases as the buffer fills:

$$\theta_{\text{eff}} = \theta \cdot \left(1 - 0.5 \cdot \frac{n_{\text{buffered}}}{\text{max\_buffer}}\right)$$

After 5 buffered steps, the effective threshold drops to $0.5 \times \theta$, making emission progressively easier. This creates a soft deadline that balances deliberation against latency.

### 5.5 Commitment-Gate Correlation

A key validation metric is the correlation between the predicted commitment scalar and the gate output. If the gate is functioning correctly, it should track commitment: high commitment positions should produce high gate values, and vice versa. In the smoke tests, this correlation reaches +0.93, indicating that the gate is strongly coupled to the geometric state.

---

## 6. Dual Pathway Architecture

### 6.1 Design Rationale

Neural signals in biological brains are processed at multiple temporal scales simultaneously. Fast signals (saccades, phoneme boundaries) operate at 100-200ms. Slow signals (sentence meaning, narrative structure) operate at 500ms-2s. The dual-pathway architecture mirrors this temporal hierarchy.

**Fast pathway**: $n_{\text{fast}}$ transformer blocks with local windowed attention (window size 128 in default config, 32 in small config). Processes every token. Captures syntax, local coherence, and high-frequency patterns. The windowed attention mask restricts attention to $\pm w/2$ positions from the query, where $w$ is the window size.

**Slow pathway**: $n_{\text{slow}}$ transformer blocks with full global attention (2048-token context in default config). In the current implementation, also processes every token (the original design called for processing every 4 tokens, to be implemented in future work). Captures long-range dependencies, global topic structure, and low-frequency patterns.

### 6.2 Pathway Gate

The fast and slow pathway outputs are combined by a learned gate that adapts based on the anticipation scalars and current hidden states:

$$\alpha = \sigma\left(W_{\text{gate}} \begin{bmatrix} \mathbf{s}_t \\ \mathbf{h}_t \end{bmatrix} + b_{\text{gate}}\right)$$

$$\mathbf{h}_{\text{out}} = \alpha \cdot \mathbf{h}_{\text{fast}} + (1 - \alpha) \cdot \mathbf{h}_{\text{slow}}$$

where $\mathbf{s}_t \in \mathbb{R}^7$ is the scalar vector, $\mathbf{h}_t$ is the input hidden state, and $W_{\text{gate}} \in \mathbb{R}^{1 \times (7 + d_{\text{model}})}$. The gate weight matrix has dimensions $1 \times (7 + d_{\text{model}})$, allowing the model to learn when to favor local vs. global context based on the geometric state.

**Intuition**: When commitment is high (the model is in a ballistic, committed trajectory), the fast pathway's local attention should dominate, because the relevant information is nearby. When uncertainty is high (the model is exploring), the slow pathway's global attention should dominate, because the model needs to consider distant context to resolve ambiguity.

### 6.3 Comparison with Existing Dual-Scale Approaches

Mixture-of-Experts (MoE) architectures (Shazeer et al., 2017; Fedus et al., 2022) use a router to select different expert FFNs per token, achieving conditional computation. The pathway gate is conceptually similar but operates at the attention level rather than the FFN level, and the gating signal is grounded in geometric trajectory analysis rather than learned from data alone.

Mamba (Gu & Dao, 2023) and other state-space models process sequences with linear-time complexity by maintaining a recurrent state. The slow pathway could be replaced with a state-space model for efficiency, though I use standard global attention in the current implementation for simplicity and interpretability.

---

## 7. Implementation

### 7.1 Codebase Structure

The implementation comprises 2,867 lines of Python across six files:

| File | Lines | Description |
|------|-------|-------------|
| `model.py` | 1,119 | All model components: TrajectoryBiasNetwork, TrajectoryAttention, DualPathwayBlock, AnticipationHead, CommitmentGate, AnticipatoryTransformer, BaselineTransformer |
| `evaluate.py` | 608 | Five evaluation metrics: perplexity, commitment correlation, context efficiency, scalar accuracy, head orthogonality |
| `train.py` | 576 | Training loop with external scalar supervision, cosine LR warmup, gradient clipping, synthetic data generation |
| `geometry_bridge.py` | 410 | NumPy-to-PyTorch bridge for anticipation scalar computation |
| `config.py` | 111 | Dataclass configuration with small() preset for smoke tests |
| `__init__.py` | 43 | Package exports |

### 7.2 Small Configuration (Smoke Test)

For rapid validation, I define a small configuration:

| Parameter | Value |
|-----------|-------|
| `d_model` | 128 |
| `n_heads` | 4 |
| `d_head` | 32 |
| `n_fast_layers` | 2 |
| `n_slow_layers` | 2 |
| `fast_window` | 32 |
| `d_ff` | 256 |
| `vocab_size` | 1,000 |
| `max_seq_len` | 256 |
| `dropout` | 0.1 |
| `trajectory_dims` | 7 |
| `commitment_threshold` | 0.8 |
| `max_buffer` | 5 |
| `lambda_ortho` | 0.1 |

This configuration produces 678,206 total parameters, distributed as follows:

| Component | Parameters |
|-----------|-----------|
| Token embedding | 128,000 |
| Positional encoding | 0 (sinusoidal, fixed) |
| Trajectory bias network | ~3,500 |
| Fast pathway (2 layers) | ~200,000 |
| Slow pathway (2 layers) | ~200,000 |
| Pathway gate | ~135 |
| Anticipation heads (2x) | ~18,000 |
| Commitment gate | ~131 |
| Final LayerNorm | 256 |
| LM head | 0 (weight-tied) |
| **Total** | **678,206** |

### 7.3 Default Configuration

The default (full-scale) configuration targets a ~36M parameter model:

| Parameter | Value |
|-----------|-------|
| `d_model` | 512 |
| `n_heads` | 8 |
| `n_fast_layers` | 3 |
| `n_slow_layers` | 3 |
| `fast_window` | 128 |
| `d_ff` | 2,048 |
| `vocab_size` | 32,000 |
| `max_seq_len` | 2,048 |

### 7.4 Training Pipeline

Training proceeds with external scalar supervision from the GeometryBridge:

1. **Data loading**: Synthetic data with deliberate structure (repeating motifs for high commitment, random segments for high uncertainty, gradual convergence for transition pressure), or WikiText-2 with character-level tokenization.

2. **External scalar computation**: For each batch, the GeometryBridge computes the seven scalars from the embedding trajectory. This computation is performed with `torch.no_grad()` to prevent gradient flow through the external supervisor.

3. **Forward pass**: The model receives both `input_ids` and `external_scalars`. The trajectory bias is computed from the external scalars (not the model's predictions), and the AnticipationHead's predictions are compared against the external scalars via MSE loss.

4. **Loss computation**: The total loss is:

$$\mathcal{L} = \mathcal{L}_{\text{LM}} + \mathcal{L}_{\text{scalar}} + \lambda_{\text{ortho}} \cdot \mathcal{L}_{\text{ortho}}$$

where $\mathcal{L}_{\text{LM}}$ is standard cross-entropy, $\mathcal{L}_{\text{scalar}}$ is the MSE between predicted and external scalars, and $\mathcal{L}_{\text{ortho}}$ is the orthogonality penalty.

5. **Optimization**: AdamW with $\beta = (0.9, 0.95)$, weight decay 0.01, gradient clipping at norm 1.0, cosine LR schedule with linear warmup.

### 7.5 Relationship to Original Design

The original design (January 2026, recovered source at `cc-anticipatory-transformer/phase0/`) used 5 trajectory dimensions (temporal, semantic, depth, homogeneity, salience) and per-dimension embedding with separate `nn.Linear(1, embed_dim)` modules for each dimension. The current implementation uses the 7 scalars from Anticipation Geometry and a unified MLP, which is simpler and aligns with the formal framework. The original design also included xFormers/FlashAttention compatibility and a target of ~2M parameters with WikiText-103 training. The core architectural choices (additive bias, dual pathway, commitment gating, orthogonality regularization) are preserved from the original design.

---

## 8. Preliminary Results

To be clear, these results are from **50 training steps on synthetic data with a 678K-parameter model**. They demonstrate that the mechanism works as intended, not that it achieves competitive performance on benchmarks. Full-scale evaluation is deferred to future work.

### 8.1 Training Dynamics

Over 50 training steps with batch size 8 and learning rate $10^{-3}$:

| Metric | Step 5 | Step 50 | Trend |
|--------|--------|---------|-------|
| Total loss | 7.19 | 6.83 | Decreasing |
| LM loss | ~6.9 | ~6.7 | Decreasing |
| Scalar MSE | 0.15 | 0.07 | Decreasing (53% reduction) |
| Ortho penalty | ~0.05 | 0.005 | Converging (90% reduction) |

The loss decrease from 7.19 to 6.83 in 50 steps is modest but monotonic, indicating stable training with no divergence. The scalar MSE halving from 0.15 to 0.07 shows the AnticipationHead is learning to predict the external geometry. The orthogonality penalty converging to 0.005 indicates the heads are successfully diversifying.

### 8.2 Commitment Gate Behavior

The commitment-gate correlation reaches **+0.93** within 50 training steps. This is a strong positive correlation, indicating that:

1. The gate output tracks the commitment scalar closely.
2. High-commitment positions receive high gate values (the model is willing to emit).
3. Low-commitment positions receive low gate values (the model prefers to defer).

This correlation emerges rapidly because the gate architecture directly receives the commitment scalar as input (via `gate_combine`), providing a strong inductive bias. The question for future work is whether this correlation improves generation quality, not whether it exists.

### 8.3 Head Specialization

After 50 steps, the four attention heads (in the small configuration) specialize to **3 out of 4 unique dominant scalars**. That is, three of the four heads have a different "most important" anticipation scalar in their trajectory bias weights. The fourth head shares its dominant scalar with one of the other three.

This is a direct consequence of the orthogonality penalty: without it, all heads would converge to the same scalar weighting. The penalty drives diversification, and 3/4 unique scalars after only 50 steps is a strong early signal.

### 8.4 Scalar Prediction Accuracy

The AnticipationHead's ability to predict the seven scalars from hidden states improves throughout training:

| Metric | Step 5 | Step 50 |
|--------|--------|---------|
| Total MSE | 0.15 | 0.07 |

At step 50, the model predicts the geometric state of its own embedding trajectory with an average error of 0.07 per scalar. Given that most scalars are in $[0, 1]$, this corresponds to approximately 7% average error, which is sufficient for the trajectory bias to provide meaningful steering signal.

### 8.5 Generation

The model generates coherent token sequences when prompted, with the commitment gate successfully modulating output. In hard-gating mode (inference), the model buffers 1-3 steps before emitting tokens at high-commitment positions, demonstrating the variable-rate generation mechanism. However, at this scale and training duration, the generated tokens do not exhibit semantic coherence beyond local patterns.

### 8.6 Baseline Comparison

The codebase includes a `BaselineTransformer` with identical architecture but without trajectory bias, anticipation heads, or commitment gate. In the evaluation framework (`evaluate.py`), five metrics are compared between anticipatory and baseline models: perplexity, commitment-loss correlation, context efficiency at multiple context sizes, scalar prediction accuracy, and head orthogonality. Full comparative results require training beyond the smoke test regime.

---

## 9. Applications

Three application domains are identified where application domains where the trajectory-bias mechanism provides advantages over standard position encodings. These are speculative and untested at the time of writing.

### 9.1 Agent Reasoning

Modern AI agents (ReAct, Toolformer, Claude's tool use) execute multi-step plans involving tool calls, intermediate reasoning, and decision points. The trajectory through the agent's internal state space exhibits the same geometric patterns that Anticipation Geometry captures: exploration phases (high uncertainty, low commitment), tool-selection phases (rising transition pressure), and execution phases (high commitment, high phase stiffness).

An Anticipatory Transformer serving as the agent's backbone could use trajectory bias to:

- **Attend narrowly** during committed execution (the plan is set, focus on local details).
- **Attend broadly** during exploration (consider all available tools and context).
- **Defer emission** at decision points (buffer hidden states until commitment exceeds the threshold, then commit to a tool call).

The commitment gate is particularly relevant here: premature tool invocation is a known failure mode of agent systems. A gate that defers until geometric commitment is high could reduce wasted tool calls.

### 9.2 Multi-Hop Knowledge Graph Traversal

Knowledge graph reasoning requires attending to different parts of the input depending on the current hop in the traversal. At hop 1, the model should attend to the query and source entity. At hop 3, the model should attend to accumulated path context and the remaining candidate entities. Standard position encodings do not distinguish these phases.

Trajectory bias, computed from the geometric state of the traversal through entity-embedding space, naturally encodes this progression. As the path narrows (increasing commitment, decreasing uncertainty), the bias would shift attention toward the accumulated path and away from the broader query context. This aligns with the finding in Paper 1 that anticipation-augmented rewards achieve 81.0% pairwise accuracy on KG path discrimination.

The DSS framework (Belova et al., 2025) provides the training curriculum: KG paths ordered by complexity (2-hop before 5-hop) produce a natural progression from high-transition-pressure-quickly to sustained-moderate-transition-pressure, which the trajectory bias can encode directly.

### 9.3 Motion-to-Audio Synthesis

The Anticipation Geometry implementation operates at 50 Hz on quaternion motion data (8,173 lines of Rust in the anticipation crate). The trajectory scalars characterize physical movements in real time: high commitment during a committed dance move, high novelty at style transitions, high phase stiffness during rhythmic patterns.

An Anticipatory Transformer mapping motion trajectories to audio parameters (pitch, timbre, rhythm) could use trajectory bias to align attention windows with the temporal structure of the movement. During a sustained, rhythmic motion (high phase stiffness, high stability), the model would attend to local temporal context and produce rhythmically consistent audio. During a transition (high novelty, high transition pressure), the model would broaden its attention to capture the gestural intent of the new movement phase.

This application requires the dual-pathway architecture: the fast pathway captures per-frame motion dynamics at 50 Hz, while the slow pathway captures phrase-level structure at 2-5 Hz. The pathway gate, modulated by the anticipation scalars, determines the mixing in real time.

---

## 10. Discussion

### 10.1 Limitations

**Scale**: All reported results are from a 678K-parameter model trained for 50 steps on synthetic data. We cannot make claims about the architecture's behavior at the 100M+ parameter scale where transformers become practically useful. The mechanism may be washed out by the sheer capacity of large models, or it may become more important as reasoning depth increases. We do not know.

**External scalar bottleneck**: During training, the GeometryBridge computes scalars per-batch by iterating over the trajectory in Python/NumPy. This is the computational bottleneck: for a batch of size 8 with sequence length 256, the bridge must compute 2,048 trajectory analyses. This is not parallelized across the batch dimension in the current implementation. A CUDA implementation of the core scalar computations would be necessary for training at scale.

**Synthetic data limitations**: The synthetic dataset is designed with deliberate patterns (repeating motifs, random segments, gradual convergence) to test whether the model can learn trajectory geometry. Real language data has more complex and noisy trajectory structure. The transfer from synthetic to natural language is unvalidated.

**Commitment gate evaluation**: While the gate achieves high correlation with commitment (+0.93), I have not demonstrated that this correlation improves generation quality. The gate could be perfectly correlated with commitment and still produce worse text than ungated generation if commitment is not actually predictive of output quality in the language domain. The companion paper shows commitment is predictive in conversation and KG domains, but not specifically in autoregressive language modeling.

**Slow pathway subsampling**: The original design (January 2026) specified that the slow pathway should process every 4th token, not every token. The current implementation processes every token in both pathways, meaning the "slow" pathway is distinguished only by its attention window (global vs. local), not by its update frequency. Implementing true temporal subsampling requires careful handling of residual connections and is deferred to future work.

### 10.2 What Full Training Would Require

To move from smoke test to benchmark-competitive evaluation, the following would be needed:

1. **Scale**: Train the default configuration (~36M parameters) on WikiText-103 or a comparable dataset for 10K-50K steps with proper tokenization (BPE, not character-level).

2. **GPU compute**: The default configuration with batch size 32, gradient accumulation 4, and sequence length 2048 requires ~2 GPU-hours on an A100 for 10K steps (estimated).

3. **CUDA bridge**: Port the GeometryBridge scalar computations to CUDA to eliminate the Python/NumPy bottleneck. The computations are elementwise or nearest-neighbor, both of which are highly parallelizable.

4. **Proper baselines**: Compare against (a) a standard transformer of identical parameter count, (b) a transformer with ALiBi, (c) a transformer with RoPE, and (d) the same architecture with random (non-geometric) bias. The random-bias ablation is critical: it distinguishes the contribution of geometric information from the contribution of any additive bias.

5. **Downstream evaluation**: Evaluate on tasks that specifically require multi-step reasoning: GSM8K, ARC-Challenge, multi-hop QA. The hypothesis is that trajectory bias helps most on tasks where the reasoning trajectory has meaningful geometric structure.

### 10.3 Relationship to Paper 1

This paper is a direct extension of Anticipation Geometry (Diomande, 2026). Paper 1 establishes the mathematical framework and demonstrates that the seven scalars carry domain-general predictive signal. This paper takes the next logical step: if the scalars are informative, inject them into the model that generates the trajectories.

The key open question is whether this injection creates a *virtuous cycle*. If the model steers attention based on geometric state, and the geometric state is computed from the model's own embeddings, then the model's behavior is coupled to its own geometry. Does this coupling stabilize (the model learns to maintain high-quality geometric trajectories) or destabilize (small perturbations in geometry amplify through the feedback loop)?

The preliminary results suggest stability: the orthogonality penalty converges, the scalar prediction MSE decreases, and the training loss decreases monotonically. But 50 steps is insufficient to detect instabilities that might emerge at longer training horizons.

### 10.4 Why Additive, Not Multiplicative

The trajectory bias is additive, not multiplicative. This is a deliberate design choice for three reasons:

1. **Numerical stability**: Multiplicative attention bias can cause gradient explosion when the bias values are far from 1. Additive bias shifts logits, which softmax normalizes.

2. **Zero-initialization**: With small initial weights, the additive bias starts near zero, and the model begins as a standard transformer. It can gradually learn to use the geometric signal without disrupting early training dynamics.

3. **ALiBi precedent**: ALiBi (Press et al., 2022) demonstrated that additive bias to attention logits is effective and efficient. The approach generalizes ALiBi from distance-based bias to geometry-based bias.

---

## 11. Conclusion

This paper presents the Anticipatory Transformer, a modified transformer architecture where attention is steered by the geometric state of the reasoning trajectory. The architecture introduces three mechanisms:

1. **Trajectory bias**: Seven anticipation scalars are mapped to per-head attention biases via a learned network, enabling heads to specialize to different geometric dimensions.

2. **Commitment gating**: A threshold-based mechanism allows the model to defer token emission at low-commitment positions, enabling variable-rate generation.

3. **Dual pathway**: Fast (local) and slow (global) attention pathways are combined by a trajectory-aware gate, enabling multi-scale processing.

The complete implementation is 2,867 lines of Python, trains on CPU in seconds at the smoke-test scale, and demonstrates that the trajectory bias mechanism functions as intended: the commitment gate tracks commitment (+0.93 correlation), heads diversify (3/4 unique dominant scalars), and the model learns to predict its own geometric state (MSE 0.15 to 0.07 in 50 steps).

I do not claim that this architecture outperforms standard transformers on any benchmark. We claim that it represents a viable and novel approach to attention steering that encodes *process geometry* rather than *sequential position*, and that the preliminary evidence justifies further investigation at scale. If the seven scalars of Anticipation Geometry carry predictive signal about reasoning quality across physical, conversational, and graph-traversal domains (as Paper 1 demonstrates), then an architecture that uses these scalars to steer its own attention has the potential to produce more efficient, more interpretable, and more controllable reasoning.

The source code is available at `anticipation-geometry/transformer/`.

---

## References

Belova, M., et al. (2026). Societies of domain-specific superintelligence. *arXiv preprint arXiv:2603.14147*.

Bronstein, M. M., Bruna, J., Cohen, T., & Velickovic, P. (2021). Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. *arXiv preprint arXiv:2104.13478*.

Cai, T., et al. (2024). Medusa: Simple LLM inference acceleration framework with multiple decoding heads. *Proceedings of ICML*.

Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science. *Behavioral and Brain Sciences*, 36(3), 181-204.

Diomande, M. (2026). Anticipation Geometry: Domain-general trajectory characterization with knowledge graph-grounded rewards. *Manuscript*.

Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120), 1-39.

Friston, K. (2010). The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.

Goyal, S., et al. (2024). Think before you speak: Training language models with pause tokens. *Proceedings of ICLR*.

Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*.

Kansal, R., et al. (2025). GraphMERT: Distilling knowledge graph structure into text encoders. *arXiv preprint arXiv:2510.09580*.

Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast inference from transformers via speculative decoding. *Proceedings of ICML*.

Pezzulo, G., Butz, M. V., Castelfranchi, C., & Falcone, R. (2008). The challenge of anticipation: A unifying framework for the analysis and design of artificial cognitive systems. *Springer*.

Press, O., Smith, N. A., & Lewis, M. (2022). Train short, test long: Attention with linear biases enables input length extrapolation. *Proceedings of ICLR*.

Press, O., & Wolf, L. (2017). Using the output embedding to improve language models. *Proceedings of EACL*.

Raffel, C., et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*, 21(140), 1-67.

Shazeer, N., et al. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *Proceedings of ICLR*.

Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). Self-attention with relative position representations. *Proceedings of NAACL*.

Su, J., et al. (2024). RoFormer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568, 127063.

Sumers, T. R., et al. (2024). Cognitive architectures for language agents. *Transactions on Machine Learning Research*.

Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 5998-6008.

Velickovic, P., et al. (2018). Graph Attention Networks. *Proceedings of ICLR*.

Xiao, Y., et al. (2025). QwQ-Med-3: Knowledge graph-grounded medical reasoning. *arXiv preprint arXiv:2507.13966*.

---

## Appendix A: Parameter Count Derivation

For the small configuration ($d = 128$, $H = 4$, $L_f = L_s = 2$, $d_{\text{ff}} = 256$, $V = 1000$):

**Embedding**: $V \times d = 1000 \times 128 = 128{,}000$

**Per transformer layer** (both pathways use the same structure):
- QKV projection: $d \times 3d = 128 \times 384 = 49{,}152$ (no bias)
- Output projection: $d \times d = 128 \times 128 = 16{,}384$ (no bias)
- FFN up: $d \times d_{\text{ff}} + d_{\text{ff}} = 128 \times 256 + 256 = 33{,}024$
- FFN down: $d_{\text{ff}} \times d + d = 256 \times 128 + 128 = 32{,}896$
- LayerNorm (2x): $2 \times 2d = 512$
- **Per layer total**: $\approx 131{,}968$

**4 layers total** (2 fast + 2 slow): $\approx 527{,}872$

**Trajectory bias network**:
- Linear(7, 56): $7 \times 56 + 56 = 448$
- Linear(56, 56): $56 \times 56 + 56 = 3{,}192$
- Linear(56, 4): $56 \times 4 + 4 = 228$
- Distance params: $4 + 4 = 8$
- **Total**: $\approx 3{,}876$

**Anticipation heads (2x)**:
- LayerNorm: $2 \times 128 = 256$
- Linear(128, 64): $128 \times 64 + 64 = 8{,}256$
- Linear(64, 7): $64 \times 7 + 7 = 455$
- **Per head**: $\approx 8{,}967$. **Two heads**: $\approx 17{,}934$

**Pathway gate**: Linear($7 + 128$, $1$) = $136$

**Commitment gate**: Linear($128$, $1$) + Linear($2$, $1$) = $131$

**Final LayerNorm**: $2 \times 128 = 256$

**LM head**: weight-tied with embedding, $0$ additional parameters.

**Total**: $128{,}000 + 527{,}872 + 3{,}876 + 17{,}934 + 136 + 131 + 256 \approx 678{,}205$

The implementation reports 678,206, consistent with this derivation up to rounding.

---

## Appendix B: Synthetic Data Design

The synthetic dataset (`make_synthetic_dataset` in `train.py`) generates sequences with deliberate structure to test trajectory geometry learning:

**Pattern 1 -- Repeating motif** (tokens 0-20, approximately): A random motif of 3-8 tokens is repeated 2-5 times. This should produce high commitment (the trajectory is locked into a repeating pattern) and high phase stiffness (the velocity autocorrelation is high because the same embedding subsequence repeats).

**Pattern 2 -- Random segment** (tokens 20-35, approximately): 5-15 random tokens from the full vocabulary. This should produce high uncertainty (the continuation is unpredictable) and high novelty (each state is far from previous states in embedding space).

**Pattern 3 -- Gradual convergence** (tokens 35-45, approximately): Tokens converge toward a center value with exponentially decreasing noise. This should produce rising transition pressure (futures are collapsing) and rising commitment (the trajectory is narrowing).

The remainder is filled with random tokens for padding. This three-phase structure provides a controlled test bed for verifying that the model's predicted scalars match the expected geometric patterns.

---

## Appendix C: Evaluation Metrics

The evaluation framework (`evaluate.py`, 608 lines) implements five comparative metrics:

1. **Perplexity**: Standard per-token cross-entropy loss, exponentiated. Measured on a held-out validation set.

2. **Commitment-loss correlation**: Pearson correlation between the predicted commitment scalar and the per-position cross-entropy loss. If commitment is meaningful, high commitment should correlate with *low* loss (the model is more accurate when committed). The correlation should be negative.

3. **Commitment-gate correlation**: Pearson correlation between the predicted commitment scalar and the gate output. Should be positive (+0.93 in the smoke test).

4. **Context efficiency**: Perplexity measured at multiple context lengths (4, 8, 16, 32 tokens). An anticipatory model should achieve equivalent perplexity with *less* context, because trajectory bias provides geometric context that compensates for shorter sequences.

5. **Head orthogonality**: Off-diagonal mean of the Gram matrix of per-head weight vectors from the trajectory bias network. Lower values indicate more diverse heads. Also reports the dominant scalar for each head (the scalar with the largest absolute weight).

---

## Appendix D: Relationship to the Production Rust Implementation

The Anticipation Geometry framework has a production implementation in Rust (anticipation crate, 26,602 lines across the full `crates/` directory, with 2,572 lines in the core anticipation crate). The Rust implementation operates at sub-2ms latency per frame for real-time motion capture at 50 Hz.

The Python implementation used by the Anticipatory Transformer (`anticipation_geometry/generalized_anticipation.py` and the `geometry_bridge.py` wrapper, totaling approximately 3,287 lines) is a faithful port of the core scalar computations. The Rust and Python implementations agree on scalar values to within floating-point precision.

For training at scale, the GeometryBridge would be the computational bottleneck. Three acceleration paths are available:

1. **CUDA kernels**: The scalar computations are elementwise or $k$-nearest-neighbor, both highly parallelizable. A custom CUDA kernel could compute all seven scalars for a batch in a single kernel launch.

2. **Torch compilation**: The scalar computations can be expressed as pure tensor operations (diff, norm, cdist, cummax) and compiled with `torch.compile` for automatic optimization.

3. **Rust FFI**: The existing Rust implementation could be called directly from Python via PyO3 bindings, gaining the performance of the production crate without reimplementation.
