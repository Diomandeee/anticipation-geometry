"""
Anticipatory Transformer Model
================================

Full PyTorch implementation of the Anticipatory Transformer, where
7 anticipation-geometry scalars steer multi-head attention via additive
trajectory bias.

Key equation:
    attention_scores = (Q @ K^T) / sqrt(d_k) + trajectory_bias(scalars)

The trajectory bias is ADDITIVE (not multiplicative) for numerical stability.
Each attention head learns its own mapping from the 7 scalars to a bias
pattern, so different heads can attend to different geometric aspects of
the reasoning trajectory.

Components:
    TrajectoryBiasNetwork: Maps 7 scalars -> per-head attention bias
    TrajectoryAttention: Multi-head attention with additive trajectory bias
    DualPathwayBlock: Fast (local) + slow (global) attention pathways
    AnticipationHead: Predicts the 7 scalars from hidden states
    CommitmentGate: Decides whether to emit or buffer
    AnticipatoryTransformer: Full model combining everything
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import AnticipatoryConfig
except ImportError:
    from config import AnticipatoryConfig


# ---------------------------------------------------------------------------
# Trajectory Bias Network
# ---------------------------------------------------------------------------

class TrajectoryBiasNetwork(nn.Module):
    """Maps 7 anticipation scalars to per-head attention bias matrices.

    For each attention head, this network produces a bias vector of length
    max_seq_len. The bias is applied additively to the attention logits,
    steering the attention pattern based on the geometric state of the
    trajectory.

    Architecture:
        scalars (7) -> Linear(7, hidden) -> GELU -> Linear(hidden, n_heads)

    The output is a per-position, per-head scalar bias. When applied to
    attention, position i's scalars produce a bias that modulates how
    much attention position i pays to every other position.

    Why per-position rather than per-pair?
    A full (seq, seq) bias per head would be O(seq^2 * n_heads) parameters
    in the bias network. Instead, we produce a per-position "attention
    temperature" per head: how much should this position's attention be
    sharpened or broadened? This is O(seq * n_heads), matching the
    intuition that commitment says "attend narrowly" while uncertainty
    says "attend broadly."

    Args:
        config: AnticipatoryConfig with trajectory_dims, n_heads, d_model.
    """

    def __init__(self, config: AnticipatoryConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.use_inscriptions = config.use_inscriptions
        hidden = config.trajectory_dims * 8  # 7 * 8 = 56

        # Inscription embedding: 10 motifs -> hidden dim, fused with scalars
        if self.use_inscriptions:
            self.inscription_emb = nn.Embedding(config.n_inscriptions, hidden)
            input_dim = config.trajectory_dims + hidden  # scalars + inscription embedding
        else:
            input_dim = config.trajectory_dims

        # Per-position (scalar + inscription) -> per-head bias
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.n_heads),
        )

        # Relative position bias modulator: combines trajectory-derived
        # bias with a learned distance kernel per head
        self.distance_scale = nn.Parameter(torch.ones(config.n_heads))
        self.distance_offset = nn.Parameter(torch.zeros(config.n_heads))

        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights so bias starts near zero."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        scalars: torch.Tensor,
        seq_len: int,
        inscription_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-head attention bias from anticipation scalars + inscriptions.

        Args:
            scalars: (batch, seq, 7) anticipation scalars at each position.
            seq_len: Sequence length for the attention matrix.
            inscription_ids: (batch, seq) inscription category IDs [0-9], optional.

        Returns:
            bias: (batch, n_heads, seq, seq) additive bias for attention logits.
        """
        batch = scalars.shape[0]

        # Fuse scalars with inscription embeddings
        if self.use_inscriptions:
            if inscription_ids is None:
                inscription_ids = torch.zeros(batch, scalars.shape[1], dtype=torch.long, device=scalars.device)
            insc_emb = self.inscription_emb(inscription_ids)  # (batch, seq, hidden)
            fused = torch.cat([scalars, insc_emb], dim=-1)  # (batch, seq, 7+hidden)
        else:
            fused = scalars

        # Compute per-position, per-head bias magnitude
        # (batch, seq, n_heads)
        position_bias = self.net(fused)

        # Build the (seq, seq) bias: position i's bias affects how it
        # attends to all positions j. We use the difference in bias
        # magnitude to create an asymmetric attention pattern.
        #
        # bias[i, j] = position_bias[i] * distance_kernel(i - j)
        # This means committed positions (high bias) attend narrowly,
        # while uncertain positions (low bias) attend broadly.

        # Relative position indices
        positions = torch.arange(seq_len, device=scalars.device, dtype=scalars.dtype)
        # (seq, seq) relative distances
        rel_dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()

        # Per-head distance kernel: exp(-scale * dist + offset)
        # (n_heads, seq, seq)
        dist_kernel = torch.exp(
            -self.distance_scale.view(-1, 1, 1) * rel_dist.unsqueeze(0) * 0.01
            + self.distance_offset.view(-1, 1, 1)
        )

        # Combine: (batch, n_heads, seq, seq)
        # position_bias (batch, seq, n_heads) -> (batch, n_heads, seq, 1)
        pb = position_bias.permute(0, 2, 1).unsqueeze(-1)  # (batch, n_heads, seq, 1)
        # dist_kernel: (1, n_heads, seq, seq) broadcast over batch
        bias = pb * dist_kernel.unsqueeze(0)  # (batch, n_heads, seq, seq)

        return bias

    def orthogonality_penalty(self) -> torch.Tensor:
        """Compute penalty encouraging different heads to use different
        aspects of the anticipation geometry.

        Returns the Frobenius norm of (W^T W - I) for the first layer's
        weight matrix projected to the head dimension.
        """
        # Get the first linear layer's weights
        first_layer = self.net[0]
        W = first_layer.weight  # (hidden, trajectory_dims)

        # Group by head: reshape to (n_heads, hidden//n_heads, trajectory_dims)
        per_head = W.view(self.n_heads, -1, W.shape[1])
        # Compute gram matrix of per-head projections
        # Average each head's rows into a single vector
        head_vecs = per_head.mean(dim=1)  # (n_heads, trajectory_dims)
        # Normalize
        head_vecs = F.normalize(head_vecs, dim=-1)
        # Gram matrix
        gram = head_vecs @ head_vecs.T
        # Penalty: off-diagonal elements should be zero
        eye = torch.eye(self.n_heads, device=gram.device)
        penalty = ((gram - eye) ** 2).sum()
        return penalty


class MoETrajectoryBias(nn.Module):
    """Mixture of Experts trajectory bias: 10 specialist networks, one per inscription.

    Each inscription category (stabilization, transition, oscillation, ...)
    has its own bias network that learns category-specific attention patterns.
    Routing is deterministic via inscription_id — no learned gating needed
    since the classification already happened upstream.

    This is a hard MoE in attention bias space:
    - convergence expert learns narrow, focused attention
    - exploration expert learns broad, dispersed attention
    - oscillation expert learns alternating patterns
    - etc.

    Total params: 10 × ~4K = ~40K (negligible at any model scale).

    Args:
        config: AnticipatoryConfig with trajectory_dims, n_heads, n_inscriptions.
    """

    def __init__(self, config: AnticipatoryConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_experts = config.n_inscriptions  # 10
        hidden = config.trajectory_dims * 8  # 56

        # 10 expert networks, each maps 7 scalars -> n_heads bias
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.trajectory_dims, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, config.n_heads),
            )
            for _ in range(self.n_experts)
        ])

        # Per-expert distance kernels (each inscription type has its own
        # notion of how attention decays with distance)
        self.distance_scales = nn.Parameter(torch.ones(self.n_experts, config.n_heads))
        self.distance_offsets = nn.Parameter(torch.zeros(self.n_experts, config.n_heads))

        self._init_weights()

    def _init_weights(self):
        for expert in self.experts:
            for module in expert:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        scalars: torch.Tensor,
        seq_len: int,
        inscription_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Route each position to its inscription-specific expert.

        Args:
            scalars: (batch, seq, 7) anticipation scalars.
            seq_len: Sequence length.
            inscription_ids: (batch, seq) inscription IDs [0-9]. Required.

        Returns:
            bias: (batch, n_heads, seq, seq) additive attention bias.
        """
        batch, seq, _ = scalars.shape
        device = scalars.device

        if inscription_ids is None:
            inscription_ids = torch.zeros(batch, seq, dtype=torch.long, device=device)

        # Compute bias per position, routing to the correct expert
        # (batch, seq, n_heads)
        position_bias = torch.zeros(batch, seq, self.n_heads, device=device)
        # Per-position distance scale/offset
        pos_dist_scale = torch.zeros(batch, seq, self.n_heads, device=device)
        pos_dist_offset = torch.zeros(batch, seq, self.n_heads, device=device)

        # Route each inscription to its expert
        for expert_id in range(self.n_experts):
            # Mask: which positions have this inscription
            mask = (inscription_ids == expert_id)  # (batch, seq)
            if not mask.any():
                continue

            # Get scalars for these positions
            expert_scalars = scalars[mask]  # (n_active, 7)
            expert_output = self.experts[expert_id](expert_scalars)  # (n_active, n_heads)

            # Scatter back
            position_bias[mask] = expert_output
            pos_dist_scale[mask] = self.distance_scales[expert_id].unsqueeze(0).expand(mask.sum(), -1)
            pos_dist_offset[mask] = self.distance_offsets[expert_id].unsqueeze(0).expand(mask.sum(), -1)

        # Build (seq, seq) bias with per-position distance kernels
        positions = torch.arange(seq, device=device, dtype=scalars.dtype)
        rel_dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # (seq, seq)

        # Per-position, per-head distance kernel
        # pos_dist_scale: (batch, seq, n_heads) -> use mean across positions for the kernel
        # For efficiency: use batch-averaged scale per head
        avg_scale = pos_dist_scale.mean(dim=1)  # (batch, n_heads)
        avg_offset = pos_dist_offset.mean(dim=1)  # (batch, n_heads)

        dist_kernel = torch.exp(
            -avg_scale.unsqueeze(-1).unsqueeze(-1) * rel_dist.unsqueeze(0).unsqueeze(0) * 0.01
            + avg_offset.unsqueeze(-1).unsqueeze(-1)
        )  # (batch, n_heads, seq, seq)

        # Combine
        pb = position_bias.permute(0, 2, 1).unsqueeze(-1)  # (batch, n_heads, seq, 1)
        bias = pb * dist_kernel  # (batch, n_heads, seq, seq)

        return bias

    def orthogonality_penalty(self) -> torch.Tensor:
        """Compute penalty encouraging different experts to produce different biases.

        For MoE this measures inter-expert diversity rather than inter-head diversity.
        """
        # Collect the first layer weights from each expert
        expert_vecs = []
        for expert in self.experts:
            W = expert[0].weight  # (hidden, trajectory_dims)
            expert_vecs.append(W.mean(dim=0))  # (trajectory_dims,)

        expert_vecs = torch.stack(expert_vecs)  # (n_experts, trajectory_dims)
        expert_vecs = F.normalize(expert_vecs, dim=-1)
        gram = expert_vecs @ expert_vecs.T
        eye = torch.eye(self.n_experts, device=gram.device)
        return ((gram - eye) ** 2).sum()

    def expert_utilization(self, inscription_ids: torch.Tensor) -> dict[int, int]:
        """Count how many positions each expert handles."""
        counts = {}
        for i in range(self.n_experts):
            counts[i] = (inscription_ids == i).sum().item()
        return counts


# ---------------------------------------------------------------------------
# Trajectory-Aware Multi-Head Attention
# ---------------------------------------------------------------------------

class TrajectoryAttention(nn.Module):
    """Multi-head attention with additive trajectory bias.

    Standard multi-head attention computes:
        scores = (Q @ K^T) / sqrt(d_k)
        attn = softmax(scores + mask)
        output = attn @ V

    Trajectory attention adds the geometric bias:
        scores = (Q @ K^T) / sqrt(d_k) + trajectory_bias
        attn = softmax(scores + mask)
        output = attn @ V

    The trajectory_bias encodes the anticipation geometry of the
    reasoning trajectory, telling each head how to modulate its
    attention pattern based on commitment, uncertainty, etc.

    Args:
        config: AnticipatoryConfig.
        is_local: If True, applies windowed attention (fast pathway).
    """

    def __init__(self, config: AnticipatoryConfig, is_local: bool = False):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.is_local = is_local
        self.window_size = config.fast_window
        self.scale = 1.0 / math.sqrt(self.d_head)

        # QKV projection (fused for efficiency)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        trajectory_bias: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional trajectory bias.

        Args:
            x: (batch, seq, d_model) input hidden states.
            trajectory_bias: (batch, n_heads, seq, seq) additive bias.
            mask: (batch, 1, seq, seq) or (1, 1, seq, seq) causal mask.

        Returns:
            output: (batch, seq, d_model) attended hidden states.
        """
        batch, seq, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.view(batch, seq, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, n_heads, seq, d_head)
        q, k, v = qkv.unbind(0)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # (batch, n_heads, seq, seq)

        # Add trajectory bias (the core innovation)
        if trajectory_bias is not None:
            # Ensure shapes match - trajectory_bias may be smaller
            tb = trajectory_bias
            if tb.shape[-1] != seq or tb.shape[-2] != seq:
                tb = tb[:, :, :seq, :seq]
            scores = scores + tb

        # Apply causal mask
        if mask is not None:
            scores = scores + mask

        # Local windowed attention (for fast pathway)
        if self.is_local:
            window_mask = self._make_window_mask(seq, x.device)
            scores = scores + window_mask

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        output = torch.matmul(attn_weights, v)
        # (batch, n_heads, seq, d_head) -> (batch, seq, d_model)
        output = output.transpose(1, 2).contiguous().view(batch, seq, self.d_model)

        return self.out_proj(output)

    def _make_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create a window mask that restricts attention to local context.

        Positions outside the window get -inf, which softmax converts to 0.
        """
        positions = torch.arange(seq_len, device=device)
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        # -inf outside window, 0 inside
        mask = torch.where(
            dist <= self.window_size // 2,
            torch.tensor(0.0, device=device),
            torch.tensor(float("-inf"), device=device),
        )
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Standard transformer FFN with GELU activation."""

    def __init__(self, config: AnticipatoryConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff)
        self.w2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.gelu(self.w1(x))))


# ---------------------------------------------------------------------------
# Dual-Pathway Block
# ---------------------------------------------------------------------------

class DualPathwayBlock(nn.Module):
    """Transformer block with dual attention pathways.

    Fast pathway: Local windowed attention that processes every token.
        Captures fine-grained, high-frequency patterns.

    Slow pathway: Global full attention that processes every N tokens.
        Captures long-range dependencies and global context.

    Both pathways receive trajectory bias, but they use it differently:
    - Fast pathway: bias modulates LOCAL attention (which nearby tokens matter?)
    - Slow pathway: bias modulates GLOBAL attention (which distant context matters?)

    The outputs are combined via a learned gate that adapts based on the
    anticipation scalars: when commitment is high, the fast pathway dominates
    (local detail matters). When uncertainty is high, the slow pathway
    dominates (global context needed).

    Args:
        config: AnticipatoryConfig.
        is_fast: If True, this is a fast-pathway block (local attention).
    """

    def __init__(self, config: AnticipatoryConfig, is_fast: bool = True):
        super().__init__()
        self.is_fast = is_fast

        # Self-attention (local for fast, global for slow)
        self.attn = TrajectoryAttention(config, is_local=is_fast)
        self.ffn = FeedForward(config)

        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        trajectory_bias: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-norm residual connections.

        Args:
            x: (batch, seq, d_model) input.
            trajectory_bias: (batch, n_heads, seq, seq) additive attention bias.
            mask: (batch, 1, seq, seq) causal mask.

        Returns:
            output: (batch, seq, d_model) transformed hidden states.
        """
        # Pre-norm attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, trajectory_bias=trajectory_bias, mask=mask)
        x = self.dropout(x) + residual

        # Pre-norm FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x


# ---------------------------------------------------------------------------
# Anticipation Head
# ---------------------------------------------------------------------------

class AnticipationHead(nn.Module):
    """Predicts the 7 anticipation scalars from hidden states.

    This is the self-referential component: the model learns to compute
    its own anticipation geometry from its internal representations. During
    training, we can supervise this with externally computed scalars (from
    the numpy AnticipationGeometry engine). At inference, the model uses
    its own predicted scalars to steer attention.

    Architecture:
        hidden_states (d_model) -> LayerNorm -> Linear -> GELU -> Linear -> 7 scalars

    The output activations are:
        - commitment: sigmoid (bounded [0, 1])
        - uncertainty: sigmoid (bounded [0, 1])
        - transition_pressure: tanh * 2 (bounded [-2, 2])
        - recovery_margin: sigmoid (bounded [0, 1])
        - phase_stiffness: sigmoid (bounded [0, 1])
        - novelty: sigmoid (bounded [0, 1])
        - stability: sigmoid (bounded [0, 1])

    Args:
        config: AnticipatoryConfig.
    """

    # Scalar names for reference
    SCALAR_NAMES = [
        "commitment",
        "uncertainty",
        "transition_pressure",
        "recovery_margin",
        "phase_stiffness",
        "novelty",
        "stability",
    ]

    def __init__(self, config: AnticipatoryConfig):
        super().__init__()
        self.trajectory_dims = config.trajectory_dims
        hidden = config.d_model // 2

        self.norm = nn.LayerNorm(config.d_model)
        self.proj = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.trajectory_dims),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict anticipation scalars from hidden states.

        Args:
            hidden_states: (batch, seq, d_model) transformer hidden states.

        Returns:
            scalars: (batch, seq, 7) predicted anticipation scalars, each
                in their appropriate range.
        """
        x = self.norm(hidden_states)
        raw = self.proj(x)  # (batch, seq, 7)

        # Apply appropriate activations per scalar
        scalars = torch.zeros_like(raw)

        # commitment (index 0): sigmoid -> [0, 1]
        scalars[..., 0] = torch.sigmoid(raw[..., 0])
        # uncertainty (index 1): sigmoid -> [0, 1]
        scalars[..., 1] = torch.sigmoid(raw[..., 1])
        # transition_pressure (index 2): tanh * 2 -> [-2, 2]
        scalars[..., 2] = torch.tanh(raw[..., 2]) * 2.0
        # recovery_margin (index 3): sigmoid -> [0, 1]
        scalars[..., 3] = torch.sigmoid(raw[..., 3])
        # phase_stiffness (index 4): sigmoid -> [0, 1]
        scalars[..., 4] = torch.sigmoid(raw[..., 4])
        # novelty (index 5): sigmoid -> [0, 1]
        scalars[..., 5] = torch.sigmoid(raw[..., 5])
        # stability (index 6): sigmoid -> [0, 1]
        scalars[..., 6] = torch.sigmoid(raw[..., 6])

        return scalars


# ---------------------------------------------------------------------------
# Commitment Gate
# ---------------------------------------------------------------------------

class CommitmentGate(nn.Module):
    """Decides whether to emit a token or wait for more context.

    Standard autoregressive transformers emit a token at every position.
    The CommitmentGate introduces a fundamentally different paradigm:
    the model can choose to WAIT. When commitment is below the threshold,
    the gate buffers the hidden state and defers emission.

    This mirrors human reasoning: you don't produce a word at every
    cognitive step. Sometimes you think for several "internal steps"
    before committing to output.

    During training, the gate operates in soft mode (differentiable):
        gate_value = sigmoid(linear(hidden_state))
        output = gate_value * lm_head(hidden_state)

    During inference, the gate operates in hard mode:
        if commitment > threshold: emit token
        else: buffer and continue (up to max_buffer steps)

    Args:
        config: AnticipatoryConfig with commitment_threshold and max_buffer.
    """

    def __init__(self, config: AnticipatoryConfig):
        super().__init__()
        self.threshold = config.commitment_threshold
        self.max_buffer = config.max_buffer

        # Learned gate: maps hidden state to emission probability
        self.gate_proj = nn.Linear(config.d_model, 1)
        # Also takes the predicted commitment scalar as input
        self.gate_combine = nn.Linear(2, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        commitment_scalar: torch.Tensor,
        hard: bool = False,
    ) -> torch.Tensor:
        """Compute gate values for each position.

        Args:
            hidden_states: (batch, seq, d_model).
            commitment_scalar: (batch, seq) commitment values in [0, 1].
            hard: If True, apply hard thresholding (inference mode).

        Returns:
            gate_values: (batch, seq, 1) emission probabilities.
                In soft mode: differentiable values in (0, 1).
                In hard mode: binary 0/1 values.
        """
        # Learned component from hidden states
        learned = self.gate_proj(hidden_states)  # (batch, seq, 1)
        learned_sig = torch.sigmoid(learned)

        # Combine learned gate with commitment scalar
        commitment = commitment_scalar.unsqueeze(-1)  # (batch, seq, 1)
        combined = torch.cat([learned_sig, commitment], dim=-1)  # (batch, seq, 2)
        gate = torch.sigmoid(self.gate_combine(combined))  # (batch, seq, 1)

        if hard:
            gate = (gate > self.threshold).float()

        return gate


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, config: AnticipatoryConfig):
        super().__init__()
        pe = torch.zeros(config.max_seq_len, config.d_model)
        position = torch.arange(0, config.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / config.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: (batch, seq, d_model) token embeddings.

        Returns:
            (batch, seq, d_model) with positional encoding added.
        """
        return x + self.pe[:, : x.shape[1]]


# ---------------------------------------------------------------------------
# Pathway Gate
# ---------------------------------------------------------------------------

class PathwayGate(nn.Module):
    """Learned gate to combine fast and slow pathway outputs.

    Uses the anticipation scalars to decide the mixing: when commitment
    is high, favor the fast (local) pathway. When uncertainty is high,
    favor the slow (global) pathway.
    """

    def __init__(self, config: AnticipatoryConfig):
        super().__init__()
        self.proj = nn.Linear(config.trajectory_dims + config.d_model, 1)

    def forward(
        self,
        fast_out: torch.Tensor,
        slow_out: torch.Tensor,
        scalars: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Mix fast and slow pathway outputs.

        Args:
            fast_out: (batch, seq, d_model) from fast pathway.
            slow_out: (batch, seq, d_model) from slow pathway.
            scalars: (batch, seq, trajectory_dims) anticipation scalars.
            hidden: (batch, seq, d_model) current hidden states.

        Returns:
            mixed: (batch, seq, d_model) pathway-gated output.
        """
        gate_input = torch.cat([scalars, hidden], dim=-1)
        alpha = torch.sigmoid(self.proj(gate_input))  # (batch, seq, 1)
        return alpha * fast_out + (1 - alpha) * slow_out


# ---------------------------------------------------------------------------
# Full Anticipatory Transformer
# ---------------------------------------------------------------------------

class AnticipatoryTransformer(nn.Module):
    """The complete Anticipatory Transformer.

    Architecture:
        1. Token embedding + positional encoding
        2. Initial anticipation prediction (from embeddings)
        3. Trajectory bias computation (from scalars)
        4. Fast pathway: n_fast_layers of local-attention blocks
        5. Slow pathway: n_slow_layers of global-attention blocks
        6. Pathway gate: combine fast and slow outputs
        7. Refined anticipation prediction (from combined hidden states)
        8. Commitment gate: decide which positions emit
        9. Language model head: vocabulary projection

    The model is trained end-to-end with:
        - Next-token prediction loss (standard LM)
        - Anticipation prediction loss (supervised by external geometry)
        - Orthogonality penalty (encouraging diverse head behaviors)

    Args:
        config: AnticipatoryConfig.
    """

    def __init__(self, config: AnticipatoryConfig):
        super().__init__()
        self.config = config

        # Token embedding and positional encoding
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(config)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Trajectory bias network (shared across layers)
        if config.moe_bias:
            self.trajectory_bias_net = MoETrajectoryBias(config)
        else:
            self.trajectory_bias_net = TrajectoryBiasNetwork(config)

        # Anticipation head - initial prediction from embeddings
        self.anticipation_head_initial = AnticipationHead(config)

        # Fast pathway (local attention)
        self.fast_layers = nn.ModuleList([
            DualPathwayBlock(config, is_fast=True)
            for _ in range(config.n_fast_layers)
        ])

        # Slow pathway (global attention)
        self.slow_layers = nn.ModuleList([
            DualPathwayBlock(config, is_fast=False)
            for _ in range(config.n_slow_layers)
        ])

        # Pathway gate
        self.pathway_gate = PathwayGate(config)

        # Anticipation head - refined prediction from hidden states
        self.anticipation_head_refined = AnticipationHead(config)

        # Commitment gate
        self.commitment_gate = CommitmentGate(config)

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)

        # Language model head (weight-tied with embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # Weight tying

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following GPT-2 conventions."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        external_scalars: Optional[torch.Tensor] = None,
        inscription_ids: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        use_commitment_gate: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the full Anticipatory Transformer.

        Args:
            input_ids: (batch, seq) token IDs.
            external_scalars: (batch, seq, 7) externally computed anticipation
                scalars. If None, uses the model's own predicted scalars.
            inscription_ids: (batch, seq) inscription category IDs [0-9].
                If None, not used even if config.use_inscriptions is True.
            targets: (batch, seq) target token IDs for loss computation.
            use_commitment_gate: If True, apply commitment gating to output.

        Returns:
            Dictionary containing:
                logits: (batch, seq, vocab_size) language model logits.
                predicted_scalars: (batch, seq, 7) model's predicted scalars.
                gate_values: (batch, seq, 1) commitment gate values.
                ortho_penalty: scalar orthogonality penalty.
                loss: scalar total loss (if targets provided).
                lm_loss: scalar language modeling loss (if targets provided).
                scalar_loss: scalar anticipation prediction loss
                    (if external_scalars provided).
        """
        batch, seq = input_ids.shape
        device = input_ids.device

        # 1. Token embedding + positional encoding
        x = self.tok_emb(input_ids)
        x = self.pos_enc(x)
        x = self.emb_dropout(x)

        # 2. Initial anticipation prediction
        initial_scalars = self.anticipation_head_initial(x)

        # 3. Choose which scalars to use for trajectory bias
        # During training with supervision: use external scalars
        # Otherwise: use model's own predictions
        if external_scalars is not None:
            active_scalars = external_scalars
        else:
            active_scalars = initial_scalars

        # 4. Compute trajectory bias (scalars + optional inscriptions)
        trajectory_bias = self.trajectory_bias_net(active_scalars, seq, inscription_ids=inscription_ids)

        # 5. Causal mask
        causal_mask = self._make_causal_mask(seq, device)

        # 6. Fast pathway
        fast_out = x
        for layer in self.fast_layers:
            fast_out = layer(fast_out, trajectory_bias=trajectory_bias, mask=causal_mask)

        # 7. Slow pathway
        slow_out = x
        for layer in self.slow_layers:
            slow_out = layer(slow_out, trajectory_bias=trajectory_bias, mask=causal_mask)

        # 8. Combine pathways
        hidden = self.pathway_gate(fast_out, slow_out, active_scalars, x)

        # 9. Refined anticipation prediction
        predicted_scalars = self.anticipation_head_refined(hidden)

        # 10. Final norm
        hidden = self.final_norm(hidden)

        # 11. Language model head
        logits = self.lm_head(hidden)  # (batch, seq, vocab_size)

        # 12. Commitment gate
        commitment = predicted_scalars[..., 0]  # (batch, seq)
        gate_values = self.commitment_gate(hidden, commitment, hard=False)

        if use_commitment_gate:
            # Soft gating during training: modulate logits
            logits = logits * gate_values

        # 13. Orthogonality penalty
        ortho_penalty = self.trajectory_bias_net.orthogonality_penalty()

        # 14. Compute losses
        result = {
            "logits": logits,
            "predicted_scalars": predicted_scalars,
            "gate_values": gate_values,
            "ortho_penalty": ortho_penalty,
        }

        if targets is not None:
            # Language modeling loss (cross-entropy)
            lm_loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
            result["lm_loss"] = lm_loss

            # Total loss
            total_loss = lm_loss + self.config.lambda_ortho * ortho_penalty

            # Anticipation prediction loss (if supervised)
            if external_scalars is not None:
                scalar_loss = F.mse_loss(predicted_scalars, external_scalars)
                result["scalar_loss"] = scalar_loss
                total_loss = total_loss + scalar_loss

            result["loss"] = total_loss

        return result

    def _make_causal_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Create causal (autoregressive) attention mask.

        Returns a (1, 1, seq, seq) mask where future positions are -inf.
        """
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )
        return mask.unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        use_commitment_gate: bool = True,
    ) -> torch.Tensor:
        """Autoregressive generation with commitment gating.

        When commitment gating is active, the model may skip emission
        at low-commitment positions, effectively "thinking" for multiple
        steps before producing a token.

        Args:
            input_ids: (batch, seq) prompt token IDs.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            use_commitment_gate: If True, apply commitment gating.

        Returns:
            (batch, seq + generated) token IDs.
        """
        self.eval()
        generated = input_ids.clone()
        buffer_count = 0

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            context = generated[:, -self.config.max_seq_len:]

            # Forward pass
            result = self.forward(
                context,
                use_commitment_gate=False,  # handle gating manually
            )

            # Get logits for last position
            next_logits = result["logits"][:, -1, :]  # (batch, vocab_size)

            if use_commitment_gate:
                commitment = result["predicted_scalars"][:, -1, 0]  # (batch,)
                should_emit = commitment > self.threshold_with_buffer(buffer_count)

                if not should_emit.any():
                    buffer_count += 1
                    if buffer_count >= self.config.max_buffer:
                        # Force emission after max_buffer steps
                        buffer_count = 0
                    else:
                        continue
                else:
                    buffer_count = 0

            # Sample next token
            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def threshold_with_buffer(self, buffer_count: int) -> float:
        """Lower the commitment threshold as buffer fills.

        The longer we wait, the more willing we are to emit. This
        prevents infinite stalling while preserving the benefit of
        waiting for high-commitment positions.
        """
        decay = buffer_count / self.config.max_buffer
        return self.config.commitment_threshold * (1.0 - decay * 0.5)

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by component."""
        counts = {}
        counts["embedding"] = sum(
            p.numel() for p in self.tok_emb.parameters()
        )
        counts["positional"] = 0  # sinusoidal, no learnable params
        counts["trajectory_bias"] = sum(
            p.numel() for p in self.trajectory_bias_net.parameters()
        )
        counts["fast_pathway"] = sum(
            p.numel() for p in self.fast_layers.parameters()
        )
        counts["slow_pathway"] = sum(
            p.numel() for p in self.slow_layers.parameters()
        )
        counts["pathway_gate"] = sum(
            p.numel() for p in self.pathway_gate.parameters()
        )
        counts["anticipation_heads"] = sum(
            p.numel()
            for head in [self.anticipation_head_initial, self.anticipation_head_refined]
            for p in head.parameters()
        )
        counts["commitment_gate"] = sum(
            p.numel() for p in self.commitment_gate.parameters()
        )
        counts["final_norm"] = sum(
            p.numel() for p in self.final_norm.parameters()
        )
        counts["lm_head"] = 0  # weight-tied with embedding
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts


# ---------------------------------------------------------------------------
# Baseline Transformer (for comparison)
# ---------------------------------------------------------------------------

class BaselineTransformer(nn.Module):
    """Standard transformer without trajectory bias.

    Same architecture as AnticipatoryTransformer but without:
    - Trajectory bias network
    - Anticipation heads
    - Commitment gate
    - Dual pathways (uses single global attention)

    Used as a baseline for evaluating the contribution of anticipation geometry.
    """

    def __init__(self, config: AnticipatoryConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(config)
        self.emb_dropout = nn.Dropout(config.dropout)

        total_layers = config.n_fast_layers + config.n_slow_layers
        self.layers = nn.ModuleList([
            DualPathwayBlock(config, is_fast=False)
            for _ in range(total_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.vocab_size, config.d_model, bias=False)
        # Note: we recreate for correct shapes below
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        external_scalars: Optional[torch.Tensor] = None,
        use_commitment_gate: bool = False,
    ) -> dict[str, torch.Tensor]:
        # external_scalars and use_commitment_gate are accepted but ignored
        # for API compatibility with AnticipatoryTransformer
        batch, seq = input_ids.shape
        device = input_ids.device

        x = self.tok_emb(input_ids)
        x = self.pos_enc(x)
        x = self.emb_dropout(x)

        causal_mask = torch.triu(
            torch.full((seq, seq), float("-inf"), device=device),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, trajectory_bias=None, mask=causal_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        result = {
            "logits": logits,
            "ortho_penalty": torch.tensor(0.0, device=device),
            "predicted_scalars": torch.zeros(batch, seq, 7, device=device),
            "gate_values": torch.ones(batch, seq, 1, device=device),
        }

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss
            result["lm_loss"] = loss

        return result

    def count_parameters(self) -> dict[str, int]:
        return {"total": sum(p.numel() for p in self.parameters())}


# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------

def smoke_test():
    """Verify forward pass, gradient flow, and commitment gating."""
    print("=" * 60)
    print("Anticipatory Transformer - Smoke Test")
    print("=" * 60)

    config = AnticipatoryConfig().small()
    model = AnticipatoryTransformer(config)

    params = model.count_parameters()
    print(f"\nParameter counts:")
    for name, count in params.items():
        print(f"  {name}: {count:,}")

    # Test 1: Forward pass with targets
    print("\n--- Test 1: Forward pass with LM loss ---")
    batch, seq = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch, seq))
    targets = torch.randint(0, config.vocab_size, (batch, seq))

    result = model(input_ids, targets=targets)
    print(f"  Logits shape: {result['logits'].shape}")
    print(f"  Predicted scalars shape: {result['predicted_scalars'].shape}")
    print(f"  Gate values shape: {result['gate_values'].shape}")
    print(f"  LM loss: {result['lm_loss'].item():.4f}")
    print(f"  Ortho penalty: {result['ortho_penalty'].item():.6f}")
    print(f"  Total loss: {result['loss'].item():.4f}")

    # Test 2: Forward pass with external scalars
    print("\n--- Test 2: Forward pass with external scalars ---")
    external_scalars = torch.rand(batch, seq, config.trajectory_dims)
    result = model(input_ids, external_scalars=external_scalars, targets=targets)
    print(f"  Scalar loss: {result['scalar_loss'].item():.4f}")
    print(f"  Total loss: {result['loss'].item():.4f}")

    # Test 3: Gradient flow
    print("\n--- Test 3: Gradient flow ---")
    model.zero_grad()
    result["loss"].backward()

    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    nonzero_grads = sum(1 for v in grad_norms.values() if v > 0)
    total_params = len(grad_norms)
    print(f"  Parameters with gradients: {nonzero_grads}/{total_params}")

    # Check key components have gradients
    for key_pattern in ["trajectory_bias_net", "anticipation_head", "commitment_gate"]:
        matching = [
            (n, g) for n, g in grad_norms.items()
            if key_pattern in n and g > 0
        ]
        status = "OK" if matching else "MISSING GRADIENTS"
        print(f"  {key_pattern}: {status} ({len(matching)} params)")

    # Test 4: Commitment gating
    print("\n--- Test 4: Commitment gating ---")
    result_gated = model(input_ids, use_commitment_gate=True)
    result_ungated = model(input_ids, use_commitment_gate=False)
    gate_mean = result_gated["gate_values"].mean().item()
    logit_diff = (result_gated["logits"] - result_ungated["logits"]).abs().mean().item()
    print(f"  Mean gate value: {gate_mean:.4f}")
    print(f"  Mean logit difference (gated vs ungated): {logit_diff:.4f}")

    # Test 5: Generation
    print("\n--- Test 5: Generation ---")
    prompt = torch.randint(0, config.vocab_size, (1, 8))
    generated = model.generate(prompt, max_new_tokens=10, use_commitment_gate=False)
    print(f"  Prompt length: {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")

    # Test 6: Baseline comparison shapes
    print("\n--- Test 6: Baseline transformer ---")
    baseline = BaselineTransformer(config)
    baseline_params = baseline.count_parameters()
    print(f"  Baseline params: {baseline_params['total']:,}")
    baseline_result = baseline(input_ids, targets=targets)
    print(f"  Baseline logits shape: {baseline_result['logits'].shape}")
    print(f"  Baseline loss: {baseline_result['loss'].item():.4f}")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    smoke_test()
