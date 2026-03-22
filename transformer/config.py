"""
Anticipatory Transformer Configuration
========================================

All hyperparameters for the Anticipatory Transformer architecture.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnticipatoryConfig:
    """Configuration for the Anticipatory Transformer.

    Architecture parameters:
        d_model: Hidden dimension throughout the model.
        n_heads: Number of attention heads (d_model must be divisible by n_heads).
        n_fast_layers: Number of fast-pathway (local window) transformer blocks.
        n_slow_layers: Number of slow-pathway (global context) transformer blocks.
        fast_window: Window size for local attention in fast pathway.
        d_ff: Feed-forward intermediate dimension.
        vocab_size: Vocabulary size for token embedding and output projection.
        max_seq_len: Maximum sequence length for positional encoding.
        dropout: Dropout rate applied throughout the model.

    Trajectory parameters:
        trajectory_dims: Number of anticipation scalars (7 for the full set:
            commitment, uncertainty, transition_pressure, recovery_margin,
            phase_stiffness, novelty, stability).

    Commitment gating:
        commitment_threshold: Minimum commitment value to emit a token.
            Below this threshold, the CommitmentGate buffers the hidden state
            and waits for more context.
        max_buffer: Maximum number of positions the gate can buffer before
            forcing emission. Prevents infinite stalling.

    Regularization:
        lambda_ortho: Weight for the orthogonality penalty on trajectory
            bias vectors across heads. Encourages each head to use
            different aspects of the anticipation geometry.
    """

    # Architecture
    d_model: int = 512
    n_heads: int = 8
    n_fast_layers: int = 3
    n_slow_layers: int = 3
    fast_window: int = 128
    d_ff: int = 2048
    vocab_size: int = 32000
    max_seq_len: int = 2048
    dropout: float = 0.1

    # Trajectory
    trajectory_dims: int = 7

    # Commitment gating
    commitment_threshold: float = 0.8
    max_buffer: int = 5

    # Regularization
    lambda_ortho: float = 0.1

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )

    @property
    def d_head(self) -> int:
        """Per-head dimension."""
        return self.d_model // self.n_heads

    def small(self) -> "AnticipatoryConfig":
        """Return a small config suitable for smoke tests and CPU training.

        ~200K parameters, trains in seconds.
        """
        return AnticipatoryConfig(
            d_model=128,
            n_heads=4,
            n_fast_layers=2,
            n_slow_layers=2,
            fast_window=32,
            d_ff=256,
            vocab_size=1000,
            max_seq_len=256,
            dropout=0.1,
            trajectory_dims=self.trajectory_dims,
            commitment_threshold=self.commitment_threshold,
            max_buffer=self.max_buffer,
            lambda_ortho=self.lambda_ortho,
        )

    def param_estimate(self) -> int:
        """Rough parameter count estimate (millions)."""
        # Embedding: vocab_size * d_model
        embed = self.vocab_size * self.d_model
        # Per layer: 4 * d_model^2 (QKV + out) + 2 * d_model * d_ff (FFN) + norms
        per_layer = 4 * self.d_model ** 2 + 2 * self.d_model * self.d_ff + 4 * self.d_model
        total_layers = self.n_fast_layers + self.n_slow_layers
        # Trajectory bias network: trajectory_dims -> n_heads * max_seq_len
        traj_bias = self.trajectory_dims * 64 + 64 * self.n_heads
        # Anticipation head: d_model -> trajectory_dims
        antic_head = self.d_model * self.trajectory_dims
        # Output projection: same as embedding
        total = embed + total_layers * per_layer + traj_bias + antic_head + embed
        return total
