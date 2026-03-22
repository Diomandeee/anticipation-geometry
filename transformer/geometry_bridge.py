"""
Geometry Bridge
================

Connects the numpy-based AnticipationGeometry engine to the PyTorch
Anticipatory Transformer.

The bridge does two things:

1. FORWARD: Given a sequence of embedding vectors (from the transformer's
   embedding layer), compute the 7 anticipation scalars using numpy and
   convert them to a PyTorch tensor suitable for the trajectory bias network.

2. INVERSE: Given the model's predicted scalars (PyTorch tensors), convert
   them back to an AnticipationPacket for analysis and visualization.

The 7 scalars are:
    [0] commitment - how locked-in the trajectory is (original)
    [1] uncertainty - how many futures remain (original)
    [2] transition_pressure - rate of convergence (original)
    [3] recovery_margin - ease of backtracking (original)
    [4] phase_stiffness - rigidity of current phase (NEW)
    [5] novelty - how new the current state is (NEW)
    [6] stability - smoothness of the trajectory (NEW)

The first 4 come directly from AnticipationGeometry.compute().
The last 3 are computed here as extensions:
    - phase_stiffness: autocorrelation of the velocity vector
    - novelty: inverse of distance to nearest historical state
    - stability: inverse of jerk (rate of change of acceleration)
"""

from __future__ import annotations

import sys
import os
from typing import Optional

import numpy as np
import torch

# Add the anticipation_geometry package to path
_AG_PATH = os.path.join(os.path.dirname(__file__), "..", "python")
if _AG_PATH not in sys.path:
    sys.path.insert(0, _AG_PATH)

from anticipation_geometry.generalized_anticipation import (
    AnticipationGeometry,
    AnticipationPacket,
)


class GeometryBridge:
    """Bridge between numpy AnticipationGeometry and PyTorch model.

    Computes all 7 anticipation scalars from embedding trajectories
    and converts between numpy and PyTorch representations.

    Args:
        k_neighbors: K parameter for uncertainty computation.
        n_angle_bins: Angular resolution for directional entropy.
        smoothing_window: Smoothing for finite differences.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        n_angle_bins: int = 8,
        smoothing_window: int = 3,
    ):
        self.geometry = AnticipationGeometry(
            k_neighbors=k_neighbors,
            n_angle_bins=n_angle_bins,
            smoothing_window=smoothing_window,
        )

    def embeddings_to_scalars(
        self,
        embeddings: torch.Tensor,
        detach: bool = True,
    ) -> torch.Tensor:
        """Compute 7 anticipation scalars from embedding trajectories.

        Args:
            embeddings: (batch, seq, d_model) embedding vectors.
                Each batch element is treated as an independent trajectory.
            detach: If True, detach embeddings from computation graph
                before converting to numpy (required for external supervision).

        Returns:
            scalars: (batch, seq, 7) anticipation scalars as a PyTorch tensor.
        """
        if detach:
            emb_np = embeddings.detach().cpu().numpy()
        else:
            emb_np = embeddings.cpu().numpy()

        batch, seq, dim = emb_np.shape
        all_scalars = np.zeros((batch, seq, 7), dtype=np.float32)

        for b in range(batch):
            trajectory = [emb_np[b, t] for t in range(seq)]
            all_scalars[b] = self._compute_7_scalars(trajectory)

        return torch.tensor(all_scalars, device=embeddings.device)

    def _compute_7_scalars(self, trajectory: list[np.ndarray]) -> np.ndarray:
        """Compute all 7 scalars for a single trajectory.

        Args:
            trajectory: List of state vectors (numpy arrays).

        Returns:
            (seq, 7) array of anticipation scalars.
        """
        n = len(trajectory)
        if n < 2:
            return np.full((n, 7), 0.5, dtype=np.float32)

        # Original 4 scalars from AnticipationGeometry
        packet = self.geometry.compute(trajectory)

        # Extended 3 scalars
        states = np.stack(trajectory)
        phase_stiffness = self._compute_phase_stiffness(states, n)
        novelty = self._compute_novelty(states, n)
        stability = self._compute_stability(states, n)

        # Stack all 7
        scalars = np.stack([
            packet.commitment,
            packet.uncertainty,
            packet.transition_pressure,
            packet.recovery_margin,
            phase_stiffness,
            novelty,
            stability,
        ], axis=-1).astype(np.float32)

        # Normalize transition_pressure from [-2, 2] to [0, 1] for the bias network
        # (the AnticipationHead handles the range internally, but for external
        # supervision we want a uniform scale)
        scalars[:, 2] = (scalars[:, 2] + 2.0) / 4.0

        return scalars

    def _compute_phase_stiffness(
        self, states: np.ndarray, n: int
    ) -> np.ndarray:
        """Phase stiffness: autocorrelation of velocity vectors.

        Measures how rigid the current motion phase is. High stiffness
        means the trajectory is moving in a consistent direction (ballistic).
        Low stiffness means the direction is changing frequently (gestural).

        Computed as the cosine similarity between consecutive velocity vectors.

        Args:
            states: (n, d) state vectors.
            n: Number of steps.

        Returns:
            (n,) stiffness values in [0, 1].
        """
        stiffness = np.full(n, 0.5, dtype=np.float32)

        if n < 3:
            return stiffness

        # Velocity vectors
        velocities = np.diff(states, axis=0)  # (n-1, d)
        vel_norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        vel_norms = np.maximum(vel_norms, 1e-10)
        unit_vel = velocities / vel_norms

        # Cosine similarity between consecutive velocities
        for t in range(2, n):
            cos_sim = np.dot(unit_vel[t - 1], unit_vel[t - 2])
            # Map from [-1, 1] to [0, 1]
            stiffness[t] = (cos_sim + 1.0) / 2.0

        # t=0 and t=1 use neutral value
        stiffness[0] = 0.5
        stiffness[1] = 0.5

        return np.clip(stiffness, 0.0, 1.0)

    def _compute_novelty(self, states: np.ndarray, n: int) -> np.ndarray:
        """Novelty: how new/unseen the current state is.

        Computed as the normalized distance to the nearest historical state.
        High novelty means the trajectory is in unexplored territory.
        Low novelty means it has been near this state before.

        Args:
            states: (n, d) state vectors.
            n: Number of steps.

        Returns:
            (n,) novelty values in [0, 1].
        """
        novelty = np.full(n, 0.5, dtype=np.float32)

        if n < 2:
            return novelty

        # Compute max range for normalization
        centroid = np.mean(states, axis=0)
        max_range = 2.0 * np.max(np.linalg.norm(states - centroid, axis=1))

        if max_range < 1e-10:
            return novelty

        novelty[0] = 1.0  # First state is always maximally novel

        for t in range(1, n):
            history = states[:t]
            dists = np.linalg.norm(history - states[t], axis=1)
            min_dist = np.min(dists)
            novelty[t] = min_dist / max_range

        return np.clip(novelty, 0.0, 1.0)

    def _compute_stability(self, states: np.ndarray, n: int) -> np.ndarray:
        """Stability: smoothness of the trajectory (inverse of jerk).

        Jerk is the rate of change of acceleration (third derivative).
        High stability means the trajectory is changing smoothly.
        Low stability means abrupt changes in acceleration.

        Computed as 1 - normalized_jerk.

        Args:
            states: (n, d) state vectors.
            n: Number of steps.

        Returns:
            (n,) stability values in [0, 1].
        """
        stability = np.full(n, 0.5, dtype=np.float32)

        if n < 4:
            return stability

        # Velocity, acceleration, jerk
        velocity = np.diff(states, axis=0)  # (n-1, d)
        acceleration = np.diff(velocity, axis=0)  # (n-2, d)
        jerk = np.diff(acceleration, axis=0)  # (n-3, d)
        jerk_norms = np.linalg.norm(jerk, axis=1)

        max_jerk = np.max(jerk_norms) if len(jerk_norms) > 0 else 0.0

        if max_jerk < 1e-10:
            stability[:] = 1.0
            return stability

        # Map to stability: low jerk = high stability
        for t in range(3, n):
            normalized_jerk = jerk_norms[t - 3] / max_jerk
            stability[t] = 1.0 - normalized_jerk

        # Edge cases
        stability[0] = 0.5
        stability[1] = 0.5
        stability[2] = 0.5

        return np.clip(stability, 0.0, 1.0)

    def scalars_to_packet(
        self,
        scalars: torch.Tensor,
    ) -> list[AnticipationPacket]:
        """Convert model-predicted scalars back to AnticipationPackets.

        Useful for analysis and visualization of what the model has learned
        about its own trajectory geometry.

        Args:
            scalars: (batch, seq, 7) predicted scalars.

        Returns:
            List of AnticipationPackets, one per batch element.
        """
        s = scalars.detach().cpu().numpy()
        batch, seq, _ = s.shape

        packets = []
        for b in range(batch):
            # De-normalize transition_pressure back to [-2, 2]
            tp = s[b, :, 2] * 4.0 - 2.0

            packet = AnticipationPacket(
                commitment=s[b, :, 0],
                uncertainty=s[b, :, 1],
                transition_pressure=tp,
                recovery_margin=s[b, :, 3],
                trajectory_length=seq,
                dimensionality=7,
            )
            packets.append(packet)

        return packets

    def batch_compute(
        self,
        input_ids: torch.Tensor,
        embedding_layer: torch.nn.Embedding,
    ) -> torch.Tensor:
        """Convenience: compute scalars directly from token IDs.

        Embeds the tokens and computes anticipation scalars from the
        embedding trajectory. This is the standard way to get external
        supervision scalars during training.

        Args:
            input_ids: (batch, seq) token IDs.
            embedding_layer: The model's token embedding layer.

        Returns:
            scalars: (batch, seq, 7) anticipation scalars.
        """
        with torch.no_grad():
            embeddings = embedding_layer(input_ids)
        return self.embeddings_to_scalars(embeddings, detach=True)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _test():
    """Verify the geometry bridge works correctly."""
    print("Geometry Bridge - Self Test")
    print("=" * 60)

    bridge = GeometryBridge(k_neighbors=3)

    # Test 1: Compute scalars from random embeddings
    print("\n--- Test 1: Random embeddings ---")
    batch, seq, dim = 2, 20, 64
    embeddings = torch.randn(batch, seq, dim)
    scalars = bridge.embeddings_to_scalars(embeddings)
    print(f"  Input shape: {embeddings.shape}")
    print(f"  Output shape: {scalars.shape}")
    assert scalars.shape == (batch, seq, 7), f"Wrong shape: {scalars.shape}"

    # Check ranges
    for i, name in enumerate([
        "commitment", "uncertainty", "transition_pressure_normalized",
        "recovery_margin", "phase_stiffness", "novelty", "stability"
    ]):
        vals = scalars[:, :, i]
        print(f"  {name}: min={vals.min():.3f}, max={vals.max():.3f}, mean={vals.mean():.3f}")
        assert vals.min() >= -0.01, f"{name} below 0: {vals.min()}"
        assert vals.max() <= 1.01, f"{name} above 1: {vals.max()}"

    # Test 2: Convert back to packets
    print("\n--- Test 2: Scalars to packets ---")
    packets = bridge.scalars_to_packet(scalars)
    assert len(packets) == batch
    for i, p in enumerate(packets):
        print(f"  Batch {i}: regime at t=5 -> {p.regime_at(5)}")
        summary = p.summary()
        print(f"  Batch {i}: commitment_mean={summary['commitment_mean']:.3f}")

    # Test 3: Structured trajectory (explore then commit)
    print("\n--- Test 3: Structured trajectory ---")
    np.random.seed(42)
    explore = [torch.randn(dim) * 0.5 for _ in range(10)]
    target = torch.ones(dim) * 2.0
    commit = []
    pos = explore[-1].clone()
    for i in range(10):
        alpha = (i + 1) / 10.0
        pos = pos * (1 - alpha * 0.3) + target * (alpha * 0.3)
        pos += torch.randn(dim) * 0.03 * (1 - alpha)
        commit.append(pos.clone())

    trajectory = torch.stack(explore + commit).unsqueeze(0)  # (1, 20, dim)
    scalars = bridge.embeddings_to_scalars(trajectory)

    # Commitment should increase from explore to commit phase
    explore_commitment = scalars[0, :10, 0].mean().item()
    commit_commitment = scalars[0, 10:, 0].mean().item()
    print(f"  Explore phase commitment: {explore_commitment:.3f}")
    print(f"  Commit phase commitment: {commit_commitment:.3f}")

    # Stability should be higher in commit phase
    explore_stability = scalars[0, :10, 6].mean().item()
    commit_stability = scalars[0, 10:, 6].mean().item()
    print(f"  Explore phase stability: {explore_stability:.3f}")
    print(f"  Commit phase stability: {commit_stability:.3f}")

    # Test 4: batch_compute from token IDs
    print("\n--- Test 4: batch_compute ---")
    vocab_size = 100
    emb = torch.nn.Embedding(vocab_size, dim)
    ids = torch.randint(0, vocab_size, (2, 15))
    scalars = bridge.batch_compute(ids, emb)
    print(f"  Input IDs shape: {ids.shape}")
    print(f"  Output scalars shape: {scalars.shape}")
    assert scalars.shape == (2, 15, 7)

    print("\n" + "=" * 60)
    print("ALL GEOMETRY BRIDGE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _test()
