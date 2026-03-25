"""
Generalized Anticipation Geometry
==================================

Domain-general anticipation scalars that work on ANY trajectory representable
as a sequence of vectors in a metric space.

The original cc-window-aligner (core/motion/) computes tension, phase, momentum,
and stability from skeletal motion data. This module abstracts those concepts
into four domain-agnostic scalars that capture the geometry of commitment,
uncertainty, transition pressure, and recovery margin for arbitrary state
trajectories.

Mathematical Foundation
-----------------------

Given a trajectory T = [s_0, s_1, ..., s_N] where each s_i is in R^d:

1. COMMITMENT c(t):
   Measures how "locked in" the trajectory is at step t.

   c(t) = 1 - ||s_t - s_{t-1}|| / max_delta

   where max_delta = max_{i} ||s_i - s_{i-1}|| over the trajectory.

   High commitment (near 1) means the trajectory is changing slowly,
   staying committed to its current direction. Low commitment (near 0)
   means the trajectory just made its largest-ever jump.

   This generalizes the motion-domain concept of "being in a sustained
   phase" vs "transitioning between phases."

2. UNCERTAINTY u(t):
   Measures how many plausible futures exist at step t, computed as
   the entropy of directions to K nearest neighbors in state space.

   u(t) = H(angles to KNN of s_t among {s_0, ..., s_{t-1}})

   where H is Shannon entropy over binned angular directions.

   High uncertainty means many different historical states are equidistant,
   suggesting the trajectory could plausibly branch in many directions.
   Low uncertainty means the trajectory's position is in a well-determined
   region of state space.

   This generalizes the motion-domain concept of "gestural vs ballistic"
   regimes (gestural = high uncertainty, ballistic = low).

3. TRANSITION PRESSURE p(t):
   Measures the rate at which commitment increases while uncertainty decreases.

   p(t) = dc/dt - du/dt

   where the derivatives are finite differences.

   High transition pressure means the system is simultaneously locking in
   (rising commitment) and ruling out alternatives (falling uncertainty).
   This signals an imminent regime change or decision point.

   In motion, this corresponds to the "gather" phase before a jump or
   the "wind-up" before a throw. In conversation, this is the moment
   when the dialogue narrows from exploration to a specific conclusion.

4. RECOVERY MARGIN r(t):
   Measures how easy it is to backtrack from the current position.

   r(t) = min_{j < t, s_j is branching point} ||s_t - s_j|| / max_range

   where branching points are trajectory positions with above-median
   uncertainty, and max_range = max pairwise distance in the trajectory.

   High recovery margin means the trajectory is near a branching point
   and could still change course. Low recovery margin means the trajectory
   is deep into a single branch with no easy return.

   In motion, this is the difference between "mid-stride" (low recovery,
   committed to the step) and "stance phase" (high recovery, can pivot).
   In reasoning, it's the difference between "deep in a proof" (low) and
   "just stated a premise" (high).

Relationship to Princeton (arXiv:2603.14147)
---------------------------------------------

Princeton's KG-path reward function evaluates the VALIDITY of a completed
reasoning path. Anticipation geometry evaluates the DYNAMICS of an ongoing
trajectory. They are complementary:

- KG-path rewards answer: "Was this path correct?"
- Anticipation geometry answers: "Where is this path going?"

Combined, they enable:
- Early detection of paths heading toward invalid territory
  (rising commitment + falling recovery toward nodes not in KG)
- Identification of high-value exploration points
  (high uncertainty + high recovery = many valid paths available)
- Optimal steering of reasoning chains
  (maximize transition pressure at decision points)

Usage:
    from generalized_anticipation import AnticipationGeometry, AnticipationPacket

    geom = AnticipationGeometry(k_neighbors=5)

    trajectory = [np.array([...]), np.array([...]), ...]  # any vectors
    packet = geom.compute(trajectory)

    print(packet.commitment)           # [0, 1] per timestep
    print(packet.uncertainty)          # [0, 1] per timestep
    print(packet.transition_pressure)  # [-2, 2] per timestep
    print(packet.recovery_margin)      # [0, 1] per timestep
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AnticipationPacket:
    """Complete anticipation state for a trajectory.

    All scalar arrays are aligned with the input trajectory: element i
    corresponds to trajectory step i. Edge effects at t=0 and t=1 are
    handled by returning neutral values (0.5 for bounded scalars, 0.0
    for transition pressure).

    Attributes:
        commitment: Per-step commitment in [0, 1].
        uncertainty: Per-step uncertainty in [0, 1].
        transition_pressure: Per-step transition pressure, unbounded but
            typically in [-2, 2].
        recovery_margin: Per-step recovery margin in [0, 1].
        trajectory_length: Number of steps in the trajectory.
        dimensionality: Dimensionality of the state vectors.
    """
    commitment: np.ndarray = field(default_factory=lambda: np.array([]))
    uncertainty: np.ndarray = field(default_factory=lambda: np.array([]))
    transition_pressure: np.ndarray = field(default_factory=lambda: np.array([]))
    recovery_margin: np.ndarray = field(default_factory=lambda: np.array([]))
    trajectory_length: int = 0
    dimensionality: int = 0

    def summary(self) -> dict:
        """Return aggregate statistics for the trajectory."""
        if self.trajectory_length == 0:
            return {"empty": True}

        return {
            "trajectory_length": self.trajectory_length,
            "dimensionality": self.dimensionality,
            "commitment_mean": float(np.mean(self.commitment)),
            "commitment_final": float(self.commitment[-1]),
            "uncertainty_mean": float(np.mean(self.uncertainty)),
            "uncertainty_final": float(self.uncertainty[-1]),
            "transition_pressure_max": float(np.max(self.transition_pressure)),
            "transition_pressure_peak_idx": int(np.argmax(self.transition_pressure)),
            "recovery_margin_min": float(np.min(self.recovery_margin)),
            "recovery_margin_final": float(self.recovery_margin[-1]),
        }

    def regime_at(self, t: int) -> str:
        """Classify the anticipation regime at step t.

        Returns one of:
          - "exploring": high uncertainty, high recovery (many options, easy to pivot)
          - "committing": rising commitment, falling uncertainty (narrowing down)
          - "locked": high commitment, low uncertainty, low recovery (deep in branch)
          - "transitioning": high transition pressure (regime change imminent)
          - "stalled": low commitment, low uncertainty (stuck, neither exploring nor committing)
        """
        if t < 0 or t >= self.trajectory_length:
            return "out_of_bounds"

        c = self.commitment[t]
        u = self.uncertainty[t]
        p = self.transition_pressure[t]
        r = self.recovery_margin[t]

        if p > 0.5:
            return "transitioning"
        if u > 0.6 and r > 0.5:
            return "exploring"
        if c > 0.7 and u < 0.3 and r < 0.4:
            return "locked"
        if c > 0.5 and u < 0.5:
            return "committing"
        return "stalled"


# ---------------------------------------------------------------------------
# Anticipation Geometry Engine
# ---------------------------------------------------------------------------

class AnticipationGeometry:
    """Domain-general anticipation geometry computer.

    Computes 4 scalar fields over any trajectory of state vectors.
    The only requirement: states must be representable as vectors in a
    metric space (i.e., as numpy arrays of the same dimensionality).

    Works on:
      - Motion vectors (3D skeleton positions over time)
      - Conversation embeddings (384/768-dim sentence embeddings)
      - KG paths (entity embeddings along a reasoning chain)
      - Task planning traces (state vectors from a planner)
      - Any other sequential vector data

    Args:
        k_neighbors: Number of nearest neighbors for uncertainty estimation.
            Higher K gives smoother uncertainty but less sensitivity to
            local structure. Default 5 is good for trajectories of 20-200 steps.
        n_angle_bins: Number of bins for angular entropy computation.
            Must be >= 2. Default 8 gives pi/4 angular resolution.
        smoothing_window: Window size for finite difference smoothing
            when computing transition pressure. Default 3.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        n_angle_bins: int = 8,
        smoothing_window: int = 3,
    ):
        self.k = k_neighbors
        self.n_bins = max(2, n_angle_bins)
        self.smooth_w = max(1, smoothing_window)

    def compute(self, trajectory: list[np.ndarray]) -> AnticipationPacket:
        """Compute anticipation scalars for a full trajectory.

        Args:
            trajectory: Ordered list of state vectors, each as np.ndarray
                of the same shape. Minimum 3 steps for meaningful output.

        Returns:
            AnticipationPacket with per-step scalar arrays.

        Raises:
            ValueError: If trajectory has fewer than 2 steps or inconsistent
                dimensionalities.
        """
        n = len(trajectory)
        if n < 2:
            return AnticipationPacket()

        # Validate dimensionality
        dim = trajectory[0].shape[0]
        states = np.stack(trajectory)  # (N, D)

        packet = AnticipationPacket(
            trajectory_length=n,
            dimensionality=dim,
        )

        # Compute pairwise deltas
        deltas = np.diff(states, axis=0)  # (N-1, D)
        delta_norms = np.linalg.norm(deltas, axis=1)  # (N-1,)

        # --- Signal 1: Commitment ---
        packet.commitment = self._compute_commitment(delta_norms, n)

        # --- Signal 2: Uncertainty ---
        packet.uncertainty = self._compute_uncertainty(states, n)

        # --- Signal 3: Transition Pressure ---
        packet.transition_pressure = self._compute_transition_pressure(
            packet.commitment, packet.uncertainty, n
        )

        # --- Signal 4: Recovery Margin ---
        packet.recovery_margin = self._compute_recovery_margin(
            states, packet.uncertainty, n
        )

        return packet

    # ----- Signal Implementations -----

    def _compute_commitment(self, delta_norms: np.ndarray, n: int) -> np.ndarray:
        """Commitment: 1 - (current step size / max step size).

        High when trajectory is changing slowly (locked in).
        Low when trajectory just made a large jump (exploring).

        Edge case: if max_delta is 0 (all states identical), commitment
        is 1.0 everywhere (perfectly committed to staying put).
        """
        commitment = np.full(n, 0.5)

        max_delta = np.max(delta_norms) if len(delta_norms) > 0 else 0.0
        if max_delta < 1e-10:
            # Trajectory is stationary
            commitment[:] = 1.0
            return commitment

        # t=0 has no predecessor, use neutral value
        commitment[0] = 0.5
        for t in range(1, n):
            commitment[t] = 1.0 - (delta_norms[t - 1] / max_delta)

        return np.clip(commitment, 0.0, 1.0)

    def _compute_uncertainty(self, states: np.ndarray, n: int) -> np.ndarray:
        """Uncertainty: entropy of directions to K nearest historical neighbors.

        At each step t, we find the K nearest previous states and compute
        the angular distribution of displacement vectors from s_t to each
        neighbor. High entropy = many directions = high uncertainty.

        For high-dimensional vectors (d > 3), we use the angle between
        displacement vectors as a proxy for directional diversity, binned
        into n_bins sectors.
        """
        uncertainty = np.full(n, 0.5)

        for t in range(1, n):
            # Collect all previous states
            history = states[:t]
            k = min(self.k, len(history))

            if k < 2:
                uncertainty[t] = 0.5
                continue

            # Find K nearest neighbors in history
            diffs = history - states[t]  # (t, D)
            dists = np.linalg.norm(diffs, axis=1)  # (t,)
            if k >= len(dists):
                nearest_idx = np.arange(len(dists))
            else:
                nearest_idx = np.argpartition(dists, k)[:k]
            nearest_diffs = diffs[nearest_idx]  # (K, D)

            # Compute pairwise angles between displacement vectors
            # This captures directional diversity without requiring
            # a specific coordinate system.
            norms = np.linalg.norm(nearest_diffs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            unit_diffs = nearest_diffs / norms

            # Compute all pairwise cosine similarities
            cos_sims = unit_diffs @ unit_diffs.T  # (K, K)
            # Convert to angles in [0, pi]
            angles = np.arccos(np.clip(cos_sims[np.triu_indices(k, k=1)], -1, 1))

            if len(angles) == 0:
                uncertainty[t] = 0.5
                continue

            # Bin angles and compute entropy
            bins = np.linspace(0, np.pi, self.n_bins + 1)
            hist, _ = np.histogram(angles, bins=bins)
            hist = hist.astype(float)
            total = hist.sum()

            if total < 1e-10:
                uncertainty[t] = 0.0
                continue

            probs = hist / total
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))

            # Normalize by max possible entropy (uniform distribution)
            max_entropy = np.log2(self.n_bins)
            uncertainty[t] = entropy / max_entropy if max_entropy > 0 else 0.0

        return np.clip(uncertainty, 0.0, 1.0)

    def _compute_transition_pressure(
        self,
        commitment: np.ndarray,
        uncertainty: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """Transition pressure: dc/dt - du/dt.

        High when commitment is rising AND uncertainty is falling simultaneously.
        This signals a decision point: the trajectory is narrowing down fast.

        We use smoothed finite differences to avoid noise spikes.

        The raw values are not clipped because transition pressure is an
        unbounded signal. Extreme values are informative.
        """
        pressure = np.zeros(n)

        if n < 3:
            return pressure

        # Smoothed finite differences
        dc = self._smoothed_derivative(commitment)
        du = self._smoothed_derivative(uncertainty)

        pressure = dc - du
        return pressure

    def _compute_recovery_margin(
        self,
        states: np.ndarray,
        uncertainty: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """Recovery margin: normalized distance to nearest branching point.

        Branching points are trajectory positions where uncertainty was
        above median (many options were available). Recovery margin
        measures how far the trajectory has traveled from those points.

        High recovery = near a branching point, can still pivot.
        Low recovery = deep into a single branch, committed.
        """
        recovery = np.full(n, 1.0)

        if n < 3:
            return recovery

        # Compute max range (max pairwise distance) for normalization.
        # For efficiency, approximate with max distance from centroid * 2.
        centroid = np.mean(states, axis=0)
        dists_to_centroid = np.linalg.norm(states - centroid, axis=1)
        max_range = 2.0 * np.max(dists_to_centroid)

        if max_range < 1e-10:
            # Trajectory is stationary, recovery is always 1
            return recovery

        # Identify branching points (uncertainty above median)
        median_u = np.median(uncertainty)
        branching_mask = uncertainty >= median_u

        for t in range(n):
            # Find nearest branching point in history (up to current step)
            branching_indices = np.where(branching_mask[:t + 1])[0]
            if len(branching_indices) == 0:
                recovery[t] = 1.0
                continue

            branching_states = states[branching_indices]
            dists = np.linalg.norm(branching_states - states[t], axis=1)
            min_dist = np.min(dists)

            # Normalize: close to branching point = high recovery
            recovery[t] = 1.0 - (min_dist / max_range)

        return np.clip(recovery, 0.0, 1.0)

    # ----- Utility Methods -----

    def _smoothed_derivative(self, signal: np.ndarray) -> np.ndarray:
        """Compute smoothed finite difference of a 1D signal.

        Uses a centered moving average of the first difference to reduce
        noise in the derivative estimate.
        """
        n = len(signal)
        raw_diff = np.zeros(n)

        # Forward difference
        for t in range(1, n):
            raw_diff[t] = signal[t] - signal[t - 1]

        # Smooth with moving average
        if self.smooth_w <= 1:
            return raw_diff

        smoothed = np.convolve(
            raw_diff,
            np.ones(self.smooth_w) / self.smooth_w,
            mode="same",
        )
        return smoothed

    def compute_incremental(
        self,
        trajectory: list[np.ndarray],
        previous_packet: Optional[AnticipationPacket] = None,
    ) -> AnticipationPacket:
        """Compute anticipation for a trajectory, reusing previous computation.

        This is an optimization for streaming scenarios where the trajectory
        grows one step at a time. Currently recomputes from scratch, but
        the interface is designed for future incremental computation.

        Args:
            trajectory: Full trajectory including new steps.
            previous_packet: Packet from previous computation (currently unused).

        Returns:
            Fresh AnticipationPacket for the full trajectory.
        """
        # TODO: Implement true incremental computation for O(K*D) per step
        # instead of O(N*K*D). For now, full recomputation is correct.
        return self.compute(trajectory)


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def demo():
    """Demonstrate anticipation geometry on a synthetic trajectory."""
    np.random.seed(42)
    geom = AnticipationGeometry(k_neighbors=5)

    # Create a trajectory that explores, then commits, then transitions
    # Phase 1: Exploration (random walk, high uncertainty)
    exploration = [np.random.randn(10) * 0.3 for _ in range(20)]

    # Phase 2: Commitment (converging toward a target)
    target = np.ones(10) * 2.0
    commitment_phase = []
    pos = exploration[-1].copy()
    for i in range(20):
        alpha = (i + 1) / 20.0
        pos = pos * (1 - alpha * 0.3) + target * (alpha * 0.3)
        pos += np.random.randn(10) * 0.05 * (1 - alpha)
        commitment_phase.append(pos.copy())

    # Phase 3: Locked in (small perturbations around target)
    locked = [target + np.random.randn(10) * 0.01 for _ in range(15)]

    trajectory = exploration + commitment_phase + locked
    packet = geom.compute(trajectory)

    print("Anticipation Geometry Demo")
    print("=" * 60)
    print(f"Trajectory: {packet.trajectory_length} steps in R^{packet.dimensionality}")
    print()

    # Show regime classification at key points
    phases = [
        (5, "early exploration"),
        (15, "late exploration"),
        (25, "early commitment"),
        (35, "late commitment"),
        (45, "locked in"),
        (50, "deep locked"),
    ]

    print(f"{'Step':>6} | {'Phase':<20} | {'Commit':>7} | {'Uncert':>7} | {'Pressure':>9} | {'Recovery':>9} | {'Regime':<15}")
    print("-" * 95)
    for t, label in phases:
        if t < packet.trajectory_length:
            print(
                f"{t:>6} | {label:<20} | {packet.commitment[t]:>7.3f} | "
                f"{packet.uncertainty[t]:>7.3f} | {packet.transition_pressure[t]:>+9.3f} | "
                f"{packet.recovery_margin[t]:>9.3f} | {packet.regime_at(t):<15}"
            )

    print()
    summary = packet.summary()
    print("Summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    demo()
