#!/usr/bin/env python3
"""
Demo: Generalized Anticipation Geometry for Conversational Trajectories

This example shows how to compute the 4 core anticipation scalars from
conversation embeddings using the Python anticipation geometry library.

The same scalar framework used for physical motion (skeleton data)
generalizes to any trajectory through a high-dimensional space,
including conversation embeddings.
"""

import numpy as np
import sys
import os

# Add the python package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from anticipation_geometry.generalized_anticipation import (
    AnticipationGeometry,
    AnticipationPacket,
)


def main():
    print("Anticipation Geometry - Conversational Trajectory Demo")
    print("=" * 55)
    print()

    # Initialize the geometry engine
    geometry = AnticipationGeometry(
        k_neighbors=5,
        n_angle_bins=8,
        smoothing_window=3,
    )

    # Simulate a conversation trajectory through embedding space
    # Each turn is a point in 384-dimensional space
    embedding_dim = 384
    np.random.seed(42)

    # Phase 1: Small talk (low commitment, high uncertainty)
    small_talk_center = np.random.randn(embedding_dim) * 0.1
    small_talk = [
        small_talk_center + np.random.randn(embedding_dim) * 0.3
        for _ in range(5)
    ]

    # Phase 2: Topic convergence (rising commitment, falling uncertainty)
    topic_direction = np.random.randn(embedding_dim)
    topic_direction /= np.linalg.norm(topic_direction)
    convergence = [
        small_talk_center + topic_direction * (i * 0.5) + np.random.randn(embedding_dim) * 0.1
        for i in range(5)
    ]

    # Phase 3: Deep discussion (high commitment, low uncertainty)
    deep_center = small_talk_center + topic_direction * 3.0
    deep = [
        deep_center + np.random.randn(embedding_dim) * 0.05
        for _ in range(5)
    ]

    # Phase 4: Topic shift (transition pressure spike)
    new_direction = np.random.randn(embedding_dim)
    new_direction /= np.linalg.norm(new_direction)
    shift = [
        deep_center + new_direction * (i * 0.4) + np.random.randn(embedding_dim) * 0.15
        for i in range(5)
    ]

    trajectory = small_talk + convergence + deep + shift
    phase_labels = (
        ["small-talk"] * 5 +
        ["convergence"] * 5 +
        ["deep-discuss"] * 5 +
        ["topic-shift"] * 5
    )

    print(f"Simulated conversation: {len(trajectory)} turns, {embedding_dim}D embeddings")
    print()

    # Compute anticipation scalars over the full trajectory
    packet = geometry.compute(trajectory)

    # Display per-turn results
    print(f"{'Turn':>4}  {'Phase':<14}  {'Commit':>7}  {'Uncert':>7}  {'Pressure':>8}  {'Recovery':>8}  {'Regime':<14}")
    print("-" * 80)

    for i in range(len(trajectory)):
        regime = packet.regime_at(i)
        print(
            f"{i:>4}  {phase_labels[i]:<14}  "
            f"{packet.commitment[i]:>7.3f}  "
            f"{packet.uncertainty[i]:>7.3f}  "
            f"{packet.transition_pressure[i]:>8.3f}  "
            f"{packet.recovery_margin[i]:>8.3f}  "
            f"{regime:<14}"
        )

    # Print summary statistics
    print()
    print("=== Summary ===")
    summary = packet.summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print()
    print("Key observations:")
    print("  - Small talk: high uncertainty (many directions possible)")
    print("  - Convergence: rising commitment, falling uncertainty")
    print("  - Deep discussion: high commitment, low uncertainty")
    print("  - Topic shift: transition pressure spike, rising novelty")


if __name__ == "__main__":
    main()
