"""Anticipation Geometry: trajectory characterization via 7 geometric scalars.

Compute commitment, uncertainty, transition pressure, recovery margin,
phase stiffness, novelty, and stability from any vector trajectory.
Supports conversation embeddings, motion capture, knowledge graph paths,
and arbitrary sequences in metric spaces.

Usage:
    from anticipation_geometry import AnticipationGeometry

    geometry = AnticipationGeometry(k_neighbors=5)
    packet = geometry.compute(trajectory)
"""

__version__ = "0.3.0"

from .core import AnticipationGeometry

__all__ = ["AnticipationGeometry", "__version__"]
