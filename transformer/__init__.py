"""
Anticipatory Transformer
=========================

Neural implementation of Anticipation Geometry: geometric scalars
steering a transformer's multi-head attention via additive trajectory bias.

The key insight: instead of attention_scores = QK^T/sqrt(d) + position_encoding,
we compute attention_scores = QK^T/sqrt(d) + trajectory_bias(anticipation_scalars),
where the 7 anticipation scalars encode the geometric state of the reasoning
trajectory at each position.

Modules:
    config: AnticipatoryConfig dataclass
    model: Full model (TrajectoryBiasNetwork, TrajectoryAttention,
           DualPathwayBlock, AnticipationHead, CommitmentGate,
           AnticipatoryTransformer)
    geometry_bridge: Bridge between numpy AnticipationGeometry and PyTorch
    train: Training loop with anticipation-aware loss
    evaluate: Comparative evaluation vs standard transformer
"""

from .config import AnticipatoryConfig
from .model import (
    TrajectoryBiasNetwork,
    TrajectoryAttention,
    DualPathwayBlock,
    AnticipationHead,
    CommitmentGate,
    AnticipatoryTransformer,
)
from .geometry_bridge import GeometryBridge

__all__ = [
    "AnticipatoryConfig",
    "TrajectoryBiasNetwork",
    "TrajectoryAttention",
    "DualPathwayBlock",
    "AnticipationHead",
    "CommitmentGate",
    "AnticipatoryTransformer",
    "GeometryBridge",
]
