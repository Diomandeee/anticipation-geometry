"""Training utilities for anticipation-conditioned fine-tuning.

Provides data loading, inscription conditioning, and trainable
anticipation modules (gate, head, scalar projection).
"""

from .data import MixedTrainDataset, EvalDataset, collate_fn, INSCRIPTION_VOCAB
from .anticipation import (
    InscriptionEmbedding,
    ScalarProjection,
    CommitmentGate,
    AnticipationHead,
    LSERewardTracker,
)

__all__ = [
    "MixedTrainDataset", "EvalDataset", "collate_fn", "INSCRIPTION_VOCAB",
    "InscriptionEmbedding", "ScalarProjection", "CommitmentGate",
    "AnticipationHead", "LSERewardTracker",
]
