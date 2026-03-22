"""
anticipation-geometry
=====================

Domain-general anticipation scalars + Princeton KG-path reward integration
for evaluating and steering reasoning trajectories.

Modules:
    kg_reward: 3-signal KG-path reward function (arXiv:2603.14147)
    generalized_anticipation: 4 anticipation scalars for arbitrary trajectories
    conversation_trajectory: Supabase conversation -> trajectory analysis
    eval_harness: Comparative evaluation of reward strategies
"""

try:
    from .kg_reward import KGPathReward, KGHop, SignalBreakdown, RewardConfig
    from .generalized_anticipation import AnticipationGeometry, AnticipationPacket
except ImportError:
    from kg_reward import KGPathReward, KGHop, SignalBreakdown, RewardConfig
    from generalized_anticipation import AnticipationGeometry, AnticipationPacket

__all__ = [
    "KGPathReward",
    "KGHop",
    "SignalBreakdown",
    "RewardConfig",
    "AnticipationGeometry",
    "AnticipationPacket",
]
