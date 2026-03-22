"""
cc-anticipation: Anticipation kernel for Computational Choreography

Converts motion state into actionable anticipatory signals.

Example:
    >>> from cc_anticipation import AnticipationKernel, AnticipationConfig, MotionWindow
    >>>
    >>> config = AnticipationConfig.default()
    >>> kernel = AnticipationKernel(config)
    >>>
    >>> # Create a test window
    >>> window = MotionWindow.still(duration_seconds=1.0, start_time=0.0)
    >>>
    >>> # Process
    >>> packet = kernel.process(window)
    >>> print(f"Commitment: {packet.commitment:.3f}")
    >>> print(f"Uncertainty: {packet.uncertainty:.3f}")
"""

# Import from the Rust extension
from cc_anticipation.cc_anticipation import (
    AnticipationConfig,
    AnticipationKernel,
    AnticipationPacket,
    MotionWindow,
    SkeletonFrame,
    LatentFrame,
    DebugTrace,
    SCHEMA_VERSION,
    MOCOPI_BONE_COUNT,
)

__version__ = "0.1.0"
__all__ = [
    "AnticipationConfig",
    "AnticipationKernel",
    "AnticipationPacket",
    "MotionWindow",
    "SkeletonFrame",
    "LatentFrame",
    "DebugTrace",
    "SCHEMA_VERSION",
    "MOCOPI_BONE_COUNT",
]
