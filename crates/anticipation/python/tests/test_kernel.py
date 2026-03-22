"""Tests for cc-anticipation Python bindings."""

import pytest
from cc_anticipation import (
    AnticipationConfig,
    AnticipationKernel,
    MotionWindow,
    SCHEMA_VERSION,
    MOCOPI_BONE_COUNT,
)


class TestConfig:
    """Test AnticipationConfig."""

    def test_default_config(self):
        config = AnticipationConfig.default()
        assert config.fps == 50.0
        assert config.window_seconds == 1.0
        assert config.regime_embedding_dim == 64
        assert abs(config.min_coverage - 0.90) < 1e-6  # f32 precision

    def test_low_latency_config(self):
        config = AnticipationConfig.low_latency()
        assert config.window_seconds == 0.5
        assert config.emit_debug is False

    def test_analysis_config(self):
        config = AnticipationConfig.analysis()
        assert config.regime_embedding_dim == 128
        assert config.emit_debug is True

    def test_custom_config(self):
        config = AnticipationConfig(
            fps=100.0,
            window_seconds=0.5,
            regime_embedding_dim=32,
            min_coverage=0.80,
            emit_debug=False,
        )
        assert config.fps == 100.0
        assert config.regime_embedding_dim == 32


class TestMotionWindow:
    """Test MotionWindow."""

    def test_still_window(self):
        window = MotionWindow.still(duration_seconds=1.0, start_time=0.0)
        assert window.t_start == 0.0
        assert window.t_end == 1.0
        assert window.fps == 50.0
        assert abs(window.coverage - 0.98) < 1e-6  # f32 precision
        assert window.frame_count == 50  # Property, not method
        assert window.has_skeleton() is True
        assert window.has_latent() is False

    def test_window_repr(self):
        window = MotionWindow.still(1.0, 0.0)
        repr_str = repr(window)
        assert "MotionWindow" in repr_str
        assert "still_window" in repr_str


class TestKernel:
    """Test AnticipationKernel."""

    def test_kernel_creation(self):
        config = AnticipationConfig.default()
        kernel = AnticipationKernel(config)
        assert repr(kernel).startswith("AnticipationKernel")

    def test_process_still_window(self):
        config = AnticipationConfig.default()
        kernel = AnticipationKernel(config)
        window = MotionWindow.still(1.0, 0.0)

        packet = kernel.process(window)

        # Check scalar bounds
        assert 0.0 <= packet.commitment <= 1.0
        assert 0.0 <= packet.uncertainty <= 1.0
        assert 0.0 <= packet.recovery_margin <= 1.0
        assert 0.0 <= packet.phase_stiffness <= 1.0
        assert 0.0 <= packet.novelty <= 1.0
        assert 0.0 <= packet.stability <= 1.0

        # Check vectors
        assert len(packet.regime_embedding) == 64
        assert len(packet.constraint_vector) == 8
        assert len(packet.derivative_summary) == 8

        # Check provenance
        assert packet.schema_version == SCHEMA_VERSION
        assert "still_window" in packet.window_id

    def test_process_low_coverage_fails(self):
        config = AnticipationConfig.default()
        kernel = AnticipationKernel(config)

        # Create window with low coverage
        window = MotionWindow(
            window_id="low_coverage",
            t_start=0.0,
            t_end=1.0,
            fps=50.0,
            skeleton_frames=None,  # No frames
            latent_frames=None,
            coverage=0.5,  # Below threshold
        )

        with pytest.raises(ValueError, match="Coverage"):
            kernel.process(window)

    def test_deterministic(self):
        config = AnticipationConfig.default()
        window = MotionWindow.still(1.0, 0.0)

        results = []
        for _ in range(3):
            kernel = AnticipationKernel(config)
            packet = kernel.process(window)
            results.append((
                packet.commitment,
                packet.uncertainty,
                packet.regime_embedding,
            ))

        # All runs should produce identical results
        for i in range(1, len(results)):
            assert results[0][0] == results[i][0], "commitment not deterministic"
            assert results[0][1] == results[i][1], "uncertainty not deterministic"
            assert results[0][2] == results[i][2], "regime_embedding not deterministic"

    def test_reset(self):
        config = AnticipationConfig.default()
        kernel = AnticipationKernel(config)

        # Process a window
        window = MotionWindow.still(1.0, 0.0)
        kernel.process(window)

        # Reset and process again - should be same as fresh kernel
        kernel.reset()

        fresh_kernel = AnticipationKernel(config)
        packet1 = kernel.process(window)
        packet2 = fresh_kernel.process(window)

        assert packet1.commitment == packet2.commitment


class TestPacket:
    """Test AnticipationPacket."""

    def test_scalars_dict(self):
        config = AnticipationConfig.default()
        kernel = AnticipationKernel(config)
        window = MotionWindow.still(1.0, 0.0)
        packet = kernel.process(window)

        scalars = packet.scalars()
        assert "commitment" in scalars
        assert "uncertainty" in scalars
        assert "transition_pressure" in scalars
        assert "recovery_margin" in scalars
        assert "phase_stiffness" in scalars
        assert "novelty" in scalars
        assert "stability" in scalars

    def test_debug_trace(self):
        config = AnticipationConfig.default()
        kernel = AnticipationKernel(config)
        window = MotionWindow.still(1.0, 0.0)
        packet = kernel.process(window)

        debug = packet.debug
        assert debug is not None
        assert isinstance(debug.raw_features, dict)
        assert isinstance(debug.intermediate_scores, dict)

    def test_validate(self):
        config = AnticipationConfig.default()
        kernel = AnticipationKernel(config)
        window = MotionWindow.still(1.0, 0.0)
        packet = kernel.process(window)

        # Should not raise
        packet.validate()


class TestConstants:
    """Test module constants."""

    def test_schema_version(self):
        assert SCHEMA_VERSION == "0.1.0"

    def test_bone_count(self):
        assert MOCOPI_BONE_COUNT == 27


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
