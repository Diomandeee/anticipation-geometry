//! LIM-RPS: Lipschitz-constrained Implicit Map for Recursive Polymodal Synthesis
//!
//! A unified equilibrium solver for cross-modal latent fusion.
//!
//! # Overview
//!
//! LIM-RPS finds equilibrium latents z* that satisfy:
//!
//! ```text
//! z* = prox(z* - γ * B(z*), encoder)
//! ```
//!
//! where:
//! - `B` is a 1-Lipschitz cross-modal operator (spectral-normalized MLP)
//! - `prox` pulls toward raw encoder output
//! - `γ` is the step size
//!
//! # Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────┐
//! │                         LIM-RPS                                │
//! └───────────────────────────────────────────────────────────────┘
//!                               │
//!       ┌───────────────────────┼───────────────────────┐
//!       │                       │                       │
//!       ▼                       ▼                       ▼
//! ┌──────────┐          ┌──────────────┐        ┌────────────┐
//! │ Spectral │          │CrossModal    │        │ Dynamics   │
//! │ Norm     │◀────────▶│ Operator     │───────▶│ Analyzer   │
//! └──────────┘          └──────────────┘        └────────────┘
//!       │                       │                       │
//!       │                       ▼                       │
//!       │               ┌──────────────┐                │
//!       └──────────────▶│ LimRpsSolver │◀───────────────┘
//!                       └──────────────┘
//!                               │
//!                               ▼
//!                       ┌──────────────┐
//!                       │ LatentState  │
//!                       └──────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use cc_core_rs::lim_rps::{LimRpsSolver, LimRpsConfig, LatentState};
//!
//! // Create solver with default config
//! let config = LimRpsConfig::default();
//! let solver = LimRpsSolver::new(128, config);
//!
//! // Solve for equilibrium
//! let encoder_latent = vec![0.5; 128];
//! let result = solver.solve(&encoder_latent, None);
//!
//! // Get rich output
//! let state = LatentState::from_result(result);
//! println!("Position: {:?}", state.position);
//! ```
//!
//! # Modules
//!
//! - [`config`] - Configuration types
//! - [`spectral`] - Spectral normalization for Lipschitz constraint
//! - [`operator`] - CrossModalOperator and trait
//! - [`solver`] - Fixed-point iteration solver
//! - [`latent_state`] - Rich output with dynamics
//! - [`dynamics`] - Phase tracking and analysis

pub mod config;
pub mod dynamics;
pub mod latent_state;
pub mod operator;
pub mod solver;
pub mod spectral;

// Re-export main types at module level
pub use config::{LimRpsConfig, ProxMode};
pub use dynamics::{DynamicsAnalyzer, PhaseTracker, PhaseCouplingDetector, SomaticEstimator};
pub use latent_state::{LatentState, LatentStateHistory};
pub use operator::{CrossModalOperator, IdentityOperator, LimRpsOperator};
pub use solver::{solve_with_operator, LimRpsResult, LimRpsSolver};
pub use spectral::{spectral_norm_estimate, spectral_normalize, SpectralNormDense};

/// LIM-RPS crate version.
pub const VERSION: &str = "1.0.0";

/// Default latent dimension.
pub const DEFAULT_LATENT_DIM: usize = 128;

/// LIM-RPS Processor (high-level convenience API).
///
/// Combines solver and dynamics analyzer into a single interface.
#[derive(Debug)]
pub struct LimRpsProcessor {
    /// Solver instance.
    solver: LimRpsSolver,
    /// Dynamics analyzer.
    dynamics: DynamicsAnalyzer,
    /// Somatic estimator.
    somatic: SomaticEstimator,
    /// Frame counter.
    frame_count: u64,
}

impl LimRpsProcessor {
    /// Create a new processor.
    ///
    /// # Arguments
    /// * `total_dim` - Dimension of the latent space
    /// * `config` - Solver configuration
    /// * `tempo_bpm` - Initial tempo for phase tracking
    pub fn new(total_dim: usize, config: LimRpsConfig, tempo_bpm: f32) -> Self {
        Self {
            solver: LimRpsSolver::new(total_dim, config),
            dynamics: DynamicsAnalyzer::new(100, tempo_bpm),
            somatic: SomaticEstimator::new(0.9),
            frame_count: 0,
        }
    }

    /// Process a single frame.
    ///
    /// # Arguments
    /// * `encoder_latent` - Raw encoder output
    /// * `timestamp_micros` - Frame timestamp
    ///
    /// # Returns
    /// Rich latent state with dynamics
    pub fn process(&mut self, encoder_latent: &[f32], timestamp_micros: u64) -> LatentState {
        // Solve for equilibrium
        let result = self.solver.solve_warm(encoder_latent);

        // Create latent state from result
        let mut state = LatentState::from_result(result);
        state.frame_index = self.frame_count;

        // Compute dynamics
        self.dynamics.process(&mut state, timestamp_micros);

        // Update somatic qualities
        self.somatic.update(&state);
        self.somatic.apply_to(&mut state);

        self.frame_count += 1;
        state
    }

    /// Process without warm start.
    pub fn process_cold(&mut self, encoder_latent: &[f32], timestamp_micros: u64) -> LatentState {
        let result = self.solver.solve(encoder_latent, None);
        let mut state = LatentState::from_result(result);
        state.frame_index = self.frame_count;

        self.dynamics.process(&mut state, timestamp_micros);
        self.somatic.update(&state);
        self.somatic.apply_to(&mut state);

        self.frame_count += 1;
        state
    }

    /// Set tempo for phase tracking.
    pub fn set_tempo(&mut self, bpm: f32) {
        self.dynamics.set_tempo(bpm);
    }

    /// Get current tempo.
    pub fn tempo(&self) -> f32 {
        self.dynamics.phase_tracker().tempo()
    }

    /// Get current phase.
    pub fn phase(&self) -> f32 {
        self.dynamics.phase_tracker().phase()
    }

    /// Get periodicity estimate.
    pub fn periodicity(&self) -> f32 {
        self.dynamics.periodicity()
    }

    /// Get internal tempo estimate.
    pub fn internal_tempo(&self) -> f32 {
        self.dynamics.internal_tempo()
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.solver.reset();
        self.dynamics.reset();
        self.somatic.reset();
        self.frame_count = 0;
    }

    /// Get solver reference.
    pub fn solver(&self) -> &LimRpsSolver {
        &self.solver
    }

    /// Get mutable solver reference.
    pub fn solver_mut(&mut self) -> &mut LimRpsSolver {
        &mut self.solver
    }

    /// Get dynamics analyzer reference.
    pub fn dynamics(&self) -> &DynamicsAnalyzer {
        &self.dynamics
    }

    /// Get frame count.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

impl Default for LimRpsProcessor {
    fn default() -> Self {
        Self::new(DEFAULT_LATENT_DIM, LimRpsConfig::default(), 120.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(VERSION, "1.0.0");
    }

    #[test]
    fn test_processor_creation() {
        let processor = LimRpsProcessor::default();
        assert_eq!(processor.frame_count(), 0);
    }

    #[test]
    fn test_processor_process() {
        let mut processor = LimRpsProcessor::new(16, LimRpsConfig::default(), 120.0);

        let encoder = vec![0.5; 16];
        let state = processor.process(&encoder, 0);

        assert_eq!(state.x_star.len(), 16);
        assert_eq!(processor.frame_count(), 1);
    }

    #[test]
    fn test_processor_multiple_frames() {
        let mut processor = LimRpsProcessor::new(8, LimRpsConfig::default(), 120.0);

        for i in 0..10 {
            let encoder = vec![i as f32 * 0.1; 8];
            let state = processor.process(&encoder, i * 20_000);
            assert_eq!(state.frame_index, i);
        }

        assert_eq!(processor.frame_count(), 10);
    }

    #[test]
    fn test_processor_reset() {
        let mut processor = LimRpsProcessor::new(8, LimRpsConfig::default(), 120.0);

        let encoder = vec![0.5; 8];
        processor.process(&encoder, 0);
        processor.process(&encoder, 20_000);

        processor.reset();

        assert_eq!(processor.frame_count(), 0);
    }

    #[test]
    fn test_processor_tempo() {
        let mut processor = LimRpsProcessor::new(8, LimRpsConfig::default(), 100.0);

        assert_eq!(processor.tempo(), 100.0);

        processor.set_tempo(140.0);
        assert_eq!(processor.tempo(), 140.0);
    }

    #[test]
    fn test_reexports() {
        // Verify all re-exports work
        let _config = LimRpsConfig::default();
        let _mode = ProxMode::L2;
        let _dim = DEFAULT_LATENT_DIM;
    }
}
