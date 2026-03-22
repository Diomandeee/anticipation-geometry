//! cc-core-rs: High-performance Rust kernels for cc-core
//!
//! This crate provides optimized implementations of real-time audio
//! processing primitives used by cc-core:
//!
//! - [`ring_buffer::LockFreeRingBuffer`] - SPSC ring buffer with atomic operations
//! - [`slew_limiter::SlewLimiter`] - First and second-order parameter smoothing
//! - [`one_euro_filter::OneEuroFilter`] - Adaptive low-pass filtering
//! - [`proximal_ops`] - Box projection, L2 prox, EMA for equilibrium solvers
//! - [`stability_metrics`] - Convergence and stability analysis
//! - [`equilibrium_kernel`] - Fast fixed-point iteration solver
//!
//! ## Performance
//!
//! These implementations are designed for real-time audio callbacks at 48kHz+:
//! - Lock-free where possible (true atomics, not Python GIL)
//! - SIMD-friendly memory layouts
//! - Zero allocations in hot paths
//! - Predictable, bounded execution time
//!
//! ## Python Integration
//!
//! Enable the `python` feature for PyO3 bindings:
//!
//! ```toml
//! [dependencies]
//! cc-core-rs = { version = "0.1", features = ["python"] }
//! ```

#![warn(missing_docs)]
#![allow(clippy::needless_return)]

pub mod ring_buffer;
pub mod slew_limiter;
pub mod one_euro_filter;
pub mod proximal_ops;
pub mod stability_metrics;
pub mod equilibrium_kernel;
pub mod matrix_ops;
pub mod lim_rps;

#[cfg(feature = "python")]
pub mod bindings;

// Re-exports for convenience
pub use ring_buffer::LockFreeRingBuffer;
pub use slew_limiter::{SlewLimiter, SecondOrderSlewLimiter};
pub use one_euro_filter::OneEuroFilter;
pub use proximal_ops::{box_project, l2_prox, l2_prox_pull, ema, l2_norm, l2_distance};
pub use stability_metrics::{
    contraction_ratio, headroom, stability_health_check,
    compute_lipschitz_estimate, estimate_convergence_time,
    StabilityHealthCheck, ConvergenceEstimate, EquilibriumDiagnostics,
};
pub use equilibrium_kernel::{
    EquilibriumKernelConfig, EquilibriumResult, LinearOperator,
    solve_linear_equilibrium, solve_equilibrium, solve_linear_equilibrium_batch,
    DenseOperator, solve_dense_equilibrium, solve_dense_equilibrium_batch,
    // Anderson Acceleration
    AndersonConfig, AndersonAccelerator,
    solve_anderson_equilibrium, solve_anderson_dense_equilibrium, solve_anderson_equilibrium_batch,
    // Parallel batch solving (Rayon)
    ParallelConfig,
    solve_linear_equilibrium_batch_parallel, solve_dense_equilibrium_batch_parallel,
    solve_anderson_equilibrium_batch_parallel, solve_anderson_dense_equilibrium_batch_parallel,
    solve_linear_equilibrium_batch_adaptive, solve_dense_equilibrium_batch_adaptive,
    solve_anderson_equilibrium_batch_adaptive,
};
pub use matrix_ops::{DenseMatrix, SparseMatrix, MatrixOperator};

// LIM-RPS exports
pub use lim_rps::{
    // Config
    LimRpsConfig, ProxMode,
    // Solver
    LimRpsSolver, LimRpsResult, solve_with_operator,
    // Operator
    CrossModalOperator, LimRpsOperator, IdentityOperator,
    // Spectral normalization
    SpectralNormDense, spectral_norm_estimate, spectral_normalize,
    // State
    LatentState, LatentStateHistory,
    // Dynamics
    DynamicsAnalyzer, PhaseTracker, PhaseCouplingDetector, SomaticEstimator,
    // High-level API
    LimRpsProcessor,
    // Constants
    DEFAULT_LATENT_DIM,
};
