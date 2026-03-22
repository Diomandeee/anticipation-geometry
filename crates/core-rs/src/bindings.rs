//! Python bindings for cc-core-rs using PyO3.
//!
//! Exposes high-performance Rust implementations to Python for integration
//! with cc-core's real-time audio processing pipeline.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyReadonlyArray1};

use crate::ring_buffer::LockFreeRingBuffer as RustRingBuffer;
use crate::slew_limiter::{SlewLimiter as RustSlewLimiter, SecondOrderSlewLimiter as RustSecondOrderSlewLimiter};
use crate::one_euro_filter::OneEuroFilter as RustOneEuroFilter;

// =============================================================================
// Ring Buffer Binding
// =============================================================================

/// Lock-free single-producer, single-consumer ring buffer.
///
/// Uses true atomic operations for wait-free read/write, unlike the Python
/// implementation which uses threading.Lock.
///
/// Args:
///     capacity: Buffer size (rounded up to power of 2)
///
/// Example:
///     >>> buf = LockFreeRingBuffer(4)
///     >>> buf.write(42.0)
///     True
///     >>> buf.read_latest()
///     42.0
#[pyclass(name = "LockFreeRingBuffer")]
pub struct PyLockFreeRingBuffer {
    inner: RustRingBuffer<f64>,
}

#[pymethods]
impl PyLockFreeRingBuffer {
    #[new]
    #[pyo3(signature = (capacity=4))]
    fn new(capacity: usize) -> PyResult<Self> {
        if capacity < 2 {
            return Err(PyValueError::new_err("Capacity must be at least 2"));
        }
        Ok(Self {
            inner: RustRingBuffer::new(capacity),
        })
    }

    /// Get the buffer capacity (may be larger than requested due to power-of-2 rounding).
    #[getter]
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Write an item to the buffer.
    ///
    /// Returns True if write succeeded without overwrite, False if buffer was full.
    fn write(&self, item: f64) -> bool {
        self.inner.write(item)
    }

    /// Read the most recent item and mark all as consumed.
    ///
    /// Returns None if buffer is empty.
    fn read_latest(&self) -> Option<f64> {
        self.inner.read_latest()
    }

    /// Peek at the most recent item without consuming.
    fn peek_latest(&self) -> Option<f64> {
        self.inner.peek_latest()
    }

    /// Check if buffer is empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Check if buffer is full.
    fn is_full(&self) -> bool {
        self.inner.is_full()
    }

    /// Get number of unread items.
    fn available_count(&self) -> usize {
        self.inner.available_count()
    }

    /// Clear the buffer (not thread-safe with concurrent access).
    fn clear(&self) {
        self.inner.clear()
    }

    fn __repr__(&self) -> String {
        format!(
            "LockFreeRingBuffer(capacity={}, available={})",
            self.inner.capacity(),
            self.inner.available_count()
        )
    }
}

// =============================================================================
// Slew Limiter Binding
// =============================================================================

/// First-order slew rate limiter for parameter smoothing.
///
/// Limits the rate of change of a parameter to prevent sudden jumps
/// that cause clicks or pops in audio.
///
/// Args:
///     max_delta_per_sample: Maximum change per sample
///     sample_rate_hz: Audio sample rate (default: 48000)
///
/// Example:
///     >>> limiter = SlewLimiter(0.001, 48000.0)
///     >>> for _ in range(1000):
///     ...     value = limiter.process_scalar(1.0)
///     >>> print(value)  # Gradually approaches 1.0
#[pyclass(name = "SlewLimiter")]
pub struct PySlewLimiter {
    inner: RustSlewLimiter,
}

#[pymethods]
impl PySlewLimiter {
    #[new]
    #[pyo3(signature = (max_delta_per_sample=0.001, sample_rate_hz=48000.0))]
    fn new(max_delta_per_sample: f32, sample_rate_hz: f32) -> Self {
        Self {
            inner: RustSlewLimiter::new(max_delta_per_sample, sample_rate_hz),
        }
    }

    /// Create with asymmetric rise/fall rates.
    #[staticmethod]
    fn with_asymmetric(rise_rate: f32, fall_rate: f32, sample_rate_hz: f32) -> Self {
        Self {
            inner: RustSlewLimiter::with_asymmetric(rise_rate, fall_rate, sample_rate_hz),
        }
    }

    /// Create from time constant (63% rise time in ms).
    #[staticmethod]
    fn from_time_constant(time_constant_ms: f32, sample_rate_hz: f32) -> Self {
        Self {
            inner: RustSlewLimiter::from_time_constant(time_constant_ms, sample_rate_hz),
        }
    }

    /// Process a single scalar value.
    fn process_scalar(&mut self, target: f32) -> f32 {
        self.inner.process_scalar(target)
    }

    /// Process a buffer of samples with constant target.
    fn process_buffer<'py>(
        &mut self,
        py: Python<'py>,
        target: f32,
        buffer_size: usize,
    ) -> &'py PyArray1<f32> {
        let mut output = vec![0.0f32; buffer_size];
        self.inner.process_buffer(target, &mut output);
        PyArray1::from_vec(py, output)
    }

    /// Reset to a new value instantly.
    fn reset(&mut self, value: f32) {
        self.inner.reset(value);
    }

    /// Get current value without updating.
    fn get_current(&self) -> f32 {
        self.inner.get_current()
    }

    /// Set maximum delta per sample.
    fn set_max_delta(&mut self, max_delta: f32) {
        self.inner.set_max_delta(max_delta);
    }

    /// Set rate based on time constant (ms).
    fn set_time_constant(&mut self, time_constant_ms: f32) {
        self.inner.set_time_constant(time_constant_ms);
    }

    fn __repr__(&self) -> String {
        format!("SlewLimiter(current={:.4})", self.inner.get_current())
    }
}

// =============================================================================
// Second-Order Slew Limiter Binding
// =============================================================================

/// Second-order slew limiter (mass-spring-damper style).
///
/// Models a physically-inspired second-order system for more natural
/// parameter transitions with inertia.
///
/// Args:
///     natural_freq_hz: Natural frequency in Hz (typical: 1-50)
///     damping_ratio: 1.0 = critically damped, <1.0 = underdamped (overshoot)
///     sample_rate_hz: Audio sample rate
///
/// Example:
///     >>> limiter = SecondOrderSlewLimiter(10.0, 1.0, 48000.0)
///     >>> for _ in range(4800):
///     ...     value = limiter.process_scalar(1.0)
#[pyclass(name = "SecondOrderSlewLimiter")]
pub struct PySecondOrderSlewLimiter {
    inner: RustSecondOrderSlewLimiter,
}

#[pymethods]
impl PySecondOrderSlewLimiter {
    #[new]
    #[pyo3(signature = (natural_freq_hz=10.0, damping_ratio=1.0, sample_rate_hz=48000.0))]
    fn new(natural_freq_hz: f32, damping_ratio: f32, sample_rate_hz: f32) -> PyResult<Self> {
        if natural_freq_hz <= 0.0 {
            return Err(PyValueError::new_err("natural_freq_hz must be > 0"));
        }
        if sample_rate_hz <= 0.0 {
            return Err(PyValueError::new_err("sample_rate_hz must be > 0"));
        }
        Ok(Self {
            inner: RustSecondOrderSlewLimiter::new(natural_freq_hz, damping_ratio, sample_rate_hz),
        })
    }

    /// Process a single scalar value.
    fn process_scalar(&mut self, target: f32) -> f32 {
        self.inner.process_scalar(target)
    }

    /// Process a buffer of samples with constant target.
    fn process_buffer<'py>(
        &mut self,
        py: Python<'py>,
        target: f32,
        buffer_size: usize,
    ) -> &'py PyArray1<f32> {
        let mut output = vec![0.0f32; buffer_size];
        self.inner.process_buffer(target, &mut output);
        PyArray1::from_vec(py, output)
    }

    /// Reset to a new value, zeroing velocity.
    fn reset(&mut self, value: f32) {
        self.inner.reset(value);
    }

    /// Get current output value.
    fn get_current(&self) -> f32 {
        self.inner.get_current()
    }

    /// Get current velocity.
    fn get_velocity(&self) -> f32 {
        self.inner.get_velocity()
    }

    /// Update natural frequency.
    fn set_natural_frequency(&mut self, freq_hz: f32) -> PyResult<()> {
        if freq_hz <= 0.0 {
            return Err(PyValueError::new_err("freq_hz must be > 0"));
        }
        self.inner.set_natural_frequency(freq_hz);
        Ok(())
    }

    /// Update damping ratio.
    fn set_damping_ratio(&mut self, zeta: f32) {
        self.inner.set_damping_ratio(zeta);
    }

    fn __repr__(&self) -> String {
        format!(
            "SecondOrderSlewLimiter(current={:.4}, velocity={:.4})",
            self.inner.get_current(),
            self.inner.get_velocity()
        )
    }
}

// =============================================================================
// One-Euro Filter Binding
// =============================================================================

/// One-Euro filter for adaptive low-pass filtering.
///
/// Automatically adjusts cutoff frequency based on signal velocity:
/// slow signals get more smoothing, fast signals get more responsiveness.
///
/// Args:
///     freq: Expected sample rate (Hz)
///     min_cutoff: Minimum cutoff frequency (Hz). Lower = more smoothing
///     beta: Speed coefficient. Higher = more responsive to fast changes
///     d_cutoff: Cutoff for derivative smoothing (Hz)
///
/// Example:
///     >>> filter = OneEuroFilter(100.0, 1.0, 0.0, 1.0)
///     >>> for x in noisy_signal:
///     ...     filtered = filter.filter(x)
#[pyclass(name = "OneEuroFilter")]
pub struct PyOneEuroFilter {
    inner: RustOneEuroFilter,
}

#[pymethods]
impl PyOneEuroFilter {
    #[new]
    #[pyo3(signature = (freq=100.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0))]
    fn new(freq: f32, min_cutoff: f32, beta: f32, d_cutoff: f32) -> PyResult<Self> {
        if freq <= 0.0 {
            return Err(PyValueError::new_err("freq must be > 0"));
        }
        Ok(Self {
            inner: RustOneEuroFilter::new(freq, min_cutoff, beta, d_cutoff),
        })
    }

    /// Create optimized for 60Hz motion tracking.
    #[staticmethod]
    fn for_motion_tracking() -> Self {
        Self {
            inner: RustOneEuroFilter::for_motion_tracking(),
        }
    }

    /// Create optimized for 100Hz control signals.
    #[staticmethod]
    fn for_control_signals() -> Self {
        Self {
            inner: RustOneEuroFilter::for_control_signals(),
        }
    }

    /// Filter one sample.
    ///
    /// Args:
    ///     x: Input value
    ///     t: Optional timestamp in seconds
    ///
    /// Returns:
    ///     Filtered value
    #[pyo3(signature = (x, t=None))]
    fn filter(&mut self, x: f32, t: Option<f32>) -> f32 {
        self.inner.filter(x, t)
    }

    /// Callable interface (same as filter).
    #[pyo3(signature = (x, t=None))]
    fn __call__(&mut self, x: f32, t: Option<f32>) -> f32 {
        self.inner.filter(x, t)
    }

    /// Filter a buffer of samples.
    fn filter_buffer<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray1<f32>,
    ) -> &'py PyArray1<f32> {
        let input_slice = input.as_slice().unwrap();
        let mut output = vec![0.0f32; input_slice.len()];
        self.inner.filter_buffer(input_slice, &mut output);
        PyArray1::from_vec(py, output)
    }

    /// Reset filter state.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get current filtered value (or None if not initialized).
    fn get_current(&self) -> Option<f32> {
        self.inner.get_current()
    }

    /// Get current smoothed derivative.
    fn get_derivative(&self) -> f32 {
        self.inner.get_derivative()
    }

    /// Set minimum cutoff frequency.
    fn set_min_cutoff(&mut self, min_cutoff: f32) {
        self.inner.set_min_cutoff(min_cutoff);
    }

    /// Set beta (speed coefficient).
    fn set_beta(&mut self, beta: f32) {
        self.inner.set_beta(beta);
    }

    /// Set derivative cutoff.
    fn set_d_cutoff(&mut self, d_cutoff: f32) {
        self.inner.set_d_cutoff(d_cutoff);
    }

    fn __repr__(&self) -> String {
        match self.inner.get_current() {
            Some(v) => format!("OneEuroFilter(current={:.4})", v),
            None => "OneEuroFilter(uninitialized)".to_string(),
        }
    }
}

// =============================================================================
// Proximal Ops Bindings
// =============================================================================

use crate::proximal_ops;
use crate::stability_metrics;
use crate::equilibrium_kernel;
use crate::lim_rps;

/// Box projection: clamp values to [lower, upper].
#[pyfunction]
#[pyo3(signature = (x, lower=-10.0, upper=10.0))]
fn box_project<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f32>,
    lower: f32,
    upper: f32,
) -> &'py PyArray1<f32> {
    let input = x.as_slice().unwrap();
    let result = proximal_ops::box_project(input, lower, upper);
    PyArray1::from_vec(py, result)
}

/// L2 proximal operator (soft thresholding).
#[pyfunction]
#[pyo3(signature = (x, tau=0.05))]
fn l2_prox<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f32>,
    tau: f32,
) -> &'py PyArray1<f32> {
    let input = x.as_slice().unwrap();
    let result = proximal_ops::l2_prox(input, tau);
    PyArray1::from_vec(py, result)
}

/// L2 proximal pull toward encoder latents.
#[pyfunction]
#[pyo3(signature = (v, enc, weight=0.05))]
fn l2_prox_pull<'py>(
    py: Python<'py>,
    v: PyReadonlyArray1<f32>,
    enc: PyReadonlyArray1<f32>,
    weight: f32,
) -> &'py PyArray1<f32> {
    let v_slice = v.as_slice().unwrap();
    let enc_slice = enc.as_slice().unwrap();
    let result = proximal_ops::l2_prox_pull(v_slice, enc_slice, weight);
    PyArray1::from_vec(py, result)
}

/// Exponentially weighted moving average.
#[pyfunction]
#[pyo3(signature = (x_prev, x_new, alpha=0.9))]
fn ema<'py>(
    py: Python<'py>,
    x_prev: PyReadonlyArray1<f32>,
    x_new: PyReadonlyArray1<f32>,
    alpha: f32,
) -> &'py PyArray1<f32> {
    let prev = x_prev.as_slice().unwrap();
    let new = x_new.as_slice().unwrap();
    let result = proximal_ops::ema(prev, new, alpha);
    PyArray1::from_vec(py, result)
}

/// EMA for an array (forward pass).
#[pyfunction]
#[pyo3(signature = (values, alpha=0.9, init=None))]
fn ema_array<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f32>,
    alpha: f32,
    init: Option<f32>,
) -> &'py PyArray1<f32> {
    let vals = values.as_slice().unwrap();
    let result = proximal_ops::ema_array(vals, alpha, init);
    PyArray1::from_vec(py, result)
}

/// Compute L2 norm of a vector.
#[pyfunction]
fn l2_norm(x: PyReadonlyArray1<f32>) -> f32 {
    proximal_ops::l2_norm(x.as_slice().unwrap())
}

/// Compute L2 distance between two vectors.
#[pyfunction]
fn l2_distance(a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>) -> f32 {
    proximal_ops::l2_distance(a.as_slice().unwrap(), b.as_slice().unwrap())
}

/// Clip state norm to max_norm.
#[pyfunction]
#[pyo3(signature = (x, max_norm=10.0))]
fn clip_state_norm<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f32>,
    max_norm: f32,
) -> &'py PyArray1<f32> {
    let input = x.as_slice().unwrap();
    let result = proximal_ops::clip_state_norm(input, max_norm);
    PyArray1::from_vec(py, result)
}

// =============================================================================
// Stability Metrics Bindings
// =============================================================================

/// Compute contraction ratio from residuals.
#[pyfunction]
#[pyo3(signature = (residuals, window_size=2))]
fn contraction_ratio(residuals: PyReadonlyArray1<f32>, window_size: usize) -> f32 {
    stability_metrics::contraction_ratio(residuals.as_slice().unwrap(), window_size)
}

/// Compute stability headroom.
#[pyfunction]
#[pyo3(signature = (operator_lipschitz, metric_scale_max=None))]
fn headroom(operator_lipschitz: f32, metric_scale_max: Option<f32>) -> f32 {
    stability_metrics::headroom(operator_lipschitz, metric_scale_max)
}

/// Stability health check result.
#[pyclass(name = "StabilityHealthCheck")]
#[derive(Clone)]
pub struct PyStabilityHealthCheck {
    #[pyo3(get)]
    is_contracting: bool,
    #[pyo3(get)]
    is_converged: bool,
    #[pyo3(get)]
    is_stable: bool,
    #[pyo3(get)]
    is_monotonic: bool,
}

#[pymethods]
impl PyStabilityHealthCheck {
    fn all_ok(&self) -> bool {
        self.is_contracting && self.is_converged && self.is_stable && self.is_monotonic
    }

    fn __repr__(&self) -> String {
        format!(
            "StabilityHealthCheck(contracting={}, converged={}, stable={}, monotonic={})",
            self.is_contracting, self.is_converged, self.is_stable, self.is_monotonic
        )
    }
}

/// Perform stability health check.
#[pyfunction]
#[pyo3(signature = (residuals, contraction_threshold=0.95, residual_threshold=0.01))]
fn stability_health_check(
    residuals: PyReadonlyArray1<f32>,
    contraction_threshold: f32,
    residual_threshold: f32,
) -> PyStabilityHealthCheck {
    let check = stability_metrics::stability_health_check(
        residuals.as_slice().unwrap(),
        contraction_threshold,
        residual_threshold,
    );
    PyStabilityHealthCheck {
        is_contracting: check.is_contracting,
        is_converged: check.is_converged,
        is_stable: check.is_stable,
        is_monotonic: check.is_monotonic,
    }
}

/// Estimate Lipschitz constant from input-output pairs.
#[pyfunction]
#[pyo3(signature = (z_sequence, output_sequence, max_samples=100))]
fn compute_lipschitz_estimate(
    z_sequence: Vec<Vec<f32>>,
    output_sequence: Vec<Vec<f32>>,
    max_samples: usize,
) -> f32 {
    stability_metrics::compute_lipschitz_estimate(&z_sequence, &output_sequence, max_samples)
}

/// Convergence estimate result.
#[pyclass(name = "ConvergenceEstimate")]
#[derive(Clone)]
pub struct PyConvergenceEstimate {
    #[pyo3(get)]
    estimated_iters: usize,
    #[pyo3(get)]
    already_converged: bool,
    #[pyo3(get)]
    contraction_rate: f32,
}

#[pymethods]
impl PyConvergenceEstimate {
    fn __repr__(&self) -> String {
        format!(
            "ConvergenceEstimate(iters={}, converged={}, rate={:.4})",
            self.estimated_iters, self.already_converged, self.contraction_rate
        )
    }
}

/// Estimate convergence time.
#[pyfunction]
#[pyo3(signature = (residuals, target_residual=0.001, contraction_rate=None))]
fn estimate_convergence_time(
    residuals: PyReadonlyArray1<f32>,
    target_residual: f32,
    contraction_rate: Option<f32>,
) -> PyConvergenceEstimate {
    let est = stability_metrics::estimate_convergence_time(
        residuals.as_slice().unwrap(),
        target_residual,
        contraction_rate,
    );
    PyConvergenceEstimate {
        estimated_iters: est.estimated_iters,
        already_converged: est.already_converged,
        contraction_rate: est.contraction_rate,
    }
}

// =============================================================================
// Equilibrium Kernel Bindings
// =============================================================================

/// Configuration for equilibrium kernel.
#[pyclass(name = "EquilibriumKernelConfig")]
#[derive(Clone)]
pub struct PyEquilibriumKernelConfig {
    #[pyo3(get, set)]
    max_iters: usize,
    #[pyo3(get, set)]
    step_size: f32,
    #[pyo3(get, set)]
    prox_l2_weight: f32,
    #[pyo3(get, set)]
    box_lower: f32,
    #[pyo3(get, set)]
    box_upper: f32,
    #[pyo3(get, set)]
    max_state_norm: f32,
    #[pyo3(get, set)]
    early_stop_eps: f32,
    #[pyo3(get, set)]
    early_stop_min_iters: usize,
    #[pyo3(get, set)]
    ema_alpha: f32,
}

#[pymethods]
impl PyEquilibriumKernelConfig {
    #[new]
    #[pyo3(signature = (
        max_iters=4,
        step_size=0.5,
        prox_l2_weight=0.05,
        box_lower=-10.0,
        box_upper=10.0,
        max_state_norm=10.0,
        early_stop_eps=0.0,
        early_stop_min_iters=2,
        ema_alpha=0.0
    ))]
    fn new(
        max_iters: usize,
        step_size: f32,
        prox_l2_weight: f32,
        box_lower: f32,
        box_upper: f32,
        max_state_norm: f32,
        early_stop_eps: f32,
        early_stop_min_iters: usize,
        ema_alpha: f32,
    ) -> Self {
        Self {
            max_iters,
            step_size,
            prox_l2_weight,
            box_lower,
            box_upper,
            max_state_norm,
            early_stop_eps,
            early_stop_min_iters,
            ema_alpha,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "EquilibriumKernelConfig(max_iters={}, step_size={:.2})",
            self.max_iters, self.step_size
        )
    }
}

impl From<&PyEquilibriumKernelConfig> for equilibrium_kernel::EquilibriumKernelConfig {
    fn from(py: &PyEquilibriumKernelConfig) -> Self {
        Self {
            max_iters: py.max_iters,
            step_size: py.step_size,
            prox_l2_weight: py.prox_l2_weight,
            box_lower: py.box_lower,
            box_upper: py.box_upper,
            max_state_norm: py.max_state_norm,
            early_stop_eps: py.early_stop_eps,
            early_stop_min_iters: py.early_stop_min_iters,
            ema_alpha: py.ema_alpha,
        }
    }
}

/// Result from equilibrium solve.
#[pyclass(name = "EquilibriumResult")]
#[derive(Clone)]
pub struct PyEquilibriumResult {
    inner: equilibrium_kernel::EquilibriumResult,
}

#[pymethods]
impl PyEquilibriumResult {
    #[getter]
    fn x_star<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        PyArray1::from_vec(py, self.inner.x_star.clone())
    }

    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        PyArray1::from_vec(py, self.inner.residuals.clone())
    }

    #[getter]
    fn k_used(&self) -> usize {
        self.inner.k_used
    }

    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    #[getter]
    fn final_residual(&self) -> f32 {
        self.inner.final_residual
    }

    #[getter]
    fn mean_residual(&self) -> f32 {
        self.inner.mean_residual
    }

    fn __repr__(&self) -> String {
        format!(
            "EquilibriumResult(k_used={}, converged={}, final_res={:.6})",
            self.inner.k_used, self.inner.converged, self.inner.final_residual
        )
    }
}

/// Linear operator for equilibrium solver.
#[pyclass(name = "LinearOperator")]
#[derive(Clone)]
pub struct PyLinearOperator {
    inner: equilibrium_kernel::LinearOperator,
}

#[pymethods]
impl PyLinearOperator {
    #[new]
    fn new(a_diag: Vec<f32>, bias: Vec<f32>) -> PyResult<Self> {
        if a_diag.len() != bias.len() {
            return Err(PyValueError::new_err("a_diag and bias must have same length"));
        }
        Ok(Self {
            inner: equilibrium_kernel::LinearOperator::new(a_diag, bias),
        })
    }

    /// Create identity operator.
    #[staticmethod]
    fn identity(dim: usize) -> Self {
        Self {
            inner: equilibrium_kernel::LinearOperator::identity(dim),
        }
    }

    /// Create contractive operator with given Lipschitz constant.
    #[staticmethod]
    fn contractive(dim: usize, lipschitz: f32) -> Self {
        Self {
            inner: equilibrium_kernel::LinearOperator::contractive(dim, lipschitz),
        }
    }

    /// Apply operator: b = A @ x + bias
    fn apply<'py>(&self, py: Python<'py>, x: PyReadonlyArray1<f32>) -> &'py PyArray1<f32> {
        let result = self.inner.apply(x.as_slice().unwrap());
        PyArray1::from_vec(py, result)
    }

    fn __repr__(&self) -> String {
        format!("LinearOperator(dim={})", self.inner.a_diag.len())
    }
}

/// Solve linear equilibrium.
#[pyfunction]
fn solve_linear_equilibrium(
    x_init: PyReadonlyArray1<f32>,
    encoder: PyReadonlyArray1<f32>,
    operator: &PyLinearOperator,
    config: &PyEquilibriumKernelConfig,
) -> PyEquilibriumResult {
    let result = equilibrium_kernel::solve_linear_equilibrium(
        x_init.as_slice().unwrap(),
        encoder.as_slice().unwrap(),
        &operator.inner,
        &config.into(),
    );
    PyEquilibriumResult { inner: result }
}

// =============================================================================
// LIM-RPS Bindings
// =============================================================================

/// LIM-RPS proximal mode enumeration.
#[pyclass(name = "ProxMode")]
#[derive(Clone, Copy)]
pub enum PyProxMode {
    /// L2 proximal (elementwise shrinkage).
    L2 = 0,
    /// Group L2 proximal (shrinks groups together).
    GroupL2 = 1,
    /// No proximal regularization.
    None = 2,
}

impl From<PyProxMode> for lim_rps::ProxMode {
    fn from(py: PyProxMode) -> Self {
        match py {
            PyProxMode::L2 => lim_rps::ProxMode::L2,
            PyProxMode::GroupL2 => lim_rps::ProxMode::GroupL2,
            PyProxMode::None => lim_rps::ProxMode::None,
        }
    }
}

impl From<lim_rps::ProxMode> for PyProxMode {
    fn from(mode: lim_rps::ProxMode) -> Self {
        match mode {
            lim_rps::ProxMode::L2 => PyProxMode::L2,
            lim_rps::ProxMode::GroupL2 => PyProxMode::GroupL2,
            lim_rps::ProxMode::None => PyProxMode::None,
        }
    }
}

/// LIM-RPS solver configuration.
///
/// Matches the Python LIMRPSConfig class for cross-language compatibility.
///
/// Args:
///     max_iters: Maximum fixed-point iterations (default: 4)
///     step_size: Forward step size γ (default: 0.5)
///     hidden_dim: MLP hidden dimension (default: 128)
///     num_layers: Number of MLP layers (default: 2)
///     prox_tau: Proximal regularization strength (default: 0.05)
///     box_lower: Box constraint lower bound (default: -10.0)
///     box_upper: Box constraint upper bound (default: 10.0)
///
/// Example:
///     >>> config = LimRpsConfig(max_iters=8, hidden_dim=256)
///     >>> solver = LimRpsSolver(128, config)
#[pyclass(name = "LimRpsConfig")]
#[derive(Clone)]
#[allow(missing_docs)]
pub struct PyLimRpsConfig {
    /// Maximum fixed-point iterations.
    #[pyo3(get, set)]
    pub max_iters: usize,
    /// Forward step size γ.
    #[pyo3(get, set)]
    pub step_size: f32,
    /// Early stopping threshold.
    #[pyo3(get, set)]
    pub early_stop_eps: f32,
    /// Minimum iterations before early stopping.
    #[pyo3(get, set)]
    pub early_stop_min_iters: usize,
    /// MLP hidden dimension.
    #[pyo3(get, set)]
    pub hidden_dim: usize,
    /// Number of MLP layers.
    #[pyo3(get, set)]
    pub num_layers: usize,
    /// Spectral normalization power iterations.
    #[pyo3(get, set)]
    pub spectral_iters: usize,
    /// Proximal regularization strength τ.
    #[pyo3(get, set)]
    pub prox_tau: f32,
    /// Box constraint lower bound.
    #[pyo3(get, set)]
    pub box_lower: f32,
    /// Box constraint upper bound.
    #[pyo3(get, set)]
    pub box_upper: f32,
    /// Maximum state norm for gradient clipping.
    #[pyo3(get, set)]
    pub max_state_norm: f32,
    /// Temporal consistency weight.
    #[pyo3(get, set)]
    pub temporal_lambda: f32,
    /// EMA smoothing factor.
    #[pyo3(get, set)]
    pub ema_alpha: f32,
}

#[pymethods]
impl PyLimRpsConfig {
    #[new]
    #[pyo3(signature = (
        max_iters=4,
        step_size=0.5,
        early_stop_eps=1e-4,
        early_stop_min_iters=2,
        hidden_dim=128,
        num_layers=2,
        spectral_iters=1,
        prox_tau=0.05,
        box_lower=-10.0,
        box_upper=10.0,
        max_state_norm=10.0,
        temporal_lambda=0.0,
        ema_alpha=0.0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_iters: usize,
        step_size: f32,
        early_stop_eps: f32,
        early_stop_min_iters: usize,
        hidden_dim: usize,
        num_layers: usize,
        spectral_iters: usize,
        prox_tau: f32,
        box_lower: f32,
        box_upper: f32,
        max_state_norm: f32,
        temporal_lambda: f32,
        ema_alpha: f32,
    ) -> Self {
        Self {
            max_iters,
            step_size,
            early_stop_eps,
            early_stop_min_iters,
            hidden_dim,
            num_layers,
            spectral_iters,
            prox_tau,
            box_lower,
            box_upper,
            max_state_norm,
            temporal_lambda,
            ema_alpha,
        }
    }

    /// Validate configuration.
    fn validate(&self) -> PyResult<()> {
        let config: lim_rps::LimRpsConfig = self.into();
        config.validate().map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "LimRpsConfig(max_iters={}, hidden_dim={}, step_size={:.2})",
            self.max_iters, self.hidden_dim, self.step_size
        )
    }
}

impl From<&PyLimRpsConfig> for lim_rps::LimRpsConfig {
    fn from(py: &PyLimRpsConfig) -> Self {
        Self {
            max_iters: py.max_iters,
            step_size: py.step_size,
            early_stop_eps: py.early_stop_eps,
            early_stop_min_iters: py.early_stop_min_iters,
            hidden_dim: py.hidden_dim,
            num_layers: py.num_layers,
            spectral_iters: py.spectral_iters,
            prox_mode: lim_rps::ProxMode::L2, // Default
            prox_tau: py.prox_tau,
            box_lower: py.box_lower,
            box_upper: py.box_upper,
            max_state_norm: py.max_state_norm,
            use_metric: false,
            use_step_field: false,
            temporal_lambda: py.temporal_lambda,
            ema_alpha: py.ema_alpha,
        }
    }
}

impl From<&lim_rps::LimRpsConfig> for PyLimRpsConfig {
    fn from(config: &lim_rps::LimRpsConfig) -> Self {
        Self {
            max_iters: config.max_iters,
            step_size: config.step_size,
            early_stop_eps: config.early_stop_eps,
            early_stop_min_iters: config.early_stop_min_iters,
            hidden_dim: config.hidden_dim,
            num_layers: config.num_layers,
            spectral_iters: config.spectral_iters,
            prox_tau: config.prox_tau,
            box_lower: config.box_lower,
            box_upper: config.box_upper,
            max_state_norm: config.max_state_norm,
            temporal_lambda: config.temporal_lambda,
            ema_alpha: config.ema_alpha,
        }
    }
}

/// Result from LIM-RPS equilibrium solve.
///
/// Attributes:
///     x_star: Equilibrium latent vector
///     residuals: Per-iteration residuals
///     k_used: Number of iterations used
///     converged: Whether early stopping triggered
///     final_residual: Final iteration residual
#[pyclass(name = "LimRpsResult")]
#[derive(Clone)]
pub struct PyLimRpsResult {
    inner: lim_rps::LimRpsResult,
}

#[pymethods]
impl PyLimRpsResult {
    #[getter]
    fn x_star<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        PyArray1::from_vec(py, self.inner.x_star.clone())
    }

    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        PyArray1::from_vec(py, self.inner.residuals.clone())
    }

    #[getter]
    fn k_used(&self) -> usize {
        self.inner.k_used
    }

    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    #[getter]
    fn final_residual(&self) -> f32 {
        self.inner.final_residual
    }

    fn __repr__(&self) -> String {
        format!(
            "LimRpsResult(k_used={}, converged={}, final_res={:.6})",
            self.inner.k_used, self.inner.converged, self.inner.final_residual
        )
    }
}

/// Rich latent state output with dynamics.
///
/// Contains the equilibrium latent plus computed dynamics like
/// position, velocity, curvature, periodicity, and somatic qualities.
///
/// Attributes:
///     x_star: Equilibrium latent vector
///     position: 2D projected position [x, y]
///     velocity: 2D velocity [vx, vy]
///     acceleration: 2D acceleration [ax, ay]
///     norm: L2 norm of equilibrium
///     micro_tension: Residual-based tension
///     curvature: Path curvature κ
///     periodicity: Estimated periodicity [0, 1]
///     internal_tempo: Estimated internal tempo (BPM)
///     phase: Current phase [0, 1]
///     grounding: Somatic grounding [0, 1]
///     verticality: Somatic verticality [0, 1]
///     tension: Somatic tension [0, 1]
///     coherence: Somatic coherence [0, 1]
#[pyclass(name = "LatentState")]
#[derive(Clone)]
pub struct PyLatentState {
    inner: lim_rps::LatentState,
}

#[pymethods]
impl PyLatentState {
    #[getter]
    fn x_star<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        PyArray1::from_vec(py, self.inner.x_star.clone())
    }

    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        PyArray1::from_vec(py, self.inner.position.to_vec())
    }

    #[getter]
    fn velocity<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        PyArray1::from_vec(py, self.inner.velocity.to_vec())
    }

    #[getter]
    fn acceleration<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        PyArray1::from_vec(py, self.inner.acceleration.to_vec())
    }

    #[getter]
    fn norm(&self) -> f32 {
        self.inner.norm
    }

    #[getter]
    fn micro_tension(&self) -> f32 {
        self.inner.micro_tension
    }

    #[getter]
    fn rotational_energy(&self) -> f32 {
        self.inner.rotational_energy
    }

    #[getter]
    fn curvature(&self) -> f32 {
        self.inner.curvature
    }

    #[getter]
    fn curvature_rate(&self) -> f32 {
        self.inner.curvature_rate
    }

    #[getter]
    fn periodicity(&self) -> f32 {
        self.inner.periodicity
    }

    #[getter]
    fn internal_tempo(&self) -> f32 {
        self.inner.internal_tempo
    }

    #[getter]
    fn phase(&self) -> f32 {
        self.inner.phase
    }

    #[getter]
    fn grounding(&self) -> f32 {
        self.inner.grounding
    }

    #[getter]
    fn verticality(&self) -> f32 {
        self.inner.verticality
    }

    #[getter]
    fn tension(&self) -> f32 {
        self.inner.tension
    }

    #[getter]
    fn coherence(&self) -> f32 {
        self.inner.coherence
    }

    #[getter]
    fn residual(&self) -> f32 {
        self.inner.residual
    }

    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    #[getter]
    fn frame_index(&self) -> u64 {
        self.inner.frame_index
    }

    #[getter]
    fn predicted_position<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        PyArray1::from_vec(py, self.inner.predicted_position.to_vec())
    }

    #[getter]
    fn prediction_confidence(&self) -> f32 {
        self.inner.prediction_confidence
    }

    fn __repr__(&self) -> String {
        format!(
            "LatentState(norm={:.3}, phase={:.2}, curvature={:.3})",
            self.inner.norm, self.inner.phase, self.inner.curvature
        )
    }
}

/// LIM-RPS equilibrium solver.
///
/// Solves for equilibrium latents using fixed-point iteration with
/// a 1-Lipschitz cross-modal operator.
///
/// Args:
///     total_dim: Dimension of the latent space
///     config: Solver configuration (optional, uses defaults if not provided)
///
/// Example:
///     >>> solver = LimRpsSolver(128)
///     >>> encoder = np.random.randn(128).astype(np.float32)
///     >>> result = solver.solve(encoder)
///     >>> print(result.x_star.shape)
///     (128,)
#[pyclass(name = "LimRpsSolver")]
pub struct PyLimRpsSolver {
    inner: lim_rps::LimRpsSolver,
}

#[pymethods]
impl PyLimRpsSolver {
    #[new]
    #[pyo3(signature = (total_dim, config=None))]
    fn new(total_dim: usize, config: Option<PyLimRpsConfig>) -> Self {
        let rust_config = config
            .as_ref()
            .map(|c| c.into())
            .unwrap_or_default();
        Self {
            inner: lim_rps::LimRpsSolver::new(total_dim, rust_config),
        }
    }

    /// Solve for equilibrium given encoder latents.
    ///
    /// Args:
    ///     encoder: Raw encoder output latent vector
    ///     z_init: Optional initial guess for warm start
    ///
    /// Returns:
    ///     LimRpsResult with equilibrium and convergence info
    #[pyo3(signature = (encoder, z_init=None))]
    fn solve(
        &self,
        encoder: PyReadonlyArray1<f32>,
        z_init: Option<PyReadonlyArray1<f32>>,
    ) -> PyLimRpsResult {
        let encoder_slice = encoder.as_slice().unwrap();
        let z_init_vec = z_init.map(|z| z.as_slice().unwrap().to_vec());
        let result = self.inner.solve(
            encoder_slice,
            z_init_vec.as_deref(),
        );
        PyLimRpsResult { inner: result }
    }

    /// Solve with warm start from previous frame.
    ///
    /// Uses the previous equilibrium as initial guess.
    fn solve_warm(&mut self, encoder: PyReadonlyArray1<f32>) -> PyLimRpsResult {
        let encoder_slice = encoder.as_slice().unwrap();
        let result = self.inner.solve_warm(encoder_slice);
        PyLimRpsResult { inner: result }
    }

    /// Get configuration.
    fn config(&self) -> PyLimRpsConfig {
        self.inner.config().into()
    }

    /// Get total latent dimension.
    fn total_dim(&self) -> usize {
        self.inner.total_dim()
    }

    /// Reset warm start state.
    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        format!(
            "LimRpsSolver(total_dim={}, max_iters={})",
            self.inner.total_dim(),
            self.inner.config().max_iters
        )
    }
}

/// High-level LIM-RPS processor with dynamics analysis.
///
/// Combines solver with dynamics analyzer for rich latent state output.
///
/// Args:
///     total_dim: Dimension of the latent space
///     config: Solver configuration (optional)
///     tempo_bpm: Initial tempo for phase tracking (default: 120.0)
///
/// Example:
///     >>> processor = LimRpsProcessor(128, tempo_bpm=140.0)
///     >>> for i, encoder in enumerate(encoder_sequence):
///     ...     state = processor.process(encoder, i * 20_000)  # 20ms frames
///     ...     print(f"Phase: {state.phase:.2f}, Curvature: {state.curvature:.3f}")
#[pyclass(name = "LimRpsProcessor")]
pub struct PyLimRpsProcessor {
    inner: lim_rps::LimRpsProcessor,
}

#[pymethods]
impl PyLimRpsProcessor {
    #[new]
    #[pyo3(signature = (total_dim, config=None, tempo_bpm=120.0))]
    fn new(total_dim: usize, config: Option<PyLimRpsConfig>, tempo_bpm: f32) -> Self {
        let rust_config = config
            .as_ref()
            .map(|c| c.into())
            .unwrap_or_default();
        Self {
            inner: lim_rps::LimRpsProcessor::new(total_dim, rust_config, tempo_bpm),
        }
    }

    /// Process a single frame.
    ///
    /// Args:
    ///     encoder: Raw encoder output latent vector
    ///     timestamp_micros: Frame timestamp in microseconds
    ///
    /// Returns:
    ///     LatentState with equilibrium and dynamics
    fn process(&mut self, encoder: PyReadonlyArray1<f32>, timestamp_micros: u64) -> PyLatentState {
        let encoder_slice = encoder.as_slice().unwrap();
        let state = self.inner.process(encoder_slice, timestamp_micros);
        PyLatentState { inner: state }
    }

    /// Process without warm start.
    fn process_cold(&mut self, encoder: PyReadonlyArray1<f32>, timestamp_micros: u64) -> PyLatentState {
        let encoder_slice = encoder.as_slice().unwrap();
        let state = self.inner.process_cold(encoder_slice, timestamp_micros);
        PyLatentState { inner: state }
    }

    /// Set tempo for phase tracking.
    fn set_tempo(&mut self, bpm: f32) {
        self.inner.set_tempo(bpm);
    }

    /// Get current tempo.
    fn tempo(&self) -> f32 {
        self.inner.tempo()
    }

    /// Get current phase [0, 1].
    fn phase(&self) -> f32 {
        self.inner.phase()
    }

    /// Get periodicity estimate [0, 1].
    fn periodicity(&self) -> f32 {
        self.inner.periodicity()
    }

    /// Get estimated internal tempo (BPM).
    fn internal_tempo(&self) -> f32 {
        self.inner.internal_tempo()
    }

    /// Get frame count.
    fn frame_count(&self) -> u64 {
        self.inner.frame_count()
    }

    /// Reset all state.
    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        format!(
            "LimRpsProcessor(frames={}, tempo={:.1})",
            self.inner.frame_count(),
            self.inner.tempo()
        )
    }
}

// =============================================================================
// Module Definition
// =============================================================================

/// High-performance Rust kernels for cc-core real-time audio processing.
///
/// Classes:
///     LockFreeRingBuffer: SPSC ring buffer with atomic operations
///     SlewLimiter: First-order parameter smoothing
///     SecondOrderSlewLimiter: Mass-spring-damper style smoothing
///     OneEuroFilter: Adaptive low-pass filtering
///     EquilibriumKernelConfig: Configuration for equilibrium solver
///     LinearOperator: Linear operator for equilibrium solver
///     StabilityHealthCheck: Health check results
///     ConvergenceEstimate: Convergence time estimate
///     LimRpsConfig: LIM-RPS solver configuration
///     LimRpsSolver: LIM-RPS equilibrium solver
///     LimRpsResult: LIM-RPS solve result
///     LatentState: Rich latent state with dynamics
///     LimRpsProcessor: High-level LIM-RPS processor
///
/// Functions:
///     box_project, l2_prox, l2_prox_pull, ema, ema_array: Proximal ops
///     contraction_ratio, headroom, stability_health_check: Stability metrics
///     solve_linear_equilibrium: Fast equilibrium solver
#[pymodule]
fn cc_core_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // Ring buffer
    m.add_class::<PyLockFreeRingBuffer>()?;

    // Slew limiters
    m.add_class::<PySlewLimiter>()?;
    m.add_class::<PySecondOrderSlewLimiter>()?;

    // Filters
    m.add_class::<PyOneEuroFilter>()?;

    // Proximal ops
    m.add_function(wrap_pyfunction!(box_project, m)?)?;
    m.add_function(wrap_pyfunction!(l2_prox, m)?)?;
    m.add_function(wrap_pyfunction!(l2_prox_pull, m)?)?;
    m.add_function(wrap_pyfunction!(ema, m)?)?;
    m.add_function(wrap_pyfunction!(ema_array, m)?)?;
    m.add_function(wrap_pyfunction!(l2_norm, m)?)?;
    m.add_function(wrap_pyfunction!(l2_distance, m)?)?;
    m.add_function(wrap_pyfunction!(clip_state_norm, m)?)?;

    // Stability metrics
    m.add_class::<PyStabilityHealthCheck>()?;
    m.add_class::<PyConvergenceEstimate>()?;
    m.add_function(wrap_pyfunction!(contraction_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(headroom, m)?)?;
    m.add_function(wrap_pyfunction!(stability_health_check, m)?)?;
    m.add_function(wrap_pyfunction!(compute_lipschitz_estimate, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_convergence_time, m)?)?;

    // Equilibrium kernel
    m.add_class::<PyEquilibriumKernelConfig>()?;
    m.add_class::<PyEquilibriumResult>()?;
    m.add_class::<PyLinearOperator>()?;
    m.add_function(wrap_pyfunction!(solve_linear_equilibrium, m)?)?;

    // LIM-RPS solver (unified cross-modal equilibrium)
    m.add_class::<PyProxMode>()?;
    m.add_class::<PyLimRpsConfig>()?;
    m.add_class::<PyLimRpsSolver>()?;
    m.add_class::<PyLimRpsResult>()?;
    m.add_class::<PyLatentState>()?;
    m.add_class::<PyLimRpsProcessor>()?;

    // Module info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
