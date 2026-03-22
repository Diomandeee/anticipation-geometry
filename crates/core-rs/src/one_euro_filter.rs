//! One-Euro filter for smooth, low-latency signal tracking.
//!
//! Combines position and velocity smoothing with an adaptive cutoff:
//!
//! ```text
//! dx = (x - x_prev) / dt
//! dx_smooth = LP(dx; d_cutoff)
//! cutoff = min_cutoff + β * |dx_smooth|
//! x_filtered = LP(x; cutoff)
//! ```
//!
//! # Reference
//!
//! Casiez et al. (2012), "1€ Filter: A Simple Speed-based Low-pass
//! Filter for Noisy Input in Interactive Systems."
//!
//! # Use Cases
//!
//! - Control signals for UI/motion
//! - Motion tracking values where responsiveness and smoothness are both needed
//! - Human input device filtering (mice, stylus, touch)

use std::f32::consts::PI;

/// One-Euro filter for adaptive low-pass filtering.
///
/// The filter automatically adjusts its cutoff frequency based on the
/// rate of change of the input signal:
///
/// - Slow signals → low cutoff → smooth output
/// - Fast signals → high cutoff → responsive output
///
/// # Parameters
///
/// - `freq`: Expected sample rate (Hz)
/// - `min_cutoff`: Minimum cutoff frequency (Hz). Lower → more smoothing
/// - `beta`: Speed coefficient. Higher → more responsive to fast changes
/// - `d_cutoff`: Cutoff for derivative smoothing (Hz)
///
/// # Example
///
/// ```
/// use cc_core_rs::OneEuroFilter;
///
/// let mut filter = OneEuroFilter::new(100.0, 1.0, 0.0, 1.0);
///
/// // Filter noisy input
/// let noisy_input = [0.0, 0.1, 0.05, 0.15, 0.1, 0.2];
/// for &x in &noisy_input {
///     let filtered = filter.filter(x, None);
///     // Output is smoothed
/// }
/// ```
#[derive(Debug, Clone)]
pub struct OneEuroFilter {
    /// Expected nominal sample rate (Hz)
    freq: f32,
    /// Minimum cutoff frequency (Hz)
    min_cutoff: f32,
    /// Speed coefficient
    beta: f32,
    /// Cutoff for derivative smoothing (Hz)
    d_cutoff: f32,
    /// Previous filtered position
    x_prev: Option<f32>,
    /// Previous smoothed derivative
    dx_prev: f32,
    /// Previous timestamp
    t_prev: Option<f32>,
}

impl OneEuroFilter {
    /// Create a new One-Euro filter.
    ///
    /// # Arguments
    ///
    /// * `freq` - Expected nominal sample rate (Hz)
    /// * `min_cutoff` - Minimum cutoff frequency (Hz). Lower → more smoothing
    /// * `beta` - Speed coefficient. Higher → more responsive to fast changes
    /// * `d_cutoff` - Cutoff for derivative smoothing (Hz)
    ///
    /// # Panics
    ///
    /// Panics if `freq` is not positive.
    pub fn new(freq: f32, min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self {
        assert!(freq > 0.0, "freq must be > 0");

        Self {
            freq,
            min_cutoff,
            beta,
            d_cutoff,
            x_prev: None,
            dx_prev: 0.0,
            t_prev: None,
        }
    }

    /// Create with default parameters for 60Hz motion tracking.
    pub fn for_motion_tracking() -> Self {
        Self::new(60.0, 1.0, 0.007, 1.0)
    }

    /// Create with default parameters for 100Hz control signals.
    pub fn for_control_signals() -> Self {
        Self::new(100.0, 1.0, 0.0, 1.0)
    }

    /// Filter one sample.
    ///
    /// # Arguments
    ///
    /// * `x` - Input value at time t
    /// * `t` - Optional timestamp (seconds). If None, assumes uniform dt = 1/freq
    ///
    /// # Returns
    ///
    /// Filtered value
    #[inline]
    pub fn filter(&mut self, x: f32, t: Option<f32>) -> f32 {
        // First sample: no filtering, just initialize
        if self.x_prev.is_none() {
            self.x_prev = Some(x);
            self.t_prev = t.or(Some(0.0));
            self.dx_prev = 0.0;
            return x;
        }

        // Compute dt
        let dt = match (t, self.t_prev) {
            (Some(t_now), Some(t_prev)) => {
                let dt = t_now - t_prev;
                if dt > 0.0 { dt } else { 1.0 / self.freq }
            }
            _ => 1.0 / self.freq,
        };

        let x_prev = self.x_prev.unwrap();

        // Estimate derivative
        let dx = (x - x_prev) / dt;

        // Smooth derivative
        let alpha_d = Self::smoothing_factor(dt, self.d_cutoff);
        let dx_smooth = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev;

        // Adaptive cutoff
        let cutoff = self.min_cutoff + self.beta * dx_smooth.abs();

        // Smooth position
        let alpha = Self::smoothing_factor(dt, cutoff);
        let x_filtered = alpha * x + (1.0 - alpha) * x_prev;

        // Update state
        self.x_prev = Some(x_filtered);
        self.dx_prev = dx_smooth;
        self.t_prev = t.or_else(|| self.t_prev.map(|t| t + dt));

        x_filtered
    }

    /// Compute exponential smoothing factor from cutoff frequency.
    #[inline]
    fn smoothing_factor(dt: f32, cutoff: f32) -> f32 {
        let cutoff = cutoff.max(1e-6);
        let tau = 1.0 / (2.0 * PI * cutoff);
        1.0 / (1.0 + tau / dt)
    }

    /// Reset filter state.
    #[inline]
    pub fn reset(&mut self) {
        self.x_prev = None;
        self.dx_prev = 0.0;
        self.t_prev = None;
    }

    /// Get current filtered value (or None if not initialized).
    #[inline]
    pub fn get_current(&self) -> Option<f32> {
        self.x_prev
    }

    /// Get current smoothed derivative.
    #[inline]
    pub fn get_derivative(&self) -> f32 {
        self.dx_prev
    }

    /// Set minimum cutoff frequency.
    #[inline]
    pub fn set_min_cutoff(&mut self, min_cutoff: f32) {
        self.min_cutoff = min_cutoff;
    }

    /// Set beta (speed coefficient).
    #[inline]
    pub fn set_beta(&mut self, beta: f32) {
        self.beta = beta;
    }

    /// Set derivative cutoff.
    #[inline]
    pub fn set_d_cutoff(&mut self, d_cutoff: f32) {
        self.d_cutoff = d_cutoff;
    }

    /// Set expected sample rate.
    #[inline]
    pub fn set_freq(&mut self, freq: f32) {
        assert!(freq > 0.0);
        self.freq = freq;
    }

    /// Process a buffer of samples (assuming uniform timing).
    pub fn filter_buffer(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        for (x, out) in input.iter().zip(output.iter_mut()) {
            *out = self.filter(*x, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_sample_passthrough() {
        let mut filter = OneEuroFilter::new(100.0, 1.0, 0.0, 1.0);

        let result = filter.filter(42.0, None);
        assert_eq!(result, 42.0); // First sample is passed through
    }

    #[test]
    fn test_smoothing_reduces_noise() {
        let mut filter = OneEuroFilter::new(100.0, 1.0, 0.0, 1.0);

        // Noisy signal around 1.0
        let noisy = [1.0, 1.2, 0.8, 1.1, 0.9, 1.0, 1.15, 0.85, 1.05, 0.95];

        let mut filtered = Vec::new();
        for &x in &noisy {
            filtered.push(filter.filter(x, None));
        }

        // Variance of filtered should be less than variance of input
        let input_var = variance(&noisy);
        let output_var = variance(&filtered);

        assert!(output_var < input_var, "Filtering should reduce variance");
    }

    #[test]
    fn test_beta_increases_responsiveness() {
        // Low beta (smoother)
        let mut filter_smooth = OneEuroFilter::new(100.0, 1.0, 0.0, 1.0);
        // High beta (more responsive)
        let mut filter_responsive = OneEuroFilter::new(100.0, 1.0, 1.0, 1.0);

        // Step input
        let step = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let mut smooth_out = Vec::new();
        let mut responsive_out = Vec::new();

        for &x in &step {
            smooth_out.push(filter_smooth.filter(x, None));
            responsive_out.push(filter_responsive.filter(x, None));
        }

        // After step, responsive filter should be closer to 1.0
        let smooth_final = *smooth_out.last().unwrap();
        let responsive_final = *responsive_out.last().unwrap();

        // Both should be approaching 1.0, but responsive should be faster
        assert!(responsive_final >= smooth_final - 0.01);
    }

    #[test]
    fn test_reset() {
        let mut filter = OneEuroFilter::new(100.0, 1.0, 0.0, 1.0);

        filter.filter(1.0, None);
        filter.filter(2.0, None);

        assert!(filter.get_current().is_some());

        filter.reset();
        assert!(filter.get_current().is_none());
    }

    fn variance(data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32
    }
}
