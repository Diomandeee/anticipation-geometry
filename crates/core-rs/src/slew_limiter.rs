//! Slew rate limiters for smooth parameter interpolation in audio callbacks.
//!
//! Prevents abrupt parameter changes that can cause clicks, pops, or
//! discontinuities in audio synthesis.
//!
//! This module provides:
//! - [`SlewLimiter`]: First-order (RC-style) rate limiting
//! - [`SecondOrderSlewLimiter`]: Mass-spring-damper style smoothing

use std::f32::consts::PI;

/// Configuration for first-order slew limiter.
#[derive(Debug, Clone, Copy)]
pub struct SlewLimiterConfig {
    /// Maximum absolute change per sample (symmetric)
    pub max_delta_per_sample: f32,
    /// Optional specific maximum delta for rising edges
    pub rise_rate: Option<f32>,
    /// Optional specific maximum delta for falling edges
    pub fall_rate: Option<f32>,
    /// Audio sample rate in Hz
    pub sample_rate_hz: f32,
}

impl Default for SlewLimiterConfig {
    fn default() -> Self {
        Self {
            max_delta_per_sample: 0.001,
            rise_rate: None,
            fall_rate: None,
            sample_rate_hz: 48000.0,
        }
    }
}

/// First-order slew rate limiter for parameter smoothing.
///
/// Limits the rate of change of a parameter to prevent sudden jumps.
/// Essential for glitch-free audio parameter modulation.
///
/// # Example
///
/// ```
/// use cc_core_rs::SlewLimiter;
///
/// let mut limiter = SlewLimiter::new(0.001, 48000.0);
///
/// // Smooth transition from 0 to 1
/// let mut value = 0.0;
/// for _ in 0..1000 {
///     value = limiter.process_scalar(1.0);
/// }
/// assert!(value > 0.9); // Approaching target
/// ```
#[derive(Debug, Clone)]
pub struct SlewLimiter {
    config: SlewLimiterConfig,
    current_value: f32,
}

impl SlewLimiter {
    /// Create a new first-order slew limiter.
    ///
    /// # Arguments
    ///
    /// * `max_delta_per_sample` - Maximum change per sample
    /// * `sample_rate_hz` - Audio sample rate
    pub fn new(max_delta_per_sample: f32, sample_rate_hz: f32) -> Self {
        Self {
            config: SlewLimiterConfig {
                max_delta_per_sample,
                sample_rate_hz,
                ..Default::default()
            },
            current_value: 0.0,
        }
    }

    /// Create with asymmetric rise/fall rates.
    pub fn with_asymmetric(
        rise_rate: f32,
        fall_rate: f32,
        sample_rate_hz: f32,
    ) -> Self {
        Self {
            config: SlewLimiterConfig {
                max_delta_per_sample: rise_rate, // Default to rise
                rise_rate: Some(rise_rate),
                fall_rate: Some(fall_rate),
                sample_rate_hz,
            },
            current_value: 0.0,
        }
    }

    /// Create from a time constant (63% rise time).
    pub fn from_time_constant(time_constant_ms: f32, sample_rate_hz: f32) -> Self {
        let tau_samples = time_constant_ms * sample_rate_hz / 1000.0;
        let max_delta = if tau_samples <= 1.0 {
            1.0
        } else {
            1.0 - (-1.0 / tau_samples).exp()
        };

        Self::new(max_delta, sample_rate_hz)
    }

    /// Process a single scalar parameter.
    ///
    /// # Arguments
    ///
    /// * `target` - Target value to approach
    ///
    /// # Returns
    ///
    /// Slew-limited value for this sample
    #[inline]
    pub fn process_scalar(&mut self, target: f32) -> f32 {
        let delta = target - self.current_value;

        // Determine slew rate (asymmetric if configured)
        let max_delta = if delta > 0.0 {
            self.config.rise_rate.unwrap_or(self.config.max_delta_per_sample)
        } else if delta < 0.0 {
            self.config.fall_rate.unwrap_or(self.config.max_delta_per_sample)
        } else {
            self.config.max_delta_per_sample
        };

        // Clamp delta and update
        let delta_clamped = delta.clamp(-max_delta, max_delta);
        self.current_value += delta_clamped;

        self.current_value
    }

    /// Process a buffer of samples with constant target.
    ///
    /// Generates a smooth ramp toward the target value.
    #[inline]
    pub fn process_buffer(&mut self, target: f32, output: &mut [f32]) {
        for sample in output.iter_mut() {
            *sample = self.process_scalar(target);
        }
    }

    /// Reset the limiter to a new value instantly.
    #[inline]
    pub fn reset(&mut self, value: f32) {
        self.current_value = value;
    }

    /// Get the current internal value without updating.
    #[inline]
    pub fn get_current(&self) -> f32 {
        self.current_value
    }

    /// Set the maximum delta per sample.
    #[inline]
    pub fn set_max_delta(&mut self, max_delta: f32) {
        self.config.max_delta_per_sample = max_delta;
    }

    /// Set slew rate based on a time constant (63% rise time).
    pub fn set_time_constant(&mut self, time_constant_ms: f32) {
        let tau_samples = time_constant_ms * self.config.sample_rate_hz / 1000.0;
        self.config.max_delta_per_sample = if tau_samples <= 1.0 {
            1.0
        } else {
            1.0 - (-1.0 / tau_samples).exp()
        };
    }
}

// =============================================================================
// Second-Order Slew Limiter
// =============================================================================

/// Configuration for second-order slew limiter.
#[derive(Debug, Clone, Copy)]
pub struct SecondOrderSlewLimiterConfig {
    /// Natural frequency in Hz
    pub natural_freq_hz: f32,
    /// Damping ratio (1.0 = critically damped)
    pub damping_ratio: f32,
    /// Audio sample rate in Hz
    pub sample_rate_hz: f32,
}

impl Default for SecondOrderSlewLimiterConfig {
    fn default() -> Self {
        Self {
            natural_freq_hz: 10.0,
            damping_ratio: 1.0,
            sample_rate_hz: 48000.0,
        }
    }
}

/// Second-order slew limiter for parameter smoothing.
///
/// Models a critically damped or under/over-damped second-order system
/// (like a mass-spring-damper):
///
/// ```text
/// d²y/dt² + 2ζωₙ dy/dt + ωₙ² y = ωₙ² x
/// ```
///
/// Where:
/// - x = target signal
/// - y = smoothed output
/// - ωₙ = natural frequency (rad/s)
/// - ζ = damping ratio
///
/// # Example
///
/// ```
/// use cc_core_rs::SecondOrderSlewLimiter;
///
/// let mut limiter = SecondOrderSlewLimiter::new(10.0, 1.0, 48000.0);
///
/// // Smooth, inertial transition
/// for _ in 0..4800 {
///     let value = limiter.process_scalar(1.0);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SecondOrderSlewLimiter {
    config: SecondOrderSlewLimiterConfig,
    /// Position (output value)
    y: f32,
    /// Velocity (first derivative)
    y_dot: f32,
    /// Precomputed omega_n
    omega_n: f32,
    /// Precomputed dt
    dt: f32,
}

impl SecondOrderSlewLimiter {
    /// Create a new second-order slew limiter.
    ///
    /// # Arguments
    ///
    /// * `natural_freq_hz` - Natural frequency in Hz (typical: 1-50)
    /// * `damping_ratio` - Damping ratio (1.0 = critically damped)
    /// * `sample_rate_hz` - Audio sample rate
    pub fn new(natural_freq_hz: f32, damping_ratio: f32, sample_rate_hz: f32) -> Self {
        assert!(natural_freq_hz > 0.0, "natural_freq_hz must be > 0");
        assert!(sample_rate_hz > 0.0, "sample_rate_hz must be > 0");

        let omega_n = 2.0 * PI * natural_freq_hz;
        let dt = 1.0 / sample_rate_hz;

        Self {
            config: SecondOrderSlewLimiterConfig {
                natural_freq_hz,
                damping_ratio,
                sample_rate_hz,
            },
            y: 0.0,
            y_dot: 0.0,
            omega_n,
            dt,
        }
    }

    /// Process a scalar target through the second-order smoother.
    ///
    /// # Arguments
    ///
    /// * `target` - Target value to move toward
    ///
    /// # Returns
    ///
    /// Smoothed value
    #[inline]
    pub fn process_scalar(&mut self, target: f32) -> f32 {
        let x = target;
        let zeta = self.config.damping_ratio;
        let omega_n = self.omega_n;
        let dt = self.dt;

        // y_ddot = ωₙ² * (x - y) - 2ζωₙ * y_dot
        let y_ddot = (omega_n * omega_n) * (x - self.y)
                   - 2.0 * zeta * omega_n * self.y_dot;

        // Explicit Euler integration
        self.y_dot += y_ddot * dt;
        self.y += self.y_dot * dt;

        self.y
    }

    /// Process a buffer of samples with constant target.
    #[inline]
    pub fn process_buffer(&mut self, target: f32, output: &mut [f32]) {
        for sample in output.iter_mut() {
            *sample = self.process_scalar(target);
        }
    }

    /// Reset the limiter to a new value, zeroing velocity.
    #[inline]
    pub fn reset(&mut self, value: f32) {
        self.y = value;
        self.y_dot = 0.0;
    }

    /// Get the current output value.
    #[inline]
    pub fn get_current(&self) -> f32 {
        self.y
    }

    /// Get the current velocity.
    #[inline]
    pub fn get_velocity(&self) -> f32 {
        self.y_dot
    }

    /// Update natural frequency.
    pub fn set_natural_frequency(&mut self, freq_hz: f32) {
        assert!(freq_hz > 0.0);
        self.config.natural_freq_hz = freq_hz;
        self.omega_n = 2.0 * PI * freq_hz;
    }

    /// Update damping ratio.
    pub fn set_damping_ratio(&mut self, zeta: f32) {
        self.config.damping_ratio = zeta;
    }

    /// Update sample rate.
    pub fn set_sample_rate(&mut self, sample_rate_hz: f32) {
        assert!(sample_rate_hz > 0.0);
        self.config.sample_rate_hz = sample_rate_hz;
        self.dt = 1.0 / sample_rate_hz;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slew_limiter_basic() {
        let mut limiter = SlewLimiter::new(0.1, 48000.0);

        // Start at 0, target 1
        let mut value = 0.0;
        for _ in 0..20 {
            value = limiter.process_scalar(1.0);
        }

        // Should be approaching 1
        assert!(value > 0.9);
        assert!(value <= 1.0);
    }

    #[test]
    fn test_slew_limiter_asymmetric() {
        let mut limiter = SlewLimiter::with_asymmetric(0.1, 0.01, 48000.0);

        // Fast rise
        limiter.reset(0.0);
        for _ in 0..15 {
            limiter.process_scalar(1.0);
        }
        let rise_value = limiter.get_current();

        // Slow fall
        for _ in 0..15 {
            limiter.process_scalar(0.0);
        }
        let fall_delta = rise_value - limiter.get_current();

        // Rise should be faster than fall
        assert!(rise_value > 0.9); // Fast rise
        assert!(fall_delta < 0.2);  // Slow fall
    }

    #[test]
    fn test_second_order_critically_damped() {
        let mut limiter = SecondOrderSlewLimiter::new(10.0, 1.0, 48000.0);

        // Step response
        let mut values = Vec::new();
        for _ in 0..4800 {
            values.push(limiter.process_scalar(1.0));
        }

        // Should approach 1 without overshooting (critically damped)
        let max_value = values.iter().cloned().fold(f32::MIN, f32::max);
        assert!(max_value <= 1.01); // No significant overshoot
        assert!(values.last().unwrap() > &0.95); // Close to target
    }

    #[test]
    fn test_second_order_underdamped() {
        let mut limiter = SecondOrderSlewLimiter::new(10.0, 0.3, 48000.0);

        // Step response should overshoot
        let mut max_value = 0.0f32;
        for _ in 0..9600 {
            let v = limiter.process_scalar(1.0);
            max_value = max_value.max(v);
        }

        // Underdamped should overshoot
        assert!(max_value > 1.0);
    }

    #[test]
    fn test_slew_limiter_reset() {
        let mut limiter = SlewLimiter::new(0.001, 48000.0);

        limiter.process_scalar(1.0);
        assert!(limiter.get_current() > 0.0);

        limiter.reset(0.5);
        assert_eq!(limiter.get_current(), 0.5);
    }
}
