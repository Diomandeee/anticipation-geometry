//! Rich Latent State for LIM-RPS.
//!
//! Extends the raw equilibrium output with derived quantities:
//! - 2D projection for visualization
//! - Velocity and acceleration
//! - Curvature and periodicity
//! - Somatic qualities (grounding, verticality, tension)
//! - Prediction for anticipation
//!
//! This is the unified output format used by the echelon and motion capture services.

use super::solver::LimRpsResult;

/// Rich latent state with derived dynamics.
///
/// This is the primary output format for real-time motion processing.
/// It extends the raw equilibrium with interpretable quantities.
#[derive(Debug, Clone, Default)]
pub struct LatentState {
    // ========================================================================
    // Core equilibrium output
    // ========================================================================

    /// Equilibrium latent vector z*.
    pub x_star: Vec<f32>,

    /// Final iteration residual.
    pub residual: f32,

    /// Whether solver converged (early stopping).
    pub converged: bool,

    /// Number of iterations used.
    pub iterations: usize,

    // ========================================================================
    // 2D projection (for visualization)
    // ========================================================================

    /// 2D position from latent projection.
    pub position: [f32; 2],

    /// 2D velocity (change in position per frame).
    pub velocity: [f32; 2],

    /// 2D acceleration (change in velocity per frame).
    pub acceleration: [f32; 2],

    // ========================================================================
    // Energy metrics
    // ========================================================================

    /// L2 norm of latent ||z*||.
    pub norm: f32,

    /// Micro-tension: high-frequency energy.
    pub micro_tension: f32,

    /// Rotational energy from angular velocity.
    pub rotational_energy: f32,

    /// Total kinetic energy (from velocity magnitude).
    pub kinetic_energy: f32,

    // ========================================================================
    // Curvature (trajectory shape)
    // ========================================================================

    /// Path curvature κ = |v × a| / |v|³.
    pub curvature: f32,

    /// Rate of change of curvature dκ/dt.
    pub curvature_rate: f32,

    // ========================================================================
    // Temporal / periodic features
    // ========================================================================

    /// Periodicity score [0, 1] from autocorrelation.
    pub periodicity: f32,

    /// Estimated internal tempo (BPM).
    pub internal_tempo: f32,

    /// Phase within the periodic cycle [0, 1].
    pub phase: f32,

    // ========================================================================
    // Somatic qualities
    // ========================================================================

    /// Grounding: how much energy is in the lower body.
    pub grounding: f32,

    /// Verticality: alignment with gravity axis.
    pub verticality: f32,

    /// Tension: overall muscular activation estimate.
    pub tension: f32,

    /// Coherence: how well body parts move together.
    pub coherence: f32,

    // ========================================================================
    // Prediction
    // ========================================================================

    /// Predicted 2D position (next frame).
    pub predicted_position: [f32; 2],

    /// Confidence in prediction [0, 1].
    pub prediction_confidence: f32,

    // ========================================================================
    // Timing
    // ========================================================================

    /// Timestamp in microseconds.
    pub timestamp_micros: u64,

    /// Frame index (for sequencing).
    pub frame_index: u64,
}

impl LatentState {
    /// Create a new empty latent state.
    pub fn new(dim: usize) -> Self {
        Self {
            x_star: vec![0.0; dim],
            ..Default::default()
        }
    }

    /// Create from solver result.
    pub fn from_result(result: LimRpsResult) -> Self {
        let dim = result.x_star.len();
        let position = if dim >= 2 {
            [result.x_star[0], result.x_star[1]]
        } else {
            [0.0, 0.0]
        };

        let norm = result
            .x_star
            .iter()
            .map(|&v| v * v)
            .sum::<f32>()
            .sqrt();

        Self {
            x_star: result.x_star,
            residual: result.final_residual,
            converged: result.converged,
            iterations: result.k_used,
            position,
            norm,
            ..Default::default()
        }
    }

    /// Create from solver result with previous state for dynamics.
    pub fn from_result_with_prev(result: LimRpsResult, prev: Option<&LatentState>) -> Self {
        let mut state = Self::from_result(result);

        if let Some(prev) = prev {
            // Compute velocity from position change
            state.velocity = [
                state.position[0] - prev.position[0],
                state.position[1] - prev.position[1],
            ];

            // Compute acceleration from velocity change
            state.acceleration = [
                state.velocity[0] - prev.velocity[0],
                state.velocity[1] - prev.velocity[1],
            ];

            // Compute curvature from velocity and acceleration
            state.curvature = compute_curvature(&state.velocity, &state.acceleration);
            state.curvature_rate = state.curvature - prev.curvature;

            // Kinetic energy from velocity
            let v_mag = (state.velocity[0].powi(2) + state.velocity[1].powi(2)).sqrt();
            state.kinetic_energy = 0.5 * v_mag.powi(2);

            // Simple prediction: linear extrapolation
            state.predicted_position = [
                state.position[0] + state.velocity[0],
                state.position[1] + state.velocity[1],
            ];

            // Prediction confidence based on velocity consistency
            let v_change = (state.acceleration[0].powi(2) + state.acceleration[1].powi(2)).sqrt();
            state.prediction_confidence = 1.0 / (1.0 + v_change);
        }

        state
    }

    /// Update projection from full latent.
    pub fn update_projection(&mut self, projection_matrix: Option<&[[f32; 2]]>) {
        if let Some(proj) = projection_matrix {
            // Project: position = proj @ x_star
            let mut pos = [0.0f32; 2];
            for (i, p) in proj.iter().enumerate() {
                if i < self.x_star.len() {
                    pos[0] += p[0] * self.x_star[i];
                    pos[1] += p[1] * self.x_star[i];
                }
            }
            self.position = pos;
        } else {
            // Default: use first two dimensions
            if self.x_star.len() >= 2 {
                self.position = [self.x_star[0], self.x_star[1]];
            }
        }
    }

    /// Get speed (magnitude of velocity).
    pub fn speed(&self) -> f32 {
        (self.velocity[0].powi(2) + self.velocity[1].powi(2)).sqrt()
    }

    /// Get angular velocity from position history.
    pub fn angular_velocity(&self) -> f32 {
        // Approximate from curvature and speed
        self.curvature * self.speed()
    }

    /// Check if motion is stationary (low velocity).
    pub fn is_stationary(&self, threshold: f32) -> bool {
        self.speed() < threshold
    }

    /// Get primary direction of motion.
    pub fn direction(&self) -> f32 {
        self.velocity[1].atan2(self.velocity[0])
    }

    /// Serialize to flat vector for neural network input.
    pub fn to_feature_vector(&self) -> Vec<f32> {
        let mut features = Vec::with_capacity(32);

        // Core latent (first 16 or all)
        let n = self.x_star.len().min(16);
        features.extend_from_slice(&self.x_star[..n]);
        features.resize(16, 0.0);

        // 2D dynamics
        features.extend_from_slice(&self.position);
        features.extend_from_slice(&self.velocity);
        features.extend_from_slice(&self.acceleration);

        // Scalar features
        features.push(self.norm);
        features.push(self.curvature);
        features.push(self.periodicity);
        features.push(self.tension);
        features.push(self.coherence);
        features.push(self.grounding);
        features.push(self.kinetic_energy);
        features.push(self.phase);

        features
    }

    /// Get a summary for debugging.
    pub fn summary(&self) -> String {
        format!(
            "LatentState {{ pos: [{:.2}, {:.2}], vel: {:.3}, curv: {:.3}, residual: {:.4}, iters: {} }}",
            self.position[0],
            self.position[1],
            self.speed(),
            self.curvature,
            self.residual,
            self.iterations
        )
    }
}

/// Compute path curvature from velocity and acceleration.
///
/// κ = |v × a| / |v|³
///
/// For 2D: |v × a| = |v_x * a_y - v_y * a_x|
fn compute_curvature(velocity: &[f32; 2], acceleration: &[f32; 2]) -> f32 {
    let v_mag = (velocity[0].powi(2) + velocity[1].powi(2)).sqrt();

    if v_mag < 1e-8 {
        return 0.0;
    }

    let cross = velocity[0] * acceleration[1] - velocity[1] * acceleration[0];
    cross.abs() / v_mag.powi(3)
}

/// Latent state history buffer for temporal analysis.
#[derive(Debug, Clone)]
pub struct LatentStateHistory {
    /// Ring buffer of states.
    states: Vec<LatentState>,
    /// Current write position.
    head: usize,
    /// Number of valid entries.
    len: usize,
    /// Maximum capacity.
    capacity: usize,
}

impl LatentStateHistory {
    /// Create a new history buffer.
    pub fn new(capacity: usize) -> Self {
        Self {
            states: Vec::with_capacity(capacity),
            head: 0,
            len: 0,
            capacity,
        }
    }

    /// Add a new state.
    pub fn push(&mut self, state: LatentState) {
        if self.states.len() < self.capacity {
            self.states.push(state);
            self.len = self.states.len();
        } else {
            self.states[self.head] = state;
            self.head = (self.head + 1) % self.capacity;
            self.len = self.capacity;
        }
    }

    /// Get the most recent state.
    pub fn latest(&self) -> Option<&LatentState> {
        if self.len == 0 {
            return None;
        }
        let idx = if self.head == 0 {
            self.len - 1
        } else {
            self.head - 1
        };
        Some(&self.states[idx])
    }

    /// Get state at offset from latest (0 = latest, 1 = previous, etc.).
    pub fn get(&self, offset: usize) -> Option<&LatentState> {
        if offset >= self.len {
            return None;
        }
        let idx = (self.head + self.capacity - 1 - offset) % self.capacity;
        Some(&self.states[idx])
    }

    /// Get all states in chronological order.
    pub fn iter(&self) -> impl Iterator<Item = &LatentState> + '_ {
        let len = self.len;
        let capacity = self.capacity;
        let head = self.head;

        (0..len).map(move |i| {
            let idx = if len < capacity {
                // Buffer not full, elements are at 0..len
                i
            } else {
                // Buffer full, head points to next write position (oldest element)
                (head + i) % capacity
            };
            &self.states[idx]
        })
    }

    /// Number of states in history.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Clear all history.
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    /// Compute mean position over history.
    pub fn mean_position(&self) -> [f32; 2] {
        if self.len == 0 {
            return [0.0, 0.0];
        }

        let mut sum = [0.0f32; 2];
        for state in self.iter() {
            sum[0] += state.position[0];
            sum[1] += state.position[1];
        }

        [sum[0] / self.len as f32, sum[1] / self.len as f32]
    }

    /// Compute position variance over history.
    pub fn position_variance(&self) -> f32 {
        if self.len < 2 {
            return 0.0;
        }

        let mean = self.mean_position();
        let mut var = 0.0f32;

        for state in self.iter() {
            let dx = state.position[0] - mean[0];
            let dy = state.position[1] - mean[1];
            var += dx * dx + dy * dy;
        }

        var / (self.len - 1) as f32
    }

    /// Estimate periodicity using autocorrelation.
    pub fn estimate_periodicity(&self) -> f32 {
        if self.len < 10 {
            return 0.0;
        }

        // Use position as signal
        let positions: Vec<[f32; 2]> = self.iter().map(|s| s.position).collect();

        // Compute autocorrelation at various lags
        let max_lag = (self.len / 2).min(50);
        let mut best_corr = 0.0f32;

        for lag in 2..max_lag {
            let mut sum = 0.0f32;
            let mut n = 0;

            for i in 0..(positions.len() - lag) {
                let p1 = &positions[i];
                let p2 = &positions[i + lag];
                sum += p1[0] * p2[0] + p1[1] * p2[1];
                n += 1;
            }

            if n > 0 {
                let corr = sum / n as f32;
                best_corr = best_corr.max(corr);
            }
        }

        // Normalize to [0, 1]
        (best_corr / self.position_variance().max(1e-6)).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latent_state_new() {
        let state = LatentState::new(16);
        assert_eq!(state.x_star.len(), 16);
        assert_eq!(state.residual, 0.0);
        assert!(!state.converged);
    }

    #[test]
    fn test_from_result() {
        let result = LimRpsResult {
            x_star: vec![1.0, 2.0, 3.0, 4.0],
            residuals: vec![0.1, 0.05, 0.02],
            k_used: 3,
            converged: true,
            final_residual: 0.02,
            mean_residual: 0.057,
        };

        let state = LatentState::from_result(result);

        assert_eq!(state.x_star.len(), 4);
        assert_eq!(state.position, [1.0, 2.0]);
        assert!(state.converged);
        assert_eq!(state.iterations, 3);
    }

    #[test]
    fn test_from_result_with_prev() {
        let result1 = LimRpsResult {
            x_star: vec![0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        let state1 = LatentState::from_result(result1);

        let result2 = LimRpsResult {
            x_star: vec![1.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        let state2 = LatentState::from_result_with_prev(result2, Some(&state1));

        assert_eq!(state2.velocity, [1.0, 0.0]);
        assert!(state2.speed() > 0.0);
    }

    #[test]
    fn test_compute_curvature() {
        // Straight line: curvature = 0
        let v1 = [1.0, 0.0];
        let a1 = [0.0, 0.0];
        assert!(compute_curvature(&v1, &a1) < 1e-6);

        // Circular motion: v perp to a
        let v2 = [1.0, 0.0];
        let a2 = [0.0, 1.0];
        let k = compute_curvature(&v2, &a2);
        assert!(k > 0.0);
    }

    #[test]
    fn test_feature_vector() {
        let mut state = LatentState::new(8);
        state.x_star = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        state.position = [0.5, 0.5];
        state.velocity = [0.1, 0.0];

        let features = state.to_feature_vector();

        // 16 (latent) + 2 (pos) + 2 (vel) + 2 (acc) + 8 (scalars) = 30
        assert!(features.len() >= 30);
    }

    #[test]
    fn test_history_ring_buffer() {
        let mut history = LatentStateHistory::new(3);

        // Push 5 states into capacity-3 buffer
        for i in 0..5 {
            let mut state = LatentState::new(4);
            state.position = [i as f32, 0.0];
            history.push(state);
        }

        assert_eq!(history.len(), 3);

        // Latest should be the most recent
        let latest = history.latest().unwrap();
        assert_eq!(latest.position[0], 4.0);

        // Previous should be one before
        let prev = history.get(1).unwrap();
        assert_eq!(prev.position[0], 3.0);
    }

    #[test]
    fn test_history_iteration() {
        let mut history = LatentStateHistory::new(5);

        for i in 0..3 {
            let mut state = LatentState::new(2);
            state.position = [i as f32, 0.0];
            history.push(state);
        }

        let positions: Vec<f32> = history.iter().map(|s| s.position[0]).collect();
        assert_eq!(positions, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_mean_position() {
        let mut history = LatentStateHistory::new(5);

        for i in 0..4 {
            let mut state = LatentState::new(2);
            state.position = [i as f32, 0.0];
            history.push(state);
        }

        let mean = history.mean_position();
        assert!((mean[0] - 1.5).abs() < 1e-6); // (0+1+2+3)/4
    }
}
