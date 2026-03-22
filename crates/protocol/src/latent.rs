//! Latent state representation
//!
//! Output from the LIM-RPS solver - the brain's internal representation of embodied movement.
//!
//! This is the "machine's perception of the performer" - a convergence of all sensors
//! and modalities into one coherent mathematical world.

use serde::{Deserialize, Serialize};

/// Geometric properties of the latent trajectory
///
/// These capture the shape and dynamics of movement in latent space,
/// independent of the raw sensor readings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LatentGeometry {
    /// Curvature of trajectory (how sharply the path bends)
    ///
    /// - 0.0 = straight line
    /// - Positive = curving (units: 1/distance)
    /// - High curvature = tight turns, sudden direction changes
    pub curvature: f32,

    /// Rate of curvature change (jerk in latent space)
    ///
    /// Measures how quickly the curvature itself is changing.
    /// Useful for detecting transitions and acceleration patterns.
    pub curvature_rate: f32,

    /// Divergence (expansion or contraction of flow)
    ///
    /// - Positive = expanding, energy increasing
    /// - Negative = contracting, energy decreasing
    /// - Near zero = stable circulation
    pub divergence: f32,

    /// Velocity magnitude in latent space (speed)
    pub velocity_magnitude: f32,

    /// Acceleration magnitude
    pub acceleration_magnitude: f32,

    /// Direction of movement (unit vector in latent space)
    ///
    /// Only valid when velocity_magnitude > threshold.
    /// None if stationary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub velocity_direction: Option<Vec<f32>>,
}

impl LatentGeometry {
    /// Create geometry with zeros
    pub fn zero() -> Self {
        Self {
            curvature: 0.0,
            curvature_rate: 0.0,
            divergence: 0.0,
            velocity_magnitude: 0.0,
            acceleration_magnitude: 0.0,
            velocity_direction: None,
        }
    }

    /// Check if movement is essentially stationary
    pub fn is_stationary(&self) -> bool {
        self.velocity_magnitude < 0.01
    }

    /// Check if trajectory is highly curved (tight turn)
    pub fn is_highly_curved(&self) -> bool {
        self.curvature.abs() > 1.0
    }

    /// Check if flow is diverging (expanding)
    pub fn is_diverging(&self) -> bool {
        self.divergence > 0.1
    }

    /// Check if flow is converging (contracting)
    pub fn is_converging(&self) -> bool {
        self.divergence < -0.1
    }
}

impl Default for LatentGeometry {
    fn default() -> Self {
        Self::zero()
    }
}

/// The latent state produced by LIM-RPS
///
/// This is the core representation of embodied movement, capturing:
/// - Position in latent space (the "where")
/// - Dynamics (the "how it's moving")
/// - Geometry (the "shape of the movement")
/// - Temporal patterns (the "rhythm")
///
/// All downstream processing (coherence, state machine, lexicon) operates on this.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LatentState {
    /// Timestamp in microseconds
    pub timestamp_us: i64,

    /// Latent vector z*(t) - the core position in latent space
    ///
    /// Dimensionality is typically 8-64. This is the fixed point solution
    /// from the proximal point solver - it represents where all sensor
    /// modalities "agree" the body is in abstract movement space.
    pub z: Vec<f32>,

    /// Velocity in latent space (dz/dt)
    ///
    /// Same dimensionality as z. Captures the direction and speed of movement.
    pub velocity: Vec<f32>,

    /// Acceleration in latent space (d²z/dt²)
    ///
    /// Same dimensionality as z. Useful for detecting sudden changes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acceleration: Option<Vec<f32>>,

    /// Geometric properties of the trajectory
    pub geometry: LatentGeometry,

    /// Overall latent norm (magnitude of z vector)
    ///
    /// Represents total "energy" in the latent space.
    pub norm: f32,

    /// Speed (magnitude of velocity)
    pub speed: f32,

    /// Emergent periodicity detected in the latent (0.0 - 1.0)
    ///
    /// Computed from autocorrelation of velocity or other latent signals.
    /// Values > 0.5 indicate rhythmic structure.
    pub periodicity: f32,

    /// Phase within periodic cycle (0.0 - 1.0)
    ///
    /// Only meaningful if periodicity > threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<f32>,

    /// Tension accumulation (0.0 - 1.0)
    ///
    /// Builds up during high-curvature or high-divergence movement.
    /// Releases during resolution phases.
    pub tension: f32,

    /// Coherence across modalities (0.0 - 1.0)
    ///
    /// How well do different sensors agree?
    /// - 1.0 = perfect agreement
    /// - 0.0 = complete disagreement (chaotic)
    pub coherence: f32,

    /// Grounding (connection to vertical axis, 0.0 - 1.0)
    ///
    /// Represents how "rooted" the movement feels.
    /// Computed from gravity alignment and vertical acceleration patterns.
    pub grounding: f32,

    /// Confidence in this state estimate (0.0 - 1.0)
    ///
    /// Based on sensor quality, fusion agreement, and solver convergence.
    pub confidence: f32,
}

impl LatentState {
    /// Create a new latent state with given dimensionality
    pub fn new(dim: usize, timestamp_us: i64) -> Self {
        Self {
            timestamp_us,
            z: vec![0.0; dim],
            velocity: vec![0.0; dim],
            acceleration: None,
            geometry: LatentGeometry::zero(),
            norm: 0.0,
            speed: 0.0,
            periodicity: 0.0,
            phase: None,
            tension: 0.0,
            coherence: 0.0,
            grounding: 0.5,
            confidence: 0.0,
        }
    }

    /// Get dimensionality of latent space
    pub fn dim(&self) -> usize {
        self.z.len()
    }

    /// Check if state is essentially zero (no movement detected)
    pub fn is_zero(&self) -> bool {
        self.norm < 0.01 && self.speed < 0.01
    }

    /// Check if state has high tension
    pub fn is_tense(&self) -> bool {
        self.tension > 0.7
    }

    /// Check if state is coherent across modalities
    pub fn is_coherent(&self) -> bool {
        self.coherence > 0.6
    }

    /// Check if state has rhythmic structure
    pub fn has_rhythm(&self) -> bool {
        self.periodicity > 0.5
    }

    /// Get 2D projection for visualization (first two dimensions)
    pub fn position_2d(&self) -> [f32; 2] {
        if self.z.len() >= 2 {
            [self.z[0], self.z[1]]
        } else {
            [0.0, 0.0]
        }
    }

    /// Get 2D velocity for visualization
    pub fn velocity_2d(&self) -> [f32; 2] {
        if self.velocity.len() >= 2 {
            [self.velocity[0], self.velocity[1]]
        } else {
            [0.0, 0.0]
        }
    }

    /// Compute distance to another latent state
    pub fn distance_to(&self, other: &LatentState) -> f32 {
        if self.dim() != other.dim() {
            return f32::INFINITY;
        }

        let mut sum_sq = 0.0;
        for (a, b) in self.z.iter().zip(other.z.iter()) {
            sum_sq += (a - b).powi(2);
        }
        sum_sq.sqrt()
    }

    /// Interpolate between two states (linear interpolation)
    ///
    /// t = 0.0 returns self, t = 1.0 returns other
    pub fn lerp(&self, other: &LatentState, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let s = 1.0 - t;

        let mut result = self.clone();

        // Interpolate z
        for i in 0..self.dim().min(other.dim()) {
            result.z[i] = s * self.z[i] + t * other.z[i];
        }

        // Interpolate velocity
        for i in 0..self.velocity.len().min(other.velocity.len()) {
            result.velocity[i] = s * self.velocity[i] + t * other.velocity[i];
        }

        // Interpolate scalars
        result.norm = s * self.norm + t * other.norm;
        result.speed = s * self.speed + t * other.speed;
        result.tension = s * self.tension + t * other.tension;
        result.coherence = s * self.coherence + t * other.coherence;
        result.periodicity = s * self.periodicity + t * other.periodicity;

        // Geometry
        result.geometry.curvature = s * self.geometry.curvature + t * other.geometry.curvature;
        result.geometry.divergence = s * self.geometry.divergence + t * other.geometry.divergence;
        result.geometry.velocity_magnitude = s * self.geometry.velocity_magnitude + t * other.geometry.velocity_magnitude;

        // Timestamp (use other's timestamp)
        result.timestamp_us = other.timestamp_us;

        result
    }
}

impl Default for LatentState {
    fn default() -> Self {
        Self::new(16, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latent_state_creation() {
        let state = LatentState::new(8, 1000);
        assert_eq!(state.dim(), 8);
        assert_eq!(state.timestamp_us, 1000);
        assert!(state.is_zero());
    }

    #[test]
    fn test_latent_geometry_predicates() {
        let mut geo = LatentGeometry::zero();
        assert!(geo.is_stationary());

        geo.velocity_magnitude = 1.0;
        assert!(!geo.is_stationary());

        geo.curvature = 2.0;
        assert!(geo.is_highly_curved());

        geo.divergence = 0.5;
        assert!(geo.is_diverging());

        geo.divergence = -0.5;
        assert!(geo.is_converging());
    }

    #[test]
    fn test_latent_state_predicates() {
        let mut state = LatentState::new(16, 0);
        assert!(!state.is_tense());
        assert!(!state.is_coherent());
        assert!(!state.has_rhythm());

        state.tension = 0.8;
        assert!(state.is_tense());

        state.coherence = 0.7;
        assert!(state.is_coherent());

        state.periodicity = 0.6;
        assert!(state.has_rhythm());
    }

    #[test]
    fn test_position_2d() {
        let mut state = LatentState::new(16, 0);
        state.z[0] = 1.5;
        state.z[1] = -0.5;

        let pos = state.position_2d();
        assert_eq!(pos, [1.5, -0.5]);
    }

    #[test]
    fn test_distance() {
        let mut state1 = LatentState::new(4, 0);
        state1.z = vec![0.0, 0.0, 0.0, 0.0];

        let mut state2 = LatentState::new(4, 0);
        state2.z = vec![3.0, 4.0, 0.0, 0.0];

        let dist = state1.distance_to(&state2);
        assert!((dist - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_lerp() {
        let mut state1 = LatentState::new(2, 1000);
        state1.z = vec![0.0, 0.0];
        state1.norm = 0.0;

        let mut state2 = LatentState::new(2, 2000);
        state2.z = vec![10.0, 20.0];
        state2.norm = 1.0;

        let mid = state1.lerp(&state2, 0.5);
        assert_eq!(mid.z, vec![5.0, 10.0]);
        assert_eq!(mid.norm, 0.5);
        assert_eq!(mid.timestamp_us, 2000);
    }
}
