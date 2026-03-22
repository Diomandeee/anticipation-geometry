//! Dynamics Analysis for LIM-RPS.
//!
//! Computes higher-order dynamics from latent trajectories:
//! - Curvature and curvature rate
//! - Periodicity detection
//! - Phase estimation
//! - Tempo detection
//! - Somatic quality estimation
//!
//! These are the "interpretable" quantities derived from raw equilibrium output.

use super::latent_state::{LatentState, LatentStateHistory};

/// Phase tracker for beat synchronization.
#[derive(Debug, Clone)]
pub struct PhaseTracker {
    /// Current tempo in BPM.
    tempo_bpm: f32,
    /// Reference timestamp for phase computation.
    start_time_micros: Option<u64>,
    /// Last computed phase [0, 1].
    last_phase: f32,
    /// Phase velocity (for prediction).
    phase_velocity: f32,
}

impl PhaseTracker {
    /// Create a new phase tracker with initial tempo.
    pub fn new(tempo_bpm: f32) -> Self {
        Self {
            tempo_bpm,
            start_time_micros: None,
            last_phase: 0.0,
            phase_velocity: 0.0,
        }
    }

    /// Update phase from timestamp.
    ///
    /// Returns phase in [0, 1] where 0/1 = downbeat.
    pub fn update(&mut self, timestamp_micros: u64) -> f32 {
        if self.start_time_micros.is_none() {
            self.start_time_micros = Some(timestamp_micros);
            return 0.0;
        }

        let dt_sec =
            (timestamp_micros - self.start_time_micros.unwrap()) as f32 / 1_000_000.0;
        let beat_period = 60.0 / self.tempo_bpm.max(1e-6);
        let phase = (dt_sec % beat_period) / beat_period;

        // Compute phase velocity
        let phase_diff = phase - self.last_phase;
        // Handle wrap-around
        let phase_diff = if phase_diff < -0.5 {
            phase_diff + 1.0
        } else if phase_diff > 0.5 {
            phase_diff - 1.0
        } else {
            phase_diff
        };
        self.phase_velocity = phase_diff;
        self.last_phase = phase;

        phase
    }

    /// Get current tempo.
    pub fn tempo(&self) -> f32 {
        self.tempo_bpm
    }

    /// Set tempo.
    pub fn set_tempo(&mut self, bpm: f32) {
        self.tempo_bpm = bpm.clamp(20.0, 300.0);
    }

    /// Get last phase.
    pub fn phase(&self) -> f32 {
        self.last_phase
    }

    /// Predict phase at future timestamp.
    pub fn predict_phase(&self, future_timestamp_micros: u64) -> f32 {
        if self.start_time_micros.is_none() {
            return 0.0;
        }

        let dt_sec =
            (future_timestamp_micros - self.start_time_micros.unwrap()) as f32 / 1_000_000.0;
        let beat_period = 60.0 / self.tempo_bpm.max(1e-6);
        (dt_sec % beat_period) / beat_period
    }

    /// Reset phase reference.
    pub fn reset(&mut self) {
        self.start_time_micros = None;
        self.last_phase = 0.0;
        self.phase_velocity = 0.0;
    }

    /// Tap to adjust tempo.
    pub fn tap(&mut self, timestamp_micros: u64) {
        // Simple tap tempo implementation
        // TODO: More sophisticated tap tempo with averaging
        if let Some(start) = self.start_time_micros {
            let dt_sec = (timestamp_micros - start) as f32 / 1_000_000.0;
            if dt_sec > 0.2 && dt_sec < 2.0 {
                let new_bpm = 60.0 / dt_sec;
                // Smooth toward new tempo
                self.tempo_bpm = 0.7 * self.tempo_bpm + 0.3 * new_bpm;
            }
        }
        self.start_time_micros = Some(timestamp_micros);
    }
}

impl Default for PhaseTracker {
    fn default() -> Self {
        Self::new(120.0)
    }
}

/// Dynamics analyzer for latent trajectories.
#[derive(Debug)]
pub struct DynamicsAnalyzer {
    /// History buffer.
    history: LatentStateHistory,
    /// Phase tracker.
    phase_tracker: PhaseTracker,
    /// Smoothing factor for derived quantities.
    smoothing: f32,
    /// Cached periodicity.
    periodicity: f32,
    /// Cached internal tempo estimate.
    internal_tempo: f32,
}

impl DynamicsAnalyzer {
    /// Create a new dynamics analyzer.
    ///
    /// # Arguments
    /// * `history_size` - Number of frames to keep for analysis
    /// * `tempo_bpm` - Initial tempo assumption
    pub fn new(history_size: usize, tempo_bpm: f32) -> Self {
        Self {
            history: LatentStateHistory::new(history_size),
            phase_tracker: PhaseTracker::new(tempo_bpm),
            smoothing: 0.9,
            periodicity: 0.0,
            internal_tempo: tempo_bpm,
        }
    }

    /// Process a new latent state.
    ///
    /// Updates internal history and computes derived dynamics.
    pub fn process(&mut self, state: &mut LatentState, timestamp_micros: u64) {
        // Update phase
        state.phase = self.phase_tracker.update(timestamp_micros);
        state.timestamp_micros = timestamp_micros;

        // Compute dynamics from previous state
        if let Some(prev) = self.history.latest() {
            // Velocity and acceleration
            state.velocity = [
                state.position[0] - prev.position[0],
                state.position[1] - prev.position[1],
            ];
            state.acceleration = [
                state.velocity[0] - prev.velocity[0],
                state.velocity[1] - prev.velocity[1],
            ];

            // Curvature
            state.curvature = compute_curvature_2d(&state.velocity, &state.acceleration);
            state.curvature_rate = state.curvature - prev.curvature;

            // Kinetic energy
            let v_mag = (state.velocity[0].powi(2) + state.velocity[1].powi(2)).sqrt();
            state.kinetic_energy = 0.5 * v_mag.powi(2);

            // Rotational energy (from angular velocity)
            let omega = state.curvature * v_mag;
            state.rotational_energy = 0.5 * omega.powi(2);

            // Micro-tension: high-frequency energy from acceleration
            let a_mag = (state.acceleration[0].powi(2) + state.acceleration[1].powi(2)).sqrt();
            state.micro_tension = a_mag;

            // Prediction: linear extrapolation
            state.predicted_position = [
                state.position[0] + state.velocity[0],
                state.position[1] + state.velocity[1],
            ];

            // Prediction confidence
            let v_change = a_mag;
            state.prediction_confidence = 1.0 / (1.0 + v_change);
        }

        // Add to history
        self.history.push(state.clone());

        // Update periodicity (less frequently)
        if self.history.len() >= 10 {
            let new_periodicity = self.history.estimate_periodicity();
            self.periodicity = self.smoothing * self.periodicity + (1.0 - self.smoothing) * new_periodicity;
        }
        state.periodicity = self.periodicity;

        // Update internal tempo estimate
        self.estimate_internal_tempo();
        state.internal_tempo = self.internal_tempo;
    }

    /// Estimate internal tempo from motion periodicity.
    fn estimate_internal_tempo(&mut self) {
        if self.history.len() < 20 {
            return;
        }

        // Use zero-crossing analysis on velocity
        let velocities: Vec<[f32; 2]> = self.history.iter().map(|s| s.velocity).collect();

        // Count zero crossings in x velocity
        let mut crossings = 0;
        for i in 1..velocities.len() {
            if velocities[i][0] * velocities[i - 1][0] < 0.0 {
                crossings += 1;
            }
        }

        // Estimate frequency from zero crossings
        // (2 crossings = 1 period)
        if crossings > 2 {
            let periods = crossings as f32 / 2.0;
            let frames_per_period = velocities.len() as f32 / periods;
            // Assume 50 FPS
            let seconds_per_period = frames_per_period / 50.0;
            let estimated_bpm = 60.0 / seconds_per_period.max(0.1);

            // Smooth update
            if estimated_bpm > 20.0 && estimated_bpm < 300.0 {
                self.internal_tempo = self.smoothing * self.internal_tempo
                    + (1.0 - self.smoothing) * estimated_bpm;
            }
        }
    }

    /// Get current periodicity estimate.
    pub fn periodicity(&self) -> f32 {
        self.periodicity
    }

    /// Get current internal tempo estimate.
    pub fn internal_tempo(&self) -> f32 {
        self.internal_tempo
    }

    /// Set external tempo.
    pub fn set_tempo(&mut self, bpm: f32) {
        self.phase_tracker.set_tempo(bpm);
    }

    /// Get phase tracker.
    pub fn phase_tracker(&self) -> &PhaseTracker {
        &self.phase_tracker
    }

    /// Get mutable phase tracker.
    pub fn phase_tracker_mut(&mut self) -> &mut PhaseTracker {
        &mut self.phase_tracker
    }

    /// Get history.
    pub fn history(&self) -> &LatentStateHistory {
        &self.history
    }

    /// Reset analyzer state.
    pub fn reset(&mut self) {
        self.history.clear();
        self.phase_tracker.reset();
        self.periodicity = 0.0;
    }
}

impl Default for DynamicsAnalyzer {
    fn default() -> Self {
        Self::new(100, 120.0)
    }
}

/// Compute path curvature in 2D.
///
/// κ = |v × a| / |v|³
fn compute_curvature_2d(velocity: &[f32; 2], acceleration: &[f32; 2]) -> f32 {
    let v_mag = (velocity[0].powi(2) + velocity[1].powi(2)).sqrt();

    if v_mag < 1e-8 {
        return 0.0;
    }

    let cross = velocity[0] * acceleration[1] - velocity[1] * acceleration[0];
    cross.abs() / v_mag.powi(3)
}

/// Somatic quality estimator.
///
/// Estimates body-based qualities from latent trajectories:
/// - Grounding: connection to ground/stability
/// - Verticality: uprightness
/// - Tension: muscular activation
/// - Coherence: body part coordination
#[derive(Debug, Clone, Default)]
pub struct SomaticEstimator {
    /// Exponential smoothing factor.
    smoothing: f32,
    /// Current grounding estimate.
    grounding: f32,
    /// Current verticality estimate.
    verticality: f32,
    /// Current tension estimate.
    tension: f32,
    /// Current coherence estimate.
    coherence: f32,
}

impl SomaticEstimator {
    /// Create a new somatic estimator.
    pub fn new(smoothing: f32) -> Self {
        Self {
            smoothing: smoothing.clamp(0.0, 0.99),
            ..Default::default()
        }
    }

    /// Update estimates from latent state.
    ///
    /// # Heuristic Mappings
    ///
    /// These are rough mappings from latent dynamics to somatic qualities.
    /// In a full system, these would be learned or calibrated.
    pub fn update(&mut self, state: &LatentState) {
        // Grounding: inversely related to kinetic energy
        // Low movement = high grounding
        let new_grounding = 1.0 / (1.0 + state.kinetic_energy * 10.0);
        self.grounding = self.smoothing * self.grounding + (1.0 - self.smoothing) * new_grounding;

        // Verticality: based on direction of velocity
        // Upward movement has high verticality
        let v_mag = state.speed();
        let new_verticality = if v_mag > 1e-6 {
            // Vertical component normalized
            (state.velocity[1].abs() / v_mag).clamp(0.0, 1.0)
        } else {
            0.5 // Neutral when stationary
        };
        self.verticality = self.smoothing * self.verticality + (1.0 - self.smoothing) * new_verticality;

        // Tension: related to micro-tension and kinetic energy
        let new_tension = (state.micro_tension + state.kinetic_energy * 5.0).tanh();
        self.tension = self.smoothing * self.tension + (1.0 - self.smoothing) * new_tension;

        // Coherence: inversely related to residual
        // Good convergence = coherent motion
        let new_coherence = 1.0 / (1.0 + state.residual * 100.0);
        self.coherence = self.smoothing * self.coherence + (1.0 - self.smoothing) * new_coherence;
    }

    /// Apply estimates to a latent state.
    pub fn apply_to(&self, state: &mut LatentState) {
        state.grounding = self.grounding;
        state.verticality = self.verticality;
        state.tension = self.tension;
        state.coherence = self.coherence;
    }

    /// Get all somatic qualities as tuple.
    pub fn qualities(&self) -> (f32, f32, f32, f32) {
        (self.grounding, self.verticality, self.tension, self.coherence)
    }

    /// Reset to neutral.
    pub fn reset(&mut self) {
        self.grounding = 0.5;
        self.verticality = 0.5;
        self.tension = 0.0;
        self.coherence = 0.5;
    }
}

/// Phase coupling detector.
///
/// Detects how well motion is coupled to musical phase.
pub struct PhaseCouplingDetector {
    /// History of (phase, energy) pairs.
    history: Vec<(f32, f32)>,
    /// Maximum history size.
    max_size: usize,
    /// Current coupling estimate.
    coupling: f32,
}

impl PhaseCouplingDetector {
    /// Create a new coupling detector.
    pub fn new(max_size: usize) -> Self {
        Self {
            history: Vec::with_capacity(max_size),
            max_size,
            coupling: 0.0,
        }
    }

    /// Update with new phase and energy observation.
    pub fn update(&mut self, phase: f32, energy: f32) {
        self.history.push((phase, energy));
        if self.history.len() > self.max_size {
            self.history.remove(0);
        }

        // Compute phase-energy correlation
        if self.history.len() >= 20 {
            self.coupling = self.compute_coupling();
        }
    }

    /// Compute phase-energy coupling.
    ///
    /// High coupling = energy peaks align with specific phases.
    fn compute_coupling(&self) -> f32 {
        if self.history.len() < 10 {
            return 0.0;
        }

        // Bin phases and compute energy variance per bin
        let num_bins = 8;
        let mut bins: Vec<Vec<f32>> = vec![Vec::new(); num_bins];

        for &(phase, energy) in &self.history {
            let bin = ((phase * num_bins as f32) as usize).min(num_bins - 1);
            bins[bin].push(energy);
        }

        // Compute mean energy per bin
        let bin_means: Vec<f32> = bins
            .iter()
            .map(|b| {
                if b.is_empty() {
                    0.0
                } else {
                    b.iter().sum::<f32>() / b.len() as f32
                }
            })
            .collect();

        // Coupling = variance of bin means / overall variance
        let overall_mean = bin_means.iter().sum::<f32>() / num_bins as f32;
        let bin_variance =
            bin_means.iter().map(|&m| (m - overall_mean).powi(2)).sum::<f32>() / num_bins as f32;

        let all_energies: Vec<f32> = self.history.iter().map(|&(_, e)| e).collect();
        let energy_mean = all_energies.iter().sum::<f32>() / all_energies.len() as f32;
        let energy_variance = all_energies
            .iter()
            .map(|&e| (e - energy_mean).powi(2))
            .sum::<f32>()
            / all_energies.len() as f32;

        if energy_variance < 1e-6 {
            return 0.0;
        }

        (bin_variance / energy_variance).sqrt().min(1.0)
    }

    /// Get current coupling estimate.
    pub fn coupling(&self) -> f32 {
        self.coupling
    }

    /// Reset detector.
    pub fn reset(&mut self) {
        self.history.clear();
        self.coupling = 0.0;
    }
}

impl Default for PhaseCouplingDetector {
    fn default() -> Self {
        Self::new(200)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_tracker_basic() {
        let mut tracker = PhaseTracker::new(120.0);

        // First update sets reference
        let p0 = tracker.update(0);
        assert_eq!(p0, 0.0);

        // At 120 BPM, 0.25 sec = phase 0.5 (half a beat)
        // beat_period = 60/120 = 0.5 sec
        let p1 = tracker.update(250_000);
        assert!((p1 - 0.5).abs() < 0.01, "Expected ~0.5, got {}", p1);

        // At 0.5 sec = phase 0 (full beat, wrapped)
        let p2 = tracker.update(500_000);
        assert!(p2 < 0.1, "Expected ~0, got {}", p2);

        // At 0.75 sec = phase 0.5
        let p3 = tracker.update(750_000);
        assert!((p3 - 0.5).abs() < 0.01, "Expected ~0.5, got {}", p3);
    }

    #[test]
    fn test_phase_tracker_tempo_change() {
        let mut tracker = PhaseTracker::new(60.0);
        tracker.update(0);

        // At 60 BPM, 0.5 sec = phase 0.5
        let p = tracker.update(500_000);
        assert!((p - 0.5).abs() < 0.01);

        // Change to 120 BPM
        tracker.set_tempo(120.0);

        // Now 0.5 sec from start = 1 beat at 120 BPM = phase 0
        tracker.reset();
        tracker.update(0);
        let p2 = tracker.update(500_000);
        assert!(p2 < 0.1 || p2 > 0.9);
    }

    #[test]
    fn test_dynamics_analyzer_basic() {
        let mut analyzer = DynamicsAnalyzer::new(50, 120.0);

        let mut state = LatentState::new(4);
        state.position = [1.0, 0.0];

        analyzer.process(&mut state, 0);
        assert!(state.phase >= 0.0 && state.phase <= 1.0);
    }

    #[test]
    fn test_dynamics_velocity_computation() {
        let mut analyzer = DynamicsAnalyzer::new(50, 120.0);

        // First state at origin
        let mut state1 = LatentState::new(4);
        state1.position = [0.0, 0.0];
        analyzer.process(&mut state1, 0);

        // Second state moved right
        let mut state2 = LatentState::new(4);
        state2.position = [1.0, 0.0];
        analyzer.process(&mut state2, 20_000);

        assert_eq!(state2.velocity, [1.0, 0.0]);
    }

    #[test]
    fn test_somatic_estimator() {
        let mut estimator = SomaticEstimator::new(0.5);

        let mut state = LatentState::new(4);
        state.kinetic_energy = 0.1;
        state.micro_tension = 0.2;
        state.residual = 0.01;
        state.velocity = [0.0, 0.1];

        estimator.update(&state);
        estimator.apply_to(&mut state);

        assert!(state.grounding >= 0.0 && state.grounding <= 1.0);
        assert!(state.tension >= 0.0 && state.tension <= 1.0);
    }

    #[test]
    fn test_phase_coupling_detector() {
        let mut detector = PhaseCouplingDetector::new(100);

        // Simulate energy peaks at phase ~0.5
        for i in 0..50 {
            let phase = (i as f32 / 10.0) % 1.0;
            let energy = if (phase - 0.5).abs() < 0.2 { 1.0 } else { 0.0 };
            detector.update(phase, energy);
        }

        // Should detect some coupling
        assert!(detector.coupling() > 0.0);
    }

    #[test]
    fn test_curvature_2d() {
        // Straight line
        let v1 = [1.0, 0.0];
        let a1 = [0.0, 0.0];
        assert!(compute_curvature_2d(&v1, &a1) < 1e-6);

        // Sharp turn
        let v2 = [1.0, 0.0];
        let a2 = [0.0, 1.0];
        assert!(compute_curvature_2d(&v2, &a2) > 0.0);
    }
}
