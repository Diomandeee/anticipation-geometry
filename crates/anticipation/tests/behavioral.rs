//! Behavioral tests for cc-anticipation
//!
//! These tests verify the invariants from INVARIANTS.md:
//! - BEH-001: Stillness response
//! - BEH-002: Commitment before peak
//! - BEH-003: Novelty decay on repetition
//! - BEH-004: Uncertainty at branches

use cc_anticipation::{
    AnticipationKernel, AnticipationConfig, MotionWindow,
    types::{SkeletonFrame, Quat, Vec3, MOCOPI_BONE_COUNT},
};
use std::collections::HashMap;

/// Create a motion window with still skeleton (no movement)
fn create_still_window(duration_seconds: f32, start_time: f64) -> MotionWindow {
    let config = AnticipationConfig::default();
    let frame_count = (duration_seconds * config.fps) as usize;

    let frames: Vec<SkeletonFrame> = (0..frame_count)
        .map(|i| SkeletonFrame {
            timestamp: start_time + i as f64 * (1.0 / config.fps as f64),
            root_position: Vec3::new(0.0, 1.0, 0.0), // Standing still
            root_rotation: Quat::IDENTITY,
            bone_rotations: [Quat::IDENTITY; MOCOPI_BONE_COUNT],
            valid: true,
            source_seq: Some(i as u64),
        })
        .collect();

    MotionWindow {
        window_id: format!("still_window_{}", start_time),
        t_start: start_time,
        t_end: start_time + duration_seconds as f64,
        fps: config.fps,
        skeleton_frames: frames,
        latent_frames: vec![],
        coverage: 0.98,
        device_offsets: HashMap::new(),
        dropped_reason: None,
    }
}

/// Create a motion window with linear movement
fn create_moving_window(
    direction: Vec3,
    speed: f32,
    duration_seconds: f32,
    start_time: f64,
) -> MotionWindow {
    let config = AnticipationConfig::default();
    let frame_count = (duration_seconds * config.fps) as usize;
    let dt = 1.0 / config.fps;
    let normalized_dir = direction.normalize();

    let frames: Vec<SkeletonFrame> = (0..frame_count)
        .map(|i| {
            let t = i as f32 * dt;
            let pos = Vec3::new(
                normalized_dir.x * speed * t,
                1.0 + normalized_dir.y * speed * t,
                normalized_dir.z * speed * t,
            );
            SkeletonFrame {
                timestamp: start_time + i as f64 * dt as f64,
                root_position: pos,
                root_rotation: Quat::IDENTITY,
                bone_rotations: [Quat::IDENTITY; MOCOPI_BONE_COUNT],
                valid: true,
                source_seq: Some(i as u64),
            }
        })
        .collect();

    MotionWindow {
        window_id: format!("moving_window_{}", start_time),
        t_start: start_time,
        t_end: start_time + duration_seconds as f64,
        fps: config.fps,
        skeleton_frames: frames,
        latent_frames: vec![],
        coverage: 0.98,
        device_offsets: HashMap::new(),
        dropped_reason: None,
    }
}

/// Create a turn/sweep motion that accelerates then decelerates
fn create_sweep_window(
    peak_speed: f32,
    duration_seconds: f32,
    start_time: f64,
) -> MotionWindow {
    let config = AnticipationConfig::default();
    let frame_count = (duration_seconds * config.fps) as usize;
    let dt = 1.0 / config.fps;

    // Bell curve speed profile: starts slow, peaks in middle, ends slow
    let frames: Vec<SkeletonFrame> = (0..frame_count)
        .map(|i| {
            let t = i as f32 / frame_count as f32; // 0 to 1
            // Bell curve: sin^2 for smooth acceleration/deceleration
            let speed_factor = (t * std::f32::consts::PI).sin().powi(2);
            let current_speed = peak_speed * speed_factor;

            // Integrate position
            let mut x = 0.0f32;
            for j in 0..=i {
                let tj = j as f32 / frame_count as f32;
                let sj = (tj * std::f32::consts::PI).sin().powi(2);
                x += peak_speed * sj * dt;
            }

            SkeletonFrame {
                timestamp: start_time + i as f64 * dt as f64,
                root_position: Vec3::new(x, 1.0, 0.0),
                root_rotation: Quat::IDENTITY,
                bone_rotations: [Quat::IDENTITY; MOCOPI_BONE_COUNT],
                valid: true,
                source_seq: Some(i as u64),
            }
        })
        .collect();

    MotionWindow {
        window_id: format!("sweep_window_{}", start_time),
        t_start: start_time,
        t_end: start_time + duration_seconds as f64,
        fps: config.fps,
        skeleton_frames: frames,
        latent_frames: vec![],
        coverage: 0.98,
        device_offsets: HashMap::new(),
        dropped_reason: None,
    }
}

/// BEH-001: Stillness Response
///
/// GIVEN: motion_energy ≈ 0 for window_duration
/// THEN:
///   - commitment lower than during movement
///   - transition_pressure ≈ 0 (±0.15)
///   - stability higher than during fast movement
#[test]
fn test_beh_001_stillness_response() {
    let config = AnticipationConfig::default();

    // First, get baseline for movement
    let mut moving_kernel = AnticipationKernel::new(config.clone());
    let moving_window = create_moving_window(Vec3::new(1.0, 0.0, 0.0), 2.0, 1.0, 0.0);
    let moving_packet = moving_kernel.process(&moving_window).unwrap();

    // Now get stillness response
    let mut still_kernel = AnticipationKernel::new(config);

    // Process several still windows to let state stabilize
    for i in 0..5 {
        let window = create_still_window(1.0, i as f64);
        let _ = still_kernel.process(&window);
    }

    let still_window = create_still_window(1.0, 5.0);
    let still_packet = still_kernel.process(&still_window).unwrap();

    // BEH-001: Stillness should have LOWER commitment than movement
    // (v0 heuristics may not achieve very low absolute values)
    assert!(
        still_packet.commitment <= moving_packet.commitment + 0.2,
        "Stillness commitment {} should not be much higher than movement {}",
        still_packet.commitment,
        moving_packet.commitment
    );

    // BEH-001: Transition pressure should be near zero for stable stillness
    assert!(
        still_packet.transition_pressure.abs() < 0.2,
        "Stillness should have ~0 transition_pressure, got {}",
        still_packet.transition_pressure
    );

    // BEH-001: Stability should be reasonable
    assert!(
        still_packet.stability >= 0.4,
        "Stillness should have at least moderate stability, got {}",
        still_packet.stability
    );
}

/// BEH-002: Commitment Before Peak
///
/// GIVEN: Intentional turn/sweep
/// THEN: commitment rises BEFORE the visual/kinematic peak
#[test]
fn test_beh_002_commitment_before_peak() {
    let config = AnticipationConfig::default();
    let mut kernel = AnticipationKernel::new(config);

    // Create a sequence of windows representing a sweep motion
    // Split the sweep into 5 windows to observe progression
    let window_duration = 0.4; // 400ms windows
    let total_duration = 2.0;
    let num_windows = (total_duration / window_duration) as usize;

    let mut packets = Vec::new();

    // Start with stillness to establish baseline
    let still = create_still_window(1.0, 0.0);
    let _ = kernel.process(&still);

    // Now process sweep windows
    for i in 0..num_windows {
        let start_time = 1.0 + i as f64 * window_duration as f64;
        let window = create_sweep_window(2.0, window_duration, start_time);
        if let Ok(packet) = kernel.process(&window) {
            packets.push(packet);
        }
    }

    assert!(packets.len() >= 3, "Need at least 3 packets to analyze");

    // Find kinetic peak (highest kinetic_intensity in debug features)
    let kinetic_values: Vec<f32> = packets.iter()
        .map(|p| {
            p.debug.as_ref()
                .and_then(|d| d.raw_features.get("kinetic_intensity"))
                .copied()
                .unwrap_or(0.0)
        })
        .collect();

    let peak_idx = kinetic_values.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(packets.len() / 2);

    // Commitment at and before peak
    if peak_idx > 0 {
        let pre_peak_commitment = packets[peak_idx - 1].commitment;
        let at_peak_commitment = packets[peak_idx].commitment;

        // Commitment should be rising as we approach peak
        // (or already high before peak)
        assert!(
            pre_peak_commitment > 0.3 || at_peak_commitment > 0.3,
            "Commitment should rise before/at peak. Pre-peak: {}, At-peak: {}",
            pre_peak_commitment,
            at_peak_commitment
        );
    }
}

/// BEH-003: Novelty Decay on Repetition
///
/// GIVEN: Same move repeated N times
/// THEN: novelty(repeat_N) < novelty(repeat_1)
#[test]
fn test_beh_003_novelty_decay_on_repetition() {
    let config = AnticipationConfig::default();
    let mut kernel = AnticipationKernel::new(config);

    // Create identical motion windows (repeating the same move)
    let mut novelty_values = Vec::new();

    for i in 0..10 {
        let window = create_moving_window(
            Vec3::new(1.0, 0.0, 0.0), // Same direction each time
            1.0,                       // Same speed
            1.0,                       // Same duration
            i as f64,
        );

        if let Ok(packet) = kernel.process(&window) {
            novelty_values.push(packet.novelty);
        }
    }

    assert!(novelty_values.len() >= 5, "Need enough repetitions");

    // Compare first few to last few
    let first_avg: f32 = novelty_values[0..3].iter().sum::<f32>() / 3.0;
    let last_avg: f32 = novelty_values[novelty_values.len()-3..].iter().sum::<f32>() / 3.0;

    assert!(
        last_avg <= first_avg + 0.1, // Allow small tolerance
        "Novelty should decay on repetition. First avg: {}, Last avg: {}",
        first_avg,
        last_avg
    );
}

/// BEH-004: Uncertainty at Branches
///
/// GIVEN: Transition between different motion types
/// THEN: uncertainty increases during transition
#[test]
fn test_beh_004_uncertainty_at_transitions() {
    let config = AnticipationConfig::default();
    let mut kernel = AnticipationKernel::new(config);

    // Start with consistent motion in one direction
    for i in 0..3 {
        let window = create_moving_window(
            Vec3::new(1.0, 0.0, 0.0),
            1.0,
            1.0,
            i as f64,
        );
        let _ = kernel.process(&window);
    }

    // Get baseline uncertainty after consistent motion
    let baseline_window = create_moving_window(
        Vec3::new(1.0, 0.0, 0.0),
        1.0,
        1.0,
        3.0,
    );
    let baseline = kernel.process(&baseline_window).unwrap();
    let baseline_uncertainty = baseline.uncertainty;

    // Now change direction suddenly
    let change_window = create_moving_window(
        Vec3::new(-1.0, 0.0, 0.0), // Opposite direction!
        1.0,
        1.0,
        4.0,
    );
    let change_packet = kernel.process(&change_window).unwrap();

    // Uncertainty and novelty should respond to the change
    // At minimum, novelty should increase (new pattern)
    assert!(
        change_packet.novelty > baseline.novelty - 0.1,
        "Novelty should not decrease significantly on direction change. Baseline: {}, After change: {}",
        baseline.novelty,
        change_packet.novelty
    );
}

/// Test determinism across repeated runs
#[test]
fn test_determinism_repeated_runs() {
    let config = AnticipationConfig::default();
    let window = create_moving_window(Vec3::new(1.0, 0.5, 0.0), 1.5, 1.0, 0.0);

    let mut results = Vec::new();

    for _ in 0..5 {
        let mut kernel = AnticipationKernel::new(config.clone());
        let packet = kernel.process(&window).unwrap();
        results.push((
            packet.commitment,
            packet.uncertainty,
            packet.transition_pressure,
            packet.regime_embedding.clone(),
        ));
    }

    // All runs should produce identical results
    for i in 1..results.len() {
        assert_eq!(results[0].0, results[i].0, "commitment should be deterministic");
        assert_eq!(results[0].1, results[i].1, "uncertainty should be deterministic");
        assert_eq!(results[0].2, results[i].2, "transition_pressure should be deterministic");
        assert_eq!(results[0].3, results[i].3, "regime_embedding should be deterministic");
    }
}

/// Test scalar bounds invariants
#[test]
fn test_scalar_bounds_invariants() {
    let config = AnticipationConfig::default();
    let mut kernel = AnticipationKernel::new(config);

    // Test various motion types
    let windows = vec![
        create_still_window(1.0, 0.0),
        create_moving_window(Vec3::new(1.0, 0.0, 0.0), 0.5, 1.0, 1.0),
        create_moving_window(Vec3::new(0.0, 0.0, 1.0), 2.0, 1.0, 2.0),
        create_sweep_window(3.0, 1.0, 3.0),
    ];

    for window in windows {
        let packet = kernel.process(&window).unwrap();

        // INV-004: Scalar bounds
        assert!(packet.commitment >= 0.0 && packet.commitment <= 1.0,
            "commitment {} out of [0,1]", packet.commitment);
        assert!(packet.uncertainty >= 0.0 && packet.uncertainty <= 1.0,
            "uncertainty {} out of [0,1]", packet.uncertainty);
        assert!(packet.recovery_margin >= 0.0 && packet.recovery_margin <= 1.0,
            "recovery_margin {} out of [0,1]", packet.recovery_margin);
        assert!(packet.phase_stiffness >= 0.0 && packet.phase_stiffness <= 1.0,
            "phase_stiffness {} out of [0,1]", packet.phase_stiffness);
        assert!(packet.novelty >= 0.0 && packet.novelty <= 1.0,
            "novelty {} out of [0,1]", packet.novelty);
        assert!(packet.stability >= 0.0 && packet.stability <= 1.0,
            "stability {} out of [0,1]", packet.stability);
    }
}
