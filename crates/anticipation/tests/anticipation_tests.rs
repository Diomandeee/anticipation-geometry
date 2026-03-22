//! Integration tests for cc-anticipation crate
//!
//! Tests the anticipation types, config, and feature computations.

use cc_anticipation::types::{
    AnticipationPacket, DebugTrace, LatentFrame, MotionWindow, Quat, SkeletonFrame, Vec3,
    MOCOPI_BONE_COUNT, SCHEMA_VERSION,
};
use cc_anticipation::config::AnticipationConfig;
use std::collections::HashMap;

// ============================================================================
// QUATERNION TESTS
// ============================================================================

#[test]
fn test_quat_identity() {
    let q = Quat::IDENTITY;
    assert_eq!(q.w, 1.0);
    assert_eq!(q.x, 0.0);
    assert_eq!(q.y, 0.0);
    assert_eq!(q.z, 0.0);
    assert!((q.magnitude() - 1.0).abs() < 1e-6);
}

#[test]
fn test_quat_creation() {
    // Quat::new(w, x, y, z)
    let q = Quat::new(0.707, 0.0, 0.707, 0.0);
    assert!((q.w - 0.707).abs() < 0.001);
    assert!((q.y - 0.707).abs() < 0.001); // y component
    assert_eq!(q.x, 0.0);
    assert_eq!(q.z, 0.0);
}

#[test]
fn test_quat_normalize() {
    let q = Quat::new(2.0, 0.0, 0.0, 0.0);
    let normalized = q.normalize();
    assert!((normalized.magnitude() - 1.0).abs() < 1e-6);
}

#[test]
fn test_quat_conjugate() {
    let q = Quat::new(0.707, 0.1, 0.2, 0.3);
    let conj = q.conjugate();
    assert_eq!(conj.w, q.w);
    assert_eq!(conj.x, -q.x);
    assert_eq!(conj.y, -q.y);
    assert_eq!(conj.z, -q.z);
}

#[test]
fn test_quat_dot() {
    let q1 = Quat::IDENTITY;
    let q2 = Quat::IDENTITY;
    let dot = q1.dot(&q2);
    assert!((dot - 1.0).abs() < 1e-6);
}

#[test]
fn test_quat_slerp_identity() {
    let q1 = Quat::IDENTITY;
    let q2 = Quat::IDENTITY;
    let result = q1.slerp(&q2, 0.5);
    assert!((result.w - 1.0).abs() < 1e-6);
}

#[test]
fn test_quat_slerp_hemisphere() {
    // Quaternions on opposite hemispheres (same rotation, opposite sign)
    let q1 = Quat::new(1.0, 0.0, 0.0, 0.0);
    let q2 = Quat::new(-1.0, 0.0, 0.0, 0.0);

    // Should take shortest path (stay at identity)
    let result = q1.slerp(&q2, 0.5);
    assert!((result.w.abs() - 1.0).abs() < 1e-6);
}

#[test]
fn test_quat_mul() {
    let q1 = Quat::IDENTITY;
    let q2 = Quat::IDENTITY;
    let result = q1.mul(&q2);
    assert!((result.w - 1.0).abs() < 1e-6);
}

// ============================================================================
// VEC3 TESTS
// ============================================================================

#[test]
fn test_vec3_zero() {
    let v = Vec3::ZERO;
    assert_eq!(v.x, 0.0);
    assert_eq!(v.y, 0.0);
    assert_eq!(v.z, 0.0);
}

#[test]
fn test_vec3_creation() {
    let v = Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(v.x, 1.0);
    assert_eq!(v.y, 2.0);
    assert_eq!(v.z, 3.0);
}

#[test]
fn test_vec3_add() {
    let v1 = Vec3::new(1.0, 2.0, 3.0);
    let v2 = Vec3::new(4.0, 5.0, 6.0);
    let result = v1.add(&v2);
    assert_eq!(result.x, 5.0);
    assert_eq!(result.y, 7.0);
    assert_eq!(result.z, 9.0);
}

#[test]
fn test_vec3_sub() {
    let v1 = Vec3::new(5.0, 7.0, 9.0);
    let v2 = Vec3::new(1.0, 2.0, 3.0);
    let result = v1.sub(&v2);
    assert_eq!(result.x, 4.0);
    assert_eq!(result.y, 5.0);
    assert_eq!(result.z, 6.0);
}

#[test]
fn test_vec3_scale() {
    let v = Vec3::new(1.0, 2.0, 3.0);
    let result = v.scale(2.0);
    assert_eq!(result.x, 2.0);
    assert_eq!(result.y, 4.0);
    assert_eq!(result.z, 6.0);
}

#[test]
fn test_vec3_dot() {
    let v1 = Vec3::new(1.0, 2.0, 3.0);
    let v2 = Vec3::new(4.0, 5.0, 6.0);
    let dot = v1.dot(&v2);
    assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 32
}

#[test]
fn test_vec3_magnitude() {
    let v = Vec3::new(3.0, 4.0, 0.0);
    assert!((v.magnitude() - 5.0).abs() < 1e-6);
}

#[test]
fn test_vec3_normalize() {
    let v = Vec3::new(3.0, 4.0, 0.0);
    let normalized = v.normalize();
    assert!((normalized.magnitude() - 1.0).abs() < 1e-6);
}

#[test]
fn test_vec3_lerp() {
    let a = Vec3::new(0.0, 0.0, 0.0);
    let b = Vec3::new(2.0, 4.0, 6.0);
    let mid = a.lerp(&b, 0.5);
    assert!((mid.x - 1.0).abs() < 1e-6);
    assert!((mid.y - 2.0).abs() < 1e-6);
    assert!((mid.z - 3.0).abs() < 1e-6);
}

// ============================================================================
// SKELETON FRAME TESTS
// ============================================================================

#[test]
fn test_skeleton_frame_invalid() {
    let frame = SkeletonFrame::invalid(1.0);
    assert!(!frame.valid);
    assert_eq!(frame.timestamp, 1.0);
    assert_eq!(frame.root_position.x, 0.0);
}

#[test]
fn test_skeleton_frame_bones_count() {
    let frame = SkeletonFrame::invalid(0.0);
    assert_eq!(frame.bone_rotations.len(), MOCOPI_BONE_COUNT);
    assert_eq!(frame.bone_rotations.len(), 27);
}

#[test]
fn test_skeleton_frame_interpolate() {
    let mut f1 = SkeletonFrame::invalid(0.0);
    f1.valid = true;
    f1.root_position = Vec3::new(0.0, 0.0, 0.0);

    let mut f2 = SkeletonFrame::invalid(1.0);
    f2.valid = true;
    f2.root_position = Vec3::new(2.0, 4.0, 6.0);

    let mid = f1.interpolate(&f2, 0.5);
    assert!(mid.valid);
    assert!((mid.timestamp - 0.5).abs() < 1e-6);
    assert!((mid.root_position.x - 1.0).abs() < 1e-6);
}

// ============================================================================
// LATENT FRAME TESTS
// ============================================================================

#[test]
fn test_latent_frame_invalid() {
    let frame = LatentFrame::invalid(2.5, 25);
    assert!(!frame.valid);
    assert_eq!(frame.timestamp, 2.5);
    assert_eq!(frame.z.len(), 25);
}

#[test]
fn test_latent_frame_interpolate() {
    let f1 = LatentFrame {
        timestamp: 0.0,
        z: vec![0.0, 0.0],
        dz_dt: None,
        valid: true,
    };
    let f2 = LatentFrame {
        timestamp: 1.0,
        z: vec![2.0, 4.0],
        dz_dt: None,
        valid: true,
    };

    let mid = f1.interpolate(&f2, 0.5);
    assert!(mid.valid);
    assert!((mid.timestamp - 0.5).abs() < 1e-6);
    assert!((mid.z[0] - 1.0).abs() < 1e-6);
    assert!((mid.z[1] - 2.0).abs() < 1e-6);
}

// ============================================================================
// MOTION WINDOW TESTS
// ============================================================================

#[test]
fn test_motion_window_empty() {
    let window = MotionWindow {
        window_id: "test".to_string(),
        t_start: 0.0,
        t_end: 1.0,
        fps: 50.0,
        skeleton_frames: Vec::new(),
        latent_frames: Vec::new(),
        coverage: 0.0,
        device_offsets: HashMap::new(),
        dropped_reason: None,
    };

    assert!(!window.has_skeleton());
    assert!(!window.has_latent());
    assert_eq!(window.duration(), 1.0);
    assert_eq!(window.frame_count(), 0);
}

#[test]
fn test_motion_window_with_skeleton() {
    let mut frames = Vec::new();
    for i in 0..50 {
        let mut frame = SkeletonFrame::invalid(i as f64 * 0.02);
        frame.valid = true;
        frames.push(frame);
    }

    let window = MotionWindow {
        window_id: "test_skeleton".to_string(),
        t_start: 0.0,
        t_end: 1.0,
        fps: 50.0,
        skeleton_frames: frames,
        latent_frames: Vec::new(),
        coverage: 1.0,
        device_offsets: HashMap::new(),
        dropped_reason: None,
    };

    assert!(window.has_skeleton());
    assert!(!window.has_latent());
    assert_eq!(window.frame_count(), 50);
}

#[test]
fn test_motion_window_with_latent() {
    let frames: Vec<LatentFrame> = (0..25)
        .map(|i| LatentFrame {
            timestamp: i as f64 * 0.04,
            z: vec![0.0; 25],
            dz_dt: None,
            valid: true,
        })
        .collect();

    let window = MotionWindow {
        window_id: "test_latent".to_string(),
        t_start: 0.0,
        t_end: 1.0,
        fps: 50.0,
        skeleton_frames: Vec::new(),
        latent_frames: frames,
        coverage: 1.0,
        device_offsets: HashMap::new(),
        dropped_reason: None,
    };

    assert!(!window.has_skeleton());
    assert!(window.has_latent());
    assert_eq!(window.frame_count(), 25);
}

// ============================================================================
// ANTICIPATION CONFIG TESTS
// ============================================================================

#[test]
fn test_anticipation_config_defaults() {
    let config = AnticipationConfig::default();

    assert_eq!(config.fps, 50.0);
    assert_eq!(config.window_seconds, 1.0);
    assert_eq!(config.regime_embedding_dim, 64);
    assert_eq!(config.constraint_vector_dim, 8);
    assert_eq!(config.derivative_summary_dim, 8);
    assert!(config.validate().is_ok());
}

#[test]
fn test_anticipation_config_low_latency() {
    let config = AnticipationConfig::low_latency();

    assert_eq!(config.window_seconds, 0.5);
    assert_eq!(config.neighbor_k, 5);
    assert!(!config.emit_debug);
    assert!(config.validate().is_ok());
}

#[test]
fn test_anticipation_config_analysis() {
    let config = AnticipationConfig::analysis();

    assert_eq!(config.window_seconds, 2.0);
    assert_eq!(config.regime_embedding_dim, 128);
    assert_eq!(config.neighbor_k, 20);
    assert!(config.emit_debug);
    assert!(config.validate().is_ok());
}

#[test]
fn test_anticipation_config_validation() {
    let valid = AnticipationConfig::default();
    assert!(valid.validate().is_ok());

    // Invalid fps
    let invalid = AnticipationConfig {
        fps: 0.0,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());

    // Invalid window_seconds
    let invalid = AnticipationConfig {
        window_seconds: -1.0,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());

    // Invalid min_coverage
    let invalid = AnticipationConfig {
        min_coverage: 1.5,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());
}

#[test]
fn test_anticipation_config_computed_values() {
    let config = AnticipationConfig::default();
    assert_eq!(config.window_frame_count(), 50); // 1.0s * 50Hz
    assert!((config.frame_duration() - 0.02).abs() < 1e-6); // 1/50Hz
}

// ============================================================================
// ANTICIPATION PACKET TESTS
// ============================================================================

fn create_valid_packet() -> AnticipationPacket {
    AnticipationPacket {
        commitment: 0.7,
        uncertainty: 0.3,
        transition_pressure: 0.2,
        recovery_margin: 0.8,
        phase_stiffness: 0.5,
        novelty: 0.4,
        stability: 0.9,
        regime_embedding: vec![0.1; 64],
        constraint_vector: vec![0.0; 8],
        derivative_summary: vec![0.0; 8],
        debug: None,
        window_id: "test_window".to_string(),
        timestamp: 1234.567,
        schema_version: SCHEMA_VERSION.to_string(),
        source_identity: None,
    }
}

#[test]
fn test_anticipation_packet_creation() {
    let packet = create_valid_packet();

    assert!((packet.commitment - 0.7).abs() < 0.001);
    assert!((packet.uncertainty - 0.3).abs() < 0.001);
    assert_eq!(packet.regime_embedding.len(), 64);
}

#[test]
fn test_anticipation_packet_validation() {
    let packet = create_valid_packet();
    assert!(packet.validate().is_ok());
}

#[test]
fn test_anticipation_packet_validation_bounds() {
    // commitment out of bounds
    let mut packet = create_valid_packet();
    packet.commitment = 1.5;
    assert!(packet.validate().is_err());

    // uncertainty out of bounds
    let mut packet = create_valid_packet();
    packet.uncertainty = -0.1;
    assert!(packet.validate().is_err());

    // recovery_margin out of bounds
    let mut packet = create_valid_packet();
    packet.recovery_margin = 1.1;
    assert!(packet.validate().is_err());

    // phase_stiffness out of bounds
    let mut packet = create_valid_packet();
    packet.phase_stiffness = -0.5;
    assert!(packet.validate().is_err());

    // novelty out of bounds
    let mut packet = create_valid_packet();
    packet.novelty = 2.0;
    assert!(packet.validate().is_err());

    // stability out of bounds
    let mut packet = create_valid_packet();
    packet.stability = -1.0;
    assert!(packet.validate().is_err());
}

#[test]
fn test_anticipation_packet_validation_schema() {
    let mut packet = create_valid_packet();
    packet.schema_version = "999.0.0".to_string();
    assert!(packet.validate().is_err());
}

#[test]
fn test_anticipation_packet_serialization() {
    let packet = create_valid_packet();

    let json = serde_json::to_string(&packet).expect("serialize");
    let parsed: AnticipationPacket = serde_json::from_str(&json).expect("deserialize");

    assert!((parsed.commitment - 0.7).abs() < 0.001);
    assert_eq!(parsed.window_id, "test_window");
    assert_eq!(parsed.regime_embedding.len(), 64);
}

// ============================================================================
// DEBUG TRACE TESTS
// ============================================================================

#[test]
fn test_debug_trace_default() {
    let trace = DebugTrace::default();
    assert!(trace.raw_features.is_empty());
    assert!(trace.neighbor_ids.is_none());
    assert!(trace.intermediate_scores.is_empty());
}

#[test]
fn test_debug_trace_with_data() {
    let mut trace = DebugTrace::default();
    trace.raw_features.insert("kinetic_energy".to_string(), 0.75);
    trace.raw_features.insert("angular_momentum".to_string(), 0.5);
    trace.intermediate_scores.insert("commitment_raw".to_string(), 0.8);
    trace.neighbor_ids = Some(vec!["phrase_1".to_string(), "phrase_2".to_string()]);

    assert_eq!(trace.raw_features.len(), 2);
    assert!(trace.neighbor_ids.is_some());
    assert_eq!(trace.neighbor_ids.as_ref().unwrap().len(), 2);
}

#[test]
fn test_debug_trace_serialization() {
    let mut trace = DebugTrace::default();
    trace.raw_features.insert("energy".to_string(), 0.5);

    let json = serde_json::to_string(&trace).expect("serialize");
    let parsed: DebugTrace = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(parsed.raw_features.get("energy"), Some(&0.5));
}

// ============================================================================
// SCHEMA VERSION TESTS
// ============================================================================

#[test]
fn test_schema_version_format() {
    assert_eq!(SCHEMA_VERSION, "0.2.0");
    // Verify it's a valid semver format
    let parts: Vec<&str> = SCHEMA_VERSION.split('.').collect();
    assert_eq!(parts.len(), 3);
}

// ============================================================================
// BONE INDEX TESTS
// ============================================================================

#[test]
fn test_bone_indices() {
    use cc_anticipation::types::bones;

    // Verify key bone indices
    assert_eq!(bones::HIPS, 0);
    assert_eq!(bones::HEAD, 5);
    assert_eq!(bones::LEFT_HAND, 9);
    assert_eq!(bones::RIGHT_HAND, 13);
    assert_eq!(bones::LEFT_FOOT, 16);
    assert_eq!(bones::RIGHT_FOOT, 20);
}

#[test]
fn test_bone_count_constant() {
    assert_eq!(MOCOPI_BONE_COUNT, 27);
}

// ============================================================================
// FEATURE COMPUTATION TESTS (Mathematical Concepts)
// ============================================================================

#[test]
fn test_commitment_from_velocity() {
    // High velocity -> high commitment (sigmoid function)
    let high_velocity_magnitude: f32 = 5.0;
    let commitment_high = 1.0 / (1.0 + (-high_velocity_magnitude + 2.0).exp());

    let low_velocity_magnitude: f32 = 0.5;
    let commitment_low = 1.0 / (1.0 + (-low_velocity_magnitude + 2.0).exp());

    assert!(commitment_high > commitment_low);
    assert!(commitment_high > 0.9); // High velocity = high commitment
    assert!(commitment_low < 0.5); // Low velocity = lower commitment
}

#[test]
fn test_uncertainty_from_variance() {
    // High variance in positions -> high uncertainty
    let values = vec![0.1, 0.9, 0.2, 0.8, 0.3];
    let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
    let variance: f32 =
        values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;

    let stable_values = vec![0.5, 0.51, 0.49, 0.5, 0.5];
    let stable_mean: f32 = stable_values.iter().sum::<f32>() / stable_values.len() as f32;
    let stable_variance: f32 = stable_values
        .iter()
        .map(|v| (v - stable_mean).powi(2))
        .sum::<f32>()
        / stable_values.len() as f32;

    assert!(variance > stable_variance);
    assert!(variance > 0.05); // High variance
    assert!(stable_variance < 0.001); // Low variance
}

#[test]
fn test_transition_pressure_from_direction_change() {
    // Direction change detection
    let positions_straight = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]];

    let positions_turn = [
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.1, 0.1, 0.0], // Changed direction
        [0.1, 0.2, 0.0],
    ];

    fn direction_change(positions: &[[f32; 3]]) -> f32 {
        let mut total_change = 0.0;
        for i in 2..positions.len() {
            let dir1 = [
                positions[i - 1][0] - positions[i - 2][0],
                positions[i - 1][1] - positions[i - 2][1],
                positions[i - 1][2] - positions[i - 2][2],
            ];
            let dir2 = [
                positions[i][0] - positions[i - 1][0],
                positions[i][1] - positions[i - 1][1],
                positions[i][2] - positions[i - 1][2],
            ];

            let dot = dir1[0] * dir2[0] + dir1[1] * dir2[1] + dir1[2] * dir2[2];
            let mag1 = (dir1[0].powi(2) + dir1[1].powi(2) + dir1[2].powi(2)).sqrt();
            let mag2 = (dir2[0].powi(2) + dir2[1].powi(2) + dir2[2].powi(2)).sqrt();

            if mag1 > 0.001 && mag2 > 0.001 {
                let cos_angle = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
                total_change += 1.0 - cos_angle; // 0 for same direction, 2 for opposite
            }
        }
        total_change
    }

    let straight_change = direction_change(&positions_straight);
    let turn_change = direction_change(&positions_turn);

    assert!(turn_change > straight_change);
    assert!(straight_change < 0.1); // Almost no change
    assert!(turn_change > 0.5); // Significant change
}
