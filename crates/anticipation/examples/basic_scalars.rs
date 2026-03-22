//! Basic example: compute the 7 anticipation scalars from a motion window
//!
//! This demonstrates the core workflow:
//! 1. Create an AnticipationConfig
//! 2. Build a MotionWindow from skeleton frames
//! 3. Process through the AnticipationKernel
//! 4. Read the 7 scalars from the output packet
//!
//! Run with: cargo run --example basic_scalars

use cc_anticipation::{
    AnticipationConfig, AnticipationKernel,
    MotionWindow, SkeletonFrame, Quat, Vec3,
    SCHEMA_VERSION,
};
use std::collections::HashMap;

fn main() {
    println!("Anticipation Geometry - Basic Scalars Example");
    println!("Schema version: {}", SCHEMA_VERSION);
    println!();

    // 1. Create configuration
    let config = AnticipationConfig::default();
    println!("Config: fps={}, window={}s, embedding_dim={}",
        config.fps, config.window_seconds, config.regime_embedding_dim);

    // 2. Build kernel
    let mut kernel = AnticipationKernel::new(config.clone());

    // 3. Create a motion window with walking-like skeleton frames
    let window = create_walking_window(&config);
    println!("Window: {} frames, coverage={:.2}", window.frame_count(), window.coverage);

    // 4. Process
    match kernel.process(&window) {
        Ok(packet) => {
            println!();
            println!("=== 7 Anticipation Scalars ===");
            println!();
            println!("  commitment:          {:.4}  (how irreversible the current motion is)", packet.commitment);
            println!("  uncertainty:         {:.4}  (how many plausible futures remain)", packet.uncertainty);
            println!("  transition_pressure: {:.4}  (rate at which futures are collapsing)", packet.transition_pressure);
            println!("  recovery_margin:     {:.4}  (distance to balance/attractor loss)", packet.recovery_margin);
            println!("  phase_stiffness:     {:.4}  (how locked to internal metronome)", packet.phase_stiffness);
            println!("  novelty:             {:.4}  (distance from recent regimes)", packet.novelty);
            println!("  stability:           {:.4}  (local stationarity of dynamics)", packet.stability);
            println!();
            println!("Regime embedding: {} dimensions", packet.regime_embedding.len());
            println!("Constraint vector: {} dimensions", packet.constraint_vector.len());

            // Validate the packet
            match packet.validate() {
                Ok(()) => println!("\nPacket validation: PASSED"),
                Err(e) => println!("\nPacket validation: FAILED - {}", e),
            }
        }
        Err(e) => {
            eprintln!("Processing error: {}", e);
        }
    }

    // 5. Process multiple windows to see temporal dynamics
    println!();
    println!("=== Temporal Dynamics (10 windows) ===");
    println!();
    println!("{:<6} {:>10} {:>10} {:>10} {:>10}", "Frame", "Commit", "Uncert", "Pressure", "Novelty");
    println!("{}", "-".repeat(50));

    kernel.reset();

    for i in 0..10 {
        let w = create_moving_window(&config, i as f64);
        if let Ok(p) = kernel.process(&w) {
            println!("{:<6} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                i, p.commitment, p.uncertainty, p.transition_pressure, p.novelty);
        }
    }
}

/// Create a window simulating walking motion
fn create_walking_window(config: &AnticipationConfig) -> MotionWindow {
    let frame_count = config.window_frame_count();
    let dt = config.frame_duration() as f64;

    let frames: Vec<SkeletonFrame> = (0..frame_count)
        .map(|i| {
            let t = i as f64 * dt;

            // Simulate walking: sinusoidal root position
            let x = (t * 2.0 * std::f64::consts::PI * 0.5).sin() as f32 * 0.1;
            let z = t as f32 * 1.2; // forward movement at ~1.2 m/s

            SkeletonFrame {
                timestamp: t,
                root_position: Vec3::new(x, 0.95, z),
                root_rotation: Quat::IDENTITY,
                bone_rotations: [Quat::IDENTITY; 27],
                valid: true,
                source_seq: Some(i as u64),
            }
        })
        .collect();

    MotionWindow {
        window_id: "walking_window".to_string(),
        t_start: 0.0,
        t_end: config.window_seconds as f64,
        fps: config.fps,
        skeleton_frames: frames,
        latent_frames: vec![],
        coverage: 0.98,
        device_offsets: HashMap::new(),
        dropped_reason: None,
    }
}

/// Create a window with varying motion intensity over time
fn create_moving_window(config: &AnticipationConfig, start_time: f64) -> MotionWindow {
    let frame_count = config.window_frame_count();
    let dt = config.frame_duration() as f64;

    let speed = 0.5 + start_time * 0.3; // Increasing speed over time

    let frames: Vec<SkeletonFrame> = (0..frame_count)
        .map(|i| {
            let t = start_time + i as f64 * dt;
            let x = (t * speed).sin() as f32 * 0.2;
            let z = t as f32 * speed as f32;

            SkeletonFrame {
                timestamp: t,
                root_position: Vec3::new(x, 0.95, z),
                root_rotation: Quat::IDENTITY,
                bone_rotations: [Quat::IDENTITY; 27],
                valid: true,
                source_seq: Some((start_time as u64) * 50 + i as u64),
            }
        })
        .collect();

    MotionWindow {
        window_id: format!("window_{}", start_time),
        t_start: start_time,
        t_end: start_time + config.window_seconds as f64,
        fps: config.fps,
        skeleton_frames: frames,
        latent_frames: vec![],
        coverage: 0.98,
        device_offsets: HashMap::new(),
        dropped_reason: None,
    }
}
