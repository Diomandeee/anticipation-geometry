//! Kernel performance benchmarks
//!
//! Target: < 2ms per kernel tick

use criterion::{criterion_group, criterion_main, Criterion, black_box};
use cc_anticipation::{AnticipationKernel, AnticipationConfig, MotionWindow};
use cc_anticipation::types::{SkeletonFrame, Quat, Vec3};
use std::collections::HashMap;

fn create_benchmark_window() -> MotionWindow {
    let frames: Vec<SkeletonFrame> = (0..50)
        .map(|i| SkeletonFrame {
            timestamp: i as f64 * 0.02,
            root_position: Vec3::new(
                (i as f32 * 0.1).sin(),
                0.0,
                (i as f32 * 0.1).cos(),
            ),
            root_rotation: Quat::IDENTITY,
            bone_rotations: [Quat::IDENTITY; 27],
            valid: true,
            source_seq: Some(i as u64),
        })
        .collect();

    MotionWindow {
        window_id: "bench_window".to_string(),
        t_start: 0.0,
        t_end: 1.0,
        fps: 50.0,
        skeleton_frames: frames,
        latent_frames: vec![],
        coverage: 0.98,
        device_offsets: HashMap::new(),
        dropped_reason: None,
    }
}

fn kernel_process_benchmark(c: &mut Criterion) {
    let config = AnticipationConfig::default();
    let mut kernel = AnticipationKernel::new(config);
    let window = create_benchmark_window();

    c.bench_function("kernel_process", |b| {
        b.iter(|| {
            let _ = kernel.process(black_box(&window));
        })
    });
}

fn kernel_process_sequence_benchmark(c: &mut Criterion) {
    let config = AnticipationConfig::default();
    let mut kernel = AnticipationKernel::new(config);

    // Create 100 windows for sequence processing
    let windows: Vec<MotionWindow> = (0..100)
        .map(|i| {
            let mut w = create_benchmark_window();
            w.t_start = i as f64;
            w.t_end = i as f64 + 1.0;
            w.window_id = format!("bench_window_{}", i);
            w
        })
        .collect();

    c.bench_function("kernel_process_100_windows", |b| {
        b.iter(|| {
            kernel.reset();
            for w in &windows {
                let _ = kernel.process(black_box(w));
            }
        })
    });
}

criterion_group!(benches, kernel_process_benchmark, kernel_process_sequence_benchmark);
criterion_main!(benches);
