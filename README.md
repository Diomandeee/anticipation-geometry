# Anticipation Geometry

**Domain-general trajectory characterization via 7 geometric scalars.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)

A mathematical framework that characterizes trajectories through arbitrary state spaces using seven geometric scalars. Originally developed for real-time motion capture at 50 Hz, the framework generalizes to conversational reasoning, knowledge graph traversal, and any sequence of vectors in a metric space.

Transition pressure, defined as `d(commitment)/dt - d(uncertainty)/dt`, predicts conversation convergence at **71.8% accuracy** (z = 2.72, p < 0.007) and discriminates valid from hard-negative KG paths with **81.0% pairwise accuracy** (Cohen's d = 2.23), with no domain-specific training.

## Architecture

```
                        anticipation-geometry/
                        =====================

  crates/anticipation/          The main library. Converts MotionWindow
  (anticipation)             or LatentFrame sequences into 7 scalars.
        |                       Sub-2ms per frame. Zero hot-path allocation.
        |
        +-- scalars/            commitment, uncertainty, transition_pressure,
        |                       recovery_margin, phase_stiffness, novelty, stability
        |
        +-- features/           Kinematic + latent dynamics feature extraction
        |   +-- forward_kinematics.rs   27-bone FK with world-space keypoints
        |   +-- kinematics.rs           Velocity, jerk, directional persistence
        |   +-- latent_dynamics.rs      LIM-RPS latent vector analysis
        |
        +-- embedding/          Deterministic random projection (frozen seed)
        +-- constraints/        Physical boundary proximity (balance, joint limits)
        +-- replay/             Deterministic trace recording and validation
        +-- telemetry/          Dashboard-ready WebSocket messages
        +-- phrase/             Motion phrase library (optional, neighbors feature)
        +-- neighbors/          HNSW continuation dispersion (optional)
        |
  crates/core-rs/               Signal processing primitives
  (cc-core-rs)                  Lock-free ring buffer, 1-Euro filter,
        |                       slew limiters, equilibrium solvers,
        |                       LIM-RPS latent dynamics
        |
  crates/protocol/              Shared message types: clock sync,
  (cc-protocol)                 sensor packets, Strudel IR, mocopi state
        |
  crates/types/                 Foundation types: SkeletonFrame,
  (cc-types)                    MotionWindow, DeviceMask, provenance

  python/                       Pure Python implementation
  anticipation_geometry/        Works on conversation embeddings,
                                KG paths, or any vector trajectory
```

## The 7 Scalars

| Scalar | Range | What it measures |
|--------|-------|-----------------|
| **commitment** | [0, 1] | How irreversible the current trajectory state is. High when motion is deep into a sustained phase. |
| **uncertainty** | [0, 1] | How many plausible futures remain. High when many historical states are equidistant (gestural regime). |
| **transition_pressure** | unbounded | Rate at which futures are collapsing: `d(commitment)/dt - d(uncertainty)/dt`. Positive spikes predict regime changes. |
| **recovery_margin** | [0, 1] | Distance to balance/attractor loss. How easy it is to reverse course. |
| **phase_stiffness** | [0, 1] | How locked to internal metronome. High directional persistence + low jerk. |
| **novelty** | [0, 1] | Distance from recent regime embeddings. High when exploring new territory. |
| **stability** | [0, 1] | Local stationarity of dynamics. High predictability + low acceleration. |

## Quick Start

### Rust

```bash
# Build
cargo build

# Run tests (340+ tests across all crates)
cargo test

# Run the example
cargo run --example basic_scalars
```

```rust
use cc_anticipation::{AnticipationConfig, AnticipationKernel, MotionWindow};

let config = AnticipationConfig::default();
let mut kernel = AnticipationKernel::new(config);

// Process a motion window (from sensor data)
let packet = kernel.process(&window)?;

// Read the 7 scalars
println!("commitment: {}", packet.commitment);
println!("uncertainty: {}", packet.uncertainty);
println!("transition_pressure: {}", packet.transition_pressure);
println!("recovery_margin: {}", packet.recovery_margin);
println!("phase_stiffness: {}", packet.phase_stiffness);
println!("novelty: {}", packet.novelty);
println!("stability: {}", packet.stability);
```

### Python

```bash
pip install -e python/
python examples/demo_motion.py
```

```python
from anticipation_geometry.generalized_anticipation import AnticipationGeometry
import numpy as np

geometry = AnticipationGeometry(k_neighbors=5)

# Any sequence of vectors works: embeddings, positions, latent states
trajectory = [np.random.randn(384) for _ in range(50)]
packet = geometry.compute(trajectory)

# Per-step scalars
for t in range(len(trajectory)):
    print(f"Step {t}: commitment={packet.commitment[t]:.3f}, "
          f"uncertainty={packet.uncertainty[t]:.3f}, "
          f"regime={packet.regime_at(t)}")
```

## Evaluation Results

Results from the research paper evaluation on real data:

### Conversation Convergence (MiniLM, 384D)
- **5,000 dialogue turns** across 39 conversations
- Transition pressure sign predicts convergence: **71.8% accuracy**
- Statistical significance: z = 2.72, p < 0.007
- Transition pressure / commitment correlation: r = 0.455

### Knowledge Graph Path Discrimination (199 paths)
- Anticipation-augmented rewards: **81.0% pairwise accuracy**
- Effect size: Cohen's d = 2.23 (large)

### High-Dimensional Embeddings (e5-large, 1024D)
- Transition pressure std as single feature: **69.8% accuracy** (+8.1pp over baseline)
- 86 conversations evaluated

Full results in `results/`.

## Crate Details

### anticipation (main library)
The kernel converts a `MotionWindow` (sequence of skeleton or latent frames) into an `AnticipationPacket` containing all 7 scalars, regime embeddings, constraint vectors, and optional debug traces. Designed for real-time use: deterministic replay (INV-001), no hot-path heap allocation (INV-006), sub-2ms per tick (PERF-001).

### cc-core-rs (signal processing)
Lock-free SPSC ring buffer with true atomics, 1-Euro adaptive filter (Casiez et al. 2012), first/second-order slew limiters, proximal operators, Anderson-accelerated equilibrium solvers with Rayon parallelism, and the full LIM-RPS latent dynamics pipeline.

### cc-protocol (message types)
Shared protocol definitions including clock synchronization, sensor packets, mocopi state management, Strudel IR (pattern/edit/effect), and network device management.

### cc-types (foundation)
Core types for the motion processing pipeline: `SkeletonFrame` (27-bone), `MotionWindow` (schema-versioned), `DeviceMask` (bitfield), `FrameProvenance` (per-DOF tracking), and raw sensor packet types.

## Citation

```bibtex
@article{diomande2025anticipation,
  title={Anticipation Geometry: Domain-General Trajectory Characterization
         with Knowledge Graph-Grounded Rewards},
  author={Diomande, Mohamed},
  year={2025},
  note={Independent research. Full paper in paper/paper.md}
}
```

## License

MIT. See [LICENSE](LICENSE).
