# cc-anticipation: System Glossary

**Document Version**: 0.1.0
**Created**: 2025-12-26
**Parent**: [PROJECT_CHARTER.md](./PROJECT_CHARTER.md)

---

## Core Concepts

### AnticipationPacket
The output contract of cc-anticipation. A structured object containing scalars (commitment, uncertainty, etc.), vectors (regime_embedding, constraint_vector), and debug traces. Emitted deterministically every frame. The only way "the future" is allowed to speak to the present.

### Commitment
Scalar [0-1]. How irreversible the current motion has become. Rises when momentum, posture, and coordination patterns make reversal expensive. Computed from: low uncertainty + constraint proximity + directional momentum persistence.

### Constraint Proximity
How close the body is to physical boundaries where options vanish: balance edge, angular momentum thresholds, joint range limits, speed saturation. When constraints close, commitment rises.

### Continuation Dispersion
The spread of next-window embeddings among K nearest neighbors in the MotionPhrase library. Low dispersion → low uncertainty (futures converge). High dispersion → high uncertainty (futures branch).

### Derivative Summary
An 8-dimensional vector summarizing: kinetic intensity, angular intensity, jerk energy, cross-limb coherence. Window-aggregated, not per-frame.

### Forward Kinematics (FK)
Computing keypoint world positions from root transform + per-bone local rotations. Required when skeleton provides only rotations.

### Kernel FPS
The canonical frame rate for MotionWindow (e.g., 50Hz). All streams are resampled to this rate. Derivatives, phase, and anticipation horizons depend on this being stable.

### Latent Dynamics
Features computed from the LIM-RPS latent vector z: dz/dt, d²z/dt², norms, coherence, local predictability. Complements skeleton kinematics.

### LatentFrame
A frame containing a fixed-length feature vector z (e.g., 25D) plus optional pre-computed derivatives. Part of MotionWindow when LIM-RPS stream is available.

### Local Dynamics
The first family of computations: velocity, acceleration, jerk in body-centric normalized space. Plus coordination statistics: cross-limb coherence, torso-limb phase lag.

### Manifold Proximity
How close the current state is to states in the MotionPhrase library that historically precede specific kinds of next events. Computed via regime embedding + neighbor search.

### MotionWindow
The input contract to cc-anticipation. A fixed-length ring of aligned frames at canonical fps. Contains SkeletonFrame[], LatentFrame[], timestamps, coverage, and provenance.

### Novelty
Scalar [0-1]. Distance of current regime_embedding from recent history centroid. Prevents looping and hallucinating "growth" when repeating. High novelty = exploring new territory.

### Phase Stiffness
Scalar [0-1]. How tightly the body is entrained to periodic timing. High = locked to internal metronome. Low = timing is loose/exploratory. Computed from phase derivative stability.

### Recovery Margin
Scalar. How much "wiggle room" exists to return to a neutral/stable attractor. Depends on: COM-to-support-boundary distance, velocity direction, current angular momentum.

### Regime
The hidden structure that makes prediction possible. Examples: preparation, initiation, travel, accent, hold, rebound, turn-setup, turn-release, drop, rise, suspend, reset. Not labeled; inferred from kinematics.

### Regime Embedding
A 64-256 dimensional vector representing the current regime in a way that's consistent across time. Initially a fixed projection; later a trained encoder. Query key for RAG++.

### SkeletonFrame
A frame containing: root transform (position + orientation) + per-bone local rotations (quaternions) for a fixed bone list. Part of MotionWindow when mocopi stream is available.

### Stability
Scalar [0-1]. Local stationarity of dynamics. High when residual from linear prediction is low (stable regime). Low when motion is transitional or chaotic.

### Transition Pressure
Scalar. The derivative of commitment combined with the rate at which uncertainty is falling. Spikes when entering a change. Formula: d(commitment)/dt - d(uncertainty)/dt, smoothed.

### Uncertainty
Scalar [0-1]. How many plausible futures remain. Falls when state approaches a corridor where only a few continuations are physically consistent. Computed from continuation dispersion among neighbors.

### Watermark
The safety margin that ensures windows are complete before emission. end_time = ingest_time - watermark_seconds. Guarantees replay stability.

---

## Abbreviations

| Abbrev | Full Form |
|--------|-----------|
| ANN | Approximate Nearest Neighbor |
| COM | Center of Mass |
| FK | Forward Kinematics |
| HNSW | Hierarchical Navigable Small World |
| IMU | Inertial Measurement Unit |
| LIM-RPS | Limb-based Inertial Motion - Rotational Phase Space |
| RAG++ | Retrieval-Augmented Generation Plus Plus |
| SLERP | Spherical Linear Interpolation |
| SPSC | Single Producer Single Consumer |

---

## Signal Flow Terminology

| Term | Meaning |
|------|---------|
| **Upstream** | Toward sensors (mocopi, phones, watches) |
| **Downstream** | Toward outputs (Conductor, MotionGen, audio) |
| **Hot path** | Real-time code that must never block or allocate |
| **Frozen** | Immutable once defined; changes require version bump |
| **Packet** | A structured output emitted at regular intervals |
| **Window** | A fixed-duration slice of aligned motion frames |

---

## Units and Conventions

| Quantity | Unit | Notes |
|----------|------|-------|
| Time | seconds (float64) | Canonical server time domain |
| Position | meters | Body-centric, forward=+Z |
| Rotation | quaternion (w,x,y,z) | Unit quaternions, hemisphere-corrected |
| Velocity | m/s or rad/s | Finite differences from window |
| Frame rate | Hz | 50Hz canonical for cc-anticipation |
| Embedding dim | int | 64-256 for regime_embedding |

---

**Next Document**: [INVARIANTS.md](./INVARIANTS.md)
