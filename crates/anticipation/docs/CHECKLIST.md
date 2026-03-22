# cc-anticipation: Living Implementation Checklist

**Document Version**: 0.1.0
**Created**: 2025-12-26
**Status**: Phases 1-9 COMPLETE (v0 core + Python + Neighbors + Replay + Dashboard)
**Parent**: [PROJECT_CHARTER.md](./PROJECT_CHARTER.md)

---

## Phase Zero: Documentation (CURRENT)

### 0.1 Canonical Documentation Set
- [x] **0.1.1** Project Charter ([PROJECT_CHARTER.md](./PROJECT_CHARTER.md))
- [x] **0.1.2** System Glossary ([GLOSSARY.md](./GLOSSARY.md))
- [x] **0.1.3** Assumptions & Invariants Ledger ([INVARIANTS.md](./INVARIANTS.md))

### 0.2 Implementation Guide
- [x] **0.2.1** Module structure defined
- [x] **0.2.2** Core types specified (MotionWindow, AnticipationPacket)
- [x] **0.2.3** Configuration schema defined
- [x] **0.2.4** Kernel implementation outlined
- [x] **0.2.5** Feature computation specified
- [x] **0.2.6** Integration points documented (cc-core-rs, rag_plusplus)
- [x] **0.2.7** Python bindings outlined
- [x] **0.2.8** Testing strategy defined
- [x] **0.2.9** Benchmark strategy defined

### 0.3 Living Checklist
- [x] **0.3.1** This checklist created

### 0.4 Continuation Protocol
- [x] **0.4.1** Continuation protocol documented ([CONTINUATION.md](./CONTINUATION.md))

---

## Phase 1: Core Infrastructure

### 1.1 Project Setup
- [x] **1.1.1** Create `Cargo.toml` with dependencies
- [x] **1.1.2** Create `src/lib.rs` with module structure
- [x] **1.1.3** Add cc-core-rs as dependency
- [x] **1.1.4** Add rag_plusplus as dependency (optional feature)
- [x] **1.1.5** Configure PyO3/maturin for Python bindings

### 1.2 Core Types
- [x] **1.2.1** Implement `types.rs` (Quat, Vec3, SkeletonFrame, LatentFrame)
- [x] **1.2.2** Implement `MotionWindow` struct
- [x] **1.2.3** Implement `AnticipationPacket` struct
- [x] **1.2.4** Implement `DebugTrace` struct
- [x] **1.2.5** Add schema version constant

### 1.3 Configuration
- [x] **1.3.1** Implement `config.rs` with `AnticipationConfig`
- [x] **1.3.2** Implement `Default` trait
- [x] **1.3.3** Add serialization (serde)

### 1.4 Kernel Skeleton
- [x] **1.4.1** Implement `AnticipationKernel` struct
- [x] **1.4.2** Implement `new()` constructor with pre-allocation
- [x] **1.4.3** Implement `process()` skeleton (coverage check, error handling)
- [x] **1.4.4** Add state fields (prev_commitment, prev_uncertainty, etc.)
- [x] **1.4.5** Implement novelty history ring buffer

---

## Phase 2: Feature Computation

### 2.1 Kinematic Features
- [x] **2.1.1** Implement `features/mod.rs`
- [x] **2.1.2** Implement `KinematicFeatures` struct
- [x] **2.1.3** Implement `compute_kinematic_features()`
- [x] **2.1.4** Implement `compute_angular_intensity()`
- [x] **2.1.5** Implement `compute_jerk_energy()`
- [x] **2.1.6** Implement `compute_directional_persistence()`
- [x] **2.1.7** Implement `compute_cross_limb_coherence()` (requires FK)
- [x] **2.1.8** Implement `compute_balance_margin()` (requires FK)

### 2.2 Forward Kinematics
- [x] **2.2.1** Define mocopi bone hierarchy (27 bones)
- [x] **2.2.2** Implement quaternion utilities (mul, conj, rotate_vec)
- [x] **2.2.3** Implement FK: rotations → keypoint positions
- [x] **2.2.4** Test FK against known poses

### 2.3 Latent Dynamics Features
- [x] **2.3.1** Implement `LatentFeatures` struct
- [x] **2.3.2** Implement `compute_latent_features()`
- [x] **2.3.3** Compute dz/dt from window
- [x] **2.3.4** Compute d²z/dt² from window
- [x] **2.3.5** Compute local predictability (linear fit residual)

### 2.4 Coordination Features
- [ ] **2.4.1** Implement `coordination.rs`
- [ ] **2.4.2** Compute torso-limb phase lag
- [ ] **2.4.3** Compute head-torso prediction

---

## Phase 3: Embedding & Constraints

### 3.1 Regime Embedding (v0: Fixed Projection)
- [x] **3.1.1** Implement `embedding/mod.rs`
- [x] **3.1.2** Implement `projection.rs` with fixed random matrix
- [x] **3.1.3** Implement `project_to_regime_embedding()`
- [x] **3.1.4** Freeze projection matrix (deterministic seed)
- [x] **3.1.5** Test embedding stability across runs

### 3.2 Constraint Vector
- [x] **3.2.1** Implement `constraints/mod.rs`
- [ ] **3.2.2** Implement `balance.rs` (COM/support polygon)
- [ ] **3.2.3** Implement `limits.rs` (joint range, speed saturation)
- [x] **3.2.4** Implement `compute_constraint_vector()`

---

## Phase 4: Scalar Computation

### 4.1 Uncertainty
- [x] **4.1.1** Implement `scalars/uncertainty.rs` (in scalars/mod.rs)
- [x] **4.1.2** Heuristic uncertainty (v0, no neighbors)
- [ ] **4.1.3** Neighbor-based uncertainty (v1, with HNSW)

### 4.2 Commitment
- [x] **4.2.1** Implement `scalars/commitment.rs` (in scalars/mod.rs)
- [x] **4.2.2** Combine uncertainty + constraints + momentum

### 4.3 Transition Pressure
- [x] **4.3.1** Implement `scalars/transition_pressure.rs` (in scalars/mod.rs)
- [x] **4.3.2** Temporal derivative with EMA smoothing

### 4.4 Recovery Margin
- [x] **4.4.1** Implement `scalars/recovery_margin.rs` (in scalars/mod.rs)
- [x] **4.4.2** Extract from constraint vector

### 4.5 Phase Stiffness
- [x] **4.5.1** Implement `scalars/phase_stiffness.rs` (in scalars/mod.rs)
- [x] **4.5.2** Compute from periodic components

### 4.6 Novelty
- [x] **4.6.1** Implement `scalars/novelty.rs` (in kernel.rs)
- [x] **4.6.2** Distance from history centroid
- [x] **4.6.3** Ring buffer update

### 4.7 Stability
- [x] **4.7.1** Implement `scalars/stability.rs` (in scalars/mod.rs)
- [x] **4.7.2** Linear prediction residual

---

## Phase 5: Integration & Testing

### 5.1 Kernel Integration
- [x] **5.1.1** Wire all components in `kernel.rs`
- [x] **5.1.2** Implement `fuse_features()`
- [x] **5.1.3** Implement `compute_derivative_summary()`
- [x] **5.1.4** Build complete `process()` pipeline

### 5.2 Determinism Tests
- [x] **5.2.1** Implement `tests/determinism.rs` (in kernel.rs tests)
- [x] **5.2.2** Test identical output on same input
- [ ] **5.2.3** Test replay from trace file
- [ ] **5.2.4** Checksum comparison harness

### 5.3 Behavioral Tests
- [x] **5.3.1** Implement `tests/behavioral.rs`
- [x] **5.3.2** Stillness test (BEH-001)
- [x] **5.3.3** Commitment-before-peak test (BEH-002)
- [x] **5.3.4** Novelty decay test (BEH-003)
- [x] **5.3.5** Uncertainty at branches test (BEH-004)

### 5.4 Benchmarks
- [x] **5.4.1** Implement `benches/kernel_bench.rs`
- [x] **5.4.2** Verify < 2ms per tick (~0.2-0.4ms achieved)
- [ ] **5.4.3** Profile for allocations

---

## Phase 6: Python Bindings

### 6.1 PyO3 Setup
- [x] **6.1.1** Implement `bindings.rs` (src/bindings.rs)
- [x] **6.1.2** Create `PyAnticipationKernel` class
- [x] **6.1.3** Create `PyAnticipationPacket` class
- [x] **6.1.4** Create `PyMotionWindow` class

### 6.2 Build & Package
- [x] **6.2.1** Configure `pyproject.toml` for maturin
- [x] **6.2.2** Build wheel (cc_anticipation-0.1.0-cp39-cp39-macosx_11_0_arm64.whl)
- [x] **6.2.3** Test import from Python (16/16 tests passing)
- [ ] **6.2.4** Add to cc-core Python package

---

## Phase 7: Neighbor Integration (v1)

### 7.1 HNSW Integration
- [x] **7.1.1** Add rag_plusplus dependency (feature-gated)
- [x] **7.1.2** Implement `neighbors/mod.rs`
- [x] **7.1.3** Implement `dispersion.rs`
- [x] **7.1.4** Query regime_embedding → K neighbors
- [x] **7.1.5** Compute continuation dispersion

### 7.2 MotionPhrase Index
- [x] **7.2.1** Define index schema (MotionPhraseIndex, ContinuationData)
- [ ] **7.2.2** Build index from phrase library (requires motion corpus)
- [x] **7.2.3** Test neighbor retrieval (5 tests passing)

---

## Phase 8: Replay Harness

### 8.1 Trace Recording
- [x] **8.1.1** Implement `replay/mod.rs` (TraceRecorder, TraceFile, TraceEntry)
- [x] **8.1.2** Define trace file format (JSON with format_version 1.0.0)
- [x] **8.1.3** Record MotionWindow + AnticipationPacket pairs

### 8.2 Replay Validation
- [x] **8.2.1** Implement ReplayHarness with validation
- [x] **8.2.2** Load trace, re-run kernel, compare checksums
- [ ] **8.2.3** Add to CI pipeline (requires CI setup)

---

## Phase 9: Dashboard Integration

### 9.1 Telemetry Module (Rust)
- [x] **9.1.1** Implement `telemetry/mod.rs` with TelemetryMessage
- [x] **9.1.2** Implement AnticipationData (dashboard-friendly format)
- [x] **9.1.3** Implement TelemetryBuffer for sparklines
- [x] **9.1.4** Derive regime labels and colors from scalars
- [x] **9.1.5** Compute risk_level, transition_readiness, motion_quality

### 9.2 TypeScript Integration
- [x] **9.2.1** Create `lib/anticipation/types.ts` (matching Rust types)
- [x] **9.2.2** Create `lib/anticipation/store.ts` (Zustand store)
- [x] **9.2.3** Update WebSocket client to handle 'anticipation' messages
- [x] **9.2.4** Add sparkline buffer utilities

### 9.3 Dashboard Visualization
- [x] **9.3.1** Create scalar gauge components (ScalarBar, BipolarBar)
- [x] **9.3.2** Create sparkline charts (AnticipationSparklines)
- [x] **9.3.3** Display regime state with color (RegimeIndicator)
- [x] **9.3.4** Create combined AnticipationPanel
- [x] **9.3.5** Create mini indicator (AnticipationMini)
- [ ] **9.3.6** Show embedding_summary heatmap (future)
- [ ] **9.3.7** Debug panel with raw_features (future)

---

## Completion Criteria

### Phase 0 (Documentation)
- [x] All Phase Zero documents created and consistent

### Phase 1-4 (Core)
- [x] `cargo build` succeeds
- [x] `cargo test` passes (39 tests: 33 unit + 6 behavioral)
- [x] Types match frozen schemas

### Phase 5 (Testing)
- [x] Determinism tests pass
- [x] Behavioral tests pass (BEH-001 through BEH-004)
- [x] Benchmark meets < 2ms target (~0.2-0.4ms achieved)

### Phase 6 (Python)
- [x] `import cc_anticipation` works
- [x] Python tests pass (16/16)

### Phase 7 (Neighbor Integration)
- [x] HNSW integration with rag_plusplus (feature-gated)
- [x] Continuation dispersion computation
- [x] Neighbor-based uncertainty in kernel (44 tests passing with neighbors feature)
- [ ] Build index from motion corpus (requires data)

### Phase 8 (Replay Harness)
- [x] TraceRecorder captures windows + packets
- [x] TraceFile JSON serialization
- [x] ReplayHarness validates determinism (4 tests passing)

### Phase 9 (Dashboard)
- [x] Telemetry module in Rust (TelemetryMessage, AnticipationData)
- [x] TypeScript types and Zustand store
- [x] WebSocket integration with 'anticipation' message type
- [x] 49 tests passing (43 unit + 6 behavioral)
- [x] Visualization components (AnticipationTelemetry, Sparklines, Panel, Mini)
- [ ] Embedding heatmap (future enhancement)
- [ ] Debug panel with raw_features (future enhancement)

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| `[ ]` | Not started |
| `[~]` | In progress |
| `[x]` | Complete |
| `[!]` | Blocked |
| `[?]` | Needs clarification |

---

**Last Updated**: 2025-12-27
**Next Action**: Integration testing - connect live mocopi data to see anticipation in action

---

**Next Document**: [CONTINUATION.md](./CONTINUATION.md)
