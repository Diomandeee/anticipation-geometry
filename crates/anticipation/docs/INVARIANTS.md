# cc-anticipation: Assumptions & Invariants Ledger

**Document Version**: 0.1.0
**Created**: 2025-12-26
**Parent**: [PROJECT_CHARTER.md](./PROJECT_CHARTER.md)

---

## 1. System Invariants

These properties MUST hold at all times. Violation indicates a bug.

### INV-001: Deterministic Replay
```
GIVEN: Same MotionWindow bytes + same config version
THEN:  Identical AnticipationPacket bytes
```
**Enforcement**: Checksum comparison in replay harness. No parallel reductions, no random seeds, no system time in computation.

### INV-002: Monotonic Window IDs
```
GIVEN: Two consecutive window emissions
THEN:  window_id(t+1) > window_id(t) lexicographically OR t_end(t+1) > t_end(t)
```
**Enforcement**: Window ID includes t_end in hash. Aligner advances by fixed hop.

### INV-003: Coverage Gating
```
GIVEN: MotionWindow with coverage < MIN_COVERAGE_THRESHOLD
THEN:  AnticipationPacket has dropped_reason set OR emission refused
```
**Enforcement**: Check at packet emission entry point. MIN_COVERAGE_THRESHOLD = 0.90 (configurable).

### INV-004: Scalar Bounds
```
FOR ALL scalars in {commitment, uncertainty, phase_stiffness, novelty, stability}:
  0.0 <= scalar <= 1.0
```
**Enforcement**: Clamp after computation. Assert in debug builds.

### INV-005: Embedding Dimension Stability
```
GIVEN: config.regime_embedding_dim = D
THEN:  len(packet.regime_embedding) == D for all packets
```
**Enforcement**: Fixed buffer allocation. Panic if mismatch.

### INV-006: No Allocation in Hot Path
```
GIVEN: Kernel tick (MotionWindow → AnticipationPacket)
THEN:  Zero heap allocations
```
**Enforcement**: Pre-allocated buffers, ring buffer reuse. Profile with allocator instrumentation.

### INV-007: Schema Version Match
```
GIVEN: AnticipationPacket.schema_version
THEN:  Must match current SCHEMA_VERSION constant
```
**Enforcement**: Version check at emission. Reject mismatched packets in consumers.

---

## 2. Architectural Assumptions

These are believed true but not enforced by cc-anticipation. If violated, behavior is undefined.

### ASM-001: Aligned Input
**Assumption**: MotionWindow arrives already resampled to canonical fps with consistent timestamps.
**Source**: cc-window-aligner contract
**If violated**: Derivatives will be wrong, regime detection will drift.

### ASM-002: Skeleton Hierarchy
**Assumption**: SkeletonFrame uses mocopi 27-bone hierarchy with consistent parent indices.
**Source**: mocopi SDK documentation
**If violated**: FK will produce incorrect keypoints.

### ASM-003: Quaternion Normalization
**Assumption**: All quaternions in SkeletonFrame are unit quaternions.
**Source**: mocopi-udp-ingest normalization step
**If violated**: Slerp interpolation will produce incorrect rotations.

### ASM-004: Latent Stability
**Assumption**: LatentFrame.z has stable semantics (same dimensions, same meaning) across versions.
**Source**: LIM-RPS encoder frozen at specific version
**If violated**: Latent dynamics will be meaningless.

### ASM-005: MotionPhrase Index Availability
**Assumption**: For neighbor-based uncertainty, HNSW index over MotionPhrase library is loaded and queryable.
**Source**: rag_plusplus initialization
**If violated**: Uncertainty degrades to heuristic-only mode.

### ASM-006: Clock Monotonicity
**Assumption**: t_end of successive windows is strictly increasing.
**Source**: cc-window-aligner watermark logic
**If violated**: Temporal deltas (transition_pressure, etc.) will be undefined.

---

## 3. Computational Invariants

Properties that must hold within the kernel computation.

### COMP-001: Body-Centric Coordinates
```
FOR ALL kinematic features:
  Computed in body-centric frame (root orientation defines forward/right/up)
```
**Rationale**: Makes features rotation-invariant in world space.

### COMP-002: Window Aggregation
```
FOR ALL derivative/summary features:
  Aggregated over window, not per-frame
```
**Rationale**: Stability; single-frame features are too noisy.

### COMP-003: Finite Difference Stencil
```
FOR ALL derivative computations:
  Use frozen stencil (e.g., central difference with fixed coefficients)
```
**Rationale**: Determinism; different stencils → different derivatives.

### COMP-004: Slerp Hemisphere Correction
```
FOR ALL quaternion interpolation:
  If dot(q1, q2) < 0: negate q2 before slerp
```
**Rationale**: Ensures shortest-path interpolation, deterministic choice.

### COMP-005: Neighbor K Consistency
```
GIVEN: neighbor search with K = config.neighbor_k
THEN:  Always request exactly K neighbors (may return fewer if index small)
```
**Rationale**: Uncertainty computation depends on consistent K.

---

## 4. Performance Invariants

### PERF-001: Latency Budget
```
GIVEN: One kernel tick (MotionWindow → AnticipationPacket)
THEN:  Complete within 2ms on reference hardware
```
**Reference hardware**: Apple M1, single core, release build.

### PERF-002: Memory Budget
```
GIVEN: Kernel running at 50Hz
THEN:  Peak memory < 50MB (excluding MotionPhrase index)
```
**Enforcement**: Profile during acceptance testing.

### PERF-003: No Blocking I/O
```
GIVEN: Kernel tick
THEN:  No file reads, network calls, or mutex waits
```
**Enforcement**: All data pre-loaded; neighbor search is in-memory.

---

## 5. Contract Invariants

Properties of the interface contracts.

### CTR-001: Input Completeness
```
MotionWindow MUST contain at least one of:
  - Non-empty SkeletonFrame[] with coverage > 0
  - Non-empty LatentFrame[] with coverage > 0
```
**If both empty**: Refuse emission, set dropped_reason.

### CTR-002: Output Completeness
```
AnticipationPacket MUST contain:
  - All 7 scalar fields (commitment, uncertainty, transition_pressure,
    recovery_margin, phase_stiffness, novelty, stability)
  - regime_embedding of configured dimension
  - window_id matching input window
  - schema_version
```
**Enforcement**: Constructor validation.

### CTR-003: Debug Trace Presence
```
IF config.emit_debug = true
THEN: packet.debug MUST contain raw_features and intermediate_scores
```
**Rationale**: Telemetry for dashboard/replay viewer.

---

## 6. Behavioral Invariants

Expected behaviors under specific conditions.

### BEH-001: Stillness Response
```
GIVEN: motion_energy ≈ 0 for window_duration
THEN:
  - commitment < 0.2
  - transition_pressure ≈ 0 (±0.05)
  - stability > 0.8
```

### BEH-002: Commitment Before Peak
```
GIVEN: Intentional turn or sweep
THEN:  commitment rises BEFORE the visual/kinematic peak, not after
```
**Enforcement**: Time-to-event behavioral test.

### BEH-003: Novelty Decay on Repetition
```
GIVEN: Same move repeated N times
THEN:  novelty(repeat_N) < novelty(repeat_1)
```

### BEH-004: Uncertainty at Branches
```
GIVEN: Ambiguous motion (feints, pauses before choice)
THEN:  uncertainty > 0.6
```

---

## 7. Violation Response

| Invariant Type | Response on Violation |
|----------------|----------------------|
| System Invariant | Panic in debug, log + refuse packet in release |
| Architectural Assumption | Undefined behavior; document and escalate |
| Computational Invariant | Assert in debug, silent corruption in release (must fix) |
| Performance Invariant | Log warning, continue (track for optimization) |
| Contract Invariant | Refuse emission, set dropped_reason |
| Behavioral Invariant | Test failure, not runtime error |

---

## 8. Invariant Verification Schedule

| Invariant | When Verified |
|-----------|---------------|
| INV-001 (Determinism) | Every CI run with replay harness |
| INV-006 (No Alloc) | Weekly profiling, pre-release |
| BEH-001-004 | Behavioral test suite, pre-release |
| PERF-001-003 | Benchmark suite, pre-release |

---

**Next Document**: [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md)
