# cc-anticipation: Project Charter

**Document Version**: 0.1.0
**Created**: 2025-12-26
**Status**: Phase Zero - Active
**Canonical Input**: [Anchor.md](../../../../docs/Anchor.md)

---

## 1. Mission Statement

cc-anticipation converts stabilized motion state into actionable anticipatory signals. It answers one question: **"What futures are cheap vs expensive given what's happening now?"**

It does not generate motion. It does not generate sound. It does not decide policy.
It **measures the geometry of possibility** around the present moment.

---

## 2. Scope Boundaries

### 2.1 In Scope

| Responsibility | Description |
|----------------|-------------|
| **MotionWindow consumption** | Accept aligned, canonical motion windows from cc-window-aligner |
| **Kinematic feature computation** | Compute velocity, acceleration, jerk, coordination from skeleton |
| **Latent dynamics computation** | Compute dz/dt, d²z/dt² from LIM-RPS latent stream |
| **Regime embedding** | Map motion window to regime embedding (64-256 dims) |
| **Scalar signal emission** | Emit commitment, uncertainty, transition_pressure, recovery_margin, phase_stiffness, novelty, stability |
| **Constraint proximity** | Compute balance margin, joint limit proximity, rotation commitment |
| **AnticipationPacket emission** | Emit frozen-schema packets every frame, deterministically |
| **Neighbor-based uncertainty** | Use continuation dispersion from MotionPhrase library |
| **Deterministic replay** | Same input → same output, always |

### 2.2 Out of Scope

| Excluded | Rationale |
|----------|-----------|
| Raw sensor ingestion | Handled by cc-mcs-headless, mocopi-udp-ingest |
| Time alignment | Handled by cc-window-aligner |
| Policy decisions | Handled by Conductor |
| Motion generation | Handled by cc-motiongen |
| Audio generation | Handled by MotionStrudel, EchelonDiT |
| Semantic labeling | No "spin coming" labels; only actionable scalars |

### 2.3 Interface Contracts

```
INPUT:  MotionWindow (from cc-window-aligner)
        ├── SkeletonFrame[] (optional)
        ├── LatentFrame[] (optional)
        ├── fps, window_length, coverage
        └── window_id (deterministic hash)

OUTPUT: AnticipationPacket
        ├── scalars: commitment, uncertainty, transition_pressure,
        │            recovery_margin, phase_stiffness, novelty, stability
        ├── vectors: regime_embedding, constraint_vector, derivative_summary
        └── debug: raw_features, neighbor_ids, intermediate_scores
```

---

## 3. Success Criteria

### 3.1 Determinism (Non-Negotiable)

- [ ] Same MotionWindow + same config → identical AnticipationPacket bytes
- [ ] Replay of recorded session produces checksum-identical packets
- [ ] No floating-point nondeterminism in feature computation

### 3.2 Behavioral Tests

| Test | Criterion |
|------|-----------|
| **Stillness** | motion_energy ≈ 0 for 1s → commitment↓, transition_pressure≈0, stability↑ |
| **Intentional commit** | Turn/sweep → commitment↑, uncertainty↓ BEFORE visual peak |
| **Branching** | Ambiguous feints → uncertainty↑ |
| **Repetition** | Same move repeated → novelty↓ across repeats |

### 3.3 Integration Tests

- [ ] AnticipationPacket feeds Conductor policy signals correctly
- [ ] regime_embedding queries RAG++ MotionPhrase index successfully
- [ ] coverage < threshold → packet refused or marked degraded

---

## 4. Architecture Position

```
Sensors → cc-mcs-headless → cc-window-aligner → [cc-anticipation] → Conductor
                                     ↓                    ↓
                              (MotionWindow)      (AnticipationPacket)
                                                         ↓
                                               RAG++ / MotionGen / MotionStrudel
```

### 4.1 Upstream Dependencies

| Module | Provides |
|--------|----------|
| cc-window-aligner | MotionWindow with canonical fps, aligned frames |
| cc-mcs-headless | LIM-RPS latent stream, device fusion |
| mocopi-udp-ingest | SkeletonFrame stream |

### 4.2 Downstream Consumers

| Consumer | Uses |
|----------|------|
| Conductor | scalars (commitment, uncertainty, transition_pressure, etc.) |
| RAG++ | regime_embedding for neighbor retrieval |
| MotionGen | uncertainty → variance control |
| MotionStrudel | transition_pressure → event gating |

---

## 5. Implementation Phases

### Phase v0: Deterministic Rule-Grounded Anticipator
- Kinematic features from skeleton (FK, keypoints, derivatives)
- Latent dynamics from LIM-RPS (dz/dt, d²z/dt²)
- Fixed projection for regime_embedding (no learning)
- Heuristic-based scalar computation
- Full replay determinism
- Behavioral test suite

### Phase v1: Learned Residuals
- Train encoder on MotionPhrase library
- Learn continuation dispersion → uncertainty
- Personalized "tells" from user's movement vocabulary
- Same packet schema, improved accuracy

### Phase v2: RAG++ Integration
- Query MotionPhrase neighbors using regime_embedding
- Retrieved continuation statistics sharpen priors
- Memory-augmented anticipation

---

## 6. Frozen Schemas

These schemas are immutable once defined. Changes require version bump.

### 6.1 MotionWindow (Input)
```typescript
interface MotionWindow {
  window_id: string;           // deterministic hash
  t_start: number;             // canonical time
  t_end: number;
  fps: number;                 // e.g., 50
  frames: (SkeletonFrame | LatentFrame)[];
  coverage: number;            // 0-1
  device_offset: Record<string, number>;
  dropped_reason?: string;
}
```

### 6.2 AnticipationPacket (Output)
```typescript
interface AnticipationPacket {
  // Scalars (control-ready)
  commitment: number;          // 0-1, irreversibility
  uncertainty: number;         // 0-1, continuation dispersion
  transition_pressure: number; // d(commitment)/dt - d(uncertainty)/dt
  recovery_margin: number;     // distance to balance/attractor loss
  phase_stiffness: number;     // how locked to internal metronome
  novelty: number;             // distance from recent regimes
  stability: number;           // local stationarity

  // Vectors (query/conditioning)
  regime_embedding: Float32Array;  // 64-256 dims
  constraint_vector: Float32Array; // ~8 dims
  derivative_summary: Float32Array; // ~8 dims

  // Debug/Telemetry
  debug: {
    raw_features: Record<string, number>;
    neighbor_ids?: string[];
    intermediate_scores: Record<string, number>;
  };

  // Provenance
  window_id: string;
  timestamp: number;
  schema_version: string;
}
```

---

## 7. Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Primary language** | Rust (cc-core-rs pattern) | Performance, determinism, PyO3 bindings |
| **Python bindings** | PyO3/maturin | Interop with cc-core, dashboard |
| **Ring buffer** | LockFreeRingBuffer from cc-core-rs | Already proven, zero-copy |
| **Filtering** | OneEuroFilter, SlewLimiter from cc-core-rs | Existing infrastructure |
| **ANN search** | HNSW from rag_plusplus | Regime embedding queries |
| **Statistics** | Welford/OutcomeStats from rag_plusplus | Running mean/variance |

---

## 8. Risk Register

| Risk | Mitigation |
|------|------------|
| Floating-point nondeterminism | Use deterministic math, avoid parallel reductions |
| Latency in kernel path | Profile early, no allocations in hot path |
| Skeleton FK errors | Validate bone hierarchy, unit tests |
| Regime embedding drift | Freeze projection matrix, version embeddings |
| Undefined behavior at boundaries | Explicit coverage checks, refuse low-quality windows |

---

## 9. References

- [Anchor.md](../../../../docs/Anchor.md) - Canonical specification
- [cc-core-rs](../../cc-core-rs/) - Ring buffers, filters, equilibrium solvers
- [rag_plusplus](../../rag_plusplus/) - HNSW index, OutcomeStats
- [RAG_PLUS_PLUS.md](../../cc-trajectory/docs/guides/RAG_PLUS_PLUS.md) - RAG++ architecture

---

**Next Document**: [GLOSSARY.md](./GLOSSARY.md)
