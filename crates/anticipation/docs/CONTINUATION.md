# cc-anticipation: Continuation Protocol

**Document Version**: 0.1.0
**Created**: 2025-12-26
**Parent**: [PROJECT_CHARTER.md](./PROJECT_CHARTER.md)

---

## Purpose

This document defines the protocol for continuing work on cc-anticipation across sessions, ensuring consistent context restoration and preventing drift from canonical specifications.

---

## 1. Mandatory Reading Order

When resuming work on cc-anticipation, read documents in this order:

### 1.1 Context Documents (Always Read)
1. **[PROJECT_CHARTER.md](./PROJECT_CHARTER.md)** - Mission, scope, interfaces
2. **[CHECKLIST.md](./CHECKLIST.md)** - Current status, next actions
3. **[INVARIANTS.md](./INVARIANTS.md)** - Non-negotiable properties

### 1.2 Reference Documents (Read as Needed)
4. **[GLOSSARY.md](./GLOSSARY.md)** - Terminology definitions
5. **[IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md)** - Code structure, patterns
6. **[Anchor.md](../../../../docs/Anchor.md)** - Canonical specification

### 1.3 Integration Context (Read for Integration Work)
7. **[cc-core-rs/src/lib.rs](../../cc-core-rs/src/lib.rs)** - Available primitives
8. **[rag_plusplus types](../../rag_plusplus/)** - HNSW, OutcomeStats

---

## 2. Status Check Protocol

Before any implementation work, execute this checklist:

### 2.1 Repository State
```bash
cd /Users/mohameddiomande/Desktop/Comp-Core/core/cc-anticipation

# Check current state
ls -la src/ 2>/dev/null || echo "src/ not yet created"
cat Cargo.toml 2>/dev/null || echo "Cargo.toml not yet created"

# Check for any existing tests
ls tests/ 2>/dev/null || echo "tests/ not yet created"
```

### 2.2 Dependency State
```bash
# Verify cc-core-rs builds
cd ../cc-core-rs && cargo check

# Verify rag_plusplus builds (if used)
cd ../rag_plusplus && cargo check
```

### 2.3 Checklist State
Open [CHECKLIST.md](./CHECKLIST.md) and identify:
- Last completed item
- Current in-progress item
- Next pending item

---

## 3. Continuation Entry Points

### 3.1 Starting Phase 1 (Project Setup)
**Prerequisites**: Phase 0 complete
**Entry**: Create Cargo.toml, module structure
**Verification**: `cargo check` succeeds

### 3.2 Resuming Mid-Phase
**Prerequisites**: Identify last completed checklist item
**Entry**: Continue from next unchecked item
**Verification**: All prior items still pass

### 3.3 Bug Fix Session
**Prerequisites**: Identify failing test or invariant
**Entry**: Read relevant INVARIANTS.md section
**Verification**: Fix restores invariant, no regression

### 3.4 Integration Session
**Prerequisites**: Core kernel complete
**Entry**: Read integration context documents
**Verification**: Integration tests pass

---

## 4. Invariant Verification

Before marking any phase complete, verify these invariants:

### 4.1 Determinism (INV-001)
```rust
// Must pass:
#[test]
fn test_deterministic_replay() {
    let window = load_test_window();
    let mut k1 = AnticipationKernel::new(config);
    let mut k2 = AnticipationKernel::new(config);
    assert_eq!(k1.process(&window), k2.process(&window));
}
```

### 4.2 Schema Compliance
```rust
// Packet must have all fields
assert!(packet.commitment >= 0.0 && packet.commitment <= 1.0);
assert!(packet.uncertainty >= 0.0 && packet.uncertainty <= 1.0);
assert_eq!(packet.regime_embedding.len(), config.regime_embedding_dim);
assert_eq!(packet.schema_version, SCHEMA_VERSION);
```

### 4.3 Performance
```bash
cargo bench -- kernel_process
# Must complete in < 2ms
```

---

## 5. Documentation Update Protocol

When modifying implementation:

### 5.1 Checklist Updates
- Mark completed items with `[x]`
- Add discovered subtasks as new items
- Update "Last Updated" timestamp

### 5.2 Invariant Violations
If an invariant cannot be maintained:
1. Document the violation in INVARIANTS.md with justification
2. Update affected tests
3. Notify in commit message

### 5.3 Schema Changes
If frozen schema must change:
1. Increment SCHEMA_VERSION
2. Document migration in IMPLEMENTATION_GUIDE.md
3. Update all affected types
4. Add backwards-compatibility tests if needed

---

## 6. Handoff Template

When pausing work, create a handoff summary:

```markdown
## Session Handoff: [DATE]

### Completed This Session
- [List completed checklist items]

### In Progress
- [Current item being worked on]
- [Blocker if any]

### Next Steps
1. [Immediate next action]
2. [Following action]

### Open Questions
- [Any unresolved design decisions]

### Files Modified
- [List of files changed]

### Tests Status
- [x] Determinism tests: PASS
- [x] Unit tests: PASS
- [ ] Behavioral tests: NOT YET IMPLEMENTED
```

---

## 7. Recovery Procedures

### 7.1 Build Failure
```bash
# Clean and rebuild
cargo clean
cargo build 2>&1 | tee build.log

# Check for missing dependencies
grep "error\[E" build.log
```

### 7.2 Test Failure
```bash
# Run specific failing test with output
cargo test test_name -- --nocapture

# Check against invariants
grep -n "INVARIANT" docs/INVARIANTS.md
```

### 7.3 Determinism Failure
```bash
# Run replay harness
cargo run --bin replay_harness -- trace.bin

# Compare checksums
diff packet_run1.bin packet_run2.bin
```

---

## 8. Communication Artifacts

### 8.1 Commit Message Format
```
cc-anticipation: [PHASE.ITEM] Brief description

- Detailed change 1
- Detailed change 2

Checklist: [x] 1.2.3 - Item description
Invariants: INV-001 (determinism) verified
```

### 8.2 PR Template
```markdown
## cc-anticipation: [Phase Description]

### Checklist Items Completed
- [x] 1.2.1 - Description
- [x] 1.2.2 - Description

### Invariants Verified
- [x] INV-001 (Determinism)
- [x] INV-004 (Scalar bounds)

### Tests Added
- `test_deterministic_replay`
- `test_stillness_response`

### Benchmark Results
- kernel_process: 1.2ms (target: < 2ms)
```

---

## 9. Emergency Procedures

### 9.1 Spec Contradiction
If implementation contradicts Anchor.md:
1. **STOP** implementation
2. Document the contradiction
3. Defer to Anchor.md unless explicitly authorized to diverge
4. Update INVARIANTS.md with resolution

### 9.2 Performance Regression
If benchmark exceeds 2ms:
1. Profile with `cargo flamegraph`
2. Identify hot path allocations
3. Revert to last known-good commit if critical

### 9.3 Integration Break
If cc-core-rs or rag_plusplus API changes:
1. Pin to working version in Cargo.toml
2. Document required update in CHECKLIST.md
3. Create adaptation task

---

## 10. Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                 cc-anticipation Quick Reference              │
├─────────────────────────────────────────────────────────────┤
│ ALWAYS READ FIRST:                                          │
│   1. PROJECT_CHARTER.md                                     │
│   2. CHECKLIST.md (find current position)                   │
│   3. INVARIANTS.md (before any changes)                     │
├─────────────────────────────────────────────────────────────┤
│ KEY INVARIANTS:                                             │
│   INV-001: Deterministic replay                             │
│   INV-004: Scalars in [0,1]                                 │
│   INV-006: No hot-path allocations                          │
│   PERF-001: < 2ms per tick                                  │
├─────────────────────────────────────────────────────────────┤
│ FROZEN CONTRACTS:                                           │
│   - MotionWindow (input)                                    │
│   - AnticipationPacket (output)                             │
│   - SCHEMA_VERSION = "0.1.0"                                │
├─────────────────────────────────────────────────────────────┤
│ CANONICAL SPEC: docs/Anchor.md                              │
│ DEPENDENCIES: cc-core-rs, rag_plusplus (optional)           │
└─────────────────────────────────────────────────────────────┘
```

---

## 11. Session Start Checklist

Copy this for each new session:

```markdown
## Session Start: [DATE]

### Context Restoration
- [ ] Read PROJECT_CHARTER.md
- [ ] Read CHECKLIST.md, identified current position
- [ ] Read INVARIANTS.md for relevant section

### Environment Check
- [ ] cc-core-rs builds: `cargo check`
- [ ] cc-anticipation builds: `cargo check` (if exists)
- [ ] Tests pass: `cargo test`

### Current Position
- Last completed: [ITEM]
- Next action: [ITEM]

### Session Goals
1. [Goal 1]
2. [Goal 2]
```

---

**This protocol ensures continuity across sessions and prevents architectural drift.**

---

## Session Handoff: 2025-12-27 (Part 1)

### Completed This Session
- [x] 9.1.1 Implement `telemetry/mod.rs` with TelemetryMessage
- [x] 9.1.2 Implement AnticipationData (dashboard-friendly format)
- [x] 9.1.3 Implement TelemetryBuffer for sparklines
- [x] 9.1.4 Derive regime labels and colors from scalars
- [x] 9.1.5 Compute risk_level, transition_readiness, motion_quality
- [x] 9.2.1 Create `lib/anticipation/types.ts` (matching Rust types)
- [x] 9.2.2 Create `lib/anticipation/store.ts` (Zustand store)
- [x] 9.2.3 Update WebSocket client to handle 'anticipation' messages
- [x] 9.2.4 Add sparkline buffer utilities

---

## Session Handoff: 2025-12-27 (Part 2)

### Completed This Session
- [x] 9.3.1 Create scalar gauge components (ScalarBar, BipolarBar)
- [x] 9.3.2 Create sparkline charts (AnticipationSparklines)
- [x] 9.3.3 Display regime state with color (RegimeIndicator)
- [x] 9.3.4 Create combined AnticipationPanel
- [x] 9.3.5 Create mini indicator (AnticipationMini)

### In Progress
- None (Phase 9 core complete)

### Next Steps
1. Integration testing with live mocopi data
2. Embedding heatmap visualization (9.3.6)
3. Debug panel with raw_features (9.3.7)

### Open Questions
- None

### Files Modified
- `src/telemetry/mod.rs` (NEW) - Telemetry message and data types
- `src/lib.rs` - Added telemetry module and re-exports
- `apps/web/cc-dashboard/src/lib/anticipation/types.ts` (NEW)
- `apps/web/cc-dashboard/src/lib/anticipation/store.ts` (NEW)
- `apps/web/cc-dashboard/src/lib/anticipation/index.ts` (NEW)
- `apps/web/cc-dashboard/src/lib/websocket.ts` - Added anticipation message handling
- `apps/web/cc-dashboard/src/components/viz/AnticipationTelemetry.tsx` (NEW)
- `apps/web/cc-dashboard/src/components/viz/AnticipationSparklines.tsx` (NEW)
- `apps/web/cc-dashboard/src/components/viz/AnticipationPanel.tsx` (NEW)
- `docs/CHECKLIST.md` - Updated Phase 9 status

### Tests Status
- [x] Determinism tests: PASS
- [x] Unit tests: 43 PASS
- [x] Behavioral tests: 6 PASS
- [x] Telemetry tests: 6 PASS (in telemetry module)

**Total: 49 tests passing**

---

**Phase 9 Dashboard Integration Complete. Ready for integration testing.**
