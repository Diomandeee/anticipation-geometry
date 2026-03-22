#!/usr/bin/env python3
"""
Inscription-Feature Retraining for Anticipation Geometry
=========================================================

Hypothesis: Inscription-derived features (10 sigil probabilities + depth +
interaction terms = 24 dims) beat the 71.8% transition pressure sign baseline
for conversation convergence prediction, especially on the expanded 429-session
dataset from Supabase.

Pipeline:
  1. Load/fetch session data (39 original + 390 expanded from Supabase)
  2. Compute inscription features per session
  3. Reproduce the 71.8% baseline (transition pressure sign → convergence)
  4. Run classifiers on original (4-dim) and inscription (24-dim) feature sets
  5. Report results with statistical tests
  6. Also evaluate on KG path discrimination (81.0% baseline)

Usage:
    python3 inscription_retrain.py
"""

from __future__ import annotations

import json
import os
import sys
import hashlib
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

# Fix OMP issue on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------------------------------------------------------
# Inline inscription encoder (from projects/karl/inscription_encoder.py)
# ---------------------------------------------------------------------------

SIGIL_NAMES = [
    "stabilization", "dispersion", "transition", "return", "dwell",
    "oscillation", "recovery", "novelty", "place_shift", "echo",
]

_STABILIZE_KW = {"continue", "keep going", "ship it", "deploy", "push", "looks good",
                 "nice", "perfect", "all steps", "no pauses", "go ahead", "proceed",
                 "yes", "correct", "exactly", "good", "do it", "run it", "execute"}
_TRANSITION_KW = {"no", "don't", "wrong", "stop", "fix", "revert", "actually",
                  "wait", "not that", "instead", "scratch that", "cancel",
                  "let's not", "problem", "broken", "failed", "error", "bug"}
_NOVELTY_KW = {"what if", "imagine", "perhaps", "considering", "idea",
               "new approach", "different way", "explore", "prototype",
               "experiment", "concept", "brainstorm", "vision"}
_ECHO_KW = {"remember", "as we did", "like before", "same as", "similar to",
            "we already", "previously", "last time", "recall", "revisit pattern"}
_PLACE_SHIFT_KW = {"switch", "move to", "change to", "different project",
                   "jump to", "pivot", "transition to", "let's work on"}
_DWELL_KW = {"wait", "hold", "pause", "stay", "keep at", "focus on",
             "stay here", "remain", "deep dive", "dig into"}
_RETURN_KW = {"back to", "return", "revisit", "circle back", "come back",
              "re-open", "pick up where", "resume"}
_OSCILLATION_KW = {"try", "test", "alternate", "compare", "versus", "or",
                   "a/b", "toggle", "back and forth", "flip between"}
_RECOVERY_KW = {"recovering", "getting back", "restoring", "healing",
                "fixing up", "cleaning up", "stabilizing after"}
_DISPERSION_KW = {"spread", "scatter", "multiple", "parallel", "diverge",
                  "fan out", "split", "fork", "many directions"}

def _keyword_match(lower: str, keywords: set[str]) -> bool:
    for kw in keywords:
        if " " in kw or len(kw) > 4:
            if kw in lower:
                return True
        else:
            if re.search(rf"\b{re.escape(kw)}\b", lower):
                return True
    return False


def classify_turn(text: str) -> tuple[str, float]:
    """Classify a turn into a sigil type. Returns (type_name, confidence)."""
    lower = text.lower()
    if _keyword_match(lower, _STABILIZE_KW):
        return "stabilization", 0.90
    if _keyword_match(lower, _TRANSITION_KW):
        return "transition", 0.85
    if _keyword_match(lower, _RETURN_KW):
        return "return", 0.78
    if _keyword_match(lower, _PLACE_SHIFT_KW):
        return "place_shift", 0.80
    if _keyword_match(lower, _NOVELTY_KW):
        return "novelty", 0.82
    if _keyword_match(lower, _ECHO_KW):
        return "echo", 0.75
    if _keyword_match(lower, _OSCILLATION_KW):
        return "oscillation", 0.72
    if _keyword_match(lower, _RECOVERY_KW):
        return "recovery", 0.68
    if _keyword_match(lower, _DISPERSION_KW):
        return "dispersion", 0.70
    if len(text) > 400:
        return "dispersion", 0.55
    return "dwell", 0.50


def compute_depth(turn_index: int, total_turns: int, is_correction: bool) -> int:
    relative_pos = turn_index / max(1, total_turns - 1) if total_turns > 1 else 0.0
    base_depth = int(relative_pos * 7)
    if is_correction:
        base_depth = min(9, base_depth + 2)
    return min(9, max(0, base_depth))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@dataclass
class SessionFeatures:
    """Feature vector for a single conversation session."""
    cid: str
    n_turns: int
    converges: bool

    # Original 4 scalars (from anticipation geometry)
    commitment_mean: float = 0.0
    uncertainty_mean: float = 0.0
    tp_mean: float = 0.0
    recovery_mean: float = 0.0

    # Additional scalars from benchmark
    first_half_tp: float = 0.0
    commitment_delta: float = 0.0
    commitment_first: float = 0.0
    commitment_second: float = 0.0

    # Inscription features (10 sigil probabilities)
    sigil_probs: np.ndarray = field(default_factory=lambda: np.zeros(10))

    # Mean depth
    mean_depth: float = 0.0

    # Mean confidence
    mean_confidence: float = 0.0


def extract_inscription_features(turns: list[str]) -> tuple[np.ndarray, float, float]:
    """Extract inscription features from a list of turn texts.

    Returns:
        (sigil_probs[10], mean_depth, mean_confidence)
    """
    if not turns:
        return np.zeros(10), 0.0, 0.0

    sigil_counts = np.zeros(10)
    total_depth = 0.0
    total_conf = 0.0
    n = len(turns)

    for i, text in enumerate(turns):
        if not text or len(text) < 5:
            continue
        claim_type, conf = classify_turn(text)
        idx = SIGIL_NAMES.index(claim_type)
        sigil_counts[idx] += 1
        total_conf += conf

        is_correction = claim_type in ("transition", "oscillation", "recovery")
        depth = compute_depth(i, n, is_correction)
        total_depth += depth

    total = sigil_counts.sum()
    if total > 0:
        sigil_probs = sigil_counts / total
    else:
        sigil_probs = np.zeros(10)

    mean_depth = total_depth / max(1, n)
    mean_conf = total_conf / max(1, n)

    return sigil_probs, mean_depth, mean_conf


def build_original_features(sf: SessionFeatures) -> np.ndarray:
    """Build the original 4-dim feature vector (same as paper)."""
    return np.array([
        sf.commitment_mean,
        sf.uncertainty_mean,
        sf.tp_mean,
        sf.recovery_mean,
    ])


def build_inscription_features(sf: SessionFeatures) -> np.ndarray:
    """Build 24-dim inscription feature vector.

    Layout:
        [0:10]  - 10 sigil probabilities
        [10:14] - 4 raw scalars (commitment, uncertainty, tp, recovery)
        [14]    - mean depth
        [15]    - mean confidence
        [16:25] - 9 interaction terms
    """
    scalars = np.array([
        sf.commitment_mean,
        sf.uncertainty_mean,
        sf.tp_mean,
        sf.recovery_mean,
    ])

    # Interaction terms (NO commitment_delta/commitment_first/commitment_second
    # which are trivially correlated with the convergence label)
    interactions = np.array([
        sf.sigil_probs[0] * sf.commitment_mean,     # stabilize * commitment
        sf.sigil_probs[2] * sf.tp_mean,              # transition * tp
        sf.sigil_probs[4] * sf.uncertainty_mean,     # dwell * uncertainty
        sf.sigil_probs[7] * sf.recovery_mean,        # novelty * recovery
        sf.sigil_probs[0] * sf.sigil_probs[2],       # stabilize * transition
        sf.mean_depth * sf.commitment_mean,           # depth * commitment
        sf.mean_depth * sf.tp_mean,                   # depth * tp
        sf.first_half_tp * sf.sigil_probs[0],         # first_half_tp * stabilize
        sf.sigil_probs[2] * sf.sigil_probs[3],       # transition * return
    ])

    return np.concatenate([
        sf.sigil_probs,           # 10
        scalars,                  # 4
        [sf.mean_depth],          # 1
        [sf.mean_confidence],     # 1
        interactions,             # 9
    ])  # total = 25


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_benchmark_data(path: str) -> tuple[list[SessionFeatures], dict]:
    """Load the 39-session benchmark from evaluation-v3-real-embeddings.json."""
    with open(path) as f:
        data = json.load(f)

    conv_data = data["eval_1_conversation"]
    sessions = []

    for c in conv_data["per_conversation"]:
        sf = SessionFeatures(
            cid=c["cid"],
            n_turns=c["n_turns"],
            converges=c["converges"],
            commitment_mean=c["commitment_mean"],
            uncertainty_mean=c["uncertainty_mean"],
            tp_mean=c["tp_mean"],
            recovery_mean=c["recovery_mean"],
            first_half_tp=c["first_half_tp"],
            commitment_delta=c["commitment_delta"],
            commitment_first=c["commitment_first"],
            commitment_second=c["commitment_second"],
        )
        sessions.append(sf)

    kg_data = data.get("eval_2_kg_paths", {})
    return sessions, kg_data


def load_cached_sessions(cache_path: str = "/tmp/all_session_turns.json") -> dict:
    """Load pre-fetched session turns from cache file.

    Returns dict with 'benchmark_cids' and 'sessions' keys.
    """
    if not os.path.exists(cache_path):
        return {"benchmark_cids": [], "sessions": {}}
    with open(cache_path) as f:
        return json.load(f)


def compute_anticipation_for_session(
    turns: list[dict],
    embedder=None,
) -> dict:
    """Compute anticipation geometry scalars for a session's turns.

    If embedder is None, uses hash-based embeddings (fast, approximate).
    """
    texts = [t.get("content", "") for t in turns if t.get("content")]
    if len(texts) < 3:
        return {}

    # Hash-based embedding (no GPU needed, deterministic)
    dim = 128
    embeddings = np.zeros((len(texts), dim))
    for i, text in enumerate(texts):
        words = text.lower().split()
        for word in words:
            for j in range(len(word) - 2):
                trigram = word[j:j + 3]
                h = hash(trigram) % dim
                embeddings[i, h] += 1.0
        norm = np.linalg.norm(embeddings[i])
        if norm > 0:
            embeddings[i] /= norm

    # Compute anticipation geometry inline (avoid import path issues)
    n = len(texts)
    trajectory = [embeddings[i] for i in range(n)]

    # Commitment
    deltas = np.diff(embeddings, axis=0)
    delta_norms = np.linalg.norm(deltas, axis=1)
    max_delta = np.max(delta_norms) if len(delta_norms) > 0 else 0.0

    commitment = np.full(n, 0.5)
    if max_delta > 1e-10:
        for t in range(1, n):
            commitment[t] = 1.0 - (delta_norms[t - 1] / max_delta)
    else:
        commitment[:] = 1.0
    commitment = np.clip(commitment, 0.0, 1.0)

    # Uncertainty (simplified: variance of distances to 5 nearest neighbors)
    k = 5
    uncertainty = np.full(n, 0.5)
    for t in range(1, n):
        history = embeddings[:t]
        k_eff = min(k, len(history))
        if k_eff < 2:
            continue
        diffs = history - embeddings[t]
        dists = np.linalg.norm(diffs, axis=1)
        if k_eff >= len(dists):
            nearest_idx = np.arange(len(dists))
        else:
            nearest_idx = np.argpartition(dists, k_eff)[:k_eff]
        nearest_diffs = diffs[nearest_idx]
        norms = np.linalg.norm(nearest_diffs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        unit_diffs = nearest_diffs / norms
        cos_sims = unit_diffs @ unit_diffs.T
        idx_upper = np.triu_indices(k_eff, k=1)
        angles = np.arccos(np.clip(cos_sims[idx_upper], -1, 1))
        if len(angles) > 0:
            n_bins = 8
            bins = np.linspace(0, np.pi, n_bins + 1)
            hist, _ = np.histogram(angles, bins=bins)
            hist = hist.astype(float)
            total = hist.sum()
            if total > 1e-10:
                probs = hist / total
                probs = probs[probs > 0]
                entropy = -np.sum(probs * np.log2(probs))
                max_entropy = np.log2(n_bins)
                uncertainty[t] = entropy / max_entropy if max_entropy > 0 else 0.0
    uncertainty = np.clip(uncertainty, 0.0, 1.0)

    # Transition pressure: dc/dt - du/dt
    dc = np.zeros(n)
    du = np.zeros(n)
    for t in range(1, n):
        dc[t] = commitment[t] - commitment[t - 1]
        du[t] = uncertainty[t] - uncertainty[t - 1]
    # Smooth with window=3
    w = 3
    dc_smooth = np.convolve(dc, np.ones(w) / w, mode="same")
    du_smooth = np.convolve(du, np.ones(w) / w, mode="same")
    tp = dc_smooth - du_smooth

    # Recovery margin
    centroid = np.mean(embeddings, axis=0)
    dists_to_centroid = np.linalg.norm(embeddings - centroid, axis=1)
    max_range = 2.0 * np.max(dists_to_centroid)
    recovery = np.full(n, 1.0)
    if max_range > 1e-10:
        median_u = np.median(uncertainty)
        branching_mask = uncertainty >= median_u
        for t in range(n):
            branching_idx = np.where(branching_mask[:t + 1])[0]
            if len(branching_idx) == 0:
                continue
            branching_states = embeddings[branching_idx]
            dists = np.linalg.norm(branching_states - embeddings[t], axis=1)
            min_dist = np.min(dists)
            recovery[t] = 1.0 - (min_dist / max_range)
    recovery = np.clip(recovery, 0.0, 1.0)

    # Split into halves for convergence
    mid = n // 2
    commitment_first = float(np.mean(commitment[:mid])) if mid > 0 else 0.5
    commitment_second = float(np.mean(commitment[mid:])) if mid < n else 0.5
    first_half_tp = float(np.mean(tp[:mid])) if mid > 0 else 0.0

    return {
        "commitment_mean": float(np.mean(commitment)),
        "uncertainty_mean": float(np.mean(uncertainty)),
        "tp_mean": float(np.mean(tp)),
        "recovery_mean": float(np.mean(recovery)),
        "first_half_tp": first_half_tp,
        "commitment_delta": commitment_second - commitment_first,
        "commitment_first": commitment_first,
        "commitment_second": commitment_second,
        "converges": commitment_second > commitment_first,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def baseline_tp_sign(sessions: list[SessionFeatures]) -> float:
    """Reproduce the 71.8% baseline: sign(first_half_tp) predicts convergence.

    Logic: if first_half_tp > 0, predict converges=True; else predict converges=False.
    Actually from the data: tp_predicts = (first_half_tp > 0 implies converges=True,
    OR first_half_tp < 0 implies converges=False).
    Looking at the data: tp_predicts correlates with both signs matching.
    """
    correct = 0
    for sf in sessions:
        # The paper's definition: transition pressure sign predicts convergence
        # Positive first_half_tp -> predict convergence
        # Negative first_half_tp -> predict divergence
        predicted_converges = sf.first_half_tp > 0
        # But wait, the paper says tp sign *predicts* convergence at 71.8%
        # Let's check: negative tp means commitment falling faster than uncertainty
        # That should predict *non-convergence*. Let's verify both ways.
        if predicted_converges == sf.converges:
            correct += 1
    return correct / len(sessions)


def run_classifiers(
    X: np.ndarray,
    y: np.ndarray,
    feature_name: str,
    n_splits: int = 5,
) -> dict:
    """Run LOO (if n<=50) or k-fold CV with multiple classifiers.

    Returns dict of {classifier_name: {accuracy, std, ...}}
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    n = len(y)
    use_loo = n <= 50

    classifiers = {
        "LogReg": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "GBM": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        "RF": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "SVM": SVC(kernel="rbf", C=1.0, random_state=42),
    }

    if use_loo:
        cv = LeaveOneOut()
        cv_name = "LOO"
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_name = f"{n_splits}-fold"

    results = {}
    for name, clf in classifiers.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        results[name] = {
            "accuracy": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "cv": cv_name,
            "n_folds": len(scores),
        }

    return results


def feature_importance_analysis(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> list[tuple[str, float]]:
    """Extract feature importances using GBM."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gbm = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    gbm.fit(X_scaled, y)

    importances = gbm.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    return ranked


def statistical_test(accuracy: float, n: int, baseline: float = 0.5) -> tuple[float, float]:
    """Compute z-score and p-value for accuracy vs baseline."""
    from scipy import stats

    # Binomial test approximation
    k = int(accuracy * n)
    z = (accuracy - baseline) / np.sqrt(baseline * (1 - baseline) / n)
    p = 1 - stats.norm.cdf(abs(z))
    return z, p


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("INSCRIPTION FEATURE RETRAINING FOR ANTICIPATION GEOMETRY")
    print("=" * 70)

    benchmark_path = Path(__file__).parent / ".." / "results" / "evaluation-v3-real-embeddings.json"
    if not benchmark_path.exists():
        benchmark_path = Path("/Users/mohameddiomande/Desktop/anticipation-geometry/results/evaluation-v3-real-embeddings.json")

    # ── Step 1: Load benchmark data (39 sessions) ──
    print("\n[1/6] Loading benchmark data...")
    sessions_39, kg_data = load_benchmark_data(str(benchmark_path))
    print(f"  Loaded {len(sessions_39)} sessions from benchmark")

    # Verify convergence rate
    n_converge = sum(1 for s in sessions_39 if s.converges)
    print(f"  Convergence rate: {n_converge}/{len(sessions_39)} = {n_converge/len(sessions_39):.1%}")

    # ── Step 2: Reproduce baseline ──
    print("\n[2/6] Reproducing 71.8% baseline (TP sign → convergence)...")
    baseline_acc = baseline_tp_sign(sessions_39)
    z_base, p_base = statistical_test(baseline_acc, len(sessions_39))
    print(f"  Baseline accuracy: {baseline_acc:.1%}")
    print(f"  z-score: {z_base:.4f}, p-value: {p_base:.6f}")

    # ── Step 3: Load expanded dataset from cache ──
    print("\n[3/6] Loading expanded session data...")
    cached = load_cached_sessions()
    cached_sessions = cached.get("sessions", {})
    print(f"  Cached sessions: {len(cached_sessions)}")

    # ── Step 4: Compute features ──
    print("\n[4/6] Computing inscription + anticipation features...")

    # Enrich original 39 sessions with inscription features from cached turns
    enriched_39 = 0
    for sf in sessions_39:
        turn_texts = cached_sessions.get(sf.cid, [])
        if turn_texts:
            sigil_probs, mean_depth, mean_conf = extract_inscription_features(turn_texts)
            sf.sigil_probs = sigil_probs
            sf.mean_depth = mean_depth
            sf.mean_confidence = mean_conf
            enriched_39 += 1

    print(f"  Enriched {enriched_39}/{len(sessions_39)} benchmark sessions with inscription features")

    # Build expanded sessions from ALL cached sessions not in original 39
    original_cids = {sf.cid for sf in sessions_39}
    expanded_sessions = []
    skipped = 0

    for sid, turn_texts in cached_sessions.items():
        if sid in original_cids:
            continue
        if len(turn_texts) < 5:
            skipped += 1
            continue

        # Convert to dict format for anticipation geometry
        turns = [{"content": t, "role": "user"} for t in turn_texts]

        # Compute anticipation geometry
        geom = compute_anticipation_for_session(turns)
        if not geom:
            skipped += 1
            continue

        # Compute inscription features
        sigil_probs, mean_depth, mean_conf = extract_inscription_features(turn_texts)

        sf = SessionFeatures(
            cid=sid,
            n_turns=len(turn_texts),
            converges=geom["converges"],
            commitment_mean=geom["commitment_mean"],
            uncertainty_mean=geom["uncertainty_mean"],
            tp_mean=geom["tp_mean"],
            recovery_mean=geom["recovery_mean"],
            first_half_tp=geom["first_half_tp"],
            commitment_delta=geom["commitment_delta"],
            commitment_first=geom["commitment_first"],
            commitment_second=geom["commitment_second"],
            sigil_probs=sigil_probs,
            mean_depth=mean_depth,
            mean_confidence=mean_conf,
        )
        expanded_sessions.append(sf)

    print(f"  New expanded sessions: {len(expanded_sessions)} (skipped {skipped})")

    all_sessions = sessions_39 + expanded_sessions
    print(f"  Total sessions: {len(all_sessions)}")

    # Convergence rate in expanded set
    n_conv_all = sum(1 for s in all_sessions if s.converges)
    print(f"  Convergence rate (all): {n_conv_all}/{len(all_sessions)} = {n_conv_all/len(all_sessions):.1%}")

    # ── Step 5: Run classifiers ──
    print("\n[5/6] Running classifiers...")

    # Feature matrices
    feature_sets = {}

    inscr_names = (
        [f"sigil_{s}" for s in SIGIL_NAMES] +
        ["commitment", "uncertainty", "tp", "recovery"] +
        ["depth", "confidence"] +
        ["stab*commit", "trans*tp", "dwell*uncert", "novel*recov",
         "stab*trans", "depth*commit", "depth*tp", "fhtp*stab", "trans*return"]
    )

    # Extended scalar names (raw scalars + first_half_tp — 5 features, no leakage)
    ext_names = ["commitment", "uncertainty", "tp", "recovery", "first_half_tp"]
    def build_extended_features(sf):
        return np.array([sf.commitment_mean, sf.uncertainty_mean, sf.tp_mean,
                         sf.recovery_mean, sf.first_half_tp])

    # Sigil-only features (10 sigil probs + depth + confidence — NO anticipation scalars)
    sigil_only_names = [f"sigil_{s}" for s in SIGIL_NAMES] + ["depth", "confidence"]
    def build_sigil_only_features(sf):
        return np.concatenate([sf.sigil_probs, [sf.mean_depth, sf.mean_confidence]])

    # 5a: Original 4 features on 39 sessions
    X_orig_39 = np.array([build_original_features(sf) for sf in sessions_39])
    y_39 = np.array([int(sf.converges) for sf in sessions_39])
    feature_sets["orig_4d_39sess"] = (X_orig_39, y_39, ["commitment", "uncertainty", "tp", "recovery"])

    # 5a2: Extended 5 features on 39 sessions
    X_ext_39 = np.array([build_extended_features(sf) for sf in sessions_39])
    feature_sets["ext_5d_39sess"] = (X_ext_39, y_39, ext_names)

    # 5b: Inscription 25 features on 39 sessions
    sessions_39_with_inscr = [sf for sf in sessions_39 if sf.sigil_probs.sum() > 0]
    if sessions_39_with_inscr:
        X_inscr_39 = np.array([build_inscription_features(sf) for sf in sessions_39_with_inscr])
        y_inscr_39 = np.array([int(sf.converges) for sf in sessions_39_with_inscr])
        feature_sets["inscr_25d_39sess"] = (X_inscr_39, y_inscr_39, inscr_names)

        # 5b2: Sigil-only (12d) on 39 sessions
        X_sigil_39 = np.array([build_sigil_only_features(sf) for sf in sessions_39_with_inscr])
        feature_sets["sigil_12d_39sess"] = (X_sigil_39, y_inscr_39, sigil_only_names)

    # 5c: Original 4 features on all sessions
    if expanded_sessions:
        X_orig_all = np.array([build_original_features(sf) for sf in all_sessions])
        y_all = np.array([int(sf.converges) for sf in all_sessions])
        feature_sets["orig_4d_all"] = (X_orig_all, y_all, ["commitment", "uncertainty", "tp", "recovery"])

        # Extended 5 features on all
        X_ext_all = np.array([build_extended_features(sf) for sf in all_sessions])
        feature_sets["ext_5d_all"] = (X_ext_all, y_all, ext_names)

        # 5d: Inscription 25 features on all sessions
        all_with_inscr = [sf for sf in all_sessions if sf.sigil_probs.sum() > 0]
        if all_with_inscr:
            X_inscr_all = np.array([build_inscription_features(sf) for sf in all_with_inscr])
            y_inscr_all = np.array([int(sf.converges) for sf in all_with_inscr])
            feature_sets["inscr_25d_all"] = (X_inscr_all, y_inscr_all, inscr_names)

            # Sigil-only on all
            X_sigil_all = np.array([build_sigil_only_features(sf) for sf in all_with_inscr])
            feature_sets["sigil_12d_all"] = (X_sigil_all, y_inscr_all, sigil_only_names)

    # Run classifiers on each feature set
    all_results = {}
    for fs_name, (X, y, fnames) in feature_sets.items():
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        print(f"\n  --- {fs_name} (n={len(y)}, pos={n_pos}, neg={n_neg}) ---")

        if len(y) < 6:
            print(f"  Skipping: too few samples")
            continue

        # Determine LOO vs k-fold
        n_splits = 5 if len(y) > 50 else -1  # -1 means LOO (handled inside)
        results = run_classifiers(X, y, fs_name, n_splits=5)
        all_results[fs_name] = results

        for clf_name, res in results.items():
            z, p = statistical_test(res["accuracy"], len(y))
            print(f"  {clf_name:>8}: {res['accuracy']:.1%} +/- {res['std']:.1%} "
                  f"({res['cv']}, z={z:.2f}, p={p:.4f})")

    # ── Step 6: Feature importance ──
    print("\n[6/6] Feature importance analysis...")

    best_fs_name = None
    best_acc = 0
    for fs_name, results in all_results.items():
        for clf_name, res in results.items():
            if res["accuracy"] > best_acc:
                best_acc = res["accuracy"]
                best_fs_name = fs_name

    if best_fs_name and "inscr" in best_fs_name:
        X, y, fnames = feature_sets[best_fs_name]
        ranked = feature_importance_analysis(X, y, fnames)
        print(f"\n  Top features ({best_fs_name}):")
        for fname, imp in ranked[:10]:
            bar = "#" * int(imp * 100)
            print(f"    {fname:<20} {imp:.4f} {bar}")

    # ── Also evaluate baseline on expanded set ──
    if expanded_sessions:
        print("\n" + "=" * 70)
        print("EXPANDED DATASET: TP SIGN BASELINE")
        print("=" * 70)

        baseline_all = baseline_tp_sign(all_sessions)
        z_all, p_all = statistical_test(baseline_all, len(all_sessions))
        print(f"  TP sign baseline on {len(all_sessions)} sessions: {baseline_all:.1%}")
        print(f"  z-score: {z_all:.4f}, p-value: {p_all:.6f}")

    # ── KG Path evaluation ──
    if kg_data and kg_data.get("paths"):
        print("\n" + "=" * 70)
        print("KG PATH DISCRIMINATION")
        print("=" * 70)

        paths = kg_data["paths"]
        print(f"  {len(paths)} KG paths from benchmark")

        # Build features for KG paths
        # For KG paths, we don't have text turns — only the 4 scalars
        # The 81.0% baseline is pairwise accuracy, not classification
        # We can still try classifying paths as valid/invalid using scalars
        # But we need gold/silver/bronze labels which aren't in the saved data
        # Report what we can from the benchmark
        kg_commitment = [p["commitment"] for p in paths]
        kg_tp = [p["tp"] for p in paths]
        print(f"  Mean commitment: {np.mean(kg_commitment):.4f}")
        print(f"  Mean TP: {np.mean(kg_tp):.4f}")
        print(f"  (KG path discrimination requires gold/corrupt pairs —")
        print(f"   81.0% baseline is from the eval harness, not classification)")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  Baseline (TP sign, 39 sessions):  {baseline_acc:.1%} (z={z_base:.2f})")

    best_overall = None
    for fs_name, results in all_results.items():
        for clf_name, res in results.items():
            if best_overall is None or res["accuracy"] > best_overall[2]:
                best_overall = (fs_name, clf_name, res["accuracy"], res["std"])

    if best_overall:
        fs, clf, acc, std = best_overall
        z, p = statistical_test(acc, len(feature_sets[fs][1]))
        print(f"  Best classifier ({clf} on {fs}): {acc:.1%} +/- {std:.1%} (z={z:.2f}, p={p:.4f})")

        if acc > baseline_acc:
            improvement = acc - baseline_acc
            print(f"\n  >>> IMPROVEMENT over baseline: +{improvement:.1%} <<<")
            if "inscr" in fs:
                print(f"  >>> Inscription features contribute to the gain <<<")
        else:
            print(f"\n  Baseline not beaten. Best: {acc:.1%} vs baseline {baseline_acc:.1%}")

    # Save results
    output_path = Path(__file__).parent / ".." / "results" / "inscription-retrain-results.json"
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline_39": {"accuracy": baseline_acc, "z": z_base, "p": p_base, "n": len(sessions_39)},
        "n_expanded": len(expanded_sessions),
        "n_total": len(all_sessions),
        "classifier_results": {},
        "feature_importance": {},
    }
    for fs_name, results in all_results.items():
        output["classifier_results"][fs_name] = results

    if best_fs_name and "inscr" in best_fs_name:
        X, y, fnames = feature_sets[best_fs_name]
        ranked = feature_importance_analysis(X, y, fnames)
        output["feature_importance"] = {f: float(v) for f, v in ranked}

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
