#!/usr/bin/env python3
"""
Paper 1 Enhanced Evaluation: Anticipation Geometry as Classification Features
==============================================================================

Key question: Do anticipation features ADD value beyond raw embeddings?

Experiment design:
  1. Fetch 20K turns from Supabase, group into conversations (10+ turns)
  2. Embed with e5-large-v2 (1024-dim)
  3. Compute anticipation features (7 scalars + derivatives + cross-terms = ~30 features)
  4. Build convergence labels from content analysis (6-signal method)
  5. Train 4 classifiers and compare:
     a) Anticipation-only: logistic regression on 30 anticipation features
     b) Embedding-only: logistic regression on mean conversation embedding (PCA to 50)
     c) Combined: anticipation features + embedding features
     d) Baseline: majority class
  6. 5-fold stratified cross-validation
  7. Report: accuracy, precision, recall, F1

If combined > embedding-only, the geometric framework captures signal
that embeddings alone don't.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import requests

# sklearn imports deferred to avoid torch/OpenMP conflicts during embedding
# They will be imported at first use in build_feature_matrices / evaluate_classifier

# Fix OpenMP duplicate lib issue on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Must set torch threads before any torch import
import torch
torch.set_num_threads(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration
# ===========================================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://your-project.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

TOTAL_TURNS_TARGET = 20000
MIN_TURNS_PER_CONV = 10
EMBEDDING_MODEL = "intfloat/e5-large-v2"
EMBEDDING_DIM = 1024
PCA_DIM = 50
N_FOLDS = 5
RANDOM_SEED = 42

# Output path
OUTPUT_PATH = os.path.expanduser("~/eval_results/paper1_enhanced.json")


# ===========================================================================
# Step 1: Fetch turns from Supabase
# ===========================================================================

def fetch_turns(n_total: int = TOTAL_TURNS_TARGET) -> list[dict]:
    """Fetch turns from Supabase memory_turns table via paginated REST API.

    Returns list of dicts with keys: conversation_id, role, content_text, created_at
    """
    logger.info("Fetching %d turns from Supabase...", n_total)

    session = requests.Session()
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }

    all_turns = []
    page_size = 1000
    offset = 0

    while len(all_turns) < n_total:
        remaining = n_total - len(all_turns)
        limit = min(page_size, remaining)

        resp = session.get(
            f"{SUPABASE_URL}/rest/v1/memory_turns",
            headers=headers,
            params={
                "select": "conversation_id,role,content_text,created_at",
                "order": "created_at.desc",
                "limit": limit,
                "offset": offset,
                # Filter for turns with actual text content
                "content_text": "not.is.null",
                "role": "in.(user,assistant)",
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data:
            logger.info("No more data at offset %d", offset)
            break

        # Filter for turns with meaningful content (not just whitespace)
        for row in data:
            text = (row.get("content_text") or "").strip()
            if len(text) >= 10:  # At least 10 chars
                all_turns.append(row)

        offset += page_size
        if len(data) < limit:
            break

        if offset % 5000 == 0:
            logger.info("  ...fetched %d turns so far (offset=%d)", len(all_turns), offset)

    session.close()
    logger.info("Fetched %d valid turns total", len(all_turns))
    return all_turns


# ===========================================================================
# Step 2: Group into conversations
# ===========================================================================

def group_conversations(
    turns: list[dict],
    min_turns: int = MIN_TURNS_PER_CONV,
) -> dict[str, list[dict]]:
    """Group turns by conversation_id, keep only conversations with >= min_turns."""

    by_conv = defaultdict(list)
    for turn in turns:
        cid = turn.get("conversation_id")
        if cid:
            by_conv[cid].append(turn)

    # Sort each conversation by created_at
    for cid in by_conv:
        by_conv[cid].sort(key=lambda t: t.get("created_at", ""))

    # Filter by minimum length
    filtered = {
        cid: turns_list
        for cid, turns_list in by_conv.items()
        if len(turns_list) >= min_turns
    }

    logger.info(
        "Grouped into %d conversations (from %d total), "
        "%d with >= %d turns",
        len(by_conv), len(turns), len(filtered), min_turns,
    )

    return filtered


# ===========================================================================
# Step 3: Embed with e5-large-v2
# ===========================================================================

def embed_conversations(
    conversations: dict[str, list[dict]],
) -> dict[str, np.ndarray]:
    """Embed all turns in all conversations using e5-large-v2.

    Uses transformers directly (not sentence-transformers) to avoid
    segfault issues with Python 3.14 + sentence-transformers.

    Returns dict mapping conversation_id -> (n_turns, 1024) embedding array.
    """
    import torch
    import gc
    from transformers import AutoTokenizer, AutoModel

    torch.set_num_threads(4)

    logger.info("Loading embedding model: %s (direct transformers, CPU)", EMBEDDING_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL)
    model.eval()

    logger.info("Model loaded. Embedding %d conversations...", len(conversations))

    # Collect all texts with their (conv_id, turn_idx) mapping
    all_texts = []
    text_map = []  # (conv_id, turn_idx) for each text

    for cid, turns_list in conversations.items():
        for idx, turn in enumerate(turns_list):
            text = (turn.get("content_text") or "").strip()
            # e5 models expect "passage: " prefix
            prefixed = f"passage: {text[:1500]}"
            all_texts.append(prefixed)
            text_map.append((cid, idx))

    logger.info("Total texts to embed: %d", len(all_texts))

    # Batch embed
    batch_size = 16
    all_embeddings = []

    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i + batch_size]

        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pooling
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embs = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        # L2 normalize
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        all_embeddings.append(embs.numpy())

        done = min(i + batch_size, len(all_texts))
        if done % 200 == 0 or done >= len(all_texts):
            logger.info("  ...embedded %d / %d texts", done, len(all_texts))
            gc.collect()

    all_embeddings = np.vstack(all_embeddings)  # (N_total, 1024)
    logger.info("Embedding complete. Shape: %s", all_embeddings.shape)

    # Reconstruct per-conversation arrays
    conv_embeddings = {}
    for (cid, idx), emb in zip(text_map, all_embeddings):
        if cid not in conv_embeddings:
            n_turns = len(conversations[cid])
            conv_embeddings[cid] = np.zeros((n_turns, EMBEDDING_DIM))
        conv_embeddings[cid][idx] = emb

    return conv_embeddings


# ===========================================================================
# Step 4: Compute anticipation features (30 features per conversation)
# ===========================================================================

class AnticipationFeatureExtractor:
    """Extract 30 features from anticipation geometry scalars.

    Base scalars (4): commitment, uncertainty, transition_pressure, recovery_margin
    Statistics per scalar (7): mean, std, min, max, final, slope, range
    = 4 * 7 = 28 base features

    Cross-terms (additional):
      - commitment * (1 - uncertainty)   [focus score]
      - transition_pressure_max * recovery_margin_min  [decision tension]
    = 2 cross-terms

    Total: 30 features
    """

    FEATURE_NAMES = []

    def __init__(self, k_neighbors: int = 5):
        self.k = k_neighbors

        # Build feature name list
        scalars = ["commitment", "uncertainty", "transition_pressure", "recovery_margin"]
        stats = ["mean", "std", "min", "max", "final", "slope", "range"]

        self.FEATURE_NAMES = []
        for s in scalars:
            for st in stats:
                self.FEATURE_NAMES.append(f"{s}_{st}")

        self.FEATURE_NAMES.append("focus_score")
        self.FEATURE_NAMES.append("decision_tension")

    def extract(self, trajectory: np.ndarray) -> np.ndarray:
        """Extract 30 anticipation features from a trajectory.

        Args:
            trajectory: (n_steps, dim) array of state vectors

        Returns:
            (30,) feature vector
        """
        n = len(trajectory)
        if n < 3:
            return np.zeros(30)

        # Compute raw scalars
        commitment = self._compute_commitment(trajectory)
        uncertainty = self._compute_uncertainty(trajectory)
        transition_pressure = self._compute_transition_pressure(commitment, uncertainty)
        recovery_margin = self._compute_recovery_margin(trajectory, uncertainty)

        features = []

        for signal in [commitment, uncertainty, transition_pressure, recovery_margin]:
            features.append(np.mean(signal))
            features.append(np.std(signal))
            features.append(np.min(signal))
            features.append(np.max(signal))
            features.append(signal[-1])
            # Slope: linear regression coefficient (trend)
            x = np.arange(len(signal))
            if np.std(x) > 0 and np.std(signal) > 0:
                slope = np.corrcoef(x, signal)[0, 1] * np.std(signal) / np.std(x)
            else:
                slope = 0.0
            features.append(slope)
            # Range
            features.append(np.max(signal) - np.min(signal))

        # Cross-terms
        focus_score = float(np.mean(commitment * (1.0 - uncertainty)))
        decision_tension = float(np.max(transition_pressure) * np.min(recovery_margin))

        features.append(focus_score)
        features.append(decision_tension)

        return np.array(features, dtype=np.float64)

    def _compute_commitment(self, states: np.ndarray) -> np.ndarray:
        """commitment[t] = 1 - ||s_t - s_{t-1}|| / max_delta"""
        n = len(states)
        deltas = np.linalg.norm(np.diff(states, axis=0), axis=1)
        max_delta = np.max(deltas) if len(deltas) > 0 else 1e-10
        if max_delta < 1e-10:
            return np.ones(n)

        commitment = np.full(n, 0.5)
        for t in range(1, n):
            commitment[t] = 1.0 - (deltas[t - 1] / max_delta)
        return np.clip(commitment, 0.0, 1.0)

    def _compute_uncertainty(self, states: np.ndarray) -> np.ndarray:
        """Entropy of angular directions to K nearest historical neighbors."""
        n = len(states)
        uncertainty = np.full(n, 0.5)
        n_bins = 8

        for t in range(1, n):
            history = states[:t]
            k = min(self.k, len(history))
            if k < 2:
                continue

            diffs = history - states[t]
            dists = np.linalg.norm(diffs, axis=1)

            if k >= len(dists):
                nearest_idx = np.arange(len(dists))
            else:
                nearest_idx = np.argpartition(dists, k)[:k]

            nearest_diffs = diffs[nearest_idx]
            norms = np.linalg.norm(nearest_diffs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            unit_diffs = nearest_diffs / norms

            cos_sims = unit_diffs @ unit_diffs.T
            triu_idx = np.triu_indices(len(unit_diffs), k=1)
            angles = np.arccos(np.clip(cos_sims[triu_idx], -1, 1))

            if len(angles) == 0:
                continue

            bins = np.linspace(0, np.pi, n_bins + 1)
            hist, _ = np.histogram(angles, bins=bins)
            total = hist.sum()
            if total < 1:
                continue

            probs = hist / total
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            max_entropy = np.log2(n_bins)
            uncertainty[t] = entropy / max_entropy if max_entropy > 0 else 0.0

        return np.clip(uncertainty, 0.0, 1.0)

    def _compute_transition_pressure(
        self, commitment: np.ndarray, uncertainty: np.ndarray,
    ) -> np.ndarray:
        """p(t) = dc/dt - du/dt with smoothing."""
        n = len(commitment)
        if n < 3:
            return np.zeros(n)

        dc = np.zeros(n)
        du = np.zeros(n)
        for t in range(1, n):
            dc[t] = commitment[t] - commitment[t - 1]
            du[t] = uncertainty[t] - uncertainty[t - 1]

        # Moving average smoothing (window=3)
        w = 3
        dc = np.convolve(dc, np.ones(w) / w, mode="same")
        du = np.convolve(du, np.ones(w) / w, mode="same")

        return dc - du

    def _compute_recovery_margin(
        self, states: np.ndarray, uncertainty: np.ndarray,
    ) -> np.ndarray:
        """Normalized distance to nearest branching point."""
        n = len(states)
        recovery = np.full(n, 1.0)
        if n < 3:
            return recovery

        centroid = np.mean(states, axis=0)
        dists_to_centroid = np.linalg.norm(states - centroid, axis=1)
        max_range = 2.0 * np.max(dists_to_centroid)
        if max_range < 1e-10:
            return recovery

        median_u = np.median(uncertainty)
        branching_mask = uncertainty >= median_u

        for t in range(n):
            branching_indices = np.where(branching_mask[:t + 1])[0]
            if len(branching_indices) == 0:
                continue
            dists = np.linalg.norm(states[branching_indices] - states[t], axis=1)
            recovery[t] = 1.0 - (np.min(dists) / max_range)

        return np.clip(recovery, 0.0, 1.0)


# ===========================================================================
# Step 5: Build convergence labels (6-signal method)
# ===========================================================================

def build_convergence_labels(
    conversations: dict[str, list[dict]],
    conv_embeddings: dict[str, np.ndarray],
) -> dict[str, int]:
    """Build binary convergence labels using 6 signals.

    A conversation is "converged" if it shows signs of reaching
    a resolution, conclusion, or stable state.

    6 signals:
      1. Embedding velocity decay: final-third velocity < first-third
      2. Semantic clustering: final turns are closer to each other
      3. Decision language: final turns contain resolution markers
      4. Role balance: both user and assistant contributed in final segment
      5. Length stability: turn lengths stabilize in final third
      6. Topic coherence: cosine similarity between halves > threshold

    Label = 1 (converged) if >= 3 of 6 signals fire.
    """

    decision_markers = re.compile(
        r'\b(done|agreed|let\'s go|perfect|sounds good|exactly|'
        r'that works|confirmed|resolved|fixed|complete|'
        r'great|yes|ok so|alright|settled|final|'
        r'ship it|merged|deployed|pushed|committed)\b',
        re.IGNORECASE,
    )

    labels = {}

    for cid, turns_list in conversations.items():
        embs = conv_embeddings.get(cid)
        if embs is None or len(embs) < MIN_TURNS_PER_CONV:
            continue

        n = len(turns_list)
        third = max(n // 3, 1)

        signals_fired = 0

        # Signal 1: Embedding velocity decay
        first_third_embs = embs[:third]
        final_third_embs = embs[-third:]

        first_velocities = np.linalg.norm(np.diff(first_third_embs, axis=0), axis=1)
        final_velocities = np.linalg.norm(np.diff(final_third_embs, axis=0), axis=1)

        if len(first_velocities) > 0 and len(final_velocities) > 0:
            if np.mean(final_velocities) < np.mean(first_velocities) * 0.8:
                signals_fired += 1

        # Signal 2: Semantic clustering in final segment
        if len(final_third_embs) >= 3:
            final_centroid = np.mean(final_third_embs, axis=0)
            final_dists = np.linalg.norm(final_third_embs - final_centroid, axis=1)
            all_centroid = np.mean(embs, axis=0)
            all_dists = np.linalg.norm(embs - all_centroid, axis=1)

            if np.mean(final_dists) < np.mean(all_dists) * 0.9:
                signals_fired += 1

        # Signal 3: Decision language in final turns
        final_texts = [
            (t.get("content_text") or "")
            for t in turns_list[-third:]
        ]
        decision_count = sum(
            1 for text in final_texts if decision_markers.search(text)
        )
        if decision_count >= max(1, len(final_texts) // 3):
            signals_fired += 1

        # Signal 4: Role balance in final segment
        final_roles = [t.get("role", "") for t in turns_list[-third:]]
        has_user = "user" in final_roles
        has_assistant = "assistant" in final_roles
        if has_user and has_assistant:
            signals_fired += 1

        # Signal 5: Length stability
        all_lengths = [len((t.get("content_text") or "")) for t in turns_list]
        first_lengths = all_lengths[:third]
        final_lengths = all_lengths[-third:]

        if len(first_lengths) >= 2 and len(final_lengths) >= 2:
            first_cv = np.std(first_lengths) / max(np.mean(first_lengths), 1)
            final_cv = np.std(final_lengths) / max(np.mean(final_lengths), 1)
            if final_cv < first_cv:
                signals_fired += 1

        # Signal 6: Topic coherence (cosine sim between halves)
        mid = n // 2
        first_half_mean = np.mean(embs[:mid], axis=0)
        second_half_mean = np.mean(embs[mid:], axis=0)
        norm1 = np.linalg.norm(first_half_mean)
        norm2 = np.linalg.norm(second_half_mean)
        if norm1 > 0 and norm2 > 0:
            cosine_sim = np.dot(first_half_mean, second_half_mean) / (norm1 * norm2)
            if cosine_sim > 0.85:
                signals_fired += 1

        # Label: converged if >= 3 signals
        labels[cid] = 1 if signals_fired >= 3 else 0

    pos = sum(labels.values())
    neg = len(labels) - pos
    logger.info(
        "Labels: %d converged (%.1f%%), %d not-converged (%.1f%%)",
        pos, 100 * pos / max(len(labels), 1),
        neg, 100 * neg / max(len(labels), 1),
    )

    return labels


# ===========================================================================
# Step 6: Build feature matrices
# ===========================================================================

def build_feature_matrices(
    conversations: dict[str, list[dict]],
    conv_embeddings: dict[str, np.ndarray],
    labels: dict[str, int],
) -> tuple:
    """Build feature matrices for all three models.

    Returns:
        (X_ant, X_emb, X_combined, y, conv_ids)
        X_ant: (n_conv, 30) anticipation features
        X_emb: (n_conv, PCA_DIM) PCA-reduced mean embeddings
        X_combined: (n_conv, 30 + PCA_DIM) concatenated
        y: (n_conv,) binary labels
        conv_ids: list of conversation IDs in order
    """
    extractor = AnticipationFeatureExtractor(k_neighbors=5)

    ant_features = []
    emb_features = []
    y_list = []
    conv_ids = []

    for cid in labels:
        embs = conv_embeddings.get(cid)
        if embs is None:
            continue

        # Anticipation features from embedding trajectory
        ant_feat = extractor.extract(embs)
        ant_features.append(ant_feat)

        # Mean embedding for the conversation
        mean_emb = np.mean(embs, axis=0)
        emb_features.append(mean_emb)

        y_list.append(labels[cid])
        conv_ids.append(cid)

    X_ant = np.array(ant_features)  # (N, 30)
    X_emb_raw = np.array(emb_features)  # (N, 1024)
    y = np.array(y_list)

    logger.info(
        "Feature matrices: X_ant=%s, X_emb_raw=%s, y=%s (pos=%d, neg=%d)",
        X_ant.shape, X_emb_raw.shape, y.shape,
        np.sum(y == 1), np.sum(y == 0),
    )

    from sklearn.decomposition import PCA

    # PCA on embeddings
    n_components = min(PCA_DIM, X_emb_raw.shape[0] - 1, X_emb_raw.shape[1])
    logger.info("Applying PCA: %d -> %d dimensions", X_emb_raw.shape[1], n_components)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_emb = pca.fit_transform(X_emb_raw)
    logger.info(
        "PCA variance explained: %.1f%%",
        100 * sum(pca.explained_variance_ratio_),
    )

    # Combined features
    X_combined = np.hstack([X_ant, X_emb])

    # Replace any NaN/inf with 0
    for X in [X_ant, X_emb, X_combined]:
        X[~np.isfinite(X)] = 0.0

    return X_ant, X_emb, X_combined, y, conv_ids, extractor.FEATURE_NAMES


# ===========================================================================
# Step 7: Train and evaluate classifiers
# ===========================================================================

@dataclass
class ClassifierResult:
    name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy_std: float = 0.0
    precision_std: float = 0.0
    recall_std: float = 0.0
    f1_std: float = 0.0
    n_features: int = 0
    fold_results: list = field(default_factory=list)


def evaluate_classifier(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = N_FOLDS,
) -> ClassifierResult:
    """Run stratified k-fold CV for a logistic regression classifier."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    accs, precs, recs, f1s = [], [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Logistic Regression with regularization
        clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
            class_weight="balanced",
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, zero_division=0))
        recs.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))

    result = ClassifierResult(
        name=name,
        accuracy=float(np.mean(accs)),
        precision=float(np.mean(precs)),
        recall=float(np.mean(recs)),
        f1=float(np.mean(f1s)),
        accuracy_std=float(np.std(accs)),
        precision_std=float(np.std(precs)),
        recall_std=float(np.std(recs)),
        f1_std=float(np.std(f1s)),
        n_features=X.shape[1],
        fold_results=[
            {"fold": i, "accuracy": a, "precision": p, "recall": r, "f1": f}
            for i, (a, p, r, f) in enumerate(zip(accs, precs, recs, f1s))
        ],
    )

    return result


def evaluate_baseline(
    y: np.ndarray,
    n_folds: int = N_FOLDS,
) -> ClassifierResult:
    """Majority class baseline."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.dummy import DummyClassifier

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    accs, precs, recs, f1s = [], [], [], []

    # We need a dummy X for sklearn splitter
    X_dummy = np.zeros((len(y), 1))

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_dummy, y)):
        y_train, y_test = y[train_idx], y[test_idx]

        clf = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
        clf.fit(X_dummy[train_idx], y_train)
        y_pred = clf.predict(X_dummy[test_idx])

        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, zero_division=0))
        recs.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))

    return ClassifierResult(
        name="Baseline (majority)",
        accuracy=float(np.mean(accs)),
        precision=float(np.mean(precs)),
        recall=float(np.mean(recs)),
        f1=float(np.mean(f1s)),
        accuracy_std=float(np.std(accs)),
        precision_std=float(np.std(precs)),
        recall_std=float(np.std(recs)),
        f1_std=float(np.std(f1s)),
        n_features=0,
        fold_results=[
            {"fold": i, "accuracy": a, "precision": p, "recall": r, "f1": f}
            for i, (a, p, r, f) in enumerate(zip(accs, precs, recs, f1s))
        ],
    )


# ===========================================================================
# Step 8: Feature importance analysis
# ===========================================================================

def feature_importance_analysis(
    X_ant: np.ndarray,
    X_combined: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Analyze which anticipation features contribute most.

    Trains a final logistic regression on all data and reports coefficients.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Train on anticipation features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ant)

    clf = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs",
        random_state=RANDOM_SEED, class_weight="balanced",
    )
    clf.fit(X_scaled, y)

    coefs = clf.coef_[0]
    importance = sorted(
        zip(feature_names, coefs, np.abs(coefs)),
        key=lambda x: -x[2],
    )

    return {
        "top_features": [
            {"name": name, "coefficient": float(coef), "abs_importance": float(absc)}
            for name, coef, absc in importance[:15]
        ],
        "intercept": float(clf.intercept_[0]),
    }


# ===========================================================================
# Step 9: Statistical significance test
# ===========================================================================

def paired_permutation_test(
    scores_a: list[float],
    scores_b: list[float],
    n_permutations: int = 10000,
) -> float:
    """Paired permutation test: is the mean difference significant?

    Returns p-value for H0: mean(scores_a) == mean(scores_b).
    """
    diffs = np.array(scores_a) - np.array(scores_b)
    observed_diff = np.mean(diffs)

    count = 0
    rng = np.random.RandomState(RANDOM_SEED)

    for _ in range(n_permutations):
        # Randomly flip signs
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_diff = np.mean(diffs * signs)
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    return count / n_permutations


# ===========================================================================
# Main pipeline
# ===========================================================================

def main():
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("Paper 1 Enhanced Evaluation: Anticipation Geometry as Features")
    logger.info("=" * 70)

    # Step 1: Fetch data
    turns = fetch_turns(TOTAL_TURNS_TARGET)
    if len(turns) < 1000:
        logger.error("Too few turns fetched (%d). Aborting.", len(turns))
        sys.exit(1)

    # Step 2: Group into conversations
    conversations = group_conversations(turns, MIN_TURNS_PER_CONV)
    if len(conversations) < 20:
        logger.error("Too few conversations (%d). Aborting.", len(conversations))
        sys.exit(1)

    # Step 3: Embed
    conv_embeddings = embed_conversations(conversations)

    # Step 4 & 5: Build labels
    labels = build_convergence_labels(conversations, conv_embeddings)
    if len(labels) < 20:
        logger.error("Too few labeled conversations (%d). Aborting.", len(labels))
        sys.exit(1)

    # Step 6: Build feature matrices
    X_ant, X_emb, X_combined, y, conv_ids, feature_names = build_feature_matrices(
        conversations, conv_embeddings, labels,
    )

    if len(y) < N_FOLDS * 2:
        logger.error("Too few samples (%d) for %d-fold CV. Aborting.", len(y), N_FOLDS)
        sys.exit(1)

    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING AND EVALUATION")
    logger.info("=" * 70)

    # Step 7: Evaluate all classifiers
    results = {}

    logger.info("")
    logger.info("--- Model A: Anticipation-only (30 features) ---")
    res_ant = evaluate_classifier("Anticipation-only", X_ant, y)
    results["anticipation_only"] = res_ant
    logger.info("  Accuracy: %.3f +/- %.3f", res_ant.accuracy, res_ant.accuracy_std)
    logger.info("  F1:       %.3f +/- %.3f", res_ant.f1, res_ant.f1_std)

    logger.info("")
    logger.info("--- Model B: Embedding-only (%d PCA features) ---", X_emb.shape[1])
    res_emb = evaluate_classifier("Embedding-only", X_emb, y)
    results["embedding_only"] = res_emb
    logger.info("  Accuracy: %.3f +/- %.3f", res_emb.accuracy, res_emb.accuracy_std)
    logger.info("  F1:       %.3f +/- %.3f", res_emb.f1, res_emb.f1_std)

    logger.info("")
    logger.info("--- Model C: Combined (%d features) ---", X_combined.shape[1])
    res_comb = evaluate_classifier("Combined", X_combined, y)
    results["combined"] = res_comb
    logger.info("  Accuracy: %.3f +/- %.3f", res_comb.accuracy, res_comb.accuracy_std)
    logger.info("  F1:       %.3f +/- %.3f", res_comb.f1, res_comb.f1_std)

    logger.info("")
    logger.info("--- Model D: Baseline (majority class) ---")
    res_base = evaluate_baseline(y)
    results["baseline"] = res_base
    logger.info("  Accuracy: %.3f +/- %.3f", res_base.accuracy, res_base.accuracy_std)
    logger.info("  F1:       %.3f +/- %.3f", res_base.f1, res_base.f1_std)

    # Step 8: Feature importance
    logger.info("")
    logger.info("--- Feature Importance Analysis ---")
    importance = feature_importance_analysis(X_ant, X_combined, y, feature_names)
    for feat in importance["top_features"][:10]:
        logger.info("  %s: coef=%.4f", feat["name"], feat["coefficient"])

    # Step 9: Statistical tests
    logger.info("")
    logger.info("--- Statistical Significance ---")

    # Combined vs Embedding-only F1 (per-fold)
    f1_combined = [f["f1"] for f in res_comb.fold_results]
    f1_emb = [f["f1"] for f in res_emb.fold_results]
    f1_ant = [f["f1"] for f in res_ant.fold_results]

    p_comb_vs_emb = paired_permutation_test(f1_combined, f1_emb)
    p_comb_vs_ant = paired_permutation_test(f1_combined, f1_ant)
    p_ant_vs_base = paired_permutation_test(
        f1_ant,
        [f["f1"] for f in res_base.fold_results],
    )

    logger.info("  Combined vs Embedding-only: p=%.4f", p_comb_vs_emb)
    logger.info("  Combined vs Anticipation-only: p=%.4f", p_comb_vs_ant)
    logger.info("  Anticipation vs Baseline: p=%.4f", p_ant_vs_base)

    # Compute the key analysis
    delta_f1 = res_comb.f1 - res_emb.f1
    adds_value = delta_f1 > 0

    elapsed = time.time() - start_time

    # Build final report
    report = {
        "experiment": "Paper 1 Enhanced Evaluation",
        "description": "Anticipation geometry scalars as classification features for conversation convergence prediction",
        "key_question": "Do anticipation features ADD value beyond raw embeddings?",
        "answer": "YES" if adds_value else "NO",
        "delta_f1_combined_vs_embedding": round(delta_f1, 4),
        "dataset": {
            "total_turns_fetched": len(turns),
            "conversations_with_10plus_turns": len(conversations),
            "labeled_conversations": len(labels),
            "positive_label_count": int(np.sum(y == 1)),
            "negative_label_count": int(np.sum(y == 0)),
            "positive_ratio": round(float(np.mean(y)), 4),
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "pca_dim": X_emb.shape[1],
        },
        "results": {
            name: {
                "accuracy": round(r.accuracy, 4),
                "accuracy_std": round(r.accuracy_std, 4),
                "precision": round(r.precision, 4),
                "precision_std": round(r.precision_std, 4),
                "recall": round(r.recall, 4),
                "recall_std": round(r.recall_std, 4),
                "f1": round(r.f1, 4),
                "f1_std": round(r.f1_std, 4),
                "n_features": r.n_features,
                "per_fold": r.fold_results,
            }
            for name, r in results.items()
        },
        "statistical_tests": {
            "combined_vs_embedding_p": round(p_comb_vs_emb, 4),
            "combined_vs_anticipation_p": round(p_comb_vs_ant, 4),
            "anticipation_vs_baseline_p": round(p_ant_vs_base, 4),
            "method": "paired_permutation_test",
            "n_permutations": 10000,
        },
        "feature_importance": importance,
        "analysis": {
            "anticipation_above_baseline": res_ant.f1 > res_base.f1,
            "anticipation_f1_lift_over_baseline": round(res_ant.f1 - res_base.f1, 4),
            "embedding_above_baseline": res_emb.f1 > res_base.f1,
            "embedding_f1_lift_over_baseline": round(res_emb.f1 - res_base.f1, 4),
            "combined_above_embedding": res_comb.f1 > res_emb.f1,
            "combined_f1_lift_over_embedding": round(res_comb.f1 - res_emb.f1, 4),
            "combined_above_anticipation": res_comb.f1 > res_ant.f1,
            "combined_f1_lift_over_anticipation": round(res_comb.f1 - res_ant.f1, 4),
            "conclusion": (
                "Anticipation features provide additive signal beyond embeddings. "
                "The geometric framework captures trajectory dynamics that "
                "static embedding averages miss."
            ) if adds_value else (
                "Embeddings alone capture sufficient signal for convergence prediction. "
                "Anticipation features do not add value in this setting."
            ),
        },
        "timing": {
            "total_seconds": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "methodology": {
            "cross_validation": f"{N_FOLDS}-fold stratified",
            "classifier": "LogisticRegression(C=1.0, class_weight='balanced')",
            "anticipation_features": {
                "count": 30,
                "base_scalars": ["commitment", "uncertainty", "transition_pressure", "recovery_margin"],
                "stats_per_scalar": ["mean", "std", "min", "max", "final", "slope", "range"],
                "cross_terms": ["focus_score (commitment * (1-uncertainty))", "decision_tension (tp_max * rm_min)"],
            },
            "convergence_labeling": {
                "method": "6-signal voting (threshold >= 3)",
                "signals": [
                    "embedding_velocity_decay",
                    "semantic_clustering",
                    "decision_language",
                    "role_balance",
                    "length_stability",
                    "topic_coherence",
                ],
            },
        },
    }

    # Print final summary table
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    logger.info("")
    logger.info("%-25s | %8s | %8s | %8s | %8s | %s",
                "Model", "Acc", "Prec", "Recall", "F1", "Features")
    logger.info("-" * 80)
    for name, r in results.items():
        logger.info("%-25s | %8.4f | %8.4f | %8.4f | %8.4f | %d",
                    r.name, r.accuracy, r.precision, r.recall, r.f1, r.n_features)

    logger.info("")
    logger.info("KEY FINDING: Combined vs Embedding-only F1 delta = %+.4f (p=%.4f)",
                delta_f1, p_comb_vs_emb)
    logger.info("ANSWER: Anticipation features %s value beyond raw embeddings.",
                "ADD" if adds_value else "DO NOT add")
    logger.info("")
    logger.info("Total time: %.1f seconds", elapsed)

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Results saved to: %s", OUTPUT_PATH)

    return report


if __name__ == "__main__":
    main()
