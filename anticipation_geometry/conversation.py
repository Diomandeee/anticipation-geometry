"""
Conversation Trajectory Builder
================================

Converts Supabase conversation turns into trajectory vectors for
anticipation geometry computation. This demonstrates that the framework
generalizes from motion (skeletal positions in R^3) to conversation
(sentence embeddings in R^384).

The pipeline:
  1. Fetch turns from Supabase `memory_turns` table
  2. Embed each turn's text using sentence-transformers (all-MiniLM-L6-v2)
  3. Compute anticipation scalars over the embedding trajectory
  4. Produce a per-turn analysis showing commitment, uncertainty,
     transition pressure, and recovery margin

Why sentence embeddings as trajectory states?
----------------------------------------------

Each conversation turn maps to a point in a 384-dimensional semantic space.
A sequence of turns forms a trajectory through this space. The anticipation
scalars then capture:

  - commitment: How semantically stable the conversation is. High when
    consecutive turns discuss the same topic. Low when the topic shifts.

  - uncertainty: How many conversational directions are available. High
    in broad discussions, low in focused deep-dives.

  - transition_pressure: Rate of topic narrowing. Spikes when a conversation
    transitions from brainstorming to decision-making.

  - recovery_margin: How easy it is to return to a previous topic. Low
    when deep in a technical rabbit hole, high when near a general
    discussion hub.

This is the same math as motion anticipation, just in a higher-dimensional
space. The geometry doesn't care whether the vectors are joint positions
or word embeddings.

Connection to Princeton (arXiv:2603.14147)
-------------------------------------------

Princeton uses KG paths as the trajectory. We use sentence embeddings.
Both are valid instantiations of the same principle: reasoning (whether
structured as KG hops or free-form conversation) traces a path through
a metric space, and that path's geometry reveals information about the
reasoning process itself.

Usage:
    from conversation_trajectory import ConversationTrajectory

    ct = ConversationTrajectory(supabase_url="...", supabase_key="...")
    result = ct.analyze_conversation(conversation_id="abc-123")

    # Or with raw turns:
    result = ct.analyze_turns([
        "Let's discuss the architecture",
        "We should use a microservices approach",
        "Each service needs its own database",
    ])
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import requests

from generalized_anticipation import AnticipationGeometry, AnticipationPacket

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    id: str
    role: str
    content: str
    timestamp: str
    embedding: Optional[np.ndarray] = None


@dataclass
class TrajectoryAnalysis:
    """Complete analysis of a conversation as a trajectory."""
    conversation_id: str
    turns: list[ConversationTurn]
    packet: AnticipationPacket
    regime_labels: list[str] = field(default_factory=list)
    topic_shifts: list[int] = field(default_factory=list)
    decision_points: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        return {
            "conversation_id": self.conversation_id,
            "n_turns": len(self.turns),
            "summary": self.packet.summary(),
            "regime_sequence": self.regime_labels,
            "topic_shift_indices": self.topic_shifts,
            "decision_point_indices": self.decision_points,
            "per_turn": [
                {
                    "index": i,
                    "role": turn.role,
                    "content_preview": turn.content[:80] + ("..." if len(turn.content) > 80 else ""),
                    "commitment": float(self.packet.commitment[i]),
                    "uncertainty": float(self.packet.uncertainty[i]),
                    "transition_pressure": float(self.packet.transition_pressure[i]),
                    "recovery_margin": float(self.packet.recovery_margin[i]),
                    "regime": self.regime_labels[i] if i < len(self.regime_labels) else "unknown",
                }
                for i, turn in enumerate(self.turns)
            ],
        }


# ---------------------------------------------------------------------------
# Embedding Engine
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """Compute sentence embeddings using sentence-transformers.

    Uses all-MiniLM-L6-v2 (384 dimensions) as default. This model is
    small enough to run on CPU and produces embeddings that capture
    semantic similarity well enough for trajectory analysis.

    Falls back to a simple TF-IDF-like hash embedding if sentence-transformers
    is not installed, to keep the package dependency-light.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._fallback = False

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info("Loaded sentence-transformers model: %s", self.model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Falling back to hash-based embeddings. "
                "Install with: pip install sentence-transformers"
            )
            self._fallback = True

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into vectors.

        Returns:
            np.ndarray of shape (len(texts), dim) where dim is 384 for
            the default model or 128 for the fallback.
        """
        self._load_model()

        if self._fallback:
            return self._hash_embed(texts)

        embeddings = self._model.encode(texts, show_progress_bar=False)
        return np.array(embeddings)

    def _hash_embed(self, texts: list[str], dim: int = 128) -> np.ndarray:
        """Fallback embedding using deterministic hashing.

        Not semantically meaningful, but preserves the property that
        similar strings produce similar vectors (via n-gram overlap).
        This is adequate for demonstrating the trajectory geometry even
        when sentence-transformers is unavailable.
        """
        embeddings = np.zeros((len(texts), dim))

        for i, text in enumerate(texts):
            # Use character 3-grams as features
            words = text.lower().split()
            for word in words:
                for j in range(len(word) - 2):
                    trigram = word[j:j + 3]
                    # Deterministic hash to a dimension
                    h = hash(trigram) % dim
                    embeddings[i, h] += 1.0

            # L2 normalize
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm

        return embeddings


# ---------------------------------------------------------------------------
# Conversation Trajectory Analyzer
# ---------------------------------------------------------------------------

class ConversationTrajectory:
    """Analyze conversations as trajectories in embedding space.

    Fetches turns from Supabase, embeds them, and computes anticipation
    geometry scalars to characterize the conversation's dynamics.

    Args:
        supabase_url: Supabase project URL.
        supabase_key: Supabase anon or service key.
        embedding_model: Name of sentence-transformers model.
        k_neighbors: K parameter for uncertainty computation.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        k_neighbors: int = 5,
    ):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.embedder = EmbeddingEngine(model_name=embedding_model)
        self.geometry = AnticipationGeometry(k_neighbors=k_neighbors)
        self._session = requests.Session()

    def analyze_conversation(
        self,
        conversation_id: str,
        limit: int = 200,
    ) -> TrajectoryAnalysis:
        """Fetch and analyze a conversation from Supabase.

        Args:
            conversation_id: UUID of the conversation in memory_turns.
            limit: Maximum number of turns to fetch.

        Returns:
            TrajectoryAnalysis with per-turn anticipation scalars.
        """
        turns = self._fetch_turns(conversation_id, limit)
        if not turns:
            return TrajectoryAnalysis(
                conversation_id=conversation_id,
                turns=[],
                packet=AnticipationPacket(),
            )

        return self._analyze(conversation_id, turns)

    def analyze_turns(
        self,
        texts: list[str],
        roles: Optional[list[str]] = None,
        conversation_id: str = "inline",
    ) -> TrajectoryAnalysis:
        """Analyze raw text turns without Supabase.

        Useful for testing and for analyzing conversations from
        other sources (e.g., local logs, exported data).

        Args:
            texts: List of turn texts in chronological order.
            roles: Optional list of roles ("user", "assistant", etc.).
            conversation_id: Label for the analysis.

        Returns:
            TrajectoryAnalysis with per-turn anticipation scalars.
        """
        if roles is None:
            roles = ["unknown"] * len(texts)

        turns = [
            ConversationTurn(
                id=f"turn-{i}",
                role=roles[i] if i < len(roles) else "unknown",
                content=text,
                timestamp=f"2026-01-01T00:00:{i:02d}Z",
            )
            for i, text in enumerate(texts)
        ]

        return self._analyze(conversation_id, turns)

    # ----- Core Analysis -----

    def _analyze(
        self,
        conversation_id: str,
        turns: list[ConversationTurn],
    ) -> TrajectoryAnalysis:
        """Core analysis pipeline: embed, compute geometry, classify."""
        texts = [t.content for t in turns]
        embeddings = self.embedder.embed(texts)

        # Attach embeddings to turns
        for i, turn in enumerate(turns):
            turn.embedding = embeddings[i]

        # Build trajectory as list of vectors
        trajectory = [embeddings[i] for i in range(len(turns))]

        # Compute anticipation geometry
        packet = self.geometry.compute(trajectory)

        # Classify regimes
        regime_labels = [packet.regime_at(t) for t in range(packet.trajectory_length)]

        # Detect topic shifts (commitment drops below 0.3)
        topic_shifts = []
        for t in range(1, packet.trajectory_length):
            if packet.commitment[t] < 0.3:
                topic_shifts.append(t)

        # Detect decision points (transition pressure spikes above 0.5)
        decision_points = []
        for t in range(packet.trajectory_length):
            if packet.transition_pressure[t] > 0.5:
                decision_points.append(t)

        return TrajectoryAnalysis(
            conversation_id=conversation_id,
            turns=turns,
            packet=packet,
            regime_labels=regime_labels,
            topic_shifts=topic_shifts,
            decision_points=decision_points,
        )

    # ----- Supabase Interaction -----

    def _fetch_turns(
        self,
        conversation_id: str,
        limit: int = 200,
    ) -> list[ConversationTurn]:
        """Fetch conversation turns from Supabase memory_turns table.

        The memory_turns table schema (from cc-graph-kernel Supabase):
          id (uuid), conversation_id (uuid), role (text), content (text),
          created_at (timestamptz), ...

        Uses the Supabase REST API with PostgREST query syntax.
        """
        if not self.supabase_url or not self.supabase_key:
            logger.warning("Supabase credentials not configured. Cannot fetch turns.")
            return []

        try:
            resp = self._session.get(
                f"{self.supabase_url}/rest/v1/memory_turns",
                headers={
                    "apikey": self.supabase_key,
                    "Authorization": f"Bearer {self.supabase_key}",
                },
                params={
                    "conversation_id": f"eq.{conversation_id}",
                    "order": "created_at.asc",
                    "limit": limit,
                    "select": "id,role,content,created_at",
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()

            turns = []
            for row in data:
                content = row.get("content", "")
                if not content:
                    continue
                turns.append(ConversationTurn(
                    id=row["id"],
                    role=row.get("role", "unknown"),
                    content=content,
                    timestamp=row.get("created_at", ""),
                ))

            logger.info("Fetched %d turns for conversation %s", len(turns), conversation_id)
            return turns

        except (requests.RequestException, ValueError) as e:
            logger.error("Failed to fetch turns: %s", e)
            return []


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def demo():
    """Demonstrate conversation trajectory analysis with sample turns."""
    ct = ConversationTrajectory()

    # Simulate a technical discussion that explores, then commits
    turns = [
        "Let's discuss the system architecture for the new service",
        "We could use microservices or a monolith. Each has tradeoffs.",
        "Microservices give us independent scaling. What about data consistency?",
        "Event sourcing could handle consistency across service boundaries",
        "Or we could use a shared database with separate schemas",
        "Let's go with event sourcing. It aligns with our async-first principle.",
        "OK, so we need an event store. Kafka or custom?",
        "Kafka. We already have the infrastructure.",
        "Great. So: Kafka event bus, per-service materialized views, CQRS pattern.",
        "Exactly. Each service owns its projections. No shared mutable state.",
        "Now for the implementation: start with the order service first?",
        "Yes. Order service, then inventory, then notification pipeline.",
        "Order service needs: API layer, command handler, event publisher, projection.",
        "I'll scaffold the order service with our standard template.",
        "Done. Moving on to the event schema definitions.",
    ]

    roles = ["user", "assistant"] * 8
    roles = roles[:len(turns)]

    result = ct.analyze_turns(turns, roles=roles, conversation_id="arch-discussion")

    print("Conversation Trajectory Analysis")
    print("=" * 80)
    print(f"Conversation: {result.conversation_id}")
    print(f"Turns: {len(result.turns)}")
    print()

    print(f"{'Turn':>5} | {'Role':<10} | {'Content':<35} | {'C':>5} | {'U':>5} | {'P':>6} | {'R':>5} | {'Regime':<15}")
    print("-" * 110)
    for entry in result.to_dict()["per_turn"]:
        content = entry["content_preview"][:35]
        print(
            f"{entry['index']:>5} | {entry['role']:<10} | {content:<35} | "
            f"{entry['commitment']:>5.2f} | {entry['uncertainty']:>5.2f} | "
            f"{entry['transition_pressure']:>+6.2f} | {entry['recovery_margin']:>5.2f} | "
            f"{entry['regime']:<15}"
        )

    print()
    print(f"Topic shifts at turns: {result.topic_shifts}")
    print(f"Decision points at turns: {result.decision_points}")
    print()
    print("Summary:", json.dumps(result.packet.summary(), indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
