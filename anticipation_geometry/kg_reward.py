"""
KG-Path Reward Function
========================

Implements the 3-signal reward model from Princeton's "An Alternative Trajectory
for Generative AI" (arXiv:2603.14147), adapted to query Comp-Core's Graph Kernel.

Princeton's core insight: Knowledge Graph paths serve as implicit reward models
for reasoning. A valid reasoning chain should:
  1. Walk real edges in a knowledge graph (axiomatic validity)
  2. Maintain continuity between consecutive hops (chain continuity)
  3. Arrive at a node that satisfies the original query (terminal grounding)

This module queries the cc-graph-kernel REST API at localhost:8001 to evaluate
these three signals over arbitrary (subject, predicate, object) reasoning paths.

Signal weights are configurable but default to Princeton's suggested ratios:
  - Axiomatic validity:  +1 per valid edge, -5 per hallucinated edge
  - Chain continuity:    +2 per continuous hop, -2 per discontinuity
  - Terminal grounding:  +3 if terminal entity matches query, 0 otherwise

The asymmetric penalty on invalid edges (-5 vs +1) encodes the principle that
a single hallucinated step poisons the entire reasoning chain more than a valid
step supports it. This mirrors the "one bad apple" effect observed in chain-of-
thought reasoning failures.

Usage:
    from kg_reward import KGPathReward

    reward = KGPathReward(gk_url="http://localhost:8001")

    path = [
        ("spore", "built_with", "swiftui"),
        ("swiftui", "is_a", "framework"),
        ("framework", "uses", "declarative_ui"),
    ]

    score = reward.score(path, query_entity="declarative_ui")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KGHop:
    """A single hop in a KG reasoning path: (subject, predicate, object)."""
    subject: str
    predicate: str
    object: str


@dataclass
class SignalBreakdown:
    """Per-signal breakdown of a KG-path reward score.

    Attributes:
        axiomatic_validity: Sum of per-hop validity scores.
            +valid_reward per edge found in GK, +invalid_penalty per missing edge.
        chain_continuity: Sum of continuity scores between consecutive hops.
            +continuity_reward when hop_n.object == hop_{n+1}.subject,
            +discontinuity_penalty otherwise.
        terminal_grounding: Score for whether the final entity satisfies the query.
            +grounding_reward if match, 0 otherwise.
        composite: Weighted sum of the three signals.
        n_hops: Total number of hops in the path.
        n_valid_edges: Number of hops whose edge exists in the GK.
        n_continuous: Number of consecutive hop pairs with chain continuity.
        terminal_match: Whether the terminal entity matched the query.
    """
    axiomatic_validity: float = 0.0
    chain_continuity: float = 0.0
    terminal_grounding: float = 0.0
    composite: float = 0.0
    n_hops: int = 0
    n_valid_edges: int = 0
    n_continuous: int = 0
    terminal_match: bool = False


@dataclass
class RewardConfig:
    """Configuration for the 3-signal reward function.

    Default values follow Princeton's suggested asymmetric penalties.
    The key design choice: hallucinated edges are penalized 5x more than
    valid edges are rewarded, encoding the observation that reasoning chains
    degrade catastrophically with even one invalid step.
    """
    # Signal 1: Axiomatic validity
    valid_edge_reward: float = 1.0
    invalid_edge_penalty: float = -5.0

    # Signal 2: Chain continuity
    continuity_reward: float = 2.0
    discontinuity_penalty: float = -2.0

    # Signal 3: Terminal grounding
    grounding_reward: float = 3.0

    # Signal weights for composite score
    validity_weight: float = 1.0
    continuity_weight: float = 1.0
    grounding_weight: float = 1.0

    # GK query parameters
    min_confidence: float = 0.0
    query_timeout_s: float = 5.0


# ---------------------------------------------------------------------------
# KG-Path Reward Function
# ---------------------------------------------------------------------------

class KGPathReward:
    """Evaluate reasoning paths against a Knowledge Graph.

    Queries cc-graph-kernel to check edge existence, then computes
    Princeton's 3-signal composite reward.

    The GK API contract (from cc-graph-kernel/src/service/knowledge_handlers.rs):
      GET /api/knowledge?subject=X&predicate=Y&object=Z  -> {triples: [...], total: N}
      POST /api/knowledge/traverse  -> {paths: [...], stats: {...}}

    We use the query endpoint for edge-existence checks (Signal 1) and the
    traverse endpoint for terminal grounding verification (Signal 3).
    """

    def __init__(
        self,
        gk_url: str = "http://localhost:8001",
        config: Optional[RewardConfig] = None,
    ):
        self.gk_url = gk_url.rstrip("/")
        self.config = config or RewardConfig()
        self._session = requests.Session()
        self._edge_cache: dict[tuple[str, str, str], bool] = {}

    # ----- Public API -----

    def score(
        self,
        path: list[tuple[str, str, str] | KGHop],
        query_entity: Optional[str] = None,
    ) -> SignalBreakdown:
        """Score a reasoning path against the Knowledge Graph.

        Args:
            path: Ordered list of (subject, predicate, object) triples
                representing a reasoning chain. Each element can be a
                3-tuple or a KGHop instance.
            query_entity: The entity the path should terminate at.
                If None, terminal grounding is skipped (scored as 0).

        Returns:
            SignalBreakdown with per-signal scores and composite.

        The composite score can be negative. A path with all hallucinated
        edges and no continuity will score approximately
        -5 * n_hops - 2 * (n_hops - 1), which is the worst case.
        """
        hops = [self._normalize_hop(h) for h in path]
        breakdown = SignalBreakdown(n_hops=len(hops))

        if not hops:
            return breakdown

        # Signal 1: Axiomatic validity
        #   For each hop, check if the edge (subject, predicate, object)
        #   exists in the Knowledge Graph.
        for hop in hops:
            if self._edge_exists(hop):
                breakdown.axiomatic_validity += self.config.valid_edge_reward
                breakdown.n_valid_edges += 1
            else:
                breakdown.axiomatic_validity += self.config.invalid_edge_penalty

        # Signal 2: Chain continuity
        #   For each consecutive pair of hops, check that the object of
        #   hop N equals the subject of hop N+1. This ensures the reasoning
        #   chain doesn't "teleport" between unrelated entities.
        for i in range(len(hops) - 1):
            tail_of_current = hops[i].object.lower().strip()
            head_of_next = hops[i + 1].subject.lower().strip()
            if tail_of_current == head_of_next:
                breakdown.chain_continuity += self.config.continuity_reward
                breakdown.n_continuous += 1
            else:
                breakdown.chain_continuity += self.config.discontinuity_penalty

        # Signal 3: Terminal grounding
        #   Check if the final entity in the path matches the query target.
        if query_entity is not None:
            terminal = hops[-1].object.lower().strip()
            target = query_entity.lower().strip()
            if terminal == target:
                breakdown.terminal_grounding = self.config.grounding_reward
                breakdown.terminal_match = True
            else:
                # Also check if terminal is reachable from target via 1 hop
                # (allows for alias/synonym proximity)
                if self._entities_adjacent(terminal, target):
                    breakdown.terminal_grounding = self.config.grounding_reward * 0.5
                    breakdown.terminal_match = True

        # Composite: weighted sum
        breakdown.composite = (
            self.config.validity_weight * breakdown.axiomatic_validity
            + self.config.continuity_weight * breakdown.chain_continuity
            + self.config.grounding_weight * breakdown.terminal_grounding
        )

        return breakdown

    def score_normalized(
        self,
        path: list[tuple[str, str, str] | KGHop],
        query_entity: Optional[str] = None,
    ) -> float:
        """Return a normalized score in [-1, 1].

        Useful for combining with other reward signals (e.g., anticipation
        geometry scalars) that also operate in bounded ranges.

        Normalization: composite / max_possible_composite, where max is
        achieved when all edges are valid, all hops are continuous, and
        terminal grounding succeeds.
        """
        breakdown = self.score(path, query_entity)
        n = breakdown.n_hops
        if n == 0:
            return 0.0

        max_score = (
            self.config.validity_weight * n * self.config.valid_edge_reward
            + self.config.continuity_weight * max(0, n - 1) * self.config.continuity_reward
            + self.config.grounding_weight * self.config.grounding_reward
        )
        # Minimum possible score (all invalid, all discontinuous)
        min_score = (
            self.config.validity_weight * n * self.config.invalid_edge_penalty
            + self.config.continuity_weight * max(0, n - 1) * self.config.discontinuity_penalty
        )

        range_size = max_score - min_score
        if range_size == 0:
            return 0.0

        # Map [min_score, max_score] to [-1, 1]
        return 2.0 * (breakdown.composite - min_score) / range_size - 1.0

    def batch_score(
        self,
        paths: list[list[tuple[str, str, str]]],
        query_entities: Optional[list[str | None]] = None,
    ) -> list[SignalBreakdown]:
        """Score multiple paths. Shares the edge cache across all paths."""
        if query_entities is None:
            query_entities = [None] * len(paths)

        return [
            self.score(path, qe)
            for path, qe in zip(paths, query_entities)
        ]

    # ----- GK Interaction -----

    def _edge_exists(self, hop: KGHop) -> bool:
        """Check if an edge exists in the Graph Kernel.

        Uses GET /api/knowledge with subject, predicate, and object filters.
        The GK returns {triples: [...], total: N}. If total > 0, the edge exists.

        Results are cached to avoid redundant queries within a session.
        """
        cache_key = (hop.subject.lower(), hop.predicate.lower(), hop.object.lower())
        if cache_key in self._edge_cache:
            return self._edge_cache[cache_key]

        try:
            resp = self._session.get(
                f"{self.gk_url}/api/knowledge",
                params={
                    "subject": hop.subject,
                    "predicate": hop.predicate,
                    "object": hop.object,
                    "min_confidence": self.config.min_confidence,
                    "limit": 1,
                },
                timeout=self.config.query_timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            exists = data.get("total", 0) > 0
        except (requests.RequestException, ValueError) as e:
            logger.warning("GK edge check failed for %s: %s", cache_key, e)
            # On GK failure, assume edge does not exist (conservative)
            exists = False

        self._edge_cache[cache_key] = exists
        return exists

    def _entities_adjacent(self, entity_a: str, entity_b: str) -> bool:
        """Check if two entities are connected by any single edge.

        Uses the traversal endpoint with max_hops=1 starting from entity_a
        and checks if entity_b appears in any returned path.
        """
        try:
            resp = self._session.post(
                f"{self.gk_url}/api/knowledge/traverse",
                json={
                    "start": entity_a,
                    "direction": "both",
                    "max_hops": 1,
                    "max_results": 50,
                    "return_paths": True,
                },
                timeout=self.config.query_timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            for path in data.get("paths", []):
                if entity_b in [e.lower() for e in path.get("entities", [])]:
                    return True
        except (requests.RequestException, ValueError) as e:
            logger.warning("GK adjacency check failed: %s", e)

        return False

    # ----- Utilities -----

    @staticmethod
    def _normalize_hop(hop: tuple[str, str, str] | KGHop) -> KGHop:
        if isinstance(hop, KGHop):
            return hop
        return KGHop(subject=hop[0], predicate=hop[1], object=hop[2])

    def clear_cache(self) -> None:
        """Clear the edge-existence cache."""
        self._edge_cache.clear()

    def health_check(self) -> bool:
        """Verify the Graph Kernel is reachable."""
        try:
            resp = self._session.get(
                f"{self.gk_url}/health",
                timeout=2.0,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def demo():
    """Demonstrate the KG-path reward function with a sample path."""
    reward = KGPathReward()

    if not reward.health_check():
        print("Graph Kernel not available at localhost:8001")
        print("Running with mock data instead...\n")

        # Demonstrate scoring logic without GK
        path = [
            ("spore", "built_with", "swiftui"),
            ("swiftui", "is_a", "framework"),
            ("framework", "uses", "declarative_ui"),
        ]

        # Manually compute what the scores would be
        print(f"Path: {len(path)} hops")
        for i, hop in enumerate(path):
            print(f"  Hop {i}: {hop[0]} --[{hop[1]}]--> {hop[2]}")

        print("\nScoring breakdown (without GK, all edges assumed invalid):")
        breakdown = reward.score(path, query_entity="declarative_ui")
        print(f"  Axiomatic validity:  {breakdown.axiomatic_validity:+.1f}")
        print(f"    ({breakdown.n_valid_edges}/{breakdown.n_hops} edges valid)")
        print(f"  Chain continuity:    {breakdown.chain_continuity:+.1f}")
        print(f"    ({breakdown.n_continuous}/{max(0, breakdown.n_hops-1)} transitions continuous)")
        print(f"  Terminal grounding:  {breakdown.terminal_grounding:+.1f}")
        print(f"    (terminal match: {breakdown.terminal_match})")
        print(f"  Composite:           {breakdown.composite:+.1f}")
        print(f"  Normalized [-1,1]:   {reward.score_normalized(path, 'declarative_ui'):+.3f}")
        return

    # Live GK scoring
    path = [
        ("spore", "built_with", "swiftui"),
        ("swiftui", "is_a", "framework"),
    ]

    print("Scoring path against live Graph Kernel:\n")
    for i, hop in enumerate(path):
        print(f"  Hop {i}: {hop[0]} --[{hop[1]}]--> {hop[2]}")

    breakdown = reward.score(path, query_entity="framework")
    print(f"\n  Axiomatic validity:  {breakdown.axiomatic_validity:+.1f}")
    print(f"  Chain continuity:    {breakdown.chain_continuity:+.1f}")
    print(f"  Terminal grounding:  {breakdown.terminal_grounding:+.1f}")
    print(f"  Composite:           {breakdown.composite:+.1f}")
    print(f"  Normalized [-1,1]:   {reward.score_normalized(path, 'framework'):+.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
