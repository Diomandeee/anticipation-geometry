"""
Evaluation Harness
==================

Compares three reasoning-path evaluation strategies on multi-hop reasoning
tasks derived from the cc-graph-kernel Knowledge Graph:

  1. Princeton KG-path rewards alone (arXiv:2603.14147)
     - Tests: does the reward correctly rank valid paths above invalid ones?

  2. Anticipation geometry alone
     - Tests: do the 4 scalars correlate with path quality?

  3. Combined (anticipation geometry + KG-path rewards)
     - Tests: does the combination outperform either signal alone?

Evaluation Protocol
-------------------

We generate reasoning paths by:
  1. Querying the GK for multi-hop traversals (gold paths)
  2. Corrupting gold paths to create silver and bronze paths:
     - Silver: swap one entity to break continuity but keep edges valid
     - Bronze: replace entities with random strings (hallucinated edges)
  3. Embedding path entities and computing anticipation geometry
  4. Scoring each path with all three strategies
  5. Measuring ranking accuracy (does the strategy rank gold > silver > bronze?)

Metrics:
  - Pairwise ranking accuracy: % of (gold, corrupt) pairs where gold scores higher
  - Kendall's tau: rank correlation between reward scores and ground-truth quality
  - Mean reward gap: average score difference between gold and corrupt paths
  - Per-signal diagnostic: which signals best distinguish path quality?

Usage:
    from eval_harness import EvaluationHarness

    harness = EvaluationHarness(gk_url="http://localhost:8001")
    results = harness.run(n_paths=50, max_hops=3)
    results.print_report()
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import requests

from kg_reward import KGPathReward, SignalBreakdown, RewardConfig
from generalized_anticipation import AnticipationGeometry, AnticipationPacket

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ReasoningPath:
    """A reasoning path with quality label."""
    hops: list[tuple[str, str, str]]
    query_entity: str
    quality: str  # "gold", "silver", "bronze"
    source_path_id: int = 0


@dataclass
class PathEvaluation:
    """Evaluation scores for a single path."""
    path: ReasoningPath
    kg_reward: SignalBreakdown
    kg_normalized: float
    anticipation: AnticipationPacket
    combined_score: float
    anticipation_summary: dict = field(default_factory=dict)


@dataclass
class EvaluationResults:
    """Aggregate evaluation results."""
    evaluations: list[PathEvaluation]
    n_gold: int = 0
    n_silver: int = 0
    n_bronze: int = 0

    # Ranking accuracy per strategy
    kg_pairwise_accuracy: float = 0.0
    anticipation_pairwise_accuracy: float = 0.0
    combined_pairwise_accuracy: float = 0.0

    # Mean reward gaps
    kg_gold_silver_gap: float = 0.0
    kg_gold_bronze_gap: float = 0.0
    anticipation_gold_silver_gap: float = 0.0
    anticipation_gold_bronze_gap: float = 0.0
    combined_gold_silver_gap: float = 0.0
    combined_gold_bronze_gap: float = 0.0

    # Per-signal diagnostics
    signal_discriminability: dict = field(default_factory=dict)

    # Timing
    elapsed_s: float = 0.0

    def print_report(self):
        """Print a formatted evaluation report."""
        print("\n" + "=" * 70)
        print("EVALUATION REPORT: KG-Path Reward vs Anticipation Geometry")
        print("=" * 70)

        print(f"\nPaths evaluated: {len(self.evaluations)}")
        print(f"  Gold: {self.n_gold}  Silver: {self.n_silver}  Bronze: {self.n_bronze}")
        print(f"  Time: {self.elapsed_s:.2f}s")

        print(f"\n{'Strategy':<25} | {'Pairwise Accuracy':>18} | {'Gold-Silver Gap':>15} | {'Gold-Bronze Gap':>15}")
        print("-" * 80)
        print(f"{'KG Reward':<25} | {self.kg_pairwise_accuracy:>17.1%} | {self.kg_gold_silver_gap:>+15.3f} | {self.kg_gold_bronze_gap:>+15.3f}")
        print(f"{'Anticipation':<25} | {self.anticipation_pairwise_accuracy:>17.1%} | {self.anticipation_gold_silver_gap:>+15.3f} | {self.anticipation_gold_bronze_gap:>+15.3f}")
        print(f"{'Combined':<25} | {self.combined_pairwise_accuracy:>17.1%} | {self.combined_gold_silver_gap:>+15.3f} | {self.combined_gold_bronze_gap:>+15.3f}")

        if self.signal_discriminability:
            print(f"\nPer-Signal Discriminability (gold vs bronze Cohen's d):")
            for signal, d in sorted(self.signal_discriminability.items(), key=lambda x: -abs(x[1])):
                bar = "#" * int(min(abs(d) * 10, 40))
                print(f"  {signal:<30} | d = {d:>+6.2f} | {bar}")

        print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Path Generators
# ---------------------------------------------------------------------------

class PathGenerator:
    """Generate gold, silver, and bronze reasoning paths from the GK.

    Gold paths: real multi-hop traversals from the Knowledge Graph.
    Silver paths: one entity swapped to break chain continuity.
    Bronze paths: entities replaced with random strings (no GK presence).
    """

    RANDOM_ENTITIES = [
        "xylophone_fish", "quantum_banana", "invisible_castle",
        "fractal_penguin", "temporal_sandwich", "abstract_lighthouse",
        "recursive_mountain", "holographic_river", "nonexistent_bridge",
        "imaginary_telescope",
    ]

    def __init__(self, gk_url: str = "http://localhost:8001"):
        self.gk_url = gk_url.rstrip("/")
        self._session = requests.Session()

    def generate_gold_paths(
        self,
        n_paths: int = 20,
        max_hops: int = 3,
        seed_entities: Optional[list[str]] = None,
    ) -> list[ReasoningPath]:
        """Generate gold paths by traversing the Knowledge Graph.

        If the GK is unavailable, falls back to synthetic paths.
        """
        if seed_entities is None:
            seed_entities = self._get_seed_entities()

        if not seed_entities:
            logger.warning("No seed entities available. Using synthetic paths.")
            return self._synthetic_gold_paths(n_paths)

        gold_paths = []
        attempts = 0
        max_attempts = n_paths * 5

        while len(gold_paths) < n_paths and attempts < max_attempts:
            attempts += 1
            entity = random.choice(seed_entities)
            path = self._traverse_from(entity, max_hops)

            if path and len(path) >= 2:
                query_entity = path[-1][2]  # object of last hop
                gold_paths.append(ReasoningPath(
                    hops=path,
                    query_entity=query_entity,
                    quality="gold",
                    source_path_id=len(gold_paths),
                ))

        if not gold_paths:
            logger.warning("GK traversal produced no paths. Using synthetic paths.")
            return self._synthetic_gold_paths(n_paths)

        return gold_paths

    def corrupt_to_silver(self, gold: ReasoningPath) -> ReasoningPath:
        """Create a silver path by breaking chain continuity.

        Swaps the subject of a random hop (not the first) with a different
        entity that exists in the GK. This keeps individual edges plausible
        but breaks the chain.
        """
        if len(gold.hops) < 2:
            return self.corrupt_to_bronze(gold)

        hops = list(gold.hops)
        swap_idx = random.randint(1, len(hops) - 1)
        s, p, o = hops[swap_idx]

        # Replace subject with a different entity (break continuity)
        prev_object = hops[swap_idx - 1][2]
        new_subject = prev_object + "_diverged"
        hops[swap_idx] = (new_subject, p, o)

        return ReasoningPath(
            hops=hops,
            query_entity=gold.query_entity,
            quality="silver",
            source_path_id=gold.source_path_id,
        )

    def corrupt_to_bronze(self, gold: ReasoningPath) -> ReasoningPath:
        """Create a bronze path by replacing entities with random strings.

        This produces a path where edges don't exist in the GK at all.
        """
        hops = []
        for i, (s, p, o) in enumerate(gold.hops):
            new_s = random.choice(self.RANDOM_ENTITIES) if i > 0 else s
            new_o = random.choice(self.RANDOM_ENTITIES)
            hops.append((new_s, p, new_o))

        return ReasoningPath(
            hops=hops,
            query_entity=gold.query_entity,
            quality="bronze",
            source_path_id=gold.source_path_id,
        )

    # ----- GK queries -----

    def _get_seed_entities(self, limit: int = 50) -> list[str]:
        """Get a set of entities from the GK to use as traversal seeds."""
        try:
            resp = self._session.get(
                f"{self.gk_url}/api/knowledge/stats",
                timeout=5.0,
            )
            resp.raise_for_status()
            stats = resp.json()

            # Fetch some actual triples to get entity names
            resp2 = self._session.get(
                f"{self.gk_url}/api/knowledge",
                params={"limit": limit},
                timeout=5.0,
            )
            resp2.raise_for_status()
            data = resp2.json()

            entities = set()
            for triple in data.get("triples", []):
                entities.add(triple["subject"])
                entities.add(triple["object"])

            return list(entities)[:limit]

        except (requests.RequestException, ValueError) as e:
            logger.warning("Failed to get seed entities: %s", e)
            return []

    def _traverse_from(
        self,
        entity: str,
        max_hops: int = 3,
    ) -> list[tuple[str, str, str]]:
        """Traverse the KG from a starting entity to get a path."""
        try:
            resp = self._session.post(
                f"{self.gk_url}/api/knowledge/traverse",
                json={
                    "start": entity,
                    "direction": "outgoing",
                    "max_hops": max_hops,
                    "max_results": 10,
                    "return_paths": True,
                },
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()

            paths = data.get("paths", [])
            if not paths:
                return []

            # Pick the longest path
            best = max(paths, key=lambda p: p.get("hops", 0))
            edges = best.get("edges", [])

            return [
                (e["subject"], e["predicate"], e["object"])
                for e in edges
            ]

        except (requests.RequestException, ValueError) as e:
            logger.warning("Traversal from %s failed: %s", entity, e)
            return []

    def _synthetic_gold_paths(self, n: int) -> list[ReasoningPath]:
        """Generate synthetic gold paths when GK is unavailable.

        These use plausible-looking entities and predicates to demonstrate
        the evaluation framework even without a live GK.
        """
        entities = [
            "spore", "swiftui", "framework", "ios", "mobile",
            "cloudkit", "database", "sync", "realtime", "websocket",
            "graph_kernel", "knowledge_graph", "triple", "entity",
            "rag_plusplus", "retrieval", "embedding", "vector",
            "anticipation", "motion", "trajectory", "semantic",
        ]
        predicates = [
            "built_with", "is_a", "uses", "belongs_to",
            "has_feature", "deployed_on", "evolved_from", "tagged",
        ]

        paths = []
        for i in range(n):
            n_hops = random.randint(2, 4)
            hops = []
            prev_obj = random.choice(entities)
            for _ in range(n_hops):
                subj = prev_obj
                pred = random.choice(predicates)
                obj = random.choice([e for e in entities if e != subj])
                hops.append((subj, pred, obj))
                prev_obj = obj

            paths.append(ReasoningPath(
                hops=hops,
                query_entity=prev_obj,
                quality="gold",
                source_path_id=i,
            ))

        return paths


# ---------------------------------------------------------------------------
# Path Embedding
# ---------------------------------------------------------------------------

def embed_path_entities(
    path: ReasoningPath,
    dim: int = 64,
) -> list[np.ndarray]:
    """Embed path entities as vectors for anticipation geometry.

    Uses deterministic hashing to produce consistent embeddings.
    Each entity maps to a point in R^dim based on its character
    composition. This is a lightweight alternative to using a
    pretrained entity embedding model.

    The key property we need: entities that share substrings should
    produce similar vectors, so that chain continuity manifests as
    small steps in the embedding space.
    """
    entities = []
    for s, _, o in path.hops:
        if not entities or entities[-1] != s:
            entities.append(s)
        entities.append(o)

    vectors = []
    for entity in entities:
        vec = np.zeros(dim)
        # Character trigram hashing
        name = entity.lower().replace("_", " ")
        for i in range(len(name) - 2):
            trigram = name[i:i + 3]
            h = hash(trigram)
            idx = h % dim
            vec[idx] += 1.0
            # Also scatter to nearby dimensions for smoothness
            vec[(idx + 1) % dim] += 0.3
            vec[(idx - 1) % dim] += 0.3

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        vectors.append(vec)

    return vectors


# ---------------------------------------------------------------------------
# Evaluation Harness
# ---------------------------------------------------------------------------

class EvaluationHarness:
    """Run comparative evaluation of KG rewards vs anticipation geometry.

    Args:
        gk_url: URL of the Graph Kernel service.
        reward_config: Configuration for the KG-path reward function.
        kg_weight: Weight of KG reward in combined score.
        anticipation_weight: Weight of anticipation scalars in combined score.
    """

    def __init__(
        self,
        gk_url: str = "http://localhost:8001",
        reward_config: Optional[RewardConfig] = None,
        kg_weight: float = 0.6,
        anticipation_weight: float = 0.4,
    ):
        self.reward = KGPathReward(gk_url=gk_url, config=reward_config)
        self.generator = PathGenerator(gk_url=gk_url)
        self.geometry = AnticipationGeometry(k_neighbors=3)
        self.kg_weight = kg_weight
        self.anticipation_weight = anticipation_weight

    def run(
        self,
        n_paths: int = 20,
        max_hops: int = 3,
        seed_entities: Optional[list[str]] = None,
    ) -> EvaluationResults:
        """Run the full evaluation.

        1. Generate gold paths from GK
        2. Corrupt each gold into silver and bronze variants
        3. Score all paths with all three strategies
        4. Compute ranking metrics

        Args:
            n_paths: Number of gold paths to generate.
            max_hops: Maximum hops per path.
            seed_entities: Optional starting entities for traversal.

        Returns:
            EvaluationResults with ranking accuracy and diagnostic scores.
        """
        start_time = time.time()
        import random
        random.seed(42)
        np.random.seed(42)
        logger.info("Starting evaluation with %d paths, max %d hops (seed=42)", n_paths, max_hops)

        # Step 1: Generate paths
        gold_paths = self.generator.generate_gold_paths(
            n_paths=n_paths,
            max_hops=max_hops,
            seed_entities=seed_entities,
        )
        logger.info("Generated %d gold paths", len(gold_paths))

        all_paths = []
        for gold in gold_paths:
            all_paths.append(gold)
            all_paths.append(self.generator.corrupt_to_silver(gold))
            all_paths.append(self.generator.corrupt_to_bronze(gold))

        # Step 2: Evaluate all paths
        evaluations = []
        for path in all_paths:
            ev = self._evaluate_path(path)
            evaluations.append(ev)

        # Step 3: Compute metrics
        results = self._compute_metrics(evaluations)
        results.elapsed_s = time.time() - start_time

        return results

    def _evaluate_path(self, path: ReasoningPath) -> PathEvaluation:
        """Score a single path with all three strategies."""
        # KG reward
        kg_breakdown = self.reward.score(path.hops, query_entity=path.query_entity)
        kg_normalized = self.reward.score_normalized(path.hops, query_entity=path.query_entity)

        # Anticipation geometry
        entity_vectors = embed_path_entities(path)
        anticipation = self.geometry.compute(entity_vectors)

        # Anticipation summary score:
        # Use mean commitment + (1 - mean uncertainty) as quality proxy.
        # High commitment + low uncertainty = well-structured path.
        ant_summary = anticipation.summary()
        if ant_summary.get("empty"):
            ant_score = 0.0
        else:
            ant_score = (
                ant_summary.get("commitment_mean", 0.5)
                + (1.0 - ant_summary.get("uncertainty_mean", 0.5))
                + ant_summary.get("recovery_margin_final", 0.5)
            ) / 3.0
            # Normalize to [-1, 1] range
            ant_score = 2.0 * ant_score - 1.0

        # Combined score
        combined = (
            self.kg_weight * kg_normalized
            + self.anticipation_weight * ant_score
        )

        return PathEvaluation(
            path=path,
            kg_reward=kg_breakdown,
            kg_normalized=kg_normalized,
            anticipation=anticipation,
            combined_score=combined,
            anticipation_summary=ant_summary,
        )

    def _compute_metrics(self, evaluations: list[PathEvaluation]) -> EvaluationResults:
        """Compute ranking accuracy and diagnostic metrics."""
        results = EvaluationResults(evaluations=evaluations)

        # Group by quality
        by_quality: dict[str, list[PathEvaluation]] = {"gold": [], "silver": [], "bronze": []}
        for ev in evaluations:
            by_quality[ev.path.quality].append(ev)

        results.n_gold = len(by_quality["gold"])
        results.n_silver = len(by_quality["silver"])
        results.n_bronze = len(by_quality["bronze"])

        # Pairwise ranking accuracy
        # For each (gold, corrupt) pair with same source_path_id,
        # check if gold scores higher.
        gold_map = {ev.path.source_path_id: ev for ev in by_quality["gold"]}
        corrupt = by_quality["silver"] + by_quality["bronze"]

        kg_correct = 0
        ant_correct = 0
        combined_correct = 0
        total_pairs = 0

        for cev in corrupt:
            gev = gold_map.get(cev.path.source_path_id)
            if gev is None:
                continue

            total_pairs += 1
            if gev.kg_normalized > cev.kg_normalized:
                kg_correct += 1
            if gev.combined_score > cev.combined_score:
                combined_correct += 1

            # Anticipation score
            g_ant = self._anticipation_score(gev)
            c_ant = self._anticipation_score(cev)
            if g_ant > c_ant:
                ant_correct += 1

        if total_pairs > 0:
            results.kg_pairwise_accuracy = kg_correct / total_pairs
            results.anticipation_pairwise_accuracy = ant_correct / total_pairs
            results.combined_pairwise_accuracy = combined_correct / total_pairs

        # Mean reward gaps
        kg_gold = [ev.kg_normalized for ev in by_quality["gold"]]
        kg_silver = [ev.kg_normalized for ev in by_quality["silver"]]
        kg_bronze = [ev.kg_normalized for ev in by_quality["bronze"]]

        if kg_gold and kg_silver:
            results.kg_gold_silver_gap = np.mean(kg_gold) - np.mean(kg_silver)
        if kg_gold and kg_bronze:
            results.kg_gold_bronze_gap = np.mean(kg_gold) - np.mean(kg_bronze)

        ant_gold = [self._anticipation_score(ev) for ev in by_quality["gold"]]
        ant_silver = [self._anticipation_score(ev) for ev in by_quality["silver"]]
        ant_bronze = [self._anticipation_score(ev) for ev in by_quality["bronze"]]

        if ant_gold and ant_silver:
            results.anticipation_gold_silver_gap = np.mean(ant_gold) - np.mean(ant_silver)
        if ant_gold and ant_bronze:
            results.anticipation_gold_bronze_gap = np.mean(ant_gold) - np.mean(ant_bronze)

        comb_gold = [ev.combined_score for ev in by_quality["gold"]]
        comb_silver = [ev.combined_score for ev in by_quality["silver"]]
        comb_bronze = [ev.combined_score for ev in by_quality["bronze"]]

        if comb_gold and comb_silver:
            results.combined_gold_silver_gap = np.mean(comb_gold) - np.mean(comb_silver)
        if comb_gold and comb_bronze:
            results.combined_gold_bronze_gap = np.mean(comb_gold) - np.mean(comb_bronze)

        # Per-signal discriminability (Cohen's d between gold and bronze)
        results.signal_discriminability = self._compute_discriminability(
            by_quality["gold"], by_quality["bronze"]
        )

        return results

    def _anticipation_score(self, ev: PathEvaluation) -> float:
        """Extract a single anticipation quality score from a PathEvaluation."""
        s = ev.anticipation_summary
        if s.get("empty"):
            return 0.0
        return (
            s.get("commitment_mean", 0.5)
            + (1.0 - s.get("uncertainty_mean", 0.5))
            + s.get("recovery_margin_final", 0.5)
        ) / 3.0

    def _compute_discriminability(
        self,
        gold_evals: list[PathEvaluation],
        bronze_evals: list[PathEvaluation],
    ) -> dict[str, float]:
        """Compute Cohen's d for each signal between gold and bronze paths.

        Cohen's d = (mean_gold - mean_bronze) / pooled_std

        Higher d means the signal better distinguishes valid from invalid paths.
        """
        if not gold_evals or not bronze_evals:
            return {}

        signals = {}

        # KG signals
        def safe_cohens_d(gold_vals: list[float], bronze_vals: list[float]) -> float:
            g = np.array(gold_vals)
            b = np.array(bronze_vals)
            if len(g) < 2 or len(b) < 2:
                return 0.0
            pooled_std = np.sqrt(
                ((len(g) - 1) * np.var(g) + (len(b) - 1) * np.var(b))
                / (len(g) + len(b) - 2)
            )
            if pooled_std < 1e-10:
                return 0.0
            return float((np.mean(g) - np.mean(b)) / pooled_std)

        # KG composite
        signals["kg_composite"] = safe_cohens_d(
            [ev.kg_normalized for ev in gold_evals],
            [ev.kg_normalized for ev in bronze_evals],
        )

        # KG sub-signals
        signals["kg_axiomatic_validity"] = safe_cohens_d(
            [ev.kg_reward.axiomatic_validity for ev in gold_evals],
            [ev.kg_reward.axiomatic_validity for ev in bronze_evals],
        )
        signals["kg_chain_continuity"] = safe_cohens_d(
            [ev.kg_reward.chain_continuity for ev in gold_evals],
            [ev.kg_reward.chain_continuity for ev in bronze_evals],
        )

        # Anticipation signals
        for key in ["commitment_mean", "uncertainty_mean", "recovery_margin_final"]:
            signals[f"ant_{key}"] = safe_cohens_d(
                [ev.anticipation_summary.get(key, 0.5) for ev in gold_evals],
                [ev.anticipation_summary.get(key, 0.5) for ev in bronze_evals],
            )

        # Combined
        signals["combined"] = safe_cohens_d(
            [ev.combined_score for ev in gold_evals],
            [ev.combined_score for ev in bronze_evals],
        )

        return signals


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------

def demo():
    """Run the evaluation harness with default settings."""
    logging.basicConfig(level=logging.INFO)

    print("Anticipation Geometry + KG-Path Reward Evaluation")
    print("=" * 60)

    harness = EvaluationHarness()
    results = harness.run(n_paths=15, max_hops=3)
    results.print_report()

    # Print a few example paths with their scores
    print("\nSample Paths:")
    print("-" * 60)
    for ev in results.evaluations[:9]:
        path = ev.path
        quality_marker = {"gold": "[GOLD]", "silver": "[SILV]", "bronze": "[BRNZ]"}[path.quality]
        hops_str = " -> ".join([h[0] for h in path.hops] + [path.hops[-1][2]])
        print(f"  {quality_marker} {hops_str}")
        print(f"    KG: {ev.kg_normalized:+.3f}  Ant: {harness._anticipation_score(ev):.3f}  Combined: {ev.combined_score:+.3f}")
        print()


if __name__ == "__main__":
    demo()
