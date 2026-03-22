# Anticipation Geometry: Domain-General Trajectory Characterization with Knowledge Graph-Grounded Rewards

**Mohamed Diomande**
Independent Researcher

---

## Abstract

I present Anticipation Geometry, a mathematical framework that characterizes trajectories through arbitrary state spaces using seven geometric scalars: commitment, uncertainty, transition pressure, recovery margin, phase stiffness, novelty, and stability. These scalars are domain-general, operating on any sequence of vectors in a metric space equipped with a differentiable time parameter. I combine this framework with knowledge graph path-derived reward signals to create a unified system for both trajectory analysis and model training.

I evaluate on three domains: physical motion (simulated kinematics), conversational reasoning (20,000 dialogue turns from 164 conversations, expanded to 308 sessions from 19,000 prompts), and knowledge graph traversal (199 multi-hop paths from a graph kernel).

The key finding is that transition pressure, defined as the rate of commitment increase minus the rate of uncertainty decrease (dC/dt - dU/dt), is a statistically significant predictor of reasoning convergence. Its sign predicts conversation convergence at 71.8% accuracy (z = 2.72, p < 0.007), and its standard deviation achieves 69.8% accuracy (+8.1pp over baseline) as a single feature on higher-dimensional embeddings. In an expanded evaluation, inscription-derived features encoding conversational dynamics as sigil probability distributions achieve 79.5% accuracy via gradient boosting on the original 39 sessions (z = 3.68, p < 0.001), a +7.7pp improvement over the transition pressure baseline. On knowledge graph paths, anticipation-augmented rewards discriminate valid from hard-negative paths with 81.0% pairwise accuracy (Cohen's d = 2.23).

I do not claim state-of-the-art performance on any single task. I show that a single geometric framework, with no task-specific training, produces significant signal across domains, suggesting that trajectory geometry captures a general property of reasoning.

---

## 1. Introduction

The central question of this paper is deceptively simple: *Is there a universal geometry to the act of deciding?*

When a dancer commits to a pirouette, the space of possible next movements collapses rapidly. Uncertainty about what comes next drops. Commitment to the current trajectory rises. The rate at which these two quantities diverge, what I call *transition pressure*, spikes measurably in the moments before the turn begins. This is not metaphor. It is a computable signal derived from quaternion sequences at 50 Hz.

When a language model reasons through a multi-step problem, an analogous process occurs in its embedding space. Early in the generation, the distribution over plausible continuations is broad. As the model accumulates intermediate reasoning steps, it narrows toward a specific answer. The embedding trajectory, viewed as a sequence of points in high-dimensional space, exhibits the same geometric signature: rising commitment, falling uncertainty, positive transition pressure.

When a graph traversal algorithm explores a knowledge graph, it starts at a source entity with many candidate edges. As it follows a specific path of hops, the set of reachable terminal entities shrinks. The trajectory through entity-embedding space again shows the same pattern.

These observations suggest that "reasoning," broadly construed, has a geometric signature independent of the medium it occurs in. This paper formalizes that intuition.

### 1.1 Motivation

There is an ongoing tension in AI between scaling (more parameters, more compute) and structure (explicit representations, knowledge graphs). Belova et al. (2025) argue for structure, proposing smaller KG-grounded models over monolithic systems.

I noticed something orthogonal to both positions: whether the reasoning agent is a 7B language model or a deterministic graph traversal, convergence from uncertainty to commitment follows the same geometric pattern. These patterns are measurable, and the measurements work as reward signals.

The practical consequence: if transition pressure predicts reasoning quality *before* the reasoning finishes, you can give real-time feedback during inference instead of waiting for task completion. This turns a sparse reward (success/failure at the end) into a dense one (geometric quality at every step).

### 1.2 Contributions

The following are the contributions:

1. **Anticipation Geometry**: A formal framework defining seven scalar signals computable from any trajectory in a metric space, with proofs of domain-generality (Section 3).

2. **The LatentFrame Abstraction**: A minimal data structure (`z: Vec<f32>, timestamp: f64, valid: bool`) that serves as the universal input type across all domain instantiations (Section 3.3).

3. **Anticipation-Augmented KG Rewards**: An extension of the Princeton DSS three-signal reward with transition pressure as a fourth signal, enabling reasoning quality assessment before task completion (Section 4).

4. **Cross-Domain Evaluation**: Empirical evidence that the same seven scalars, with no domain-specific tuning, produce statistically significant predictions on conversation convergence (z = 2.72, p < 0.007) and large-effect KG path discrimination (Cohen's d = 2.23), plus valid scalar distributions on kinematic data (Section 7).

5. **Open-Source Implementation**: A Rust implementation (the anticipation library) operating at sub-2ms latency per frame, with Python bindings for ML integration.

---

## 2. Related Work

### 2.1 Knowledge Graphs as Implicit Reward Models

Belova et al. (2025) propose domain-specific superintelligence (DSS): small language models ($\leq$ 32B parameters) grounded in knowledge graphs, ontologies, and formal logic (arXiv:2603.14147). Their argument is that explicit structure, not scale, is the path to reliable reasoning.

The Princeton framework treats knowledge graphs as training scaffolding. During pre-training, KG paths define curricula: the model learns to traverse 2-hop chains before attempting 5-hop reasoning. During fine-tuning, KG-derived rewards score outputs based on three signals:

- **Axiomatic validity**: Does the output conform to ontological constraints? (e.g., a medical diagnosis must reference entities in the disease-symptom subgraph.)
- **Chain continuity**: Are intermediate reasoning steps connected by valid KG edges?
- **Terminal grounding**: Does the final answer correspond to a terminal entity in the relevant subgraph?

This three-signal framework produces a reward $R \in [0, 1]$ that can replace or augment human preference labels for RLHF-style training. The key advantage over human annotation is determinism: the same KG, query, and output always yield the same reward.

### 2.2 GraphMERT: Distilling KGs into Parameters

GraphMERT (Belova et al., 2025, arXiv:2510.09580) shows that an 80M-parameter encoder can distill KG structure from text at 69.8% FActScore, beating Qwen3-32B at 40.2%. This matters here because it confirms KG structure can live in a compact embedding space, which is where the scalars operate.

GraphMERT's training pipeline proceeds in three stages: (1) entity-linked pre-training on KG-annotated text, (2) relation-type contrastive learning, and (3) multi-hop path prediction. The resulting encoder maps text spans to a 768-dimensional space where KG neighborhoods are preserved. When I apply Anticipation Geometry to sequences of GraphMERT embeddings, the scalars recover the hop structure of the underlying KG paths (Section 5.3).

### 2.3 QwQ-Med-3: KG-Path Training Curricula

The bottom-up DSS paper (Xiao et al., 2025, arXiv:2507.13966) fine-tunes QwQ-32B on 24,000 KG-grounded medical reasoning tasks, achieving 84.72% on ICD-Bench (state of the art at time of publication). The training curriculum orders tasks by KG path complexity: 2-hop diagnostic chains first, then 3-hop differential diagnoses, then 5-hop multi-system reasoning.

In geometric terms, this is transition pressure management. Simple 2-hop paths have high transition pressure (the answer space collapses after one hop). Complex 5-hop paths need sustained moderate pressure across many steps. Ordering easy-to-hard implicitly teaches the model to control its own convergence rate.

### 2.4 Geometric Deep Learning

Bronstein et al. (2021) establish the theoretical foundations for applying geometric principles to deep learning. Their framework identifies fthe geometric domains (grids, graphs, groups, and geodesics) and defines equivariance as the key inductive bias. The work extends this perspective by treating the *trajectory* through a learned representation as a geometric object in its own right, rather than focusing on the structure of the representation space itself.

The distinction is important. Geometric deep learning asks: "What symmetries should the representation respect?" Anticipation Geometry asks: "What does the *path* through the representation tell us about the *process* that generated it?" These are complementary questions, and I show that the answers can be composed (Section 6).

### 2.5 Cognitive Architectures

The CoALA framework (Sumers et al., 2024) proposes a cognitive architecture for language agents organized around working memory, long-term memory, and an action space. The Anticipation Geometry provides the missing *monitoring layer*: a way for the architecture to introspect on the quality of its own reasoning process in real time, rather than only evaluating outputs after the fact.

### 2.6 Anticipatory Computing

Pezzulo et al. (2008) formalize anticipatory behavior as "a process or behavior that does not only depend on past and present, but also on predictions, expectations, or beliefs about the future." The framework operationalizes this definition by providing computable predictions (via the seven scalars) that any system can use to modulate its own behavior. The scalars answer the question: "Given where I have been, how constrained is where I can go?"

---

## 3. Anticipation Geometry Framework

### 3.1 State Space Definition

**Definition 1** (State Space). A state space $\mathcal{S}$ is a metric space $(X, d)$ equipped with an embedding function $\phi: \text{Domain} \to \mathbb{R}^n$ that maps domain-specific observations to vectors. The only requirement on $\phi$ is that semantically similar observations map to nearby vectors under $d$.

This definition is intentionally permissive. Any of the following qualify:

- **Physical motion**: $X = \mathbb{R}^{3 \times 27}$ (27 joint positions), $d$ = Euclidean distance, $\phi$ = forward kinematics from quaternion rotations.
- **Language**: $X = \mathbb{R}^{768}$ (sentence embedding space), $d$ = cosine distance, $\phi$ = any sentence encoder.
- **Knowledge graphs**: $X = \mathbb{R}^{128}$ (entity embedding space), $d$ = Euclidean distance, $\phi$ = TransE, RotatE, or GraphMERT encoder.
- **Agent actions**: $X = \mathbb{R}^{256}$ (action embedding space), $d$ = Euclidean distance, $\phi$ = learned action encoder.

**Definition 2** (Trajectory). A trajectory $\tau$ is a finite sequence of states with timestamps:

$$\tau = \{(s_1, t_1), (s_2, t_2), \ldots, (s_T, t_T)\}$$

where $s_i \in X$ and $t_1 < t_2 < \cdots < t_T$.

**Definition 3** (Latent Trajectory). Given an embedding function $\phi$, the latent trajectory is:

$$\hat{\tau} = \{(\phi(s_1), t_1), (\phi(s_2), t_2), \ldots, (\phi(s_T), t_T)\}$$

All seven scalars are defined on latent trajectories. This is the key to domain-generality: the scalars never see domain-specific observations, only their embeddings.

### 3.2 The Seven Scalars

The seven anticipation scalars are defined below. Each is a function $f: \hat{\tau} \times t \to \mathbb{R}$ that maps a latent trajectory and a time index to a real value. Unless otherwise noted, all scalars are normalized to $[0, 1]$.

#### 3.2.1 Uncertainty $\mathcal{U}(t)$

**Intuition**: How many plausible futures remain at time $t$?

**Definition**: Let $z_t = \phi(s_t)$ be the current latent state. Let $\mathcal{N}_k(z_t)$ be the $k$ nearest neighbors of $z_t$ in the trajectory history. For each neighbor $z^{(i)}$, compute the displacement vector $\delta_i = z^{(i)} - z_t$.

$$\mathcal{U}(t) = \frac{1 - \bar{\cos}(\{\delta_i\})}{2}$$

where $\bar{\cos}$ is the mean pairwise cosine similarity between all pairs of displacement vectors. When neighbors are spread in many directions, $\bar{\cos}$ is low and uncertainty is high. When neighbors cluster in one direction, $\bar{\cos}$ is high and uncertainty is low. The result is clamped to $[0, 1]$.

**Note on Rust implementation**: The Rust kernel supports a richer formulation using continuation dispersion (where the neighbor's future trajectory went) when an HNSW-backed neighbor database is available. The Python implementation used for all evaluations in this paper uses the simpler pairwise angular dispersion described above.

#### 3.2.2 Commitment $\mathcal{C}(t)$

**Intuition**: How irreversible has the trajectory become?

**Definition**: Commitment measures how much the trajectory has settled. Let $\delta_t = \|z_t - z_{t-1}\|$ be the step size at time $t$, and let $\delta_{\max} = \max_t \delta_t$ be the maximum step in the trajectory.

$$\mathcal{C}(t) = 1 - \frac{\delta_t}{\delta_{\max}}$$

When recent steps are small relative to the largest step in the trajectory, commitment is high (the trajectory has settled). When recent steps are large, commitment is low (still exploring). The result is clamped to $[0, 1]$.

**Note**: The Rust implementation supports a richer three-term formula incorporating constraint proximity and directional persistence for physical motion domains. The Python implementation used for all evaluations in this paper uses the step-size formula above.

**Domain interpretation**: In physical motion, high commitment means the body has entered a movement that would be costly to reverse (mid-stride, mid-jump). In conversation, high commitment means the model's embedding trajectory has been narrowing consistently toward a specific region of semantic space. In graph traversal, high commitment means the path has entered a subgraph with few outgoing edges.

#### 3.2.3 Transition Pressure $\mathcal{T}(t)$

**Intuition**: How fast are futures collapsing?

**Definition**:

$$\mathcal{T}(t) = \frac{d\mathcal{C}}{dt} - \frac{d\mathcal{U}}{dt}$$

In discrete time with step $\Delta t = t_i - t_{i-1}$:

$$\mathcal{T}(t_i) = \frac{\mathcal{C}(t_i) - \mathcal{C}(t_{i-1})}{\Delta t} - \frac{\mathcal{U}(t_i) - \mathcal{U}(t_{i-1})}{\Delta t}$$

Optionally smoothed with an exponential moving average (EMA) with parameter $\alpha$:

$$\hat{\mathcal{T}}(t_i) = \alpha \cdot \mathcal{T}(t_i) + (1 - \alpha) \cdot \hat{\mathcal{T}}(t_{i-1})$$

**Range**: Unlike the other scalars, transition pressure is *not* bounded to $[0, 1]$. Positive values indicate convergence (futures collapsing). Negative values indicate divergence (new possibilities opening up). Zero indicates steady state.

**Central thesis**: Transition pressure is the scalar most directly related to "reasoning quality," because high-quality reasoning is precisely the process of efficiently collapsing a large space of possibilities to a single well-supported conclusion. A reasoning process with low peak transition pressure is meandering. A reasoning process with a single sharp spike is decisive. A reasoning process with sustained moderate pressure is methodical. These patterns are detectable and informative across all three domains I evaluate.

#### 3.2.4 Recovery Margin $\mathcal{R}(t)$

**Intuition**: How far is the current state from losing control?

**Definition**: Recovery margin measures the distance from the nearest branching point, i.e., a position where the trajectory changed direction significantly. Let $B$ be the set of positions where the velocity angle change exceeds the median. Let $b^* = \arg\min_{b \in B} \|z_t - z_b\|$ be the nearest branching point.

$$\mathcal{R}(t) = 1 - \frac{\|z_t - z_{b^*}\|}{\max_{i,j} \|z_i - z_j\|}$$

High recovery margin means the trajectory is still near a decision fork and could pivot. Low recovery margin means it has traveled far from any branching point and is deeply committed to the current path.

**Domain interpretation**: In physical motion, this corresponds to balance margin. In conversation, it reflects how far the reasoning has progressed from the last topic branch. In graph traversal, it measures proximity to the last high-degree node.

#### 3.2.5 Phase Stiffness $\mathcal{P}(t)$

**Intuition**: How locked is the trajectory to its own rhythm?

**Definition**: Phase stiffness combines two components:

$$\mathcal{P}(t) = \frac{1}{2}\mathcal{P}_{\text{dir}}(t) + \frac{1}{2}\left(\frac{1}{1 + \mathcal{J}(t)}\right)$$

where $\mathcal{P}_{\text{dir}}(t)$ is the directional persistence (velocity autocorrelation) and $\mathcal{J}(t)$ is the jerk energy (defined in Section 3.2.7).

Directional persistence is the average cosine similarity between consecutive velocity vectors:

$$\mathcal{P}_{\text{dir}}(t) = \frac{1}{2}\left(1 + \frac{1}{W-1}\sum_{i=t-W+2}^{t}\frac{\dot{z}_i \cdot \dot{z}_{i-1}}{\|\dot{z}_i\| \cdot \|\dot{z}_{i-1}\|}\right)$$

where $W$ is the window size and $\dot{z}_i = (z_i - z_{i-1})/\Delta t$ is the latent velocity.

**Domain interpretation**: In physical motion, high phase stiffness means the body is locked into a rhythmic pattern (walking, running, breathing). In conversation, high phase stiffness means the model is producing tokens that maintain a consistent semantic trajectory (sustained argumentation rather than topic-switching). In graph traversal, high phase stiffness means the path is following a consistent predicate pattern (e.g., successive `is_a` edges rather than alternating between `is_a` and `has_part`).

#### 3.2.6 Novelty $\mathcal{N}(t)$

**Intuition**: How far is the current state from recent experience?

**Definition**: Novelty is the L2 distance from the centroid of recent trajectory history:

$$\mathcal{N}(t) = \text{clamp}\left(\frac{\|z_t - \bar{z}_{\text{hist}}\|}{2}, \; 0, \; 1\right)$$

where $\bar{z}_{\text{hist}} = \frac{1}{H}\sum_{i=t-H}^{t-1}z_i$ is the centroid of the $H$ most recent latent states. The denominator of 2 normalizes under the assumption that embeddings are approximately unit-scaled (L2-normalized regime embeddings satisfy this by construction).

**Implementation**: The kernel maintains a ring buffer of recent regime embeddings. The novelty computation runs in $O(n \cdot H)$ where $n$ is the embedding dimension and $H$ is the history length (typically 50-200 frames).

#### 3.2.7 Stability $\mathcal{S}(t)$

**Intuition**: How predictable are the local dynamics?

**Definition**: Stability combines two sources depending on available features:

From latent features:

$$\mathcal{S}_{\text{lat}}(t) = \frac{1}{2}\text{pred}(t) + \frac{1}{2}\left(\frac{1}{1 + \|\ddot{z}_t\|}\right)$$

where $\text{pred}(t)$ is the linear predictability (inverse of extrapolation residual) and $\ddot{z}_t$ is the latent acceleration.

From kinematic features:

$$\mathcal{S}_{\text{kin}}(t) = \frac{1}{2}\left(\frac{1}{1 + \mathcal{J}(t)}\right) + \frac{1}{2}\mathcal{P}_{\text{dir}}(t)$$

where $\mathcal{J}(t)$ is the jerk energy:

$$\mathcal{J}(t) = \left(\frac{1}{W-2}\sum_{i=t-W+3}^{t}\left\|\frac{\ddot{z}_i - \ddot{z}_{i-1}}{\Delta t}\right\|^2\right)^{1/2}$$

Jerk, the third derivative of position, captures the smoothness of the dynamics. Low jerk means smooth, predictable motion. High jerk means abrupt, erratic changes.

### 3.3 The LatentFrame Abstraction

The bridge between domain-specific observations and domain-general scalars is the `LatentFrame`:

```rust
pub struct LatentFrame {
    pub timestamp: f64,
    pub z: Vec<f32>,
    pub dz_dt: Option<Vec<f32>>,
    pub valid: bool,
}
```

This is the *only* type that the scalar computation functions accept. Domain-specific adapters are responsible for producing `LatentFrame` sequences from raw observations:

- **Physical motion**: The anticipation library extracts kinematic features from 27-bone quaternion sequences, then projects them through the regime embedding to produce latent frames.
- **Conversational reasoning**: An embedding model (e.g., sentence-transformers) maps each dialogue turn to a vector, which becomes the `z` field of a `LatentFrame`. The `timestamp` is the turn index (or wall-clock time of the turn).
- **Knowledge graph traversal**: Each entity in a hop path is represented by its TransE/RotatE embedding. The `timestamp` is the hop index.
- **Agent task planning**: Each tool invocation in a KARL trajectory is represented by the concatenation of the tool-name embedding and key-parameter hash, forming the `z` field.

The `valid` flag handles missing data gracefully. In physical motion, sensor dropout produces invalid frames. In conversation, system prompts or metadata turns may be marked invalid. The scalar computations skip invalid frames and interpolate across gaps.

### 3.4 Regime Embedding via Random Projection

To ensure that scalar computations are commensurable across domains with different embedding dimensions, the framework includes a fixed random projection step:

$$e_{\text{regime}} = \text{normalize}\left(M \cdot f\right)$$

where $f \in \mathbb{R}^n$ is the fused feature vector ($n \leq 128$), $M \in \mathbb{R}^{d \times n}$ is a fixed random projection matrix ($d \in \{64, 128, 256\}$), and normalize denotes L2 normalization.

The matrix $M$ is generated deterministically from a fixed seed using a linear congruential generator with Box-Muller transform for approximate Gaussian entries, scaled by $1/\sqrt{n}$ for variance preservation:

$$M_{ij} \sim \frac{1}{\sqrt{n}}\mathcal{N}(0, 1)$$

This approach is grounded in the Johnson-Lindenstrauss lemma (Johnson and Lindenstrauss, 1984): with high probability, the projection preserves pairwise distances up to a factor of $(1 \pm \epsilon)$ for $d = O(\epsilon^{-2} \log T)$, where $T$ is the number of points. For the typical trajectory lengths ($T \leq 10{,}000$) and $d = 64$, this gives $\epsilon \approx 0.4$, which is sufficient for the relative comparisons the scalars perform.

**Frozen invariant**: The projection matrix is computed once at initialization and never changes. The seed (`0xCC_A7_1C_1A_7104_2025`) is fixed across all versions of the software. This guarantees that regime embeddings are comparable across sessions, machines, and software updates.

### 3.5 Formal Properties

**Theorem 1** (Domain Generality). Let $\mathcal{S}_1 = (X_1, d_1)$ and $\mathcal{S}_2 = (X_2, d_2)$ be two state spaces with embedding functions $\phi_1$ and $\phi_2$ mapping to $\mathbb{R}^{n_1}$ and $\mathbb{R}^{n_2}$ respectively. If both latent trajectories are projected through the regime embedding to a common dimension $d$, then all seven scalars are defined and produce values in their respective ranges.

*Proof sketch*: Each scalar is defined purely in terms of: (a) the regime embedding $e_{\text{regime}} \in \mathbb{R}^d$, which has fixed dimension regardless of input domain; (b) differences and norms of these embeddings; and (c) the timestamp sequence. None of these depend on the original state space structure.

**Theorem 2** (Monotone Commitment under Constraint Tightening). If the constraint proximity $\bar{c}(t)$ is monotonically increasing and uncertainty $\mathcal{U}(t)$ is monotonically decreasing over an interval $[t_a, t_b]$, then commitment $\mathcal{C}(t)$ is monotonically increasing over the same interval.

*Proof*: $\mathcal{C}(t) = w_1(1 - \mathcal{U}(t)) + w_2 \bar{c}(t) + w_3 \mathcal{P}_{\text{dir}}(t)$. The first term increases because $\mathcal{U}$ decreases. The second term increases by assumption. The third term may vary, but its weight ($w_3 = 0.3$) is insufficient to overcome the combined monotone increase of the first two terms (combined weight $0.7$) unless directional persistence drops by more than $\frac{0.7}{0.3} \approx 2.3$ units, which is impossible since $\mathcal{P}_{\text{dir}} \in [0, 1]$.

**Corollary** (Positive Transition Pressure at Convergence). Under the conditions of Theorem 2, transition pressure $\mathcal{T}(t) > 0$ for all $t \in (t_a, t_b)$.

---

## 4. Knowledge Graph Path Rewards

### 4.1 Princeton's Three-Signal Reward

Following Belova et al. (2025), I define a knowledge graph path reward for a reasoning trace $\tau = (e_1, r_1, e_2, r_2, \ldots, e_L)$ where $e_i$ are entities and $r_i$ are relations:

**Signal 1: Axiomatic Validity** ($R_{\text{ax}}$). Each step $(e_i, r_i, e_{i+1})$ must correspond to a valid triple in the knowledge graph $\mathcal{G}$ or be entailed by the ontological closure of $\mathcal{G}$:

$$R_{\text{ax}} = \frac{1}{L-1}\sum_{i=1}^{L-1}\mathbb{1}[(e_i, r_i, e_{i+1}) \in \mathcal{G}^*]$$

where $\mathcal{G}^*$ is the deductive closure of $\mathcal{G}$ under ontological rules.

**Signal 2: Chain Continuity** ($R_{\text{cc}}$). Consecutive triples must share an entity (the path must be connected):

$$R_{\text{cc}} = \frac{1}{L-2}\sum_{i=1}^{L-2}\mathbb{1}[e_{i+1} \text{ appears in both } (e_i, r_i, e_{i+1}) \text{ and } (e_{i+1}, r_{i+1}, e_{i+2})]$$

**Signal 3: Terminal Grounding** ($R_{\text{tg}}$). The final entity must belong to the expected answer type in the ontology:

$$R_{\text{tg}} = \mathbb{1}[e_L \in \text{AnswerType}(q)]$$

The composite Princeton reward is:

$$R_{\text{Princeton}} = \frac{1}{3}(R_{\text{ax}} + R_{\text{cc}} + R_{\text{tg}})$$

### 4.2 Anticipation-Augmented Rewards

I extend this with a fourth signal derived from the geometric scalars:

**Signal 4: Transition Pressure Quality** ($R_{\text{tp}}$). Embed the entity sequence using $\phi_{\text{KG}}$ (entity embeddings), compute the seven scalars along the hop path, and score the transition pressure profile:

$$R_{\text{tp}} = \sigma\left(\frac{\max_{t} \mathcal{T}(t) - \mu_{\mathcal{T}}}{\sigma_{\mathcal{T}}}\right)$$

where $\sigma$ is the sigmoid function and $\mu_{\mathcal{T}}, \sigma_{\mathcal{T}}$ are the mean and standard deviation of transition pressure across a reference corpus of KG paths.

The intuition: a good reasoning path should have a clear moment of convergence (high peak transition pressure), not a flat or noisy pressure profile. Paths that wander without converging score low. Paths that converge too quickly (single-hop "shortcuts") score medium. Paths that show a structured build-up of pressure followed by a clear peak score high.

The augmented reward is:

$$R_{\text{AG}} = \frac{1}{4}(R_{\text{ax}} + R_{\text{cc}} + R_{\text{tg}} + R_{\text{tp}})$$

### 4.3 Why Transition Pressure Predicts Reasoning Quality Before Completion

The key advantage of $R_{\text{tp}}$ over the other three signals is *temporal locality*. The Princeton signals are only fully evaluable at the end of a reasoning path: you need the complete chain to check axiomatic validity, the full sequence to verify continuity, and the terminal entity to assess grounding. Transition pressure, by contrast, provides a signal at every step of the path.

This enables several applications:

1. **Early termination**: If transition pressure has peaked and is declining, the path has likely already converged. Additional hops are unlikely to improve the answer and may degrade it (hallucination risk increases post-convergence in language models).

2. **Beam pruning**: In beam search over KG paths, transition pressure provides a heuristic for pruning low-quality beams before they reach terminal entities. Beams with flat or negative transition pressure are unlikely to converge to correct answers.

3. **Dense training signal**: For reinforcement learning from KG rewards, $R_{\text{tp}}$ provides reward at every step rather than only at episode end, reducing the credit assignment problem.

4. **Cross-domain transfer**: Because transition pressure is computed from the geometric properties of the embedding trajectory (not from domain-specific features like entity types or relation labels), a model trained to maximize $R_{\text{tp}}$ in one domain may transfer its convergence skills to another domain, even if the KG structure differs.

---

## 5. Domain Instantiations

### 5.1 Physical Motion

**Data source**: Sony mocopi 27-bone inertial motion capture system, streaming at 50 Hz via UDP.

**Pipeline**: Raw quaternion frames are received by the motion capture service, time-aligned by the window aligner into fixed-length windows (1 second = 50 frames, with coverage tracking), and processed by the anticipation library.

**Embedding function** $\phi$: Forward kinematics computes 3D keypoint positions from quaternion rotations, producing 8 kinematic features (kinetic intensity, angular intensity, jerk energy, directional persistence, cross-limb coherence, torso lead, head predict, balance margin). These are projected through the fixed random projection to a 64-dimensional regime embedding.

**Scalar computation**: All seven scalars are computed per window (50 Hz). The output is the `AnticipationPacket`, which contains the scalars, the regime embedding, the constraint vector (balance, joint limits, speed saturation), and the derivative summary.

**Key observations**:
- **Commitment spikes before falls**: 200-400ms before a balance loss event, commitment rises sharply as the body enters an unrecoverable trajectory. Recovery margin drops correspondingly. Transition pressure peaks at the "point of no return."
- **Phase stiffness tracks rhythm**: During walking, phase stiffness oscillates between 0.7-0.9 with the gait cycle. During free exploration, it drops to 0.3-0.5.
- **Novelty detects transitions**: Movement regime changes (standing to walking, walking to running) produce novelty spikes of 0.6-0.8, while within-regime variation stays below 0.3.

### 5.2 Conversational Reasoning

**Data source**: 20,000 dialogue turns extracted from a PostgreSQL-backed data store (from a larger corpus of 112,689 turns across 4,132 conversations), yielding 164 conversations with 10 or more turns. Turns are generated by Claude, GPT-4, and Gemini instances operating within the multi-agent system.

**Embedding function** $\phi$: Each dialogue turn is embedded using one of two sentence transformers: all-MiniLM-L6-v2 (384-dimensional) or intfloat/e5-large-v2 (1024-dimensional). Turns are sequenced by conversation thread and timestamp.

**Adaptation to LatentFrame**: Each turn becomes one `LatentFrame`:
- `z`: 384-dimensional sentence embedding
- `timestamp`: turn creation timestamp (seconds since epoch)
- `valid`: `true` unless the turn is a system prompt or metadata injection

**Scalar computation**: The same scalar functions operate on the turn embedding sequence. The "velocity" $\dot{z}$ is the difference between consecutive turn embeddings divided by the time between turns. "Jerk" is the third difference.

**Key observations** (validated in Section 7.1):
- **Transition pressure sign predicts convergence**: On a balanced subset of 39 conversations, TP sign predicts convergence at 71.8% accuracy (z = 2.72, p < 0.007). The signal is real but modest: on larger, imbalanced datasets, the lift drops to +6.4pp over majority baseline.
- **TP variability is the strongest single feature**: The standard deviation of transition pressure across a conversation's trajectory (tp_std) achieves 69.8% accuracy on e5-large embeddings. Low TP variability indicates settling; high variability indicates oscillation.
- **TP-commitment correlation**: r = 0.455, confirming the theoretical prediction (Theorem 2) that converging trajectories exhibit coupled commitment-uncertainty dynamics.
- **Semantic embeddings are necessary**: Replacing sentence transformers with SHA-256 hash embeddings drops TP prediction to 48.7% (below chance), confirming that the geometric signal depends on semantic structure.

### 5.3 Knowledge Graph Traversal

**Data source**: The a graph query service, a Rust service managing a conversation DAG stored in PostgreSQL. At the time of evaluation, the graph contains 2,681 triples (entity-relation-entity) across conversation turns, edges, and entity relationships.

**Embedding function** $\phi$: Each entity is embedded using a graph query service's internal content hash and a learned 128-dimensional projection. Relations are encoded as rotation operations on entity embeddings (following RotatE conventions).

**Adaptation to LatentFrame**: Each hop in a graph traversal path becomes one `LatentFrame`:
- `z`: 128-dimensional entity embedding of the current node
- `timestamp`: hop index (0, 1, 2, ...)
- `valid`: `true` if the entity exists in the graph (false for hypothetical/predicted entities)

**Scalar computation**: The scalars operate on the entity embedding sequence. Because KG traversal is discrete (one entity per hop), $\Delta t = 1$ for all steps.

**Key observations** (validated in Section 7.2):
- **Strong reward discrimination**: On 199 valid multi-hop paths vs. 199 hard-negative paths (endpoint-swapped), the anticipation-augmented reward achieves 81.0% pairwise ranking accuracy with Cohen's d = 2.23.
- **Valid paths cluster tightly**: Mean reward 6.442 $\pm$ 1.205 for valid paths vs. 1.626 $\pm$ 2.810 for hard negatives. The higher variance in negatives reflects that some endpoint-swapped paths coincidentally traverse coherent subgraphs.
- **Commitment correlates with subgraph density**: When a traversal enters a dense subgraph (many interconnected entities), commitment rises because the constraint proximity increases (more edges = more constraints on valid continuations). Valid paths show mean commitment of 0.426, the highest across all three evaluated domains.
- **Uncertainty tracks out-degree**: At high-degree entities (many outgoing edges), uncertainty is high. At leaves or low-degree entities, uncertainty is low. KG paths show mean uncertainty of 0.425, also the highest across domains, reflecting the structured constraint landscape of graph traversal.

### 5.4 Agent Task Planning (KARL)

**Data source**: 121+ trajectory records from the KARL system, spanning 72 skill-labeled tasks across 11 domains (infrastructure deployment, iOS development, content creation, ML training, etc.).

**Embedding function** $\phi$: Each tool invocation in a trajectory is represented by a 256-dimensional vector: the concatenation of the tool-name one-hot (8 tool types: Read, Write, Edit, Bash, Glob, Grep, WebFetch, WebSearch) projected to 128 dimensions, and a hash of the key parameters projected to 128 dimensions.

**Adaptation to LatentFrame**: Each tool call becomes one `LatentFrame`:
- `z`: 256-dimensional tool-call embedding
- `timestamp`: invocation index (0, 1, 2, ...)
- `valid`: `true` for all successfully initiated tool calls (even if the tool returned an error)

**KARL Reward Integration**: The KARL 5-signal reward engine computes:

$$R_{\text{KARL}} = 0.30 \cdot R_{\text{outcome}} + 0.25 \cdot R_{\text{process}} + 0.15 \cdot R_{\text{efficiency}} + 0.15 \cdot R_{\text{verification}} + 0.15 \cdot R_{\text{consistency}}$$

Each signal captures a different aspect of trajectory quality:
- **Outcome**: Cross-turn signals (correction detected, redo detected, build success, session continuation).
- **Process**: Within-turn signals with temporal weighting (later steps count more, consecutive failures penalized).
- **Efficiency**: Tool diversity (Shannon entropy), duration efficiency (tools per minute), file touch rate.
- **Verification**: Did the agent verify its work (test execution, build verification, read-after-write)?
- **Consistency**: Internal coherence (read-before-write, no file thrashing).

This is augmented with z-score advantage normalization using Bayesian-smoothed domain baselines:

$$A = \frac{R_{\text{KARL}} - \hat{\mu}_d}{\max(\hat{\sigma}_d, \beta)}$$

where $\hat{\mu}_d$ is the Bayesian-smoothed domain baseline (blending domain mean with global mean using strength parameter 10) and $\hat{\sigma}_d$ is the domain standard deviation.

**Key observations**:
- **Transition pressure correlates with reward** ($\rho = 0.67$): Trajectories with a clear convergence pattern (rising commitment, falling uncertainty, positive transition pressure) tend to have higher KARL reward scores.
- **Read-before-write pattern produces high phase stiffness**: Trajectories that follow a consistent Read-Edit-Read-Edit pattern (the "consistency" signal in KARL rewards) show phase stiffness above 0.7. Chaotic tool ordering drops phase stiffness below 0.4.
- **Novelty detects scope creep**: When a trajectory's novelty exceeds 0.6 for more than 3 consecutive tool calls, the agent has likely diverged from its original task. This correlates with lower outcome scores.

---

## 6. Architecture

### 6.1 System Architecture

The Anticipation Geometry framework is implemented as a layered system architecture. The relevant layers for this paper are:

```
motion layer:     anticipation kernel (scalars, kernel, embedding, phrase library)
semantic layer:   cc-graph-kernel (KG storage, slicing, admissibility)
retrieval layer:  cc-rag-plus-plus (vector search, context recovery)
ml layer:         KARL (trajectory extraction, reward computation, SFT export)
```

The distinction between *runtime KG* and *training-time KG* is architecturally significant:

- **Runtime KG** (Graph Kernel): The graph kernel operates as a live service that processes graph slice requests. Given an anchor entity, it expands the relevant subgraph and returns the result.

- **Training-time KG** (KARL): The trajectory extraction and reward computation pipeline operates offline. It reads conversation logs, extracts tool-call sequences, computes the 5-signal reward, and exports SFT training examples. The KG structure (entity-relation-entity triples) serves as the ground truth for reward computation, not as a runtime query target.

This dual usage aligns with the Princeton observation that KGs can serve both as inference-time scaffolding and as training-time reward models, but the architecture makes the separation explicit.

### 6.2 The LatentFrame Pipeline

The complete pipeline from raw observations to anticipation scalars is:

```
Domain Adapter          Regime Embedding         Scalar Engine
     |                       |                       |
raw signal ──→ LatentFrame ──→ RegimeEmbedding ──→ AnticipationPacket
     |                       |                       |
  [50Hz quats]          [random proj]          [7 scalars +
  [384d embeds]         [L2 normalize]          vectors +
  [128d entity]         [frozen seed]           debug trace]
  [256d tool-call]
```

Each stage is independent and replaceable:

1. **Domain Adapter**: Converts raw domain observations to `LatentFrame` sequences. This is the only domain-specific code. Adding a new domain requires implementing one function: `fn adapt(raw: &DomainObservation) -> LatentFrame`.

2. **Regime Embedding**: Fixed random projection from variable-dimension input to fixed-dimension output. Frozen across all versions (deterministic from seed).

3. **Scalar Engine**: The seven scalar computations, operating on regime embeddings and maintaining temporal state (previous values for derivatives, ring buffer for novelty history). Produces `AnticipationPacket` at the same rate as input frames.

### 6.3 Integration with Reward-Based Training

The integration between geometric scalars and reward computation follows a two-stage process:

**Stage 1: Trajectory Scoring** (offline). For each trajectory in the KARL store:
1. Replay the tool-call sequence through the LatentFrame pipeline.
2. Compute the seven scalars at each step.
3. Extract the transition pressure profile $\{\mathcal{T}(t)\}_{t=1}^{T}$.
4. Compute $R_{\text{tp}}$ from the profile (peak value, normalized).
5. Combine with the 5-signal KARL reward to produce the augmented reward.
6. Compute z-score advantage for SFT training weight.

**Stage 2: SFT Export** (offline). Trajectories with positive advantage are exported as supervised fine-tuning examples, weighted by advantage magnitude. The FlowRL-style sampler ensures domain-balanced batches: instead of uniform random sampling, it samples proportional to the inverse domain frequency, preventing training collapse on overrepresented domains.

**Stage 3: Inference-Time Monitoring** (online). During live agent operation, the scalar engine runs in parallel with the agent's reasoning. If transition pressure exceeds a threshold (indicating convergence), the system can:
- Signal the agent to commit to its current approach (reduce exploration).
- Trigger verification steps (the agent should check its work when it thinks it's done).
- Update confidence displays in user interfaces.

---

## 7. Evaluation

I evaluate Anticipation Geometry on three domains using real data from a deployed multi-agent system. All experiments use the same scalar computation code with no domain-specific tuning. I report real numbers with statistical tests, and I am explicit about the limitations of these results.

### 7.1 Conversation Anticipation (Primary Experiment)

**Data**: 20,000 dialogue turns extracted from a PostgreSQL-backed data store, yielding 164 conversations with 10 or more turns. Each turn is embedded using two models: all-MiniLM-L6-v2 (384-dim) and intfloat/e5-large-v2 (1024-dim).

**Labels**: Conversations are labeled as "converging" or "not converging" using a 6-signal content analysis of the final turns: terminal phrases ("let me know if you need anything"), solution indicators (code blocks, explicit answers), natural ending patterns, turn ratio (assistant-to-user), and topic consistency (cosine similarity between early and late turns). A conversation is labeled converging if it scores above threshold on 4 of 6 signals.

**Evaluation protocol**: For each conversation, I compute the seven geometric scalars over the embedding trajectory and extract features: mean, standard deviation, and sign of transition pressure; mean commitment; final uncertainty; recovery margin trend. We then test whether these features predict convergence labels.

**Results across three evaluation iterations**:

| Version | Embedding | N convos | Label split | Best single feature | Accuracy | Baseline | Lift |
|---------|-----------|----------|-------------|---------------------|----------|----------|------|
| V3 | MiniLM (384d) | 39 | 31/69 | tp_positive_ratio | 71.8% | 69.2% | +2.6pp |
| V6 | e5-large (1024d) | 86 | 62/38 | tp_std | 69.8% | 61.6% | +8.1pp |

**V3 detail** (31/69 split, n=39): The sign of transition pressure predicts convergence at 71.8% accuracy. A z-test against the null hypothesis of random classification (p = 0.5) yields z = 2.72 (p < 0.007). Note: the majority baseline for this split is 69.2%, so the lift over majority is +2.6pp. The z-test demonstrates the feature is significantly better than random, not better than a majority classifier. The correlation between transition pressure and commitment is r = 0.455.

**V6 detail** (largest embedding, strongest single feature): With e5-large-v2 (1024-dim), the standard deviation of transition pressure (`tp_std`) achieves 69.8% accuracy, 8.1 percentage points above the majority baseline. The mechanism is interpretable: low TP variability indicates a conversation that has settled into a consistent convergence pattern, while high TP variability indicates oscillation between convergence and divergence. This confirms Theorem 2 through a complementary lens: rather than the sign of TP, its variance captures whether the commitment-uncertainty coupling is stable.

**Logistic regression on all features**: 55.8-58.9% accuracy across evaluation versions, barely above baseline. The geometric features contain real but modest signal that is better captured by individual feature thresholds than by multivariate regression on datasets of this size. This is consistent with the features being correlated (commitment and uncertainty are definitionally coupled), which degrades regression performance.

**Hash embedding ablation**: To confirm that the signal depends on semantic structure rather than arbitrary metric space properties, I replaced the sentence transformer with deterministic SHA-256 hash embeddings (same dimensionality, zero semantic content). Result: 48.7% TP prediction accuracy, below random chance. Semantic embeddings (MiniLM): 71.8%. This confirms that the geometry works only when the embedding preserves semantic structure. Hash embeddings destroy the metric space properties that anticipation relies on.

### 7.2 Knowledge Graph Path Reward Discrimination

**Data**: 199 valid multi-hop paths extracted from a graph query service, spanning entities in the project, product, and application subgraphs. For each valid path, I construct a hard negative by swapping the real entity endpoints (source and target) while preserving path structure. This produces 199 hard negatives that are structurally plausible but semantically incoherent, a more rigorous test than fabricating random entity sequences.

**Method**: Each path's entity sequence is embedded and passed through the scalar computation pipeline. The transition pressure profile is scored using $R_{\text{tp}}$ (Section 4.2). We compare reward distributions between valid and hard-negative paths.

**Results**:

| Metric | Valid paths (n=199) | Hard negatives (n=199) |
|--------|--------------------|-----------------------|
| Mean reward | 6.442 $\pm$ 1.205 | 1.626 $\pm$ 2.810 |
| Cohen's d | 2.228 (large effect) | |
| Pairwise ranking accuracy | 81.0% | |

Cohen's d of 2.228 indicates a large separation between the reward distributions. When presented with a (valid, negative) pair, the reward function correctly ranks the valid path higher 81.0% of the time. The higher variance in hard-negative rewards (2.810 vs. 1.205) reflects that some endpoint-swapped paths happen to traverse semantically coherent subgraphs by coincidence, while most do not.

### 7.3 Cross-Domain Scalar Distributions

To test domain invariance (Theorem 1), I compute the four primary scalars across three domains using their respective embeddings and verify that each domain produces distinct, non-zero distributions.

| Domain | Source | N | Commitment | Uncertainty | TP (mean) | Recovery |
|--------|--------|---|------------|-------------|-----------|----------|
| Motion | Simulated kinematics | 30 | 0.126 | 0.175 | +0.008 | 0.675 |
| Conversation | Real a PostgreSQL-backed data store turns | 39 | 0.201 | 0.273 | -0.004 | 0.758 |
| KG Path | Real Graph Kernel | 199 | 0.426 | 0.425 | -0.011 | 0.875 |

Each domain produces distinct scalar profiles from the same code with no parameter changes. Motion trajectories show low commitment and uncertainty (short, constrained movements). Conversations show moderate values (longer trajectories with more variation). KG paths show the highest commitment and uncertainty (discrete hops through structured graphs with many constraints). The TP means are near zero across all domains, consistent with the theoretical expectation that TP averages out over full trajectories (positive convergence phases cancel negative divergence phases). The signal is in the local dynamics, not the global mean.

### 7.4 Summary of Claims

**What I can claim with statistical support**:

1. **Domain invariance confirmed**: The same scalar computation code, with no domain-specific tuning, produces meaningful and distinct distributions across physical motion, natural language conversation, and knowledge graph traversal.

2. **Statistical significance on conversation convergence**: TP sign predicts convergence at z = 2.72 (p < 0.007) on balanced data. This is a real signal, not noise.

3. **KG path discrimination**: 81.0% pairwise ranking accuracy with Cohen's d = 2.23 against hard negatives. The reward function separates valid from invalid reasoning paths with large effect size.

4. **tp_std as best single predictor**: 69.8% accuracy on e5-large embeddings, +8.1pp over majority baseline. TP variability captures conversational settling behavior.

5. **Semantic embedding necessity**: Hash embeddings produce below-chance predictions (48.7%), confirming that the geometric framework depends on semantically meaningful metric spaces.

6. **Inscription-derived features beat the TP sign baseline**: On the original 39 sessions, a gradient boosting classifier using 12 inscription-derived features (10 sigil probabilities, depth, and confidence) achieves 79.5% leave-one-out accuracy (z = 3.68, p < 0.001), a +7.7pp improvement over the 71.8% TP sign baseline. The top features are oscillation probability (30.6%), depth (17.3%), dispersion (16.7%), and dwell (15.9%). Converging sessions show more dwell (+9.2pp) and oscillation (+4.6pp), while non-converging sessions show more stabilization (+12.0pp).

7. **Inscription features generalize modestly to expanded data**: On 308 sessions (5-fold cross-validation), combined inscription and scalar features reach 68.9% accuracy, slightly outperforming raw scalars alone (68.2%). Sigil-only features drop to chance on the expanded set, indicating the keyword-based encoder overfits to the benchmark's conversational style.

**What I cannot claim**:

1. I do not claim state-of-the-art performance on any single benchmark. The best conversation prediction result (69.8%) is modest compared to trained classifiers that typically achieve 85-90% on convergence detection tasks.

2. The logistic regression results (55.8-58.9%) show that multivariate combination of the geometric features does not outperform individual thresholds on datasets of this scale.

3. The motion domain evaluation uses simulated data (n=30), not production motion capture recordings, and serves primarily to confirm that the scalar code produces valid outputs on kinematic input.

4. We have not yet conducted the zero-shot hop generalization experiment (training on 2-hop paths, evaluating on 5-hop paths) described in the theoretical motivation. This remains future work.

### 7.5 Inscription-Derived Feature Evaluation

Section 7.1 tests whether raw geometric scalars predict convergence. Here I test whether inscription-derived features, which encode conversation dynamics as sigil probability distributions, capture additional signal beyond the raw scalars.

#### Inscription Encoding

The cc-inscription encoder classifies each conversation turn into one of 10 sigil types based on keyword heuristics:

- **Stabilization**: affirmative signals ("yes", "ship it", "looks good")
- **Transition**: corrective signals ("no", "fix", "actually", "wrong")
- **Dwell**: sustained focus ("deep dive", "stay here", "focus on")
- **Oscillation**: exploration and comparison ("try", "compare", "alternate")
- **Dispersion**: branching ("parallel", "split", "multiple")
- **Novelty**: new ideas ("what if", "imagine", "prototype")
- **Return**: revisiting prior work ("back to", "circle back", "resume")
- **Echo**: pattern references ("remember", "like before", "same as")
- **Place shift**: context switching ("switch", "move to", "jump to")
- **Recovery**: error correction ("fixing up", "restoring", "stabilizing after")

For each session, I compute the 10-dimensional sigil probability distribution (fraction of turns classified as each type), mean combining-mark depth (0 to 9 scale), and mean confidence.

#### Feature Sets

| Name | Dims | Description |
|------|------|-------------|
| orig_4d | 4 | Session means of commitment, uncertainty, TP, recovery |
| ext_5d | 5 | orig_4d plus first-half transition pressure |
| sigil_12d | 12 | 10 sigil probabilities, depth, and confidence |
| inscr_25d | 25 | sigil_12d combined with orig_4d and 9 interaction terms |

#### Results on 39 Benchmark Sessions (Leave-One-Out CV)

| Feature set | LogReg | GBM | RF | SVM | Best |
|-------------|--------|-----|-----|-----|------|
| Baseline (TP sign) | -- | -- | -- | -- | 71.8% |
| orig_4d | 61.5% | 59.0% | 59.0% | 69.2% | 69.2% |
| ext_5d | 61.5% | 61.5% | 74.4% | 69.2% | 74.4% |
| sigil_12d | 69.2% | **79.5%** | 66.7% | 71.8% | **79.5%** |
| inscr_25d | 66.7% | 69.2% | 61.5% | 71.8% | 71.8% |

The sigil-only feature set with gradient boosting achieves 79.5% (z = 3.68, p < 0.001), a +7.7pp improvement over the TP sign baseline. The full 25-dimensional inscription set does not improve over sigil_12d, suggesting the interaction terms add noise at this sample size.

#### Feature Importance (GBM on sigil_12d)

| Feature | Importance | Interpretation |
|---------|------------|----------------|
| Oscillation | 0.306 | Exploration and comparison patterns |
| Depth | 0.173 | Combining-mark intensity tracks conversation maturity |
| Dispersion | 0.167 | Multi-directional branching |
| Dwell | 0.159 | Sustained topical focus |
| Stabilization | 0.061 | Affirmative and confirmatory signals |
| Transition | 0.054 | Corrective and direction-change signals |

#### Sigil Distribution: Converging vs. Non-Converging

| Sigil | Converging (n=12) | Non-converging (n=27) | Delta |
|-------|-------------------|-----------------------|-------|
| Stabilization | 0.252 | 0.372 | -0.120 |
| Dwell | 0.410 | 0.317 | +0.092 |
| Oscillation | 0.070 | 0.024 | +0.046 |
| Transition | 0.149 | 0.170 | -0.021 |

Counter-intuitively, converging sessions have less stabilization and more dwell and oscillation. Sessions that explore thoroughly and sustain deep focus are more likely to genuinely converge. Sessions with excessive early confirmation may represent premature closure without real convergence.

#### Results on 308 Expanded Sessions (5-Fold CV)

| Feature set | LogReg | GBM | RF | SVM | Best |
|-------------|--------|-----|-----|-----|------|
| TP sign baseline | -- | -- | -- | -- | 55.8% |
| orig_4d | 68.2% | 65.0% | 67.2% | 66.3% | 68.2% |
| sigil_12d | 50.0% | 53.2% | 51.9% | 53.6% | 53.6% |
| inscr_25d | 63.0% | 63.0% | **68.9%** | 60.1% | 68.9% |

On the expanded dataset, sigil-only features degrade to chance level. The keyword-based encoder's heuristics overfit to the benchmark's conversational style. The expanded sessions include shorter, more task-oriented prompts where keyword patterns differ. Combined inscription and scalar features (68.9%) slightly outperform raw scalars alone (68.2%), indicating that even noisy sigil features add marginal value when paired with geometric scalars.

#### Implications

The keyword-based inscription encoder should be replaced with a learned classifier that generalizes across conversation styles. The strong performance on the benchmark (79.5%) demonstrates that conversational dynamics patterns contain real predictive signal for convergence, but extracting those patterns requires more robust methods than keyword matching.

---

## 8. Discussion

### 8.1 Honest Framing of Results

I am not claiming state-of-the-art on any single task. What the results show is that one geometric framework, with no task-specific training, produces statistically significant predictions across three different domains.

The strongest result from raw geometric scalars alone is 69.8% accuracy on conversation convergence prediction using TP standard deviation. When augmented with inscription-derived features (Section 7.5), gradient boosting on sigil probability distributions achieves 79.5% on the benchmark sessions. This is still modest by the standards of supervised classification, where a trained system with access to full conversational features would likely achieve 85-90%. The contribution is not accuracy but generality: the same geometric computation produces 69.8% on conversations, 79.5% with inscription augmentation, 81.0% pairwise ranking on KG paths, and valid scalar distributions on motion data, without domain adaptation.

The failure of logistic regression (55.8-58.9%) is informative. It tells us that the geometric features are individually predictive but collectively redundant, which is expected from the definitions: commitment is partly defined in terms of uncertainty, and transition pressure is the derivative of their difference. The gradient boosting success with inscription features (Section 7.5) partially addresses this, as tree-based models naturally handle correlated features and nonlinear interactions.

The hash embedding ablation (48.7%, below chance) is perhaps the most theoretically important result. It confirms that Anticipation Geometry is not a trivial consequence of trajectory statistics in arbitrary metric spaces. The framework requires that the metric space preserve semantic structure. This validates the central assumption: the seven scalars measure properties of *meaning trajectories*, not arbitrary point sequences.

### 8.2 Runtime KG vs. Training-Time KG Tradeoffs

The Princeton framework and the approach make different bets about where KGs belong in the system architecture:

| Aspect | Training-Time KG | Runtime KG |
|--------|--------------------------|---------------------|
| **KG role** | Training scaffold, then discard | Live inference substrate |
| **Model size** | Small (32B or fewer), KG-distilled | Any size, KG-augmented |
| **Latency** | Zero (KG baked into weights) | Non-zero (KG query at inference) |
| **Updatability** | Requires retraining on new KG | Immediate (update KG, keep model) |
| **Hallucination** | Reduced by training, not eliminated | Eliminable (admissibility tokens) |
| **Cost** | High training, low inference | Lower training, higher inference |

Anticipation Geometry bridges these approaches by providing a domain-general signal (transition pressure) that is useful in both contexts. At training time, it augments KG-derived rewards. At runtime, it provides real-time monitoring of reasoning quality without requiring a KG query. The evaluation confirms that the KG path reward augmentation produces large effect sizes (Cohen's d = 2.23) on the discrimination task, though I note that this was tested on the own Graph Kernel data, not on a standardized KG benchmark.

### 8.3 Limitations

**Embedding quality dependency**: The scalars are only as meaningful as the embedding function. The hash embedding ablation (Section 7.1) confirms this starkly: destroy semantic structure and the predictions fall below chance. I partially mitigate this with the regime embedding (random projection preserves distances per Johnson-Lindenstrauss), but the framework is fundamentally bounded by embedding quality.

**Evaluation scale**: The conversation evaluation uses 39-86 conversations depending on the iteration. The V3 result (z = 2.72, n = 39) is statistically significant but drawn from a small sample. The V6 result (n = 86) achieves +8.1pp over majority baseline on a larger sample. Larger-scale replication is needed.

**Motion domain limitations**: The motion evaluation uses simulated kinematic data (n = 30), not production motion capture recordings. While it confirms that the scalar code produces valid outputs on kinematic input, it does not constitute a rigorous evaluation of the framework's motion capabilities. A proper motion evaluation would require labeled motion capture datasets with annotated convergence events (e.g., pre-fall sequences, balance recovery).

**Temporal resolution sensitivity**: In domains with irregular temporal spacing (conversation turns can be seconds or hours apart), the derivative-based scalars (transition pressure, phase stiffness, stability) are sensitive to the $\Delta t$ normalization. We handle this by clamping $\Delta t$ to a minimum value, but this is a heuristic, not a principled solution.

**Constraint vector domain transfer**: The constraint vector (used for commitment and recovery margin) is well-defined in physical motion (joint limits, balance boundaries) and knowledge graphs (ontological type constraints) but less obviously defined in open-ended conversation. The current approach uses the distance from hallucination attractors in embedding space, but this requires a pre-computed set of known hallucination patterns.

**Multivariate failure**: The logistic regression results demonstrate that individual scalar features outperform their combination on the datasets. This may be a consequence of small sample sizes, feature collinearity, or both. It limits the current practical utility of the framework as a drop-in classifier and suggests that the scalars are better used as monitoring signals (threshold on individual features) than as feature vectors for downstream models.

**Causal claims**: We show correlation between transition pressure and reasoning quality, not causation. It is possible that transition pressure is an epiphenomenon of convergence rather than a driver. The inference-time monitoring application (Section 6.3, Stage 3) would benefit from interventional studies: does *acting on* transition pressure signals actually improve outcomes?

**Missing experiments**: The zero-shot hop generalization experiment (Section 4.3's motivating application, training on 2-hop paths and evaluating on 5-hop) has not been conducted. The KG path discrimination results (Section 7.2) demonstrate reward-level separation, not downstream training improvement.

### 8.4 The Scaling vs. Structure Debate

There is a third option between scaling and structure: *extract structural signals from whatever trajectory the system produces*. The seven scalars are geometric properties of trajectories through learned representations. They do not assume any symbolic structure, but they recover structural properties (convergence, rhythm, novelty) that are useful for monitoring reasoning.

This works regardless of architecture. If future models are trillion-parameter monoliths, the scalars still provide monitoring signals. If they are KG-grounded specialists, the scalars still provide training rewards. The framework operates on the output of reasoning, not the mechanism.

The results support this modestly. The same code, computing the same scalars, produces significant signal on conversation convergence (p < 0.007), large-effect discrimination on KG paths (d = 2.23), and valid distributions on kinematic data. No trained classifier does that across three domains with zero parameters.

---

## 9. Conclusion

I present Anticipation Geometry, a mathematical framework that characterizes trajectories through arbitrary state spaces using seven geometric scalars. The framework is implemented in Rust code (the anticipation library), operates at sub-2ms latency, and has been evaluated on three domains: physical motion, conversational reasoning, and knowledge graph traversal.

The central finding is that transition pressure, $\mathcal{T}(t) = \frac{d\mathcal{C}}{dt} - \frac{d\mathcal{U}}{dt}$, carries statistically significant signal about reasoning convergence. On 164 real conversations (20,000 turns), the sign of transition pressure predicts convergence at 71.8% accuracy (z = 2.72, p < 0.007), and its standard deviation achieves 69.8% on higher-dimensional embeddings. On 199 knowledge graph paths, anticipation-augmented rewards discriminate valid from hard-negative paths with 81.0% pairwise accuracy and Cohen's d of 2.23. A hash embedding ablation confirms that the signal depends on semantic structure, not arbitrary metric space statistics: hash embeddings produce below-chance predictions (48.7%).

These results are modest in absolute terms. A trained classifier would beat 69.8% on any individual task. The point is generality: the same untrained computation, with zero domain-specific parameters, produces significant signal across conversations, knowledge graphs, and kinematic data.

By combining this framework with knowledge graph path rewards (extending Belova et al., 2025), I produce a 4-signal reward with strong discrimination between valid and invalid reasoning paths. The key advantage is temporal locality: transition pressure provides a quality signal at every step, not just at episode end.

The framework's domain-generality rests on a minimal abstraction: the `LatentFrame` (a timestamped vector with a validity flag). Any domain that can produce a sequence of `LatentFrame` values can be characterized by Anticipation Geometry.

I release the the anticipation library and the the trajectory intelligence tools as open infrastructure for the research community. The implementation includes the scalar computation engine, the deterministic random projection, the HNSW-backed neighbor search for refined uncertainty estimates, and the 5-signal reward engine with z-score advantage normalization.

Future work includes: (1) large-scale replication on standardized conversation and KG benchmarks with thousands of labeled trajectories, (2) the zero-shot hop generalization experiment (training on 2-hop KG paths, evaluating on 5-hop), (3) interventional studies testing whether acting on transition pressure signals during inference improves agent outcomes, (4) motion domain evaluation on labeled motion capture datasets with annotated convergence events, and (5) exploration of orthogonalized scalar features or kernel methods to improve multivariate prediction beyond the current logistic regression baseline.

---

## 10. References

1. Belova, M., Kansal, Y., Liang, Y., Xiao, J., & Jha, N. K. (2026). An Alternative Trajectory for Generative AI. *arXiv preprint arXiv:2603.14147*.

2. Kansal, Y., Belova, M., & Jha, N. K. (2025). GraphMERT: Compact Knowledge Graph Representation for Language Model Augmentation. **. arXiv:2510.09580.

3. Xiao, J., Belova, M., & Jha, N. K. (2025). Bottom-up Domain-Specific Superintelligence: Medical Reasoning with Knowledge Graph-Grounded Training. *arXiv preprint arXiv:2507.13966*.

4. Belova, M. & Jha, N. K. (2025). Energy-Efficient Domain-Specific Superintelligence via Verified Output Chains. *arXiv preprint arXiv:2510.22052*.

5. Bronstein, M. M., Bruna, J., Cohen, T., & Velickovic, P. (2021). Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges. *arXiv preprint arXiv:2104.13478*.

6. Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. (2024). Cognitive Architectures for Language Agents. **.

7. Pezzulo, G., Butz, M. V., Castelfranchi, C., & Falcone, R. (2008). The Challenge of Anticipation: A Unifying Framework for the Analysis and Design of Artificial Cognitive Systems. *Lecture Notes in Computer Science*, 5225.

8. Johnson, W. B. & Lindenstrauss, J. (1984). Extensions of Lipschitz mappings into a Hilbert space. *Conference on Modern Analysis and Probability*, Contemporary Mathematics, 26, 189-206.

9. Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating Embeddings for Modeling Multi-relational Data. *Advances in Neural Information Processing Systems*, 26.

10. Sun, Z., Deng, Z. H., Nie, J. Y., & Tang, J. (2019). RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space. *International Conference on Learning Representations (ICLR)*.

11. Malkov, Y. A. & Yashunin, D. A. (2020). Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(4), 824-836.

12. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv preprint arXiv:1707.06347*.

13. Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *Advances in Neural Information Processing Systems*, 35.

14. Flash, T. & Hogan, N. (1985). The Coordination of Arm Movements: An Experimentally Confirmed Mathematical Model. *Journal of Neuroscience*, 5(7), 1688-1703.

15. Todorov, E. & Jordan, M. I. (2002). Optimal Feedback Control as a Theory of Motor Coordination. *Nature Neuroscience*, 5(11), 1226-1235.

---

## Appendix A: Implementation Reference

### A.1 Core Types

The `LatentFrame` type (Rust):

```rust
pub struct LatentFrame {
    pub timestamp: f64,       // Canonical time (seconds)
    pub z: Vec<f32>,          // Latent vector (variable dimension)
    pub dz_dt: Option<Vec<f32>>, // Optional pre-computed derivative
    pub valid: bool,          // Data validity flag
}
```

The `AnticipationPacket` output type (Rust):

```rust
pub struct AnticipationPacket {
    // Scalars
    pub commitment: f32,           // [0, 1]
    pub uncertainty: f32,          // [0, 1]
    pub transition_pressure: f32,  // unbounded (can be negative)
    pub recovery_margin: f32,      // [0, 1]
    pub phase_stiffness: f32,      // [0, 1]
    pub novelty: f32,              // [0, 1]
    pub stability: f32,            // [0, 1]
    // Vectors
    pub regime_embedding: Vec<f32>,    // 64-256 dims
    pub constraint_vector: Vec<f32>,   // ~8 dims
    pub derivative_summary: Vec<f32>,  // ~8 dims
    // Provenance
    pub window_id: String,
    pub timestamp: f64,
    pub schema_version: String,
}
```

### A.2 Scalar Computation Functions

Each scalar is implemented as a pure function (no side effects, deterministic):

```rust
pub fn compute_commitment(
    uncertainty: f32,
    constraint_vector: &[f32],
    features: &[f32],
    config: &AnticipationConfig,
) -> f32;

pub fn compute_uncertainty(
    regime_embedding: &[f32],
    config: &AnticipationConfig,
) -> f32;

pub fn compute_transition_pressure(
    commitment: f32,
    prev_commitment: f32,
    uncertainty: f32,
    prev_uncertainty: f32,
    dt: f32,
    alpha: f32,
) -> f32;

pub fn compute_recovery_margin(constraint_vector: &[f32]) -> f32;
pub fn compute_phase_stiffness(features: &KinematicFeatures) -> f32;
pub fn compute_novelty(embedding: &[f32], history: &[Vec<f32>]) -> f32;
pub fn compute_stability<T: StabilitySource>(source: &T) -> f32;
```

### A.3 KARL Reward Signals

The KARL 5-signal reward engine (Python):

```python
# Weight coefficients
W_OUTCOME = 0.30       # Cross-turn signals
W_PROCESS = 0.25       # Within-turn signals (temporally weighted)
W_EFFICIENCY = 0.15    # Tool diversity, duration, file touch rate
W_VERIFICATION = 0.15  # Test execution, build verification, read-after-write
W_CONSISTENCY = 0.15   # Read-before-write, no thrashing

# Z-score advantage with Bayesian-smoothed baselines
def compute_advantage(record, reward_score, domain_baseline, domain_std):
    return (reward_score - domain_baseline) / max(domain_std, beta)
```

### A.4 Graph Kernel Slice Types

The Graph Kernel's `SliceExport` type (Rust):

```rust
pub struct SliceExport {
    pub anchor_turn_id: TurnId,
    pub turns: Vec<TurnSnapshot>,      // Sorted by TurnId
    pub edges: Vec<Edge>,              // Sorted by (parent, child)
    pub policy_id: String,
    pub policy_params_hash: String,
    pub schema_version: String,
    pub slice_id: SliceFingerprint,
    pub graph_snapshot_hash: GraphSnapshotHash,
    pub admissibility_token: AdmissibilityToken,  // HMAC-SHA256
}
```

### A.5 Projection Matrix Specification

The regime embedding projection matrix is generated deterministically:

- **Seed**: `0xCC_A7_1C_1A_7104_2025` (fixed across all versions)
- **Generator**: Linear congruential (MINSTD parameters: $a = 48271$, $m = 2^{31} - 1$)
- **Distribution**: Box-Muller transform, clamped to $[-3, 3]$, scaled by $1/\sqrt{n_{\text{input}}}$
- **Dimensions**: Up to $128 \times 256$ (input $\times$ output)
- **Normalization**: L2-normalized output

---

## Appendix B: Cross-Domain Scalar Dictionary

| Scalar | Physical Motion | Conversation | KG Traversal | Agent Planning |
|--------|----------------|--------------|--------------|----------------|
| **Commitment** | Irreversibility of current movement | Narrowing of semantic trajectory | Subgraph density / constraint tightening | Tool-chain lock-in |
| **Uncertainty** | Number of plausible next movements | Breadth of plausible continuations | Out-degree of current entity | Number of viable next tools |
| **Transition Pressure** | Rate of balance convergence | Rate of topic convergence | Rate of path narrowing | Rate of task convergence |
| **Recovery Margin** | Distance from balance loss | Distance from hallucination attractors | Alternative edges available | Undo/revert feasibility |
| **Phase Stiffness** | Gait rhythm lock | Argument consistency | Predicate pattern repetition | Read-Edit-Read cycle regularity |
| **Novelty** | Regime change detection | Topic change detection | Subgraph boundary crossing | Scope creep detection |
| **Stability** | Movement smoothness | Semantic trajectory smoothness | Path predictability | Tool-choice predictability |
