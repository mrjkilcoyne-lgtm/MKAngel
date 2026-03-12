# Block Prophet Model (BPM) -- Design Document

**Author:** Matt Kilcoyne
**Date:** 2026-03-12
**Status:** Speculative / Pre-Validation
**Parent Project:** MKAngel GLM

---

## 1. Executive Summary

The Block Prophet Model (BPM) is a proposed extension of MKAngel's Grammar Language Model that applies block universe theory to machine reasoning. The core thesis: if spacetime is a fixed 4D manifold (as general relativity implies), then "prediction" is not generating future states from present ones -- it is discovering the structural invariants (geodesics) that constrain what configurations are possible. A model trained on the *skeletons* of reasoning rather than on surface token distributions should be able to achieve frontier-level performance on structural reasoning tasks at a fraction of the compute.

**What BPM claims:**

- Grammar derivation trees are isomorphic to causal reasoning chains, which are isomorphic to spacetime geodesics. This is the central structural identity.
- A model with 1B-13B parameters, pre-trained on grammar-derived structural skeletons (not raw text), can match or exceed frontier models on reasoning benchmarks when augmented with inference-time scaling (MCTS over reasoning forks, transient LoRA adapters).
- The power comes from *structure*, not parameters. MKAngel's existing 370K-parameter GLM demonstrates the principle at small scale; BPM scales it.

**What is proven vs speculative:**

| Claim | Status |
|-------|--------|
| Grammar derivation trees encode causal structure | Implemented in MKAngel's DerivationEngine (engine.py) |
| Bidirectional attention + grammar constraints improve prediction | Implemented in TemporalAttention (attention.py) |
| Strange loops provide recursive self-reference | Implemented in StrangeLoopAttention (attention.py) |
| Cross-domain isomorphisms exist between grammars | Implemented in DerivationEngine.find_isomorphisms() |
| MNEMO codec compresses via grammar rules (10^137+ space) | Implemented in MnemoCodec (codec.py) |
| Grammar skeletons = causal geodesics (the BPM thesis) | **Speculative -- untested** |
| 1B model + MCTS can match frontier reasoning | **Speculative -- no evidence yet** |
| Block universe framing yields practical compute savings | **Speculative -- theoretically motivated only** |

**Why this matters (if it works):**

Frontier reasoning models use hundreds of billions of parameters and enormous compute budgets. If structural understanding can substitute for parametric memorisation, then reasoning becomes accessible on consumer hardware. MKAngel's GLM already demonstrates that 370K parameters with grammar awareness can perform non-trivial derivation, prediction, and isomorphism detection. BPM asks: how far does this principle scale?

**Honest assessment:** This is a high-risk, high-reward research direction. The theoretical motivation is sound (block universe physics, formal grammar theory, and causal inference are well-established fields), but the claim that these fields combine to produce a practical reasoning engine at sub-frontier compute is unproven. The design document lays out the thesis, the architecture, and -- critically -- the kill criteria that determine whether to continue.

---

## 2. Theoretical Foundation

### 2.1 Block Universe Theory Applied to Reasoning

In Minkowski's block universe (the standard interpretation of special relativity's spacetime), the past, present, and future all exist as a single 4D manifold. There is no "flow" of time -- only a static structure viewed from different observer perspectives. Events are related by geodesics (shortest paths through spacetime), and the laws of physics are constraints on which configurations of the manifold are self-consistent.

**The analogy to reasoning:** A well-formed argument is a "geodesic" through a space of propositions. The premises constrain which conclusions are reachable, just as initial conditions constrain which future states are physically realisable. A reasoner does not "generate" conclusions from premises sequentially; it *discovers* which conclusions are structurally compatible with the premises.

This is not just a metaphor. The mathematical structure is shared:

- **Geodesics** are paths that extremise an action functional. In spacetime, this is the integral of the Lagrangian. In grammar, the analogous quantity is the derivation cost -- the total weight of rule applications along a derivation path. MKAngel's DerivationEngine already tracks this via `rule.weight` accumulated along `DerivationTree` paths.
- **Causal structure** in spacetime is determined by the light-cone -- events are causally connected if and only if a signal can travel between them. In grammar, causal structure is determined by the derivation graph -- symbols are causally connected if and only if a chain of rule applications connects them. MKAngel's `Grammar.find_loops()` and `DerivationEngine.detect_loops()` already identify the causal cycles (strange loops) in this graph.
- **Observer dependence** in relativity means that different observers see different "now" slices of the same underlying manifold. In reasoning, different "prompts" or "framings" of the same problem correspond to different slices through the same underlying logical structure. MKAngel's `compose_fugue()` already demonstrates this: the same derivation viewed from different grammar "voices" reveals different aspects.

### 2.2 Grammar Skeletons as Causal Structure Extractors

A grammar skeleton is the derivation tree of a reasoning chain, stripped of its surface tokens. What remains is pure structure: which steps depend on which others, where branches fork, where branches merge, and where loops exist.

**Example:**

Consider a chain-of-thought (CoT) reasoning trace:
```
"The equation x^2 - 5x + 6 = 0 can be factored as (x-2)(x-3) = 0,
 so x = 2 or x = 3."
```

Its grammar skeleton (using MKAngel's existing grammar primitives):
```
DERIVE: QuadraticEquation
  FORK: FactoringAttempt
    APPLY: Factoring(a=1, sum=5, product=6)
      DERIVE: (x - root1)(x - root2)
        FORK: ZeroProductProperty
          BRANCH: root1 = 2
          BRANCH: root2 = 3
        MERGE: SolutionSet = {2, 3}
```

This skeleton is *domain-independent*. The same DERIVE-FORK-APPLY-BRANCH-MERGE pattern appears in:
- Mathematical proofs (as above)
- Causal reasoning ("If A and B, then C or D")
- Scientific hypothesis testing ("Given observations X, hypotheses H1 and H2 are consistent; experiment E distinguishes them")
- Planning ("Goal G requires steps S1 and S2; S1 has two approaches A1a and A1b")

MKAngel's existing `DerivationEngine.derive()` already produces `DerivationTree` objects that capture exactly this structure. The `DerivationTree.paths()` method extracts all root-to-leaf derivation chains. The `compose_fugue()` method finds shared structural patterns across different grammar domains.

### 2.3 The Central Isomorphism

The BPM thesis rests on a triple isomorphism:

```
Spacetime Geodesics  <->  Grammar Derivation Trees  <->  Causal Reasoning Chains
        |                         |                              |
   extremise action         minimise derivation cost       minimise logical steps
   constrained by           constrained by                 constrained by
   field equations          grammar rules                  inference rules
   observer-dependent       voice-dependent                framing-dependent
   have light-cones         have derivation cones          have relevance cones
   exhibit strange loops    exhibit strange loops           exhibit self-reference
   (closed timelike         (recursive productions)        (meta-reasoning)
    curves)
```

**What MKAngel already implements of this isomorphism:**

1. Grammar derivation trees -- `DerivationTree` in `engine.py`
2. Bidirectional derivation (forward = predict, backward = reconstruct) -- `DerivationEngine.derive(direction="forward"|"backward")`
3. Derivation cost tracking -- `Rule.weight` accumulated along paths
4. Strange loop detection -- `Grammar.find_loops()`, `StrangeLoopAttention`
5. Cross-domain isomorphism detection -- `DerivationEngine.find_isomorphisms()`
6. Fugue composition (multi-voice derivation) -- `DerivationEngine.compose_fugue()`
7. Observer-dependent views -- different `Grammar` objects as "voices" viewing the same derivation

**What BPM adds:**

1. Explicit geodesic-finding algorithms (MCTS over derivation trees)
2. Special tokens encoding the block-universe structure (GEO, FORK, ETERNAL, OBSERVE, PROPHECY)
3. Training on structural skeletons rather than surface tokens
4. Inference-time scaling via grammar-guided search

### 2.4 Mapping MKAngel's GLM to Block Universe Concepts

| MKAngel Concept | Block Universe Analogue |
|----------------|------------------------|
| `Grammar` (grammar.py) | Local spacetime geometry (metric tensor) |
| `Rule` | Geodesic equation (constraint on paths) |
| `Production` | Field equation (dynamics) |
| `StrangeLoop` | Closed timelike curve (causal loop) |
| `DerivationTree` | Worldline (path through spacetime) |
| `DerivationEngine.derive(forward)` | Evolving forward along a geodesic |
| `DerivationEngine.reconstruct()` | Tracing a geodesic backward to its origin |
| `DerivationEngine.compose_fugue()` | Comparing geodesics from different reference frames |
| `DerivationEngine.find_isomorphisms()` | Coordinate transformations between reference frames |
| `FugueAttention` | Multiple observers (voices) attending to the same manifold |
| `StrangeLoopAttention` | Self-referential measurement (observer observing itself) |
| `TemporalAttention` | Bidirectional causal cone (past light-cone + future light-cone) |
| `EmbeddingSpace.find_isomorphisms()` | Diffeomorphism between manifolds |
| `MnemoCodec.compress()` | Kolmogorov complexity (minimal description of a worldline) |

---

## 3. Architecture Specification

### 3.1 Model Scale

The BPM targets the 1B-13B parameter range using Mixture of Experts (MoE) for fork efficiency:

| Tier | Total Params | Active Params (MoE) | Experts | Use Case |
|------|-------------|---------------------|---------|----------|
| BPM-1B | 1.3B | ~300M | 8 | On-device (high-end phone/laptop) |
| BPM-3B | 3.2B | ~800M | 8 | Consumer GPU (RTX 3060+) |
| BPM-7B | 7.1B | ~1.8B | 16 | Prosumer GPU (RTX 4080+) |
| BPM-13B | 13.2B | ~3.3B | 16 | Multi-GPU / cloud |

MoE is critical because reasoning forks naturally map to expert routing: different experts specialise in different derivation strategies (algebraic, geometric, causal, analogical), and the router selects experts based on the grammar skeleton of the current reasoning branch.

**Why MoE over dense:** At a 1B total parameter budget, dense models typically achieve only ~300M "effective" parameters for any given input (due to activation sparsity). MoE makes this explicit and controllable: the router can direct different reasoning branches to different experts, achieving effective specialisation without per-branch fine-tuning.

**Relationship to MKAngel's 370K GLM:** The existing GLM serves as the structural prototype. Its architecture (FugueAttention + StrangeLoopAttention + TemporalAttention + GrammarFFN) is preserved in each BPM expert. The BPM is not a replacement for the GLM -- it is the GLM scaled up, with MoE routing replacing the single-path inference.

### 3.2 Special Tokens

BPM introduces five special tokens that encode block-universe structure directly into the token vocabulary:

| Token | Name | Semantics | GLM Analogue |
|-------|------|-----------|--------------|
| `<GEO>` | Geodesic | Marks the start/end of a derivation chain (a path through reasoning space) | `Derivation` objects in engine.py |
| `<FORK>` | Fork | Marks a branch point where multiple derivation paths diverge | `DerivationTree` branching (`children` list) |
| `<ETERNAL>` | Eternal | Marks a statement that is structurally invariant (axiom, definition, tautology) | `Rule` with `weight=1.0` and bidirectional |
| `<OBSERVE>` | Observer | Marks a perspective-dependent statement (frame of reference, assumption) | Different `Grammar` "voices" in fugue composition |
| `<PROPHECY>` | Prophecy | Marks a prediction derived from grammar structure | `superforecast()` output |

These tokens are *not* decorative. They structure the training data so the model learns to distinguish between:
- Fixed structure (ETERNAL) vs observer-dependent claims (OBSERVE)
- Points where reasoning must branch (FORK) vs points where it can proceed linearly (GEO)
- Predictions that follow from structure (PROPHECY) vs mere extrapolation

**Integration with MNEMO:** Each special token maps to a MNEMO operation:
- `<GEO>` -> `*g` (universal geodesic)
- `<FORK>` -> `*f` (universal fork)
- `<ETERNAL>` -> `*e` (universal eternal)
- `<OBSERVE>` -> `*o` (universal observe)
- `<PROPHECY>` -> `*p` (universal prophecy)

### 3.3 Positional Encoding: RoPE Extended for Temporal Depth

Standard RoPE encodes token position in the sequence. BPM extends RoPE with two additional rotary axes, mirroring MKAngel's existing `TemporalEmbedding` which already encodes three temporal axes (position, derivation depth, epoch):

1. **Sequence position** (standard RoPE) -- surface order of tokens
2. **Derivation depth** -- how many rule applications deep in the grammar tree this token sits. Uses a different frequency base (1000.0 vs the standard 10000.0) to ensure disentanglement, matching MKAngel's `TemporalEmbedding._build_sinusoidal(max_depth, dim, base=1000.0)`.
3. **Causal distance** -- how many reasoning steps separate this token from the most recent `<GEO>` or `<FORK>` token. Encodes the "temporal depth" within a single reasoning geodesic.

**Implementation:** The existing `TemporalEmbedding` class in `embeddings.py` already handles multi-axis sinusoidal encoding with configurable frequency bases. For BPM, this is replaced with rotary position embeddings applied to Q and K in the attention layers, but the same three-axis design is preserved.

### 3.4 Multi-Task Training Objective

BPM trains on four objectives simultaneously, extending the GLM's existing five objectives (forward prediction, backward reconstruction, grammar induction, cross-domain alignment, strange-loop detection):

1. **Geodesic Prediction** -- Given a partial derivation chain (geodesic), predict the next step. This is the standard next-token prediction objective, but applied to grammar skeletons rather than surface tokens. Weight: 0.4.

2. **Fork Completion** -- Given a `<FORK>` token and one completed branch, predict the other branch(es). This teaches the model to reason about alternative derivation paths. Weight: 0.25.

3. **Manifold Induction** -- Given multiple geodesics from the same "manifold" (set of related reasoning problems), induce the shared grammar that generates them. This is the grammar-induction objective from MKAngel's trainer, scaled up. Weight: 0.2.

4. **Contrastive Timeline Pairs** -- Given two reasoning chains that reach different conclusions from the same premises, learn to identify the fork point where they diverged. This teaches the model about causal structure and counterfactual reasoning. Weight: 0.15.

**Relationship to existing GLM objectives:**

| GLM Objective | BPM Objective | Mapping |
|--------------|---------------|---------|
| Forward prediction | Geodesic prediction | Surface tokens -> grammar skeleton tokens |
| Backward reconstruction | (Subsumed by geodesic prediction on reversed chains) | Bidirectional is now a data augmentation |
| Grammar induction | Manifold induction | Single rule -> shared grammar across chains |
| Cross-domain alignment | (Subsumed by manifold induction across domains) | Alignment is now implicit |
| Strange-loop detection | Fork completion (loop = fork that returns to start) | Loops are forks with self-reference |

### 3.5 Attention Mechanism Extensions

Each BPM layer contains the same three attention mechanisms as a GLMLayer, extended for the larger scale:

1. **FugueAttention** -- Unchanged in principle. Temporal offsets and derivation-depth biases work identically. At 1B+ scale, the number of heads increases from 4 to 16-32, providing finer-grained "voice" decomposition.

2. **StrangeLoopAttention** -- Extended with *multi-hop* self-reference: instead of attending to only the previous layer's attention weights, BPM-StrangeLoop can attend to attention weights from up to `loop_depth=4` layers back (vs 2 in the current GLM). The loop gate is initialised at 0.05 (vs 0.1) to prevent early-training instability at larger scale.

3. **TemporalAttention** -- Extended with *grammar-masked* bidirectional attention: the transition bias table (`_transition_bias` in the current implementation) is populated from the training data's grammar skeletons rather than from hand-coded rules. This makes the grammar constraints *learned* rather than *specified*.

4. **NEW: GeodesicRouterAttention** -- A new attention mechanism specific to BPM that routes tokens through MoE experts based on their grammar context. Tokens following a `<FORK>` are routed to "divergence" experts; tokens following `<GEO>` are routed to "continuation" experts; tokens following `<ETERNAL>` are routed to "axiom" experts. This is implemented as a top-2 expert routing mechanism conditioned on the local grammar context window.

---

## 4. Training Curriculum (Synthetic Eternity Engine)

### 4.1 Overview

The BPM training curriculum comprises 5-15T tokens (depending on tier), with 80%+ being structural/eternal (grammar skeletons, formal derivations, symbolic traces) and less than 20% being bridging natural language.

| Bucket | Share | Content | Token Count (1B tier) |
|--------|-------|---------|----------------------|
| Geodesic Grammars | 35% | Recursive spacetime BNF, context-free causality ladders, derivation tree linearisations | ~1.75T |
| Quantum Traces + Symbolic Forks | 30% | GR derivation chains, Many-Worlds simulation traces, decision tree linearisations | ~1.5T |
| Abstracted Prophecy Skeletons | 20% | Delexicalized CoT traces from frontier models, with surface tokens replaced by grammar skeleton tokens | ~1.0T |
| Adversarial Perturbations | 10% | Chaotic attractor trajectories, near-miss derivations, adversarial forks designed to confuse structure | ~0.5T |
| Bridging Observer Language | 5% | Natural language reasoning with inline grammar annotations, allowing the model to translate between human language and skeleton notation | ~0.25T |

### 4.2 Bucket 1: Geodesic Grammars (35%)

Content: Linearised derivation trees from formal grammars, generated using MKAngel's existing `DerivationEngine.derive()`. Each training example is a derivation tree serialised as a token sequence with BPM special tokens:

```
<GEO> QuadraticEquation <FORK> FactoringAttempt
  <GEO> APPLY Factoring <ETERNAL> ZeroProductProperty
    <FORK> root1=2 root2=3 <PROPHECY> SolutionSet={2,3}
```

**Generating this data:** MKAngel's seven grammar domains (linguistic, etymological, chemical, biological, computational, mathematical, physics) provide the starting grammars. The DerivationEngine generates derivation trees from each grammar, and these are serialised into the BPM training format.

**Mapping MKAngel's grammar domains to curriculum buckets:**

| MKAngel Grammar Domain | Primary Curriculum Bucket | Rationale |
|------------------------|--------------------------|-----------|
| mathematical (algebra, calculus, logic, number theory) | Geodesic Grammars | Mathematical derivations are the purest form of causal chains |
| physics (mechanics, EM, thermo, quantum, relativity) | Quantum Traces + Symbolic Forks | Physics derivations often involve branching (quantum superposition, frame-dependent observations) |
| computational (syntax, types, patterns) | Geodesic Grammars | Type derivations and pattern matching are formal derivation chains |
| linguistic (syntax, phonology, morphology) | Bridging Observer Language | Natural language is the bridge between formal structure and human understanding |
| chemical (bonding, reaction, molecular) | Geodesic Grammars | Chemical reactions are formal transformations with conservation laws |
| biological (genetic, protein, evolutionary) | Quantum Traces + Symbolic Forks | Biological evolution involves branching (speciation, mutation) |
| etymological (etymology, substrate transfer, cognate detection) | Bridging Observer Language | Etymology bridges formal sound change rules with human language |

### 4.3 Bucket 2: Quantum Traces + Symbolic Forks (30%)

Content: Traces from simulated reasoning processes that involve branching and merging:

- **General relativity derivations:** Linearised Christoffel symbol computations, geodesic equation solutions, metric tensor transformations. These are pure symbol-manipulation chains.
- **Many-Worlds simulation traces:** Decision trees where each branch point is a `<FORK>` and each outcome is a `<PROPHECY>`. Generated synthetically by simulating multiple-hypothesis reasoning.
- **Proof search traces:** From automated theorem provers, capturing the branching and backtracking structure of proof search. Each failed branch teaches the model about reasoning dead-ends.

### 4.4 Bucket 3: Abstracted Prophecy Skeletons (20%)

Content: Chain-of-thought (CoT) reasoning traces from frontier models (or synthetic equivalents), *delexicalised* -- surface words replaced by grammar skeleton tokens while preserving the structural dependency graph.

**Process:**
1. Take a CoT trace: "Let's think about this step by step. First, we factor the quadratic..."
2. Parse its dependency structure (which step depends on which)
3. Replace surface tokens with structure tokens: `<GEO> STEP1 <FORK> APPROACH_A APPROACH_B <GEO> STEP2(dep=APPROACH_A) <PROPHECY> RESULT`
4. The surface content is stripped; only the reasoning *shape* remains.

**Why this works (in theory):** If the BPM thesis is correct, the *shape* of reasoning (its grammar skeleton) carries most of the information, and the surface tokens are just labels. By training on shapes, the model learns general reasoning patterns that transfer across domains.

**Risk:** This is the most speculative component. If reasoning quality depends primarily on surface-level knowledge (domain expertise, factual recall) rather than structural patterns, this bucket will be ineffective.

### 4.5 Bucket 4: Adversarial Perturbations (10%)

Content: Examples specifically designed to break structural prediction:

- **Chaotic attractor trajectories:** Sequences from chaotic dynamical systems (Lorenz attractor, logistic map) where long-term prediction is provably impossible. The model must learn to output high uncertainty (`<OBSERVE>` tokens) for these.
- **Near-miss derivations:** Derivation chains that are almost valid but contain a single error. The model must learn to detect the error point.
- **Adversarial forks:** Reasoning problems where the "obvious" structural pattern leads to the wrong answer. The model must learn to not over-rely on pattern matching.

### 4.6 Bucket 5: Bridging Observer Language (5%)

Content: Natural language reasoning with inline grammar annotations, enabling the model to translate between human language and BPM skeleton notation. This is the smallest bucket because the model's primary modality is structural, not linguistic. The bridging language serves only to interface with human users.

### 4.7 Synthetic Data Generation Using the Derivation Engine

MKAngel's `DerivationEngine` is the core data generator. The pipeline:

1. **Select a grammar** from the 7 domains (21 builder functions in `glm/grammars/`).
2. **Select a starting form** from the grammar's symbol set (`grammar.symbols()`).
3. **Run `derive(form, grammar, "forward")`** to produce a `DerivationTree`.
4. **Serialise the tree** into BPM training format (linearised with special tokens).
5. **Augment** by running the same tree backward (`derive(result, grammar, "backward")`), producing contrastive pairs.
6. **Cross-pollinate** by running `compose_fugue()` across grammar pairs, capturing isomorphic structure.

**Estimated generation rate:** The current pure-Python DerivationEngine processes ~1000 derivations/second on a modern CPU. For 5T tokens at ~100 tokens/derivation, this requires ~50B derivations, or ~50M seconds (~1.6 years) on a single core. Parallelised across 128 cores, this drops to ~4.5 days. This is feasible but highlights the need to rewrite the engine in a compiled language (Rust, C++) for production data generation.

---

## 5. Progressive Training Strategy

The BPM is not built in one shot. It evolves through three stages that progressively transfer intelligence from external LLMs into the grammar engine.

### Stage 1: Grammar Pre-Processing + LLM (Current Target -- March 2026)

**Architecture:** MKAngel's GLM processes input through its grammar domains to extract structural skeletons. These skeletons are prepended to the prompt for an external LLM (Claude API, local Llama, etc.), which generates the natural language output.

```
User Input -> GLM Grammar Analysis -> Skeleton Extraction
                                          |
                                          v
                                   Skeleton + Input -> External LLM -> Output
```

**What this proves:** Whether grammar skeletons improve reasoning quality when provided as context to an existing LLM. This is the lowest-risk test of the BPM thesis.

**Implementation using existing MKAngel:**
1. Use `DerivationEngine.derive()` to generate derivation trees from the input.
2. Serialise the trees as structured prompts.
3. Feed to an LLM with the instruction: "Use this structural analysis to guide your reasoning."
4. Measure whether the grammar-guided output is more accurate on structural reasoning benchmarks.

**What can be trained on-device:** The 370K GLM itself (grammar rule weights, attention parameters, embedding alignments) can be trained on the Pixel 10 Pro XL or Windows CPU using the existing `GLMTrainer`. At 370K parameters with SPSA gradient estimation, training takes ~2 seconds per step on CPU.

**Success criterion:** Measurable improvement on GSM8K or similar benchmarks when grammar skeletons are prepended to prompts vs baseline (no skeleton).

### Stage 2: Hybrid Grammar-Guided Inference (Target -- April 2026)

**Architecture:** The GLM's grammar analysis constrains the search space of a small (1B-3B) language model. Instead of the LLM sampling freely over its full vocabulary, the grammar analysis provides a mask or bias over the token distribution, favouring tokens that are consistent with the grammar skeleton.

```
User Input -> GLM Grammar Analysis -> Grammar Constraint Mask
                                          |
                                          v
                            Small LLM (1B-3B) with constrained decoding -> Output
```

**What this proves:** Whether grammar constraints can substitute for parametric knowledge -- i.e., whether a *small* model with structural guidance can match a *large* model without it.

**Key technical challenges:**
- Mapping grammar skeleton tokens to LLM vocabulary tokens (vocabulary alignment)
- Implementing constrained decoding that respects grammar structure without being too restrictive
- Balancing grammar guidance with the LLM's learned knowledge (too much guidance = brittle; too little = no benefit)

**Training:** The small LLM is fine-tuned on BPM training data (grammar skeletons + completions). The GLM is trained jointly to produce masks that improve the small LLM's output. This requires a differentiable interface between the GLM and the LLM, which is a significant engineering challenge.

### Stage 3: Pure Grammar Engine (Target -- Q2 2026)

**Architecture:** No external LLM. The BPM model (1B-13B parameters, MoE) trained entirely on grammar-derived data performs end-to-end reasoning. Input is parsed into grammar skeletons, skeletons are extended via geodesic MCTS (Section 7), and output is generated from the extended skeleton.

```
User Input -> BPM Tokeniser -> BPM Model (MoE) -> Grammar Skeleton Output
                                                           |
                                                           v
                                              Skeleton -> NL Surface Form
```

**What this proves:** Whether structure alone is sufficient for frontier-level reasoning. This is the full BPM thesis.

**Risk:** This is the highest-risk stage. If reasoning quality depends fundamentally on broad world knowledge encoded in billions of parameters, Stage 3 will fail. The kill criterion is defined in Section 9.

---

## 6. On-Device Training

### 6.1 The 370K GLM: Trainable Now

MKAngel's current GLM has 370K trainable parameters (verified via `GrammarLanguageModel.num_parameters`). These are distributed as:

| Component | Parameters | Notes |
|-----------|-----------|-------|
| GrammarEmbedding (vocab=256, dim=64) | ~16.6K | 256 * 64 + 3 * 64 + 64 |
| SubstrateEmbedding (8 substrates, dim=64) | ~1.0K | 8 * 64 * 2 |
| TemporalEmbedding | 0 | Sinusoidal (fixed, not trained) |
| FugueAttention x3 layers | ~49.2K per layer | 4 * 64 * 64 = 16,384 per matrix, 4 matrices |
| StrangeLoopAttention x3 layers | ~33.5K per layer | 4 * 64 * 64 + 2 * 16 * 16 + 1 |
| GrammarFFN x3 layers (dim=64, hidden=256) | ~33.3K per layer | 256 * 64 + 256 + 64 * 256 + 64 |
| TemporalAttention (model-level) | ~57.4K | 7 * 64 * 64 + 1 |
| Forward head | ~16.6K | 256 * 64 + 256 |
| Backward head | ~16.6K | 256 * 64 + 256 |
| Superforecasting head | ~25.0K | (128 + 3) * 64 + 256 * 64 + 256 |
| **Total** | **~370K** | |

**Training on Pixel 10 Pro XL:**

The Pixel 10 Pro XL includes Google's Tensor G6 chip with an integrated TPU for on-device ML. Via TFLite and pyjnius, the GLM's training loop can be offloaded to the TPU.

- **Estimated training time (SPSA, 370K params):** ~2-5 seconds per batch of 8 examples on CPU. With TPU acceleration via TFLite delegates, estimated ~0.5-1 second per batch.
- **Memory footprint:** 370K * 4 bytes (FP32) = 1.48 MB for parameters. Activations for a 512-length sequence: ~2-5 MB. Total: well within the Pixel 10's 16GB RAM.
- **Training data size:** 100K synthetic examples (generated by `GLMTrainer.generate_synthetic_data()`) at ~50 tokens each = ~5M tokens = ~20MB on disk.

**Training on Windows CPU:**

The current pure-Python implementation (no numpy/torch) runs at ~1000 forward passes/second on a modern CPU. SPSA requires 2 forward passes per gradient step, so training throughput is ~500 steps/second. For 10 epochs over 100K examples with batch size 8: ~125K steps total, or ~4 minutes.

### 6.2 Scaling Path

| Tier | Parameters | Memory (FP32) | Memory (INT8) | Training Device | Est. Time (100K examples, 10 epochs) |
|------|-----------|---------------|---------------|-----------------|--------------------------------------|
| 370K (current) | 370K | 1.5 MB | 370 KB | Phone TPU / Any CPU | ~4 minutes |
| 1M | 1M | 4 MB | 1 MB | Phone TPU / Any CPU | ~15 minutes |
| 10M | 10M | 40 MB | 10 MB | Phone TPU / Laptop CPU/GPU | ~2 hours |
| 100M | 100M | 400 MB | 100 MB | Laptop GPU / Desktop | ~20 hours |
| 1B | 1B | 4 GB | 1 GB | Desktop GPU (8GB+) / Cloud | ~200 hours (8 days) |

**Key insight:** Every tier up to 100M parameters is trainable on consumer hardware. The 1B tier requires a dedicated GPU but is still within reach of a single RTX 3060 (12GB). The MoE architecture means that only the active parameters (~300M for a 1B MoE-8) need to be in fast memory during any given forward pass.

### 6.3 Android TPU Path

**Current status:** The Pixel 10 Pro XL's Tensor G6 TPU is accessible via TFLite's delegate API. Pyjnius (already in MKAngel's build dependencies) provides the bridge from Python to Java's TFLite API.

**Steps to enable on-device training:**

1. Export the GLM's parameters as a TFLite-compatible FlatBuffer.
2. Define the forward pass as a TFLite model (requires converting the pure-Python implementation to TFLite ops).
3. Use TFLite's on-device training API (available since TFLite 2.7) for gradient computation.
4. Write results back to the Python GLM via pyjnius.

**Limitation:** TFLite's on-device training supports a limited set of ops and does not support SPSA natively. The gradient computation may need to remain in Python (CPU) while the forward pass runs on the TPU. This is still a net win: the forward pass is the bottleneck, and TPU acceleration of 5-10x is expected.

### 6.4 Quantisation Strategy

| Precision | Bits/Param | 370K Size | 1M Size | 10M Size | 100M Size | Quality Impact |
|-----------|-----------|-----------|---------|----------|-----------|---------------|
| FP32 | 32 | 1.5 MB | 4 MB | 40 MB | 400 MB | Baseline |
| FP16 | 16 | 740 KB | 2 MB | 20 MB | 200 MB | <1% degradation |
| INT8 | 8 | 370 KB | 1 MB | 10 MB | 100 MB | ~2-5% degradation (recoverable with QAT) |
| INT4 | 4 | 185 KB | 500 KB | 5 MB | 50 MB | ~5-15% degradation (experimental) |

**Recommended path:** Train in FP32, deploy in FP16 for the first three tiers, INT8 for the 100M+ tiers. Quantisation-aware training (QAT) during the last 10% of training recovers most INT8 degradation.

---

## 7. Inference-Time Scaling (Test-Time Godmode)

The BPM's key innovation is that small models can match large ones when given more compute *at inference time*. This mirrors the block universe framing: the model does not need to "know" the answer; it needs to *search* for the geodesic that connects the question to the answer.

### 7.1 Adaptive Compute Budget

Each query receives a compute budget proportional to its structural complexity:

| Query Complexity | Budget Multiplier | Description |
|-----------------|-------------------|-------------|
| Trivial (no forks) | 1x (baseline) | Single forward pass, no search |
| Simple (1-2 forks) | 4x | Limited branching, quick resolution |
| Moderate (3-5 forks) | 16x | MCTS with shallow exploration |
| Complex (6+ forks) | 64x | Full MCTS with deep exploration |
| Extreme (novel structure) | 256x | Exhaustive MCTS + LoRA evolution |

**Complexity detection:** The grammar skeleton of the input (produced by the GLM's derivation engine) determines the budget. The number of `<FORK>` tokens, the depth of the derivation tree, and the number of unseen production patterns together estimate complexity.

### 7.2 Geodesic MCTS

Monte Carlo Tree Search, guided by grammar constraints, over the space of reasoning forks:

```
Root: Query grammar skeleton
  |
  +-- FORK 1: Approach A (BPM score: 0.72)
  |     +-- GEO: Step A1 (BPM score: 0.68)
  |     +-- GEO: Step A2 (BPM score: 0.81)
  |           +-- PROPHECY: Result A (BPM score: 0.79)
  |
  +-- FORK 1: Approach B (BPM score: 0.65)
        +-- GEO: Step B1 (BPM score: 0.71)
        +-- FORK 2: Sub-approach B1a vs B1b
              +-- GEO: B1a → PROPHECY: Result B1a (BPM score: 0.44)
              +-- GEO: B1b → PROPHECY: Result B1b (BPM score: 0.83)
```

**Grammar guidance:** At each FORK, the BPM model proposes candidate branches. The grammar engine (`DerivationEngine.derive()`) filters branches that violate known grammar constraints. Only structurally valid branches are explored.

**Scoring:** Each node is scored by the BPM model's confidence estimate, which (following the GLM architecture) is derived from:
1. FugueAttention harmony -- inter-head agreement on the derivation path
2. StrangeLoopAttention gate value -- how much self-referential reasoning was activated
3. TemporalAttention forward/backward consistency -- whether the path is consistent in both temporal directions

This mirrors the existing `superforecast()` method in `glm.py`, which combines harmony, loop gates, and temporal consistency into a confidence score.

**MCTS parameters:**
- Exploration constant c_puct = 1.5 (balancing exploration vs exploitation)
- Maximum tree depth: 32 reasoning steps
- Rollout policy: grammar-constrained random derivation (using `DerivationEngine.derive()` with `max_steps=10`)
- Backup policy: average of leaf scores weighted by grammar rule weights

### 7.3 Transient LoRA Evolution

For queries that exhaust the MCTS budget without finding a high-confidence geodesic, BPM applies transient Low-Rank Adaptation:

1. **Freeze** the base BPM model.
2. **Initialise** a small LoRA adapter (rank 4-8, affecting only the GeodesicRouterAttention and the last FFN layer).
3. **Train** the adapter for 10-50 gradient steps on the query itself, using the MCTS exploration results as training signal (paths with higher BPM scores as positive examples, lower scores as negative).
4. **Re-run** inference with the adapted model.
5. **Discard** the adapter after the query is answered (transient -- not saved).

**Compute cost:** For a 1B model with rank-4 LoRA on 2 layers: ~200K trainable parameters. 50 gradient steps at ~10ms each (GPU) = 500ms total. This is within the acceptable latency for a reasoning query.

**Why this works (in theory):** The LoRA adapter specialises the model's expert routing for the specific structural pattern of the current query. It is the computational analogue of "focusing attention" on a particular aspect of the block universe.

### 7.4 Evaluator-Reranker

After MCTS produces multiple candidate geodesics (reasoning paths), a frozen copy of the BPM model scores each path end-to-end:

1. Each candidate path is serialised as a full BPM token sequence.
2. The frozen model computes perplexity (lower = more structurally consistent).
3. Paths are ranked by perplexity-weighted confidence.
4. The top path is selected as the final answer.

**Why use a frozen copy:** The evaluator must not be influenced by the LoRA adaptation. It provides an independent "second opinion" on structural consistency.

### 7.5 Self-Correction via Observer Verifiers

Inspired by the block universe's observer dependence, BPM implements self-correction by running the winning geodesic through multiple "observer" lenses:

1. **Temporal verifier:** Run the reasoning backward (using TemporalAttention's backward direction). If the backward derivation does not recover the original premises, the forward reasoning is suspect.
2. **Domain verifier:** Re-run the reasoning using a different grammar domain's rules. If the same structural pattern holds across domains (isomorphism), confidence increases.
3. **Adversarial verifier:** Apply perturbations from the adversarial training bucket. If the reasoning is fragile (small perturbations break it), flag for human review.

### 7.6 Predicted Accuracy vs Compute Curve

**Speculative -- these are targets, not measurements:**

```
Accuracy (% relative to Opus 4.6 on structural reasoning)
100% |                                              ......***
 95% |                                    .....*****
 90% |                              ...***
 85% |                        ..***
 80% |                  ..**
 75% |            ..**
 70% |      ..**
 65% | ..**
     +----+----+----+----+----+----+----+----+----+
     1x   4x   16x  64x  256x                     log(tokens)
          Inference-time compute multiplier
```

**Key prediction:** The curve should be approximately logarithmic -- each doubling of compute produces diminishing returns in accuracy. The BPM thesis claims that the curve *starts higher* (better baseline due to grammar structure) and *flattens later* (more compute-efficient scaling) compared to a pure token-prediction model of the same parameter count.

**This must be validated empirically.** The curve shape is the single most important diagnostic of whether the BPM thesis holds.

---

## 8. Evaluation Framework

### 8.1 Target Benchmarks

| Benchmark | Domain | Target Score | Why This Benchmark |
|-----------|--------|-------------|-------------------|
| ARC (AI2 Reasoning Challenge) | Multi-domain reasoning | >= 92% (ARC-Easy), >= 78% (ARC-Challenge) | Tests structural reasoning ability across domains |
| GPQA (Graduate-level Q&A) | Expert reasoning | >= 50% (diamond set) | Tests depth of reasoning at expert level |
| GSM8K | Mathematical reasoning | >= 85% | Tests multi-step arithmetic reasoning chains |
| MATH | Mathematical reasoning | >= 65% | Tests formal mathematical reasoning |
| LogiQA | Logical reasoning | >= 80% | Tests formal logic and causal reasoning |
| Custom Grammar Benchmarks | Grammar-specific | (see 8.3) | Tests the specific capabilities BPM should excel at |

### 8.2 Success Criteria

**Primary criterion:** >= 95% relative performance vs Claude Opus 4.6 on structural reasoning tasks at < 3% of Opus 4.6's estimated compute budget.

**Breaking this down:**
- "Structural reasoning tasks" = ARC-Challenge, GSM8K, MATH, LogiQA (averaged).
- "95% relative performance" = if Opus 4.6 scores X on these benchmarks, BPM must score >= 0.95X.
- "< 3% compute" = if Opus 4.6 requires ~10^18 FLOPs per query (estimated for a ~500B parameter model generating ~2K tokens), BPM must require < 3 * 10^16 FLOPs per query.

**For the 1B BPM model at 64x inference scaling:**
- Base forward pass: ~1B * 2 (FLOPs per param) * 2048 (sequence length) = ~4 * 10^12 FLOPs
- 64x MCTS: ~64 * 4 * 10^12 = ~2.56 * 10^14 FLOPs
- LoRA (50 steps): ~50 * 200K * 2 * 2048 = ~4 * 10^10 FLOPs (negligible)
- Evaluator-reranker (5 candidates): ~5 * 4 * 10^12 = ~2 * 10^13 FLOPs
- **Total: ~2.8 * 10^14 FLOPs** -- roughly 0.03% of estimated Opus 4.6 compute. This is well within the 3% budget.

**Caveat:** These FLOP estimates are very rough. Actual compute depends on sequence length, MoE routing efficiency, and hardware utilisation. The key point is that the BPM architecture is *structurally* orders of magnitude cheaper than a dense frontier model.

### 8.3 Custom Grammar-Specific Benchmarks

These test BPM's specific claimed advantages:

1. **Grammar Skeleton Completion:** Given a partial derivation tree, predict the next 5 steps. Measured by exact-match accuracy on the step sequence.
2. **Fork Resolution:** Given a `<FORK>` with two branches (one correct, one plausible-but-wrong), select the correct branch. Measured by accuracy.
3. **Isomorphism Detection:** Given two derivation chains from different domains, identify the shared structural pattern. Measured by F1 on known isomorphism pairs.
4. **Strange Loop Identification:** Given a sequence, determine whether it contains a self-referential pattern. Measured by accuracy and false-positive rate.
5. **Adversarial Robustness:** Given a reasoning chain with one deliberate error, locate the error. Measured by precision@1 (does the model identify the correct error location?).

These benchmarks are generated synthetically using MKAngel's grammar engine and can be scaled to any size.

### 8.4 Ablation Studies

To isolate the contribution of grammar structure vs raw parameter count:

| Ablation | What It Tests |
|----------|--------------|
| BPM without grammar skeletons (train on raw text) | Is grammar structure necessary, or does the architecture alone suffice? |
| BPM without MoE (dense equivalent parameter count) | Does expert routing contribute, or is the total parameter count what matters? |
| BPM without MCTS (single forward pass only) | How much does inference-time search contribute? |
| BPM without LoRA evolution | Does per-query adaptation matter? |
| BPM without special tokens | Do the block-universe tokens provide signal, or are they just noise? |
| Vanilla transformer (same param count) + grammar skeletons in data | Is the custom architecture necessary, or do grammar skeletons in standard training data suffice? |

The most informative ablation is the last one: if a vanilla transformer trained on grammar-skeleton data matches BPM, then the custom architecture is unnecessary and the value lies purely in the data. This would be a partial validation of the BPM thesis (structural data matters) but a refutation of the architectural claims.

---

## 9. Risk Analysis (Bear Case)

### 9.1 Kill Criteria

**ABORT if any of the following are observed:**

1. **< 70% on unseen forks after 64x scaling:** If the model cannot resolve novel reasoning forks (not seen in training) even with 64x compute budget, the grammar-guided MCTS is not working. This means grammar skeletons do not generalise to new structures.

2. **< 10% improvement over vanilla transformer baseline (same param count, same data):** If the architecture provides less than 10% relative improvement over a standard transformer trained on the same grammar-skeleton data, the custom attention mechanisms (FugueAttention, StrangeLoopAttention, GeodesicRouter) are not contributing enough to justify their complexity.

3. **Grammar skeleton extraction costs > 50% of total inference time:** If the DerivationEngine's analysis of the input consumes more than half the total inference budget, the grammar pre-processing is a bottleneck, not an accelerator.

### 9.2 "Quantum Fog" Risk

**Description:** Some reasoning tasks may involve genuinely chaotic dynamics where no amount of structural analysis can improve predictions beyond a baseline. This is the "Quantum Fog" -- the boundary where the block universe metaphor breaks down because the underlying system is irreducibly complex.

**Symptoms:** BPM achieves high accuracy on problems with clean causal structure (mathematical proofs, logical deductions) but performs no better than random on problems involving:
- Creative insight (lateral thinking, analogy to far domains)
- Empirical knowledge (factual recall, real-world commonsense)
- Social reasoning (theory of mind, pragmatics)

**Mitigation:** The adversarial perturbation bucket (10% of training data) is specifically designed to teach the model when to signal uncertainty. The `<OBSERVE>` token should appear in outputs when the model detects that the reasoning structure is underdetermined. If this detection is reliable, Quantum Fog is managed (not eliminated) -- the model knows when it does not know.

**Residual risk:** High. Social and creative reasoning may be fundamentally non-structural, in which case BPM is a strong tool for formal domains but not a general-purpose reasoner.

### 9.3 Grammar Brittleness

**Description:** MKAngel's grammar domains are hand-crafted (21 grammar builder functions across 7 domains). These encode *known* structural regularities but may not cover the structures that appear in real reasoning tasks.

**Symptoms:** BPM works well on tasks that happen to match the 7 grammar domains (math, physics, chemistry, biology, linguistics, computation, etymology) but fails on tasks from uncovered domains (law, economics, psychology, art).

**Mitigation:**
1. **Manifold induction (training objective 3):** BPM is trained to *induce* grammars from data, not just apply pre-defined ones. This should enable generalisation to new domains.
2. **Grammar expansion:** As new domains are encountered, new grammar builder functions can be added to MKAngel's `glm/grammars/` directory.
3. **Meta-grammar learning:** At the 1B+ scale, BPM should be able to learn grammar rules directly from the training data, reducing dependence on hand-crafted rules.

**Residual risk:** Moderate. The manifold induction objective is promising but unproven at scale.

### 9.4 Scaling Cliff

**Description:** The BPM thesis may hold at 370K parameters (where structure demonstrably matters more than scale) and at 100B+ parameters (where everything works), but fail in the 1B-13B "middle ground" where neither structure nor scale is sufficient on its own.

**Symptoms:** BPM-1B underperforms a standard 1B model despite grammar guidance. BPM-13B matches but does not exceed a standard 13B model.

**Mitigation:** The progressive training strategy (Section 5) is designed to detect this early. Stage 1 (grammar + external LLM) tests whether grammar skeletons provide signal at all. Stage 2 (hybrid inference) tests whether grammar constraints help small models. Only if both stages succeed does Stage 3 (pure grammar engine) proceed.

**Residual risk:** Moderate to high. The 1B-13B range is empirically underexplored for grammar-guided models.

---

## 10. Roadmap

### Phase 0: Thesis Validation (Current -- March 2026)

**Goals:**
- Complete the GSM-GLM integration: use the existing 370K GLM to parse GSM8K problems into grammar skeletons
- Build the benchmark harness (custom grammar-specific benchmarks from Section 8.3)
- Measure whether grammar skeletons prepended to LLM prompts improve GSM8K accuracy (Stage 1 test)

**Deliverables:**
- `glm/benchmarks/` -- benchmark harness with 5 grammar-specific tests
- `glm/integration/gsm_glm.py` -- GSM8K parser using the derivation engine
- Baseline accuracy measurements (GLM alone, LLM alone, GLM + LLM)

**Decision gate:** If grammar skeletons provide >= 5% improvement on GSM8K when prepended to LLM prompts, proceed to Phase 1. If < 5%, re-evaluate the thesis.

### Phase 1: Stage 1 Training (Late March 2026)

**Goals:**
- Train the GLM's grammar rule weights and attention parameters on GSM8K-derived data
- Implement the synthetic data generation pipeline (Section 4.7)
- Generate 1M training examples using the derivation engine
- Train the 370K GLM to convergence on grammar skeleton completion

**Deliverables:**
- Trained 370K GLM checkpoint
- 1M synthetic training examples in BPM format
- Grammar skeleton completion benchmark results

**Decision gate:** If the trained GLM achieves >= 60% on grammar skeleton completion (5-step lookahead), proceed to Phase 2. If < 60%, investigate whether the grammar representations are expressive enough.

### Phase 2: Stage 2 Hybrid Model (April 2026)

**Goals:**
- Implement grammar-constrained decoding for a small (1B-3B) LLM
- Train joint GLM + small LLM on BPM curriculum (Bucket 1 + Bucket 5)
- Measure accuracy on full benchmark suite (Section 8.1)

**Deliverables:**
- Grammar-constrained decoding module
- Hybrid model checkpoint (GLM + small LLM)
- Full benchmark results with ablations

**Decision gate:** If the hybrid model matches >= 80% of a standalone 7B model's accuracy on the benchmark suite, proceed to Phase 3. If < 80%, the grammar constraints are not providing sufficient signal to compensate for the parameter gap.

### Phase 3: Stage 3 Pure Grammar Engine (Q2 2026)

**Goals:**
- Implement the full BPM architecture (MoE, special tokens, GeodesicRouterAttention)
- Train BPM-1B on the full 5T-token curriculum
- Implement Geodesic MCTS and LoRA evolution
- Benchmark against frontier models

**Deliverables:**
- BPM-1B model checkpoint
- Geodesic MCTS implementation
- Full benchmark suite results at 1x, 4x, 16x, 64x, 256x inference scaling
- Accuracy vs compute curve

**Decision gate:** If BPM-1B at 64x scaling achieves >= 70% on unseen forks (kill criterion from Section 9.1), proceed to Phase 4. If not, the BPM thesis is falsified at this scale.

### Phase 4: Scale to 1B+ if Thesis Holds (Q3 2026)

**Goals:**
- Scale to BPM-3B and BPM-7B
- Optimise inference pipeline for production deployment
- Integrate into MKAngel Android app as the on-device reasoning engine
- Target: BPM-3B on Pixel 10 Pro XL via INT8 quantisation (~800MB model)

**Deliverables:**
- BPM-3B and BPM-7B checkpoints
- Quantised INT8 models for on-device deployment
- Android integration via TFLite
- Production benchmark results

---

## Appendix A: Parameter Count Estimates

### BPM-1B Architecture Detail

| Component | Parameters | Notes |
|-----------|-----------|-------|
| Token embeddings (32K vocab, d=2048) | 65.5M | |
| RoPE (3-axis) | 0 | Computed, not stored |
| Transformer layers x24 | | |
| -- FugueAttention (16 heads) | 16.8M per layer | 4 * 2048 * 2048 |
| -- StrangeLoopAttention (16 heads) | 17.3M per layer | 4 * 2048 * 2048 + 2 * 128 * 128 + 1 |
| -- GeodesicRouterAttention (top-2 of 8 experts) | 33.6M per layer | 8 * 2048 * 2048 (shared across experts) |
| -- GrammarFFN (hidden=8192) | 33.6M per layer | 8192 * 2048 + 8192 + 2048 * 8192 + 2048 |
| Layer total x24 | ~2.43B total, ~600M active (MoE top-2) | |
| Prediction heads | ~131M | 2 * 32K * 2048 |
| **Total** | ~1.3B, **~300M active** | |

### FLOPs per Forward Pass (1B model, seq_len=2048)

| Operation | FLOPs | Notes |
|-----------|-------|-------|
| Embedding lookup | Negligible | |
| FugueAttention x24 | 24 * 2 * 2048^2 * 2048 = ~4.0 * 10^11 | |
| StrangeLoopAttention x24 | ~4.2 * 10^11 | Slightly larger due to loop keys |
| GeodesicRouterAttention x24 | ~2.0 * 10^11 | MoE top-2: only 2/8 experts active |
| GrammarFFN x24 | ~4.0 * 10^11 | |
| Prediction heads | ~2.7 * 10^11 | |
| **Total per forward pass** | **~1.7 * 10^12** | |
| **64x MCTS** | **~1.1 * 10^14** | |

---

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| Block Universe | The view from general relativity that all of spacetime exists as a fixed 4D manifold |
| Geodesic | The "shortest path" through spacetime (or reasoning space), extremising an action |
| Grammar Skeleton | A derivation tree stripped of surface tokens, preserving only structural dependencies |
| Fork | A branch point in a derivation tree where multiple derivation paths diverge |
| Strange Loop | A self-referential cycle in a grammar (or reasoning chain) that returns to its starting point at a different level of abstraction |
| Fugue | Multiple "voices" (grammars or attention heads) following the same structural theme at different offsets |
| MCTS | Monte Carlo Tree Search -- a search algorithm that balances exploration and exploitation using random sampling |
| LoRA | Low-Rank Adaptation -- a parameter-efficient fine-tuning method that adds small trainable matrices to a frozen model |
| MoE | Mixture of Experts -- an architecture where different subnetworks (experts) specialise in different input types, with a router selecting which experts process each input |
| MNEMO | MKAngel's mnemonic encoding language where 1-3 character tokens reference grammar rules |
| Derivation Engine | MKAngel's core runtime (`glm/core/engine.py`) that applies grammars to inputs, producing derivation trees |
| GLM | Grammar Language Model -- MKAngel's existing 370K-parameter model |
| BPM | Block Prophet Model -- the proposed extension described in this document |
