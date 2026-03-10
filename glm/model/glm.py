"""The Grammar Language Model — where grammar meets neural computation.

Architecture overview
─────────────────────
The GLM is a *small* model that derives its power from structural
understanding rather than brute-force parameters.  Like a musician who
has internalised the rules of harmony and counterpoint, the GLM learns
the deep grammars underlying symbolic domains and uses them to:

* **Predict forward** — given grammatical patterns, forecast plausible
  future states (superforecasting).
* **Reconstruct backward** — given present observations, infer the
  derivation history and ancestral forms (etymology, phylogeny).
* **Detect patterns** — find self-similar structures, strange loops,
  and cross-domain isomorphisms.

The architecture combines three GEB-inspired mechanisms:

1. **Fugue attention** — multiple voices following the same grammatical
   theme at different temporal offsets, producing emergent harmony.
2. **Strange-loop attention** — self-referential layers that fold back
   on themselves, letting the model reason about its own reasoning.
3. **Temporal attention** — bidirectional attention constrained by
   grammar rules, enabling principled prediction and reconstruction.

These are stacked in ``GLMLayer`` blocks with inter-layer skip
connections (tangled hierarchy), then topped with prediction heads.

All maths use only the Python standard library.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .embeddings import EmbeddingSpace, SymbolInfo, _zeros, _randn, _add, _scale, _dot, _layer_norm, _cosine
from .attention import (
    FugueAttention,
    StrangeLoopAttention,
    TemporalAttention,
    _randn_matrix,
    _matvec,
    _softmax,
    _concat,
)

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class GLMConfig:
    """Configuration for the Grammar Language Model.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of all internal representations.  Kept small
        (64) because the model's power comes from structure, not size.
    num_heads : int
        Number of attention heads / fugue "voices".  Each voice
        attends at a different temporal offset and derivation depth.
    num_layers : int
        Number of stacked GLMLayer blocks.  Each layer can reference
        previous layers' attention weights (strange loops).
    vocab_size : int
        Total number of distinct symbols across all substrates.
    num_substrates : int
        Number of substrate domains (linguistic, chemical, etc.).
    temporal_horizon : int
        How many steps forward/backward the temporal prediction heads
        can reach.
    loop_depth : int
        How many layers back the strange-loop attention can reach.
        Higher values allow deeper self-referential reasoning but
        increase computation.
    max_seq_len : int
        Maximum input sequence length.
    ffn_hidden_dim : int
        Hidden dimension of the feed-forward network inside each layer.
        Defaults to 4× embedding_dim (standard Transformer ratio).
    dropout_rate : float
        Not used in inference, but stored for training configuration.
    """

    embedding_dim: int = 64
    num_heads: int = 4
    num_layers: int = 3
    vocab_size: int = 256
    num_substrates: int = 8
    temporal_horizon: int = 8
    loop_depth: int = 2
    max_seq_len: int = 512
    ffn_hidden_dim: int = 0  # 0 → auto = 4 * embedding_dim
    dropout_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.ffn_hidden_dim == 0:
            self.ffn_hidden_dim = 4 * self.embedding_dim


# ═══════════════════════════════════════════════════════════════════════
# Feed-forward network (grammar-aware)
# ═══════════════════════════════════════════════════════════════════════


class GrammarFFN:
    """Two-layer feed-forward network with GELU activation.

    Standard Transformer FFN but we call it "grammar-aware" because it
    operates on representations that already carry grammatical, substrate,
    and temporal information from the embedding and attention layers.
    The FFN's job is to *transform* those representations — combining
    features in nonlinear ways that capture rule-application effects
    (e.g., "if a symbol has role X at depth D, its representation should
    shift towards …").

    Parameters
    ----------
    input_dim : int
        Input (and output) dimensionality.
    hidden_dim : int
        Width of the hidden layer.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W1 = _randn_matrix(hidden_dim, input_dim)
        self.b1 = _zeros(hidden_dim)
        self.W2 = _randn_matrix(input_dim, hidden_dim)
        self.b2 = _zeros(input_dim)

    def forward(self, x: List[float]) -> List[float]:
        """FFN forward: x → LayerNorm(x + W2 · GELU(W1 · x + b1) + b2)."""
        h = _add(_matvec(self.W1, x), self.b1)
        # GELU approximation: x * sigmoid(1.702 * x)
        h = [
            v * (1.0 / (1.0 + math.exp(-1.702 * v)))
            for v in h
        ]
        out = _add(_matvec(self.W2, h), self.b2)
        return _layer_norm(_add(x, out))  # residual + norm

    @property
    def parameters(self) -> List[List[float]]:
        params: List[List[float]] = []
        params.extend(self.W1)
        params.append(self.b1)
        params.extend(self.W2)
        params.append(self.b2)
        return params


# ═══════════════════════════════════════════════════════════════════════
# GLMLayer — a single block in the tangled hierarchy
# ═══════════════════════════════════════════════════════════════════════


class GLMLayer:
    """A single layer of the Grammar Language Model.

    Combines three attention mechanisms and a feed-forward network:

    1. **FugueAttention** — multi-voice structural attention.
    2. **StrangeLoopAttention** — self-referential meta-attention,
       receiving attention weights from up to ``loop_depth`` previous
       layers.
    3. **GrammarFFN** — nonlinear feature transformation.

    The TemporalAttention is applied at the model level (spanning all
    layers) rather than inside each layer, since it needs the full
    context to do bidirectional prediction.

    Parameters
    ----------
    config : GLMConfig
        Model configuration.
    layer_idx : int
        This layer's index in the stack (used for strange-loop wiring).
    """

    def __init__(self, config: GLMConfig, layer_idx: int = 0) -> None:
        self.config = config
        self.layer_idx = layer_idx

        self.fugue = FugueAttention(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            max_offset=min(4, config.temporal_horizon),
        )
        self.strange_loop = StrangeLoopAttention(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            loop_gate_init=0.1,
        )
        self.ffn = GrammarFFN(config.embedding_dim, config.ffn_hidden_dim)

        # Inter-layer skip connection weight (learnable)
        self.skip_alpha: float = 0.1

    def forward(
        self,
        seq: List[List[float]],
        prev_attn_weights: Optional[List[List[List[float]]]] = None,
        derivation_depths: Optional[List[float]] = None,
        mask: Optional[List[List[bool]]] = None,
    ) -> List[List[float]]:
        """Process a sequence through this layer.

        Parameters
        ----------
        seq : list[list[float]]
            Input ``[seq_len, embedding_dim]``.
        prev_attn_weights : attention weights from a previous layer
            for the strange loop.
        derivation_depths : per-position derivation depths.
        mask : attention mask.

        Returns
        -------
        list[list[float]]
            Transformed sequence ``[seq_len, embedding_dim]``.
        """
        # 1. Fugue attention — voices harmonise or counterpoint
        x = self.fugue.forward(seq, derivation_depths=derivation_depths, mask=mask)

        # 2. Strange-loop attention — reason about reasoning
        x = self.strange_loop.forward(x, prev_attn_weights=prev_attn_weights, mask=mask)

        # 3. Feed-forward — nonlinear grammar transformation
        x = [self.ffn.forward(xi) for xi in x]

        # 4. Inter-layer residual (tangled hierarchy: current output
        #    blended with original input, so information can flow
        #    "backwards" in the hierarchy even without explicit loops)
        output: List[List[float]] = []
        for xi, si in zip(x, seq):
            blended = _add(
                _scale(xi, 1.0 - self.skip_alpha),
                _scale(si, self.skip_alpha),
            )
            output.append(_layer_norm(blended))

        return output

    @property
    def attn_weights(self) -> List[List[List[float]]]:
        """Return the most recent attention weights from this layer's
        fugue attention (used as input to the next layer's strange loop)."""
        return self.fugue.last_attn_weights

    @property
    def harmony(self) -> float:
        """Inter-head harmony score from the last forward pass."""
        return self.fugue.last_harmony

    @property
    def loop_gate(self) -> float:
        """Current strange-loop gate value."""
        return self.strange_loop.last_gate_value

    @property
    def parameters(self) -> List[List[float]]:
        params: List[List[float]] = []
        params.extend(self.fugue.parameters)
        params.extend(self.strange_loop.parameters)
        params.extend(self.ffn.parameters)
        params.append([self.skip_alpha])
        return params


# ═══════════════════════════════════════════════════════════════════════
# GrammarLanguageModel — the full architecture
# ═══════════════════════════════════════════════════════════════════════


class GrammarLanguageModel:
    """The Grammar Language Model.

    A small, structurally-powerful model that learns grammar-aware
    representations and uses them for temporal prediction, pattern
    detection, and superforecasting.

    Architecture:
    * **Embedding layer** — fuses grammar, substrate, and temporal
      embeddings into a unified representation.
    * **GLMLayer stack** — alternating fugue attention, strange-loop
      attention, and grammar-aware FFN, with inter-layer skip
      connections forming a tangled hierarchy.
    * **Temporal attention** — applied after the layer stack, providing
      bidirectional (forward-predictive, backward-reconstructive)
      context.
    * **Prediction heads**:
      - Forward head — predicts next symbols.
      - Backward head — reconstructs previous symbols.
      - Superforecasting head — combines grammar structure and context
        to make predictions that go beyond surface statistics.

    Parameters
    ----------
    config : GLMConfig
        Model configuration.
    """

    def __init__(self, config: Optional[GLMConfig] = None) -> None:
        self.config = config or GLMConfig()
        c = self.config

        # ── Embedding ─────────────────────────────────────────────────
        self.embedding = EmbeddingSpace(
            vocab_size=c.vocab_size,
            embedding_dim=c.embedding_dim,
            num_substrates=c.num_substrates,
            max_seq_len=c.max_seq_len,
        )

        # ── Layer stack ───────────────────────────────────────────────
        self.layers: List[GLMLayer] = [
            GLMLayer(c, layer_idx=i) for i in range(c.num_layers)
        ]

        # ── Temporal attention (model-level) ──────────────────────────
        self.temporal_attn = TemporalAttention(
            embedding_dim=c.embedding_dim,
            num_heads=c.num_heads,
        )

        # ── Prediction heads ─────────────────────────────────────────
        # Forward prediction: hidden → vocab logits
        self.forward_head = _randn_matrix(c.vocab_size, c.embedding_dim)
        self.forward_bias = _zeros(c.vocab_size)

        # Backward reconstruction: hidden → vocab logits
        self.backward_head = _randn_matrix(c.vocab_size, c.embedding_dim)
        self.backward_bias = _zeros(c.vocab_size)

        # Superforecasting head: takes concatenated fwd + bwd hidden +
        # harmony features → vocab logits
        sf_input_dim = c.embedding_dim * 2 + c.num_layers  # fwd + bwd + harmony per layer
        self.sf_project = _randn_matrix(c.embedding_dim, sf_input_dim)
        self.sf_head = _randn_matrix(c.vocab_size, c.embedding_dim)
        self.sf_bias = _zeros(c.vocab_size)

        # ── State ────────────────────────────────────────────────────
        self._last_hidden: List[List[float]] = []
        self._last_fwd: List[List[float]] = []
        self._last_bwd: List[List[float]] = []
        self._layer_harmonies: List[float] = []

    # ─── Forward pass ─────────────────────────────────────────────────

    def forward(
        self,
        input_ids: List[int],
        substrate_ids: Optional[List[int]] = None,
        derivation_depths: Optional[List[int]] = None,
        epochs: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Standard forward pass.

        Parameters
        ----------
        input_ids : list[int]
            Symbol identifiers for the input sequence.
        substrate_ids : list[int], optional
            Per-position substrate identifiers.
        derivation_depths : list[int], optional
            Per-position derivation depths.
        epochs : list[int], optional
            Per-position historical epoch indices.

        Returns
        -------
        dict
            ``forward_logits``: ``[seq_len, vocab_size]``
            ``backward_logits``: ``[seq_len, vocab_size]``
            ``hidden``: ``[seq_len, embedding_dim]``
            ``harmonies``: ``[num_layers]``
            ``loop_gates``: ``[num_layers]``
        """
        seq_len = len(input_ids)
        if substrate_ids is None:
            substrate_ids = [0] * seq_len
        if derivation_depths is None:
            derivation_depths = [0] * seq_len
        if epochs is None:
            epochs = [0] * seq_len

        # 1. Embed
        symbols = [
            SymbolInfo(
                symbol_id=sid,
                substrate_id=sub,
                position=i,
                derivation_depth=dep,
                epoch=epo,
            )
            for i, (sid, sub, dep, epo) in enumerate(
                zip(input_ids, substrate_ids, derivation_depths, epochs)
            )
        ]
        x = self.embedding.embed_sequence(symbols)

        # 2. Layer stack with strange-loop wiring
        float_depths = [float(d) for d in derivation_depths]
        prev_attn: Optional[List[List[List[float]]]] = None
        harmonies: List[float] = []
        loop_gates: List[float] = []

        for layer_idx, layer in enumerate(self.layers):
            # Strange loop: feed previous layer's attention weights
            # (within loop_depth range)
            loop_input = None
            if layer_idx > 0:
                # Use the attention from the nearest eligible previous layer
                source_idx = max(0, layer_idx - self.config.loop_depth)
                loop_input = self.layers[source_idx].attn_weights

            x = layer.forward(
                x,
                prev_attn_weights=loop_input,
                derivation_depths=float_depths,
            )
            harmonies.append(layer.harmony)
            loop_gates.append(layer.loop_gate)

        self._last_hidden = x
        self._layer_harmonies = harmonies

        # 3. Temporal attention (bidirectional)
        fwd_out, bwd_out = self.temporal_attn.forward(x, symbol_ids=input_ids)
        self._last_fwd = fwd_out
        self._last_bwd = bwd_out

        # 4. Prediction heads
        fwd_logits: List[List[float]] = []
        bwd_logits: List[List[float]] = []
        for i in range(seq_len):
            fl = _add(_matvec(self.forward_head, fwd_out[i]), self.forward_bias)
            bl = _add(_matvec(self.backward_head, bwd_out[i]), self.backward_bias)
            fwd_logits.append(fl)
            bwd_logits.append(bl)

        return {
            "forward_logits": fwd_logits,
            "backward_logits": bwd_logits,
            "hidden": x,
            "forward_hidden": fwd_out,
            "backward_hidden": bwd_out,
            "harmonies": harmonies,
            "loop_gates": loop_gates,
        }

    # ─── Prediction & forecasting ────────────────────────────────────

    def predict_future(
        self,
        sequence: List[int],
        grammar_rules: Optional[List[Tuple[int, List[int]]]] = None,
        horizon: int = 4,
    ) -> List[List[float]]:
        """Use grammar + learned patterns to forecast future symbols.

        Given an input sequence, predict the next ``horizon`` symbols
        by iteratively feeding the most likely prediction back as input.
        Grammar rules (if provided) are used to bias the temporal
        attention, favouring transitions that respect the grammar.

        Parameters
        ----------
        sequence : list[int]
            Input symbol ids.
        grammar_rules : list[tuple[int, list[int]]], optional
            Grammar rules as ``(lhs, body)`` pairs.  Used to register
            transition biases in the temporal attention.
        horizon : int
            How many future steps to predict.

        Returns
        -------
        list[list[float]]
            Predicted probability distributions over the vocabulary,
            one per future step.  Shape: ``[horizon, vocab_size]``.
        """
        if grammar_rules:
            self._register_grammar_transitions(grammar_rules)

        predictions: List[List[float]] = []
        current = list(sequence)

        for _ in range(horizon):
            result = self.forward(current)
            # Take the last position's forward logits
            last_logits = result["forward_logits"][-1]
            probs = _softmax(last_logits)
            predictions.append(probs)

            # Greedy: feed back the most likely symbol
            next_id = probs.index(max(probs))
            current.append(next_id)
            # Keep sequence bounded
            if len(current) > self.config.max_seq_len:
                current = current[-self.config.max_seq_len:]

        return predictions

    def reconstruct_past(
        self,
        sequence: List[int],
        grammar_rules: Optional[List[Tuple[int, List[int]]]] = None,
        depth: int = 4,
    ) -> List[List[float]]:
        """Work backward to reconstruct the origin/derivation history.

        Given a sequence, predict what preceded it by using backward
        attention constrained by grammar rules.

        Parameters
        ----------
        sequence : list[int]
            Current observed sequence.
        grammar_rules : list[tuple[int, list[int]]], optional
            Grammar rules for transition biasing.
        depth : int
            How many steps backward to reconstruct.

        Returns
        -------
        list[list[float]]
            Predicted distributions for past symbols (most recent
            past first).  Shape: ``[depth, vocab_size]``.
        """
        if grammar_rules:
            self._register_grammar_transitions(grammar_rules)

        result = self.forward(sequence)
        reconstructions: List[List[float]] = []

        # The backward head at position 0 gives the most likely
        # predecessor.  We iteratively prepend.
        current = list(sequence)

        for _ in range(depth):
            result = self.forward(current)
            first_bwd_logits = result["backward_logits"][0]
            probs = _softmax(first_bwd_logits)
            reconstructions.append(probs)

            prev_id = probs.index(max(probs))
            current = [prev_id] + current
            if len(current) > self.config.max_seq_len:
                current = current[: self.config.max_seq_len]

        return reconstructions

    def superforecast(
        self,
        sequence: List[int],
        context: Optional[Dict[str, Any]] = None,
        grammar_rules: Optional[List[Tuple[int, List[int]]]] = None,
        horizon: int = 4,
    ) -> Dict[str, Any]:
        """Full superforecasting — the crown jewel.

        Superforecasting goes beyond surface-level prediction by
        combining:
        1. Forward prediction (what comes next statistically).
        2. Backward reconstruction (what derivation history implies).
        3. Structural invariants (grammar rules constrain possibilities).
        4. Inter-voice harmony (when fugue voices agree, confidence is
           higher; when they disagree, uncertainty is flagged).
        5. Strange-loop meta-reasoning (the model's confidence in its
           own attention patterns modulates the forecast).

        Parameters
        ----------
        sequence : list[int]
            Input sequence.
        context : dict, optional
            Additional context (e.g., substrate, epoch, domain hints).
        grammar_rules : list[tuple[int, list[int]]], optional
            Grammar rules for structural constraints.
        horizon : int
            Steps to forecast.

        Returns
        -------
        dict
            ``predictions``: ``[horizon, vocab_size]`` — probability
                distributions per future step.
            ``confidence``: ``[horizon]`` — per-step confidence scores
                derived from harmony and loop-gate values.
            ``top_predictions``: ``[horizon]`` — greedy top symbol ids.
            ``harmony_signal``: float — overall model harmony.
            ``loop_depth_used``: float — how much strange-loop the
                model relied on.
        """
        if grammar_rules:
            self._register_grammar_transitions(grammar_rules)

        # Run the full forward pass to get all internal signals
        result = self.forward(sequence)
        harmonies = result["harmonies"]
        loop_gates = result["loop_gates"]

        predictions: List[List[float]] = []
        confidences: List[float] = []
        top_preds: List[int] = []

        current = list(sequence)

        for step in range(horizon):
            result = self.forward(current)

            # Get forward and backward signals at the last position
            fwd_hidden = result["forward_hidden"][-1]
            bwd_hidden = result["backward_hidden"][-1]
            step_harmonies = result["harmonies"]

            # Build superforecasting input: concat fwd + bwd + harmonies
            sf_input = fwd_hidden + bwd_hidden + step_harmonies
            # Pad or truncate to match sf_project input dim
            expected_dim = self.config.embedding_dim * 2 + self.config.num_layers
            if len(sf_input) < expected_dim:
                sf_input.extend([0.0] * (expected_dim - len(sf_input)))
            else:
                sf_input = sf_input[:expected_dim]

            # Project → logits
            projected = _matvec(self.sf_project, sf_input)
            # GELU activation
            projected = [
                v * (1.0 / (1.0 + math.exp(-1.702 * v)))
                for v in projected
            ]
            logits = _add(_matvec(self.sf_head, projected), self.sf_bias)
            probs = _softmax(logits)
            predictions.append(probs)

            # Confidence from harmony and loop gates
            avg_harmony = sum(step_harmonies) / max(len(step_harmonies), 1)
            avg_gate = sum(result["loop_gates"]) / max(len(result["loop_gates"]), 1)
            # High harmony + moderate gate = high confidence
            confidence = avg_harmony * (0.5 + 0.5 * avg_gate)
            # Decay confidence over longer horizons
            confidence *= math.exp(-0.1 * step)
            confidences.append(confidence)

            top_id = probs.index(max(probs))
            top_preds.append(top_id)
            current.append(top_id)
            if len(current) > self.config.max_seq_len:
                current = current[-self.config.max_seq_len:]

        return {
            "predictions": predictions,
            "confidence": confidences,
            "top_predictions": top_preds,
            "harmony_signal": sum(harmonies) / max(len(harmonies), 1),
            "loop_depth_used": sum(loop_gates) / max(len(loop_gates), 1),
        }

    # ─── Pattern detection ───────────────────────────────────────────

    def detect_patterns(
        self, sequence: List[int]
    ) -> Dict[str, Any]:
        """Find grammatical patterns in the input sequence.

        Analyses the model's internal representations to detect:
        * **Repetitions** — symbols or sub-sequences that recur.
        * **Structural motifs** — attention patterns indicating
          grammatical regularity (high harmony regions).
        * **Strange loops** — self-referential patterns where the
          model's attention feeds back on itself (high loop-gate
          regions).

        Parameters
        ----------
        sequence : list[int]
            Input symbol sequence.

        Returns
        -------
        dict
            ``repetitions``: list of ``(symbol, count)`` pairs.
            ``harmony_profile``: per-layer harmony scores.
            ``attention_entropy``: per-position entropy of attention
                (low entropy = focused = pattern detected).
            ``motifs``: list of ``(start, end, score)`` for high-harmony
                sub-regions.
        """
        result = self.forward(sequence)

        # Repetition analysis
        counts: Dict[int, int] = {}
        for s in sequence:
            counts[s] = counts.get(s, 0) + 1
        repetitions = sorted(counts.items(), key=lambda t: t[1], reverse=True)

        # Attention entropy per position (from last layer's fugue attention)
        last_layer = self.layers[-1]
        attn_entropy: List[float] = []
        if last_layer.attn_weights:
            head_0_attn = last_layer.attn_weights[0]
            for row in head_0_attn:
                entropy = 0.0
                for w in row:
                    if w > 1e-10:
                        entropy -= w * math.log(w)
                attn_entropy.append(entropy)

        # Motif detection: sliding window harmony
        motifs: List[Tuple[int, int, float]] = []
        window = max(2, len(sequence) // 4)
        if len(sequence) >= window:
            for start in range(len(sequence) - window + 1):
                sub_seq = sequence[start : start + window]
                sub_result = self.forward(sub_seq)
                avg_h = sum(sub_result["harmonies"]) / max(
                    len(sub_result["harmonies"]), 1
                )
                if avg_h > 0.5:
                    motifs.append((start, start + window, avg_h))

        return {
            "repetitions": repetitions,
            "harmony_profile": result["harmonies"],
            "attention_entropy": attn_entropy,
            "motifs": motifs,
        }

    def find_strange_loops(self) -> Dict[str, Any]:
        """Introspect the model's own strange loops.

        Examines the strange-loop attention gates and inter-layer
        attention feedback to identify where the model is engaging
        in self-referential reasoning.

        This is meta-cognition made computational: the model examining
        how it examines things.

        Returns
        -------
        dict
            ``loop_gates``: per-layer gate values (higher = more
                self-referential).
            ``active_loops``: list of ``(layer_from, layer_to, strength)``
                for loops with gate > 0.3.
            ``total_loop_strength``: sum of all active loop strengths.
            ``is_self_referential``: bool — whether any loop is strongly
                active.
        """
        loop_gates = [layer.loop_gate for layer in self.layers]

        active_loops: List[Tuple[int, int, float]] = []
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx > 0:
                source_idx = max(0, layer_idx - self.config.loop_depth)
                strength = layer.loop_gate
                if strength > 0.3:
                    active_loops.append((source_idx, layer_idx, strength))

        total = sum(s for _, _, s in active_loops)

        return {
            "loop_gates": loop_gates,
            "active_loops": active_loops,
            "total_loop_strength": total,
            "is_self_referential": total > 0.5,
        }

    # ─── Helpers ─────────────────────────────────────────────────────

    def _register_grammar_transitions(
        self, rules: List[Tuple[int, List[int]]]
    ) -> None:
        """Register grammar rules as transition biases in the temporal
        attention layer."""
        for lhs, body in rules:
            # Register forward transitions: body[i] → body[i+1]
            for i in range(len(body) - 1):
                self.temporal_attn.register_transition(body[i], body[i + 1], 1.0)
            # Register derivation transitions: lhs → body[0]
            if body:
                self.temporal_attn.register_transition(lhs, body[0], 0.8)
            # Also inform the grammar embedding
            self.embedding.grammar.register_rule(lhs, body)

    @property
    def parameters(self) -> List[List[float]]:
        """All trainable parameters, flattened for the trainer."""
        params: List[List[float]] = []
        params.extend(self.embedding.parameters)
        for layer in self.layers:
            params.extend(layer.parameters)
        params.extend(self.temporal_attn.parameters)
        params.extend(self.forward_head)
        params.append(self.forward_bias)
        params.extend(self.backward_head)
        params.append(self.backward_bias)
        params.extend(self.sf_project)
        params.extend(self.sf_head)
        params.append(self.sf_bias)
        return params

    @property
    def num_parameters(self) -> int:
        """Total number of trainable scalar parameters."""
        return sum(len(p) for p in self.parameters)

    def summary(self) -> str:
        """Human-readable model summary."""
        c = self.config
        lines = [
            "Grammar Language Model",
            "=" * 40,
            f"  embedding_dim:     {c.embedding_dim}",
            f"  num_heads:         {c.num_heads} (fugue voices)",
            f"  num_layers:        {c.num_layers}",
            f"  vocab_size:        {c.vocab_size}",
            f"  num_substrates:    {c.num_substrates}",
            f"  temporal_horizon:  {c.temporal_horizon}",
            f"  loop_depth:        {c.loop_depth}",
            f"  ffn_hidden_dim:    {c.ffn_hidden_dim}",
            f"  max_seq_len:       {c.max_seq_len}",
            f"  total parameters:  {self.num_parameters:,}",
            "=" * 40,
        ]
        return "\n".join(lines)
