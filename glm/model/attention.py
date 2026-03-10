"""Fugue-inspired attention mechanisms for the Grammar Language Model.

Bach's fugues demonstrate that extraordinary complexity can arise from
simple rules applied contrapuntally: multiple *voices* (attention heads)
following the same *subject* (grammatical theme) but entering at
different times and transposed to different keys.  When voices agree
(consonance), confidence is high.  When they disagree *productively*
(dissonance resolving to consonance), new structural insight emerges.

Three attention mechanisms embody this philosophy:

* **FugueAttention** — Multi-head attention where heads share the same
  structural "theme" (grammar-derived Q/K biases) but attend at
  different temporal offsets and derivation depths, like voices entering
  a fugue at different beats and in different registers.

* **StrangeLoopAttention** — Attention that attends to its *own*
  attention weights from previous layers.  A strange loop in the sense
  of Hofstadter: a tangled hierarchy where the model reasons about
  how it is reasoning.  Computationally: the attention distribution
  from layer *l-1* is projected and used as an additional key/value
  signal in layer *l*.

* **TemporalAttention** — Bidirectional temporal attention that
  simultaneously attends to past context (for reconstruction) and
  future context (for prediction), using grammar-derived constraints
  to limit the space of plausible continuations in each direction.

All maths use only the Python standard library.
Vectors are plain Python lists of floats.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ═══════════════════════════════════════════════════════════════════════
# Vector / matrix helpers (self-contained, no numpy)
# ═══════════════════════════════════════════════════════════════════════


def _zeros(n: int) -> List[float]:
    return [0.0] * n


def _randn(n: int, scale: float = 0.02) -> List[float]:
    return [random.gauss(0.0, scale) for _ in range(n)]


def _randn_matrix(rows: int, cols: int, scale: float = 0.02) -> List[List[float]]:
    return [_randn(cols, scale) for _ in range(rows)]


def _add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def _scale(a: List[float], s: float) -> List[float]:
    return [x * s for x in a]


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _matvec(mat: List[List[float]], vec: List[float]) -> List[float]:
    """Matrix-vector product: mat @ vec."""
    return [_dot(row, vec) for row in mat]


def _outer(a: List[float], b: List[float]) -> List[List[float]]:
    """Outer product a ⊗ b."""
    return [[ai * bj for bj in b] for ai in a]


def _softmax(logits: List[float]) -> List[float]:
    """Numerically stable softmax."""
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    if s < 1e-12:
        n = len(logits)
        return [1.0 / n] * n
    return [e / s for e in exps]


def _layer_norm(v: List[float], eps: float = 1e-5) -> List[float]:
    n = len(v)
    if n == 0:
        return v
    mean = sum(v) / n
    var = sum((x - mean) ** 2 for x in v) / n
    inv_std = 1.0 / math.sqrt(var + eps)
    return [(x - mean) * inv_std for x in v]


def _concat(vectors: List[List[float]]) -> List[float]:
    out: List[float] = []
    for v in vectors:
        out.extend(v)
    return out


def _split(vec: List[float], n_parts: int) -> List[List[float]]:
    """Split a vector into n_parts equal-length sub-vectors."""
    k = len(vec) // n_parts
    return [vec[i * k : (i + 1) * k] for i in range(n_parts)]


# ═══════════════════════════════════════════════════════════════════════
# FugueAttention
# ═══════════════════════════════════════════════════════════════════════


class FugueAttention:
    """Multi-head attention modelled on a musical fugue.

    Each head is a *voice* that processes the same input but with:
    * A different **temporal offset** — voice *i* attends primarily to
      positions shifted by ``offset_i`` steps, like a fugue voice
      entering later.
    * A different **derivation-depth bias** — voice *i* favours tokens
      at a particular depth in the derivation tree, like a voice
      transposed to a different register.
    * Shared **structural Q/K/V projections** — the same grammatical
      "subject" (weight matrices) for all voices, ensuring coherence.

    After each head produces its own weighted sum, the outputs are
    concatenated and projected back to ``embedding_dim``.  A
    **harmony score** is computed from inter-head agreement — high
    harmony means the voices concur and confidence should be high;
    low harmony signals creative counterpoint where new patterns may
    be emerging.

    Parameters
    ----------
    embedding_dim : int
        Input / output dimensionality.
    num_heads : int
        Number of "voices" (must divide embedding_dim).
    max_offset : int
        Maximum temporal offset across heads.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 4,
        max_offset: int = 4,
    ) -> None:
        assert embedding_dim % num_heads == 0, (
            f"embedding_dim ({embedding_dim}) must be divisible by "
            f"num_heads ({num_heads})"
        )
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Shared Q, K, V projections (all heads use the same weights,
        # but slice different dimensions — structural unity).
        self.W_q = _randn_matrix(embedding_dim, embedding_dim)
        self.W_k = _randn_matrix(embedding_dim, embedding_dim)
        self.W_v = _randn_matrix(embedding_dim, embedding_dim)
        self.W_o = _randn_matrix(embedding_dim, embedding_dim)

        # Per-head temporal offsets and depth biases
        self.offsets = [
            int(i * max_offset / max(num_heads - 1, 1))
            for i in range(num_heads)
        ]
        self.depth_biases = [
            float(i) / num_heads for i in range(num_heads)
        ]

        # Last attention weights and harmony score (for introspection)
        self.last_attn_weights: List[List[List[float]]] = []
        self.last_harmony: float = 0.0

    # ── forward ───────────────────────────────────────────────────────

    def forward(
        self,
        seq: List[List[float]],
        derivation_depths: Optional[List[float]] = None,
        mask: Optional[List[List[bool]]] = None,
    ) -> List[List[float]]:
        """Run fugue attention over a sequence.

        Parameters
        ----------
        seq : list[list[float]]
            Input sequence of shape ``[seq_len, embedding_dim]``.
        derivation_depths : list[float], optional
            Per-position derivation depths for depth-bias computation.
        mask : list[list[bool]], optional
            ``mask[i][j] = True`` means position *i* may attend to *j*.

        Returns
        -------
        list[list[float]]
            Output sequence, same shape as input.
        """
        seq_len = len(seq)
        if seq_len == 0:
            return []

        if derivation_depths is None:
            derivation_depths = [0.0] * seq_len

        # Project Q, K, V for the whole sequence
        Q = [_matvec(self.W_q, x) for x in seq]
        K = [_matvec(self.W_k, x) for x in seq]
        V = [_matvec(self.W_v, x) for x in seq]

        scale = 1.0 / math.sqrt(self.head_dim)

        head_outputs: List[List[List[float]]] = []  # [num_heads][seq_len][head_dim]
        all_attn_weights: List[List[List[float]]] = []  # [num_heads][seq_len][seq_len]

        for h in range(self.num_heads):
            lo = h * self.head_dim
            hi = lo + self.head_dim
            offset = self.offsets[h]
            depth_bias = self.depth_biases[h]

            head_out: List[List[float]] = []
            head_attn: List[List[float]] = []

            for i in range(seq_len):
                q_i = Q[i][lo:hi]
                logits: List[float] = []

                for j in range(seq_len):
                    # Shifted index: this voice "hears" the sequence
                    # shifted by its offset — the fugue entrance delay.
                    j_shifted = (j + offset) % seq_len
                    k_j = K[j_shifted][lo:hi]

                    score = _dot(q_i, k_j) * scale

                    # Depth-bias: boost attention to positions whose
                    # derivation depth matches this voice's register.
                    depth_diff = abs(
                        derivation_depths[j_shifted] - depth_bias
                    )
                    score -= 0.1 * depth_diff

                    # Apply mask
                    if mask is not None and not mask[i][j]:
                        score = -1e9

                    logits.append(score)

                weights = _softmax(logits)
                head_attn.append(weights)

                # Weighted sum of values (also shifted)
                out = _zeros(self.head_dim)
                for j, w in enumerate(weights):
                    j_shifted = (j + offset) % seq_len
                    v_j = V[j_shifted][lo:hi]
                    out = _add(out, _scale(v_j, w))

                head_out.append(out)

            head_outputs.append(head_out)
            all_attn_weights.append(head_attn)

        # Store for introspection / strange loop feedback
        self.last_attn_weights = all_attn_weights

        # Concatenate heads and project
        output: List[List[float]] = []
        for i in range(seq_len):
            concat = _concat([head_outputs[h][i] for h in range(self.num_heads)])
            projected = _matvec(self.W_o, concat)
            # Residual connection + layer norm
            output.append(_layer_norm(_add(seq[i], projected)))

        # Compute harmony score: average pairwise agreement between heads
        self.last_harmony = self._compute_harmony(all_attn_weights)

        return output

    def _compute_harmony(
        self, all_weights: List[List[List[float]]]
    ) -> float:
        """Measure inter-head agreement (harmony).

        High harmony → voices agree → high confidence.
        Low harmony → counterpoint → new patterns emerging.

        Uses Jensen-Shannon-style comparison: average KL divergence
        between each head's attention distribution and the mean.
        Returns 1.0 for perfect agreement, 0.0 for maximal disagreement.
        """
        if len(all_weights) < 2 or not all_weights[0]:
            return 1.0

        num_heads = len(all_weights)
        seq_len = len(all_weights[0])
        total_kl = 0.0
        count = 0

        for i in range(seq_len):
            # Compute mean distribution
            mean_dist = _zeros(len(all_weights[0][i]))
            for h in range(num_heads):
                mean_dist = _add(mean_dist, all_weights[h][i])
            mean_dist = _scale(mean_dist, 1.0 / num_heads)

            # KL divergence of each head from mean
            for h in range(num_heads):
                for p, q in zip(all_weights[h][i], mean_dist):
                    if p > 1e-10 and q > 1e-10:
                        total_kl += p * math.log(p / q)
                count += 1

        avg_kl = total_kl / max(count, 1)
        # Map to [0, 1]: harmony = exp(-kl)
        return math.exp(-avg_kl)

    @property
    def parameters(self) -> List[List[float]]:
        params: List[List[float]] = []
        params.extend(self.W_q)
        params.extend(self.W_k)
        params.extend(self.W_v)
        params.extend(self.W_o)
        return params


# ═══════════════════════════════════════════════════════════════════════
# StrangeLoopAttention
# ═══════════════════════════════════════════════════════════════════════


class StrangeLoopAttention:
    """Self-referential attention — the strange loop made computational.

    In *Gödel, Escher, Bach*, Hofstadter describes strange loops as
    systems that, by moving through a hierarchy of levels, unexpectedly
    arrive back at the starting level.  StrangeLoopAttention implements
    this: the attention weights computed by a previous layer are
    projected into key/value space and fed *back* as additional context,
    allowing the current layer to reason about *how* the previous layer
    was reasoning.

    Concretely:

    1. Take the attention weight matrix ``A`` from layer *l-1*
       (shape ``[seq_len, seq_len]``).
    2. Project each row of ``A`` (an attention distribution) through a
       learned linear map into key and value vectors of dimension
       ``head_dim``.
    3. Concatenate these "meta-keys" and "meta-values" with the normal
       keys and values, so the current layer can attend both to the
       input content *and* to the previous layer's attention patterns.
    4. A gating mechanism controls how much the strange loop influences
       the output — too much self-reference leads to fixation; too
       little loses the benefit.

    Parameters
    ----------
    embedding_dim : int
        Model dimensionality.
    num_heads : int
        Number of attention heads.
    loop_gate_init : float
        Initial value of the loop gate (0 = ignore loop, 1 = full loop).
        Starting small prevents early-training instability.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 4,
        loop_gate_init: float = 0.1,
    ) -> None:
        assert embedding_dim % num_heads == 0
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Standard Q/K/V projections
        self.W_q = _randn_matrix(embedding_dim, embedding_dim)
        self.W_k = _randn_matrix(embedding_dim, embedding_dim)
        self.W_v = _randn_matrix(embedding_dim, embedding_dim)
        self.W_o = _randn_matrix(embedding_dim, embedding_dim)

        # Strange loop projections: attention_distribution → key/value
        # Each attention distribution has seq_len entries, but we use a
        # small fixed projection from a padded/truncated representation.
        # We project from num_heads (one weight per head's attn) to head_dim.
        self.W_loop_k = _randn_matrix(self.head_dim, self.head_dim)
        self.W_loop_v = _randn_matrix(self.head_dim, self.head_dim)

        # Gating scalar (learnable) — sigmoid applied at runtime
        self.loop_gate_logit: float = math.log(
            loop_gate_init / (1.0 - loop_gate_init + 1e-8)
        )

        # Storage for introspection
        self.last_attn_weights: List[List[List[float]]] = []
        self.last_gate_value: float = 0.0

    def forward(
        self,
        seq: List[List[float]],
        prev_attn_weights: Optional[List[List[List[float]]]] = None,
        mask: Optional[List[List[bool]]] = None,
    ) -> List[List[float]]:
        """Forward pass with optional strange-loop feedback.

        Parameters
        ----------
        seq : list[list[float]]
            Input sequence ``[seq_len, embedding_dim]``.
        prev_attn_weights : list[list[list[float]]], optional
            Attention weights from the previous layer, shape
            ``[num_heads, seq_len, seq_len]``.  If ``None``, the
            strange loop is inactive (first layer).
        mask : list[list[bool]], optional
            Attention mask.

        Returns
        -------
        list[list[float]]
            Output sequence ``[seq_len, embedding_dim]``.
        """
        seq_len = len(seq)
        if seq_len == 0:
            return []

        gate = 1.0 / (1.0 + math.exp(-self.loop_gate_logit))  # sigmoid
        self.last_gate_value = gate

        Q = [_matvec(self.W_q, x) for x in seq]
        K = [_matvec(self.W_k, x) for x in seq]
        V = [_matvec(self.W_v, x) for x in seq]

        scale = 1.0 / math.sqrt(self.head_dim)

        # Prepare loop keys/values if previous attention is available
        loop_keys: Optional[List[List[List[float]]]] = None  # [head][seq_len][head_dim]
        loop_vals: Optional[List[List[List[float]]]] = None

        if prev_attn_weights is not None and gate > 0.01:
            loop_keys = []
            loop_vals = []
            prev_heads = len(prev_attn_weights)
            for h in range(self.num_heads):
                hk: List[List[float]] = []
                hv: List[List[float]] = []
                for i in range(seq_len):
                    # Gather this position's attention weights across heads
                    # from the previous layer, then project.
                    meta_vec = _zeros(self.head_dim)
                    for ph in range(min(prev_heads, self.head_dim)):
                        if i < len(prev_attn_weights[ph]):
                            # Summarise: mean attention weight for position i
                            row = prev_attn_weights[ph][i]
                            meta_val = sum(row) / max(len(row), 1)
                            if ph < self.head_dim:
                                meta_vec[ph] = meta_val
                    hk.append(_matvec(self.W_loop_k, meta_vec))
                    hv.append(_matvec(self.W_loop_v, meta_vec))
                loop_keys.append(hk)
                loop_vals.append(hv)

        all_attn: List[List[List[float]]] = []
        head_outputs: List[List[List[float]]] = []

        for h in range(self.num_heads):
            lo = h * self.head_dim
            hi = lo + self.head_dim

            head_out: List[List[float]] = []
            head_attn: List[List[float]] = []

            for i in range(seq_len):
                q_i = Q[i][lo:hi]
                logits: List[float] = []

                # Standard content attention
                for j in range(seq_len):
                    k_j = K[j][lo:hi]
                    score = _dot(q_i, k_j) * scale
                    if mask is not None and not mask[i][j]:
                        score = -1e9
                    logits.append(score)

                # Strange loop: additional "meta" keys
                loop_logits: List[float] = []
                if loop_keys is not None:
                    for j in range(seq_len):
                        lk = loop_keys[h][j]
                        score = _dot(q_i, lk) * scale * gate
                        loop_logits.append(score)

                # Combine: interleave content and loop logits
                if loop_logits:
                    # Merge by treating loop positions as extra columns
                    combined_logits = logits + [l * gate for l in loop_logits]
                    combined_weights = _softmax(combined_logits)
                    content_weights = combined_weights[:seq_len]
                    loop_weights = combined_weights[seq_len:]
                else:
                    content_weights = _softmax(logits)
                    loop_weights = []

                head_attn.append(content_weights)

                # Weighted sum
                out = _zeros(self.head_dim)
                for j, w in enumerate(content_weights):
                    v_j = V[j][lo:hi]
                    out = _add(out, _scale(v_j, w))

                # Add loop values
                if loop_vals is not None and loop_weights:
                    for j, w in enumerate(loop_weights):
                        lv = loop_vals[h][j]
                        out = _add(out, _scale(lv, w))

                head_out.append(out)

            head_outputs.append(head_out)
            all_attn.append(head_attn)

        self.last_attn_weights = all_attn

        # Concat heads, project, residual + norm
        output: List[List[float]] = []
        for i in range(seq_len):
            concat = _concat([head_outputs[h][i] for h in range(self.num_heads)])
            projected = _matvec(self.W_o, concat)
            output.append(_layer_norm(_add(seq[i], projected)))

        return output

    @property
    def parameters(self) -> List[List[float]]:
        params: List[List[float]] = []
        params.extend(self.W_q)
        params.extend(self.W_k)
        params.extend(self.W_v)
        params.extend(self.W_o)
        params.extend(self.W_loop_k)
        params.extend(self.W_loop_v)
        params.append([self.loop_gate_logit])
        return params


# ═══════════════════════════════════════════════════════════════════════
# TemporalAttention
# ═══════════════════════════════════════════════════════════════════════


class TemporalAttention:
    """Bidirectional temporal attention for prediction and reconstruction.

    Unlike standard causal or bidirectional attention, TemporalAttention
    explicitly models *two temporal directions*:

    * **Forward (predictive)** — given the past, attend to plausible
      future continuations constrained by grammar rules.
    * **Backward (reconstructive)** — given the present, attend to
      plausible past origins constrained by grammar rules.

    Each direction has its own Q/K/V projections.  The outputs are
    combined via a learned gate that balances reconstruction vs.
    prediction depending on the task.

    Grammar constraints enter as soft biases on the attention logits:
    transitions that violate known grammar rules receive a penalty,
    while transitions that follow rules receive a bonus.  This is how
    the model's predictions go beyond surface statistics — they respect
    the *structural invariants* encoded in the grammar.

    Parameters
    ----------
    embedding_dim : int
        Model dimensionality.
    num_heads : int
        Number of attention heads (shared across both directions).
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 4,
    ) -> None:
        assert embedding_dim % num_heads == 0
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Forward (predictive) projections
        self.W_q_fwd = _randn_matrix(embedding_dim, embedding_dim)
        self.W_k_fwd = _randn_matrix(embedding_dim, embedding_dim)
        self.W_v_fwd = _randn_matrix(embedding_dim, embedding_dim)

        # Backward (reconstructive) projections
        self.W_q_bwd = _randn_matrix(embedding_dim, embedding_dim)
        self.W_k_bwd = _randn_matrix(embedding_dim, embedding_dim)
        self.W_v_bwd = _randn_matrix(embedding_dim, embedding_dim)

        # Output projection (shared)
        self.W_o = _randn_matrix(embedding_dim, embedding_dim)

        # Direction gate logit (learnable): sigmoid → balance fwd/bwd
        self.direction_gate_logit: float = 0.0  # starts balanced

        # Grammar-rule transition bias table:
        # Maps (from_symbol, to_symbol) → bias float.
        # Populated externally via ``register_transition``.
        self._transition_bias: Dict[Tuple[int, int], float] = {}

        # Last attention weights for introspection
        self.last_fwd_weights: List[List[List[float]]] = []
        self.last_bwd_weights: List[List[List[float]]] = []

    def register_transition(
        self,
        from_sym: int,
        to_sym: int,
        bias: float = 1.0,
    ) -> None:
        """Register a grammar-derived transition bias.

        Positive bias = this transition is grammatically plausible.
        Negative bias = this transition is implausible.
        """
        self._transition_bias[(from_sym, to_sym)] = bias

    def forward(
        self,
        seq: List[List[float]],
        symbol_ids: Optional[List[int]] = None,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Bidirectional temporal attention.

        Parameters
        ----------
        seq : list[list[float]]
            Input sequence ``[seq_len, embedding_dim]``.
        symbol_ids : list[int], optional
            Symbol identifiers per position, used to look up
            grammar-derived transition biases.

        Returns
        -------
        tuple[list[list[float]], list[list[float]]]
            ``(forward_output, backward_output)`` — each of shape
            ``[seq_len, embedding_dim]``.  The caller can combine them
            as needed (e.g., sum, gate, concatenate).
        """
        seq_len = len(seq)
        if seq_len == 0:
            return [], []

        gate = 1.0 / (1.0 + math.exp(-self.direction_gate_logit))
        scale = 1.0 / math.sqrt(self.head_dim)

        # Forward pass: causal mask (i can attend to j <= i)
        fwd_out = self._attend(
            seq,
            self.W_q_fwd, self.W_k_fwd, self.W_v_fwd,
            scale,
            causal_direction="forward",
            symbol_ids=symbol_ids,
        )
        self.last_fwd_weights = self._last_head_attn

        # Backward pass: reverse causal mask (i can attend to j >= i)
        bwd_out = self._attend(
            seq,
            self.W_q_bwd, self.W_k_bwd, self.W_v_bwd,
            scale,
            causal_direction="backward",
            symbol_ids=symbol_ids,
        )
        self.last_bwd_weights = self._last_head_attn

        # Combine via gated residual
        combined_fwd: List[List[float]] = []
        combined_bwd: List[List[float]] = []
        for i in range(seq_len):
            f = _add(seq[i], _scale(fwd_out[i], gate))
            b = _add(seq[i], _scale(bwd_out[i], 1.0 - gate))
            combined_fwd.append(_layer_norm(f))
            combined_bwd.append(_layer_norm(b))

        return combined_fwd, combined_bwd

    # ── internals ─────────────────────────────────────────────────────

    _last_head_attn: List[List[List[float]]]  # set during _attend

    def _attend(
        self,
        seq: List[List[float]],
        W_q: List[List[float]],
        W_k: List[List[float]],
        W_v: List[List[float]],
        scale: float,
        causal_direction: str,
        symbol_ids: Optional[List[int]],
    ) -> List[List[float]]:
        seq_len = len(seq)
        Q = [_matvec(W_q, x) for x in seq]
        K = [_matvec(W_k, x) for x in seq]
        V = [_matvec(W_v, x) for x in seq]

        all_head_attn: List[List[List[float]]] = []
        head_outputs: List[List[List[float]]] = []

        for h in range(self.num_heads):
            lo = h * self.head_dim
            hi = lo + self.head_dim
            h_out: List[List[float]] = []
            h_attn: List[List[float]] = []

            for i in range(seq_len):
                q_i = Q[i][lo:hi]
                logits: List[float] = []
                for j in range(seq_len):
                    # Causal mask
                    if causal_direction == "forward" and j > i:
                        logits.append(-1e9)
                        continue
                    if causal_direction == "backward" and j < i:
                        logits.append(-1e9)
                        continue

                    k_j = K[j][lo:hi]
                    score = _dot(q_i, k_j) * scale

                    # Grammar transition bias
                    if symbol_ids is not None:
                        pair = (symbol_ids[i], symbol_ids[j])
                        bias = self._transition_bias.get(pair, 0.0)
                        score += bias * 0.5  # tempered

                    logits.append(score)

                weights = _softmax(logits)
                h_attn.append(weights)

                out = _zeros(self.head_dim)
                for j, w in enumerate(weights):
                    v_j = V[j][lo:hi]
                    out = _add(out, _scale(v_j, w))
                h_out.append(out)

            head_outputs.append(h_out)
            all_head_attn.append(h_attn)

        self._last_head_attn = all_head_attn

        # Concat + project
        output: List[List[float]] = []
        for i in range(seq_len):
            concat = _concat([head_outputs[h][i] for h in range(self.num_heads)])
            projected = _matvec(self.W_o, concat)
            output.append(projected)

        return output

    @property
    def parameters(self) -> List[List[float]]:
        params: List[List[float]] = []
        params.extend(self.W_q_fwd)
        params.extend(self.W_k_fwd)
        params.extend(self.W_v_fwd)
        params.extend(self.W_q_bwd)
        params.extend(self.W_k_bwd)
        params.extend(self.W_v_bwd)
        params.extend(self.W_o)
        params.append([self.direction_gate_logit])
        return params
