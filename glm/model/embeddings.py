"""Grammar-aware embeddings for the Grammar Language Model.

In a fugue, the *subject* is not merely a sequence of notes — it carries
within it the harmonic implications, the rhythmic skeleton, and the
contrapuntal potential that the entire piece will exploit.  A grammar-aware
embedding does the same: it encodes a symbol not just as an opaque vector
but as a *grammatical object* — carrying information about its role in
production rules, how deep in a derivation tree it sits, which substrate
it belongs to, and where it falls in temporal/evolutionary time.

Three complementary embeddings are fused into a single space:

* **GrammarEmbedding** — encodes rule roles and derivation history.
* **SubstrateEmbedding** — encodes the domain/medium a symbol lives in
  (phonemes, molecules, codons, tokens …), enabling isomorphism detection
  across substrates.
* **TemporalEmbedding** — encodes *two* kinds of time: sequence position
  (surface order) and derivation depth (structural depth in the parse
  tree / rule application chain), plus an optional historical epoch axis
  for etymological / evolutionary reasoning.

All maths use only the Python standard library (``math``, ``random``).
Vectors are plain Python lists of floats.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ═══════════════════════════════════════════════════════════════════════
# Lightweight vector arithmetic — no external dependencies
# ═══════════════════════════════════════════════════════════════════════


def _zeros(n: int) -> List[float]:
    return [0.0] * n


def _randn(n: int, scale: float = 0.02) -> List[float]:
    """Xavier-style small random init."""
    return [random.gauss(0.0, scale) for _ in range(n)]


def _add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def _sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def _scale(a: List[float], s: float) -> List[float]:
    return [x * s for x in a]


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: List[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _cosine(a: List[float], b: List[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return _dot(a, b) / (na * nb)


def _elementwise_mul(a: List[float], b: List[float]) -> List[float]:
    return [x * y for x, y in zip(a, b)]


def _layer_norm(v: List[float], eps: float = 1e-5) -> List[float]:
    """Layer normalisation over a single vector."""
    n = len(v)
    mean = sum(v) / n
    var = sum((x - mean) ** 2 for x in v) / n
    inv_std = 1.0 / math.sqrt(var + eps)
    return [(x - mean) * inv_std for x in v]


# ═══════════════════════════════════════════════════════════════════════
# GrammarEmbedding
# ═══════════════════════════════════════════════════════════════════════


class GrammarEmbedding:
    """Embeds symbols as *grammatical objects*.

    Each symbol's embedding encodes:
    * **Lexical identity** — a learned vector unique to the symbol.
    * **Rule-role encoding** — whether the symbol appears as head,
      left-hand side, or body element in known production rules, and
      how many rules it participates in.
    * **Derivation-depth signal** — a soft indicator of how "deep" in
      a derivation tree the symbol typically appears (terminals are
      deep; the start symbol is shallow).

    These signals are fused via element-wise addition (the way positional
    and token embeddings combine in a Transformer) and then
    layer-normalised.

    Parameters
    ----------
    vocab_size : int
        Number of distinct symbols.
    embedding_dim : int
        Dimensionality of the embedding vectors.
    """

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Learnable lookup table: symbol → vector
        self.table: List[List[float]] = [
            _randn(embedding_dim) for _ in range(vocab_size)
        ]

        # Rule-role projection weights (3 binary features → embedding_dim)
        # Features: is_lhs, is_body, is_terminal
        self.rule_role_weights: List[List[float]] = [
            _randn(embedding_dim) for _ in range(3)
        ]

        # Derivation-depth scalar projection
        self.depth_weight: List[float] = _randn(embedding_dim)

        # Metadata: rule participation counts per symbol
        self._rule_counts: Dict[int, Dict[str, int]] = {}

    # ── public API ────────────────────────────────────────────────────

    def register_rule(
        self,
        lhs_id: int,
        body_ids: List[int],
        *,
        terminal_ids: Optional[List[int]] = None,
    ) -> None:
        """Inform the embedding about a grammar rule so role encodings
        can be computed.

        Parameters
        ----------
        lhs_id : int
            Symbol id of the left-hand side.
        body_ids : list[int]
            Symbol ids in the rule body.
        terminal_ids : list[int], optional
            Which body ids are terminals (if known).
        """
        terminal_set = set(terminal_ids or [])
        self._ensure_meta(lhs_id)
        self._rule_counts[lhs_id]["lhs"] += 1

        for sid in body_ids:
            self._ensure_meta(sid)
            self._rule_counts[sid]["body"] += 1
            if sid in terminal_set:
                self._rule_counts[sid]["terminal"] = 1

    def embed(
        self,
        symbol_id: int,
        derivation_depth: float = 0.0,
    ) -> List[float]:
        """Return the grammar-aware embedding for *symbol_id*.

        Parameters
        ----------
        symbol_id : int
            Index into the vocabulary.
        derivation_depth : float
            How deep in the current derivation tree this occurrence sits.
            Normalised internally via tanh so the magnitude stays bounded.

        Returns
        -------
        list[float]
            The fused, layer-normalised embedding vector.
        """
        if symbol_id < 0 or symbol_id >= self.vocab_size:
            raise ValueError(
                f"symbol_id {symbol_id} out of range [0, {self.vocab_size})"
            )

        # 1. Lexical identity
        vec = list(self.table[symbol_id])

        # 2. Rule-role encoding
        meta = self._rule_counts.get(symbol_id, {})
        features = [
            1.0 if meta.get("lhs", 0) > 0 else 0.0,
            1.0 if meta.get("body", 0) > 0 else 0.0,
            1.0 if meta.get("terminal", 0) > 0 else 0.0,
        ]
        for feat_val, weight_vec in zip(features, self.rule_role_weights):
            if feat_val > 0:
                vec = _add(vec, _scale(weight_vec, feat_val))

        # 3. Derivation depth (bounded via tanh)
        depth_signal = math.tanh(derivation_depth / 10.0)
        vec = _add(vec, _scale(self.depth_weight, depth_signal))

        return _layer_norm(vec)

    # ── internals ─────────────────────────────────────────────────────

    def _ensure_meta(self, sid: int) -> None:
        if sid not in self._rule_counts:
            self._rule_counts[sid] = {"lhs": 0, "body": 0, "terminal": 0}

    @property
    def parameters(self) -> List[List[float]]:
        """All trainable parameter vectors (flat list for the trainer)."""
        params: List[List[float]] = []
        params.extend(self.table)
        params.extend(self.rule_role_weights)
        params.append(self.depth_weight)
        return params


# ═══════════════════════════════════════════════════════════════════════
# SubstrateEmbedding
# ═══════════════════════════════════════════════════════════════════════


class SubstrateEmbedding:
    """Embeds the *substrate* (domain) a symbol belongs to.

    In the GLM every symbol lives on a substrate — phonemes on the
    phonological substrate, atoms on the chemical substrate, codons on
    the biological substrate, tokens on the computational substrate.

    By giving each substrate its own embedding, the model can learn to
    detect *isomorphisms* across domains — structural parallels where
    the grammar rules of one substrate mirror those of another.  This is
    the mathematical heart of the fugue: the same theme (grammar)
    played on different instruments (substrates).

    Parameters
    ----------
    num_substrates : int
        Number of distinct substrate domains.
    embedding_dim : int
        Dimensionality (shared with GrammarEmbedding).
    """

    def __init__(self, num_substrates: int, embedding_dim: int) -> None:
        self.num_substrates = num_substrates
        self.embedding_dim = embedding_dim

        self.table: List[List[float]] = [
            _randn(embedding_dim) for _ in range(num_substrates)
        ]

        # Cross-substrate alignment matrix: maps substrate i → shared
        # isomorphism space.  Stored as list of projection vectors.
        self.alignment_proj: List[List[float]] = [
            _randn(embedding_dim) for _ in range(num_substrates)
        ]

    def embed(self, substrate_id: int) -> List[float]:
        """Return the substrate embedding."""
        if substrate_id < 0 or substrate_id >= self.num_substrates:
            raise ValueError(
                f"substrate_id {substrate_id} out of range "
                f"[0, {self.num_substrates})"
            )
        return list(self.table[substrate_id])

    def project_to_shared_space(self, substrate_id: int) -> List[float]:
        """Project a substrate embedding into the shared isomorphism
        space where cross-domain similarities can be measured.

        This is the key operation for fugue-style cross-domain
        reasoning: different substrates mapped into a common harmonic
        space.
        """
        raw = self.embed(substrate_id)
        proj = self.alignment_proj[substrate_id]
        # Element-wise gating: alignment_proj acts as a soft mask
        # selecting the dimensions most relevant for cross-domain
        # comparison.
        gated = _elementwise_mul(raw, [math.tanh(p) for p in proj])
        return _layer_norm(gated)

    def substrate_similarity(self, id_a: int, id_b: int) -> float:
        """Cosine similarity between two substrates in the shared
        isomorphism space."""
        return _cosine(
            self.project_to_shared_space(id_a),
            self.project_to_shared_space(id_b),
        )

    @property
    def parameters(self) -> List[List[float]]:
        params: List[List[float]] = []
        params.extend(self.table)
        params.extend(self.alignment_proj)
        return params


# ═══════════════════════════════════════════════════════════════════════
# TemporalEmbedding
# ═══════════════════════════════════════════════════════════════════════


class TemporalEmbedding:
    """Encodes *time* in three complementary senses.

    1. **Sequence position** — where in the surface sequence a symbol
       occurs (analogous to standard positional encoding).
    2. **Derivation depth** — how many rule applications deep in the
       derivation tree this occurrence sits.  This is *structural* time,
       orthogonal to surface order.
    3. **Historical epoch** — for etymological / evolutionary reasoning,
       an optional axis encoding when in real-world historical time a
       form existed (Proto-Indo-European → Latin → Old French → English).

    Position is encoded with sinusoidal functions (à la Vaswani et al.)
    but extended to all three axes, each using a different frequency
    base so the model can disentangle them.

    Parameters
    ----------
    embedding_dim : int
        Must be divisible by 2 (pairs of sin/cos).
    max_seq_len : int
        Maximum supported sequence length.
    max_depth : int
        Maximum derivation depth.
    max_epoch : int
        Maximum historical epoch index.
    """

    def __init__(
        self,
        embedding_dim: int,
        max_seq_len: int = 512,
        max_depth: int = 64,
        max_epoch: int = 32,
    ) -> None:
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even")
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.max_depth = max_depth
        self.max_epoch = max_epoch

        # Dimension budget: split equally among three axes
        # Each axis gets embedding_dim // 3 dims (remainder → position)
        third = embedding_dim // 3
        self._pos_dim = embedding_dim - 2 * third  # absorbs remainder
        self._depth_dim = third
        self._epoch_dim = third

        # Pre-compute sinusoidal tables for each axis
        self._pos_table = self._build_sinusoidal(
            max_seq_len, self._pos_dim, base=10000.0
        )
        self._depth_table = self._build_sinusoidal(
            max_depth, self._depth_dim, base=1000.0
        )
        self._epoch_table = self._build_sinusoidal(
            max_epoch, self._epoch_dim, base=500.0
        )

    # ── public API ────────────────────────────────────────────────────

    def embed(
        self,
        position: int,
        derivation_depth: int = 0,
        epoch: int = 0,
    ) -> List[float]:
        """Return the temporal embedding for the given coordinates.

        The three axes are concatenated (not summed) so downstream
        attention can learn which temporal axis matters for each query.

        Parameters
        ----------
        position : int
            Surface sequence position (0-indexed).
        derivation_depth : int
            Structural depth in the derivation tree.
        epoch : int
            Historical epoch index (0 = most ancient).

        Returns
        -------
        list[float]
            Temporal embedding of length ``embedding_dim``.
        """
        pos_vec = self._lookup(self._pos_table, position, self._pos_dim)
        dep_vec = self._lookup(
            self._depth_table, derivation_depth, self._depth_dim
        )
        epo_vec = self._lookup(self._epoch_table, epoch, self._epoch_dim)
        return pos_vec + dep_vec + epo_vec  # list concatenation

    # ── internals ─────────────────────────────────────────────────────

    @staticmethod
    def _build_sinusoidal(
        max_len: int, dim: int, base: float
    ) -> List[List[float]]:
        """Pre-compute a [max_len × dim] sinusoidal table.

        Even indices get sin, odd indices get cos, with geometrically
        spaced frequencies (the classic Transformer recipe, but with a
        configurable base to differentiate axes).
        """
        table: List[List[float]] = []
        half = dim // 2 if dim > 0 else 0
        for pos in range(max_len):
            row: List[float] = []
            for i in range(half):
                freq = 1.0 / (base ** (2 * i / max(dim, 1)))
                row.append(math.sin(pos * freq))
                row.append(math.cos(pos * freq))
            # If dim is odd, pad with zero
            if dim % 2 == 1:
                row.append(0.0)
            table.append(row[:dim])
        return table

    @staticmethod
    def _lookup(
        table: List[List[float]], index: int, dim: int
    ) -> List[float]:
        if index < len(table):
            return list(table[index])
        # Extrapolate beyond pre-computed range using the formula
        half = dim // 2 if dim > 0 else 0
        row: List[float] = []
        for i in range(half):
            freq = 1.0 / (10000.0 ** (2 * i / max(dim, 1)))
            row.append(math.sin(index * freq))
            row.append(math.cos(index * freq))
        if dim % 2 == 1:
            row.append(0.0)
        return row[:dim]

    @property
    def parameters(self) -> List[List[float]]:
        """Sinusoidal tables are fixed (not trained), so return empty."""
        return []


# ═══════════════════════════════════════════════════════════════════════
# EmbeddingSpace — the combined embedding manifold
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SymbolInfo:
    """Minimal metadata attached to a symbol for embedding purposes."""

    symbol_id: int
    substrate_id: int = 0
    position: int = 0
    derivation_depth: int = 0
    epoch: int = 0
    label: str = ""


class EmbeddingSpace:
    """The unified embedding space that fuses grammar, substrate, and
    temporal information into a single vector per symbol occurrence.

    The fusion strategy is element-wise addition followed by layer
    normalisation — the same approach that combines token + position
    embeddings in a Transformer, extended to three axes.

    The space also provides geometric utilities:

    * ``similarity(a, b)`` — cosine similarity between two embeddings.
    * ``find_isomorphisms(domain_a, domain_b)`` — discover structural
      parallels between two substrate domains by comparing their
      symbol embeddings in the shared isomorphism space.

    Parameters
    ----------
    vocab_size : int
        Number of distinct symbols across all substrates.
    embedding_dim : int
        Shared dimensionality for all embedding components.
    num_substrates : int
        Number of substrate domains.
    max_seq_len : int
        Maximum sequence length for temporal encoding.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_substrates: int = 8,
        max_seq_len: int = 512,
    ) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.grammar = GrammarEmbedding(vocab_size, embedding_dim)
        self.substrate = SubstrateEmbedding(num_substrates, embedding_dim)
        self.temporal = TemporalEmbedding(embedding_dim, max_seq_len=max_seq_len)

        # Fusion projection: optional learned linear that mixes the
        # three signals before normalisation.  Initialised to identity-
        # like so the default behaviour is pure addition.
        self.fusion_weights: List[float] = [1.0, 1.0, 1.0]

    # ── core operations ───────────────────────────────────────────────

    def embed(self, symbol: SymbolInfo) -> List[float]:
        """Produce the full embedding for a symbol occurrence.

        This is the main entry point: given a ``SymbolInfo`` carrying
        identity, substrate, position, and depth, return the fused
        vector in the unified embedding space.
        """
        g = self.grammar.embed(symbol.symbol_id, float(symbol.derivation_depth))
        s = self.substrate.embed(symbol.substrate_id)
        t = self.temporal.embed(
            symbol.position, symbol.derivation_depth, symbol.epoch
        )

        # Weighted fusion
        fused = _add(
            _add(
                _scale(g, self.fusion_weights[0]),
                _scale(s, self.fusion_weights[1]),
            ),
            _scale(t, self.fusion_weights[2]),
        )
        return _layer_norm(fused)

    def embed_sequence(
        self, symbols: Sequence[SymbolInfo]
    ) -> List[List[float]]:
        """Embed an entire sequence, returning a list of vectors."""
        return [self.embed(sym) for sym in symbols]

    def similarity(self, a: List[float], b: List[float]) -> float:
        """Cosine similarity between two embedding vectors."""
        return _cosine(a, b)

    def find_isomorphisms(
        self,
        domain_a_ids: List[int],
        domain_b_ids: List[int],
        substrate_a: int,
        substrate_b: int,
        threshold: float = 0.5,
    ) -> List[Tuple[int, int, float]]:
        """Discover cross-domain isomorphisms.

        Compares symbols from *domain_a* to symbols from *domain_b* in
        the shared isomorphism space.  Returns pairs whose cosine
        similarity exceeds *threshold*, sorted by similarity descending.

        This is the computational analogue of noticing that
        ``NP → Det N`` in syntax mirrors ``H₂O → H-O-H`` in chemistry:
        structurally analogous productions on different substrates.

        Parameters
        ----------
        domain_a_ids, domain_b_ids : list[int]
            Symbol ids belonging to each domain.
        substrate_a, substrate_b : int
            Substrate indices for each domain.
        threshold : float
            Minimum cosine similarity to report.

        Returns
        -------
        list[tuple[int, int, float]]
            ``(symbol_a, symbol_b, similarity)`` triples.
        """
        # Embed each symbol with its substrate, at position 0
        def _emb(sid: int, sub: int) -> List[float]:
            info = SymbolInfo(symbol_id=sid, substrate_id=sub)
            return self.embed(info)

        results: List[Tuple[int, int, float]] = []
        for a_id in domain_a_ids:
            va = _emb(a_id, substrate_a)
            for b_id in domain_b_ids:
                vb = _emb(b_id, substrate_b)
                sim = _cosine(va, vb)
                if sim >= threshold:
                    results.append((a_id, b_id, sim))

        results.sort(key=lambda t: t[2], reverse=True)
        return results

    @property
    def parameters(self) -> List[List[float]]:
        """All trainable parameters from sub-embeddings."""
        params: List[List[float]] = []
        params.extend(self.grammar.parameters)
        params.extend(self.substrate.parameters)
        params.extend(self.temporal.parameters)
        # fusion_weights as a single-element list
        params.append(self.fusion_weights)
        return params
