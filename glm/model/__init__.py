"""Neural model layer for the Grammar Language Model.

This is where grammar meets neural computation.  The model learns
grammar-aware representations and uses them for temporal prediction
and superforecasting.

Architecture inspired by three pillars from *Gödel, Escher, Bach*:

* **Strange loops** — Self-referential attention layers that fold back on
  themselves, letting the model reason about *how* it reasons.
* **Fugues** — Multiple attention heads as contrapuntal "voices", each
  following the same grammatical theme but offset in time or derivation
  depth, producing emergent harmony when they agree and creative
  counterpoint when they diverge.
* **Tangled hierarchies** — Inter-layer connections that break strict
  feed-forward ordering, enabling the model to reference its own
  intermediate representations at any depth.

The design favours *structural understanding over brute-force
parameters*.  A small model that truly grasps grammar can outperform a
large model that only memorises surfaces.

Modules
───────
    embeddings   Grammar-aware, substrate-aware, temporal embeddings
    attention    Fugue, strange-loop, and temporal attention mechanisms
    glm          The Grammar Language Model itself (config, layers, model)
    trainer      Training loop with multi-objective gradient descent
"""

from .embeddings import (
    GrammarEmbedding,
    SubstrateEmbedding,
    TemporalEmbedding,
    EmbeddingSpace,
)
from .attention import (
    FugueAttention,
    StrangeLoopAttention,
    TemporalAttention,
)
from .glm import (
    GLMConfig,
    GLMLayer,
    GrammarLanguageModel,
)
from .trainer import (
    GLMTrainer,
    TrainingConfig,
)

__all__ = [
    # Embeddings
    "GrammarEmbedding",
    "SubstrateEmbedding",
    "TemporalEmbedding",
    "EmbeddingSpace",
    # Attention
    "FugueAttention",
    "StrangeLoopAttention",
    "TemporalAttention",
    # Model
    "GLMConfig",
    "GLMLayer",
    "GrammarLanguageModel",
    # Training
    "GLMTrainer",
    "TrainingConfig",
]
