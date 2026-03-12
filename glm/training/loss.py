"""Loss functions for grammar-based training of the GLM.

The GLM is trained with four simultaneous objectives, each targeting a
different facet of grammatical understanding:

1. **DerivationLoss** -- Cross-entropy on next-symbol prediction in
   grammar derivations.  The bread-and-butter language-modelling
   objective, but applied to grammatical derivation sequences rather
   than raw text.

2. **ReconstructionLoss** -- How well backward derivation recovers the
   input from the output.  Forces the model to learn reversible
   representations, echoing the bidirectional nature of grammar rules.

3. **IsomorphismLoss** -- Contrastive loss that pushes same-structure
   pairs close together in embedding space and different-structure
   pairs apart.  This is the fugue objective: same theme, different
   instruments.

4. **LoopLoss** -- Rewards detecting genuine strange loops and
   penalises false positives.  Teaches the model to recognise
   self-referential structure.

All losses operate on Tensors from the autograd module and support
full backpropagation.
"""

from __future__ import annotations

import math
from typing import List, Optional

from .autograd import Tensor, cross_entropy


# =======================================================================
# DerivationLoss
# =======================================================================

class DerivationLoss:
    """Cross-entropy loss on next-symbol prediction in grammar derivations.

    Given a sequence of forward logits (one per position) and a sequence
    of target symbol IDs, computes the average cross-entropy loss.

    This is the standard autoregressive language-modelling objective
    adapted for grammar derivation sequences: the model predicts the
    next symbol in a derivation chain, with the loss driving it to
    assign high probability to the correct next step.
    """

    def __init__(self, label_smoothing: float = 0.0) -> None:
        self.label_smoothing = label_smoothing

    def __call__(
        self,
        logits: List[Tensor],
        targets: List[int],
    ) -> Tensor:
        """Compute derivation loss.

        Parameters
        ----------
        logits : list of Tensor, each shape (vocab_size,)
            Per-position forward logits from the model.
        targets : list of int
            Target symbol IDs, one per position.

        Returns
        -------
        Tensor
            Scalar loss with gradient support.
        """
        if not logits or not targets:
            return Tensor([0.0], shape=(1,))

        n = min(len(logits), len(targets))
        total = Tensor([0.0], shape=(1,))

        for i in range(n):
            vocab_size = logits[i].shape[0]
            if 0 <= targets[i] < vocab_size:
                ce = cross_entropy(logits[i], targets[i])
                total = total + ce

        # Average
        if n > 0:
            total = total * (1.0 / n)

        return total


# =======================================================================
# ReconstructionLoss
# =======================================================================

class ReconstructionLoss:
    """Measures how well backward derivation recovers input from output.

    Applies cross-entropy loss to backward logits, where the targets
    are the original input symbols.  This forces the model to learn
    representations that support bidirectional derivation -- a key
    property of grammar rules.

    Optionally weighted by a ``reconstruction_weight`` that can
    emphasise or de-emphasise reconstruction relative to forward
    prediction.
    """

    def __init__(self, weight: float = 0.5) -> None:
        self.weight = weight

    def __call__(
        self,
        backward_logits: List[Tensor],
        original_ids: List[int],
    ) -> Tensor:
        """Compute reconstruction loss.

        Parameters
        ----------
        backward_logits : list of Tensor, each shape (vocab_size,)
            Per-position backward logits from the model.
        original_ids : list of int
            The original input symbols to reconstruct.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        if not backward_logits or not original_ids:
            return Tensor([0.0], shape=(1,))

        n = min(len(backward_logits), len(original_ids))
        total = Tensor([0.0], shape=(1,))

        for i in range(n):
            vocab_size = backward_logits[i].shape[0]
            if 0 <= original_ids[i] < vocab_size:
                ce = cross_entropy(backward_logits[i], original_ids[i])
                total = total + ce

        if n > 0:
            total = total * (self.weight / n)

        return total


# =======================================================================
# IsomorphismLoss
# =======================================================================

class IsomorphismLoss:
    """Contrastive loss for cross-domain isomorphism detection.

    Given pairs of hidden representations:
    - **Positive pairs**: same grammatical structure in different domains.
      These should embed close together (high cosine similarity).
    - **Negative pairs**: different structures.
      These should embed far apart (low cosine similarity).

    Uses a margin-based contrastive formulation:
        L_pos = (1 - cos(a, b))^2
        L_neg = max(0, cos(a, b) - margin)^2

    Parameters
    ----------
    margin : float
        Negative pairs should have cosine similarity below this margin.
    """

    def __init__(self, margin: float = 0.2) -> None:
        self.margin = margin

    def __call__(
        self,
        anchor: Tensor,
        positive: Tensor,
        negative: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute isomorphism contrastive loss.

        Parameters
        ----------
        anchor : Tensor, shape (D,)
            Hidden representation of the anchor example.
        positive : Tensor, shape (D,)
            Hidden representation from the same structural class.
        negative : Tensor, shape (D,), optional
            Hidden representation from a different structural class.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        # Positive loss: push anchor and positive together
        pos_cos = _cosine_similarity(anchor, positive)
        # (1 - cos)^2
        one = Tensor([1.0], shape=(1,))
        pos_diff = one - pos_cos
        pos_loss = pos_diff * pos_diff

        if negative is None:
            return pos_loss

        # Negative loss: push anchor and negative apart
        neg_cos = _cosine_similarity(anchor, negative)
        # max(0, cos - margin)^2
        margin_t = Tensor([self.margin], shape=(1,))
        neg_excess = neg_cos - margin_t
        # ReLU-style clamping
        neg_val = max(0.0, neg_excess.data[0])
        neg_loss = Tensor([neg_val * neg_val], shape=(1,))
        # Attach backward manually for the clamp
        neg_loss._prev = (neg_excess,)

        def _backward_clamp() -> None:
            if neg_excess.grad is not None and neg_val > 0:
                neg_excess.grad[0] += neg_loss.grad[0] * 2.0 * neg_val

        neg_loss._backward = _backward_clamp

        return pos_loss + neg_loss


def _cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
    """Cosine similarity between two 1-D tensors, returned as a scalar Tensor."""
    dot_val = sum(x * y for x, y in zip(a.data, b.data))
    norm_a = math.sqrt(sum(x * x for x in a.data) + 1e-12)
    norm_b = math.sqrt(sum(x * x for x in b.data) + 1e-12)
    cos_val = dot_val / (norm_a * norm_b)

    out = Tensor([cos_val], shape=(1,))
    out._prev = (a, b)

    def _backward() -> None:
        g = out.grad[0] if out.grad else 0.0
        if g == 0.0:
            return
        # d(cos)/da_i = (b_i / (||a|| * ||b||)) - cos * (a_i / ||a||^2)
        if a.grad is not None:
            for i in range(len(a.data)):
                da = (b.data[i] / (norm_a * norm_b)) - cos_val * (a.data[i] / (norm_a * norm_a))
                a.grad[i] += g * da
        if b.grad is not None:
            for i in range(len(b.data)):
                db = (a.data[i] / (norm_a * norm_b)) - cos_val * (b.data[i] / (norm_b * norm_b))
                b.grad[i] += g * db

    out._backward = _backward
    return out


# =======================================================================
# LoopLoss
# =======================================================================

class LoopLoss:
    """Binary classification loss for strange-loop detection.

    The model's loop-gate signals (from StrangeLoopAttention) should
    fire strongly when the input genuinely contains a self-referential
    pattern and remain quiet otherwise.

    Uses binary cross-entropy:
        L = -[y * log(p) + (1 - y) * log(1 - p)]

    where y is the ground-truth label (1 = has loop, 0 = no loop)
    and p is the model's loop-strength prediction (sigmoid of the
    total loop gate value).

    Parameters
    ----------
    threshold : float
        Offset for the sigmoid mapping of raw loop strength.
    false_positive_penalty : float
        Extra weight on false positives (predicting a loop that
        does not exist is worse than missing one, since it leads
        to spurious self-referential reasoning).
    """

    def __init__(
        self,
        threshold: float = 0.5,
        false_positive_penalty: float = 1.5,
    ) -> None:
        self.threshold = threshold
        self.fp_penalty = false_positive_penalty

    def __call__(
        self,
        loop_strength: Tensor,
        has_loop: bool,
    ) -> Tensor:
        """Compute loop detection loss.

        Parameters
        ----------
        loop_strength : Tensor, shape (1,) or scalar-like
            The model's raw loop-strength signal (sum of loop gates).
        has_loop : bool
            Ground-truth label.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        raw = loop_strength.data[0]
        # Sigmoid mapping
        sig_input = raw - self.threshold
        if sig_input > 500:
            pred = 1.0 - 1e-7
        elif sig_input < -500:
            pred = 1e-7
        else:
            pred = 1.0 / (1.0 + math.exp(-sig_input))

        target = 1.0 if has_loop else 0.0
        eps = 1e-7

        # Binary cross-entropy
        bce = -(target * math.log(pred + eps) + (1.0 - target) * math.log(1.0 - pred + eps))

        # Extra penalty for false positives
        if not has_loop and pred > 0.5:
            bce *= self.fp_penalty

        out = Tensor([bce], shape=(1,))
        out._prev = (loop_strength,)

        def _backward() -> None:
            if loop_strength.grad is not None:
                # d(BCE)/d(raw) = d(BCE)/d(pred) * d(pred)/d(raw)
                # d(pred)/d(raw) = pred * (1 - pred)   [sigmoid derivative]
                d_pred = pred * (1.0 - pred)
                if target > 0.5:
                    d_bce = -1.0 / (pred + eps)
                else:
                    d_bce = 1.0 / (1.0 - pred + eps)
                    if pred > 0.5:
                        d_bce *= self.fp_penalty
                loop_strength.grad[0] += out.grad[0] * d_bce * d_pred

        out._backward = _backward
        return out


# =======================================================================
# CombinedLoss -- weighted sum of all objectives
# =======================================================================

class CombinedLoss:
    """Weighted combination of all GLM training objectives.

    Parameters
    ----------
    derivation_weight : float
        Weight for forward derivation prediction loss.
    reconstruction_weight : float
        Weight for backward reconstruction loss.
    isomorphism_weight : float
        Weight for cross-domain isomorphism loss.
    loop_weight : float
        Weight for strange-loop detection loss.
    """

    def __init__(
        self,
        derivation_weight: float = 1.0,
        reconstruction_weight: float = 0.5,
        isomorphism_weight: float = 0.3,
        loop_weight: float = 0.2,
    ) -> None:
        self.derivation = DerivationLoss()
        self.reconstruction = ReconstructionLoss(weight=1.0)  # weighting handled here
        self.isomorphism = IsomorphismLoss()
        self.loop = LoopLoss()

        self.w_deriv = derivation_weight
        self.w_recon = reconstruction_weight
        self.w_iso = isomorphism_weight
        self.w_loop = loop_weight
