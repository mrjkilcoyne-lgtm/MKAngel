"""Training loop for the Grammar Language Model.

The GLM is trained with *five* simultaneous objectives, each
corresponding to a different facet of grammatical understanding:

1. **Forward prediction** — predict the next symbol given context,
   constrained by grammar.  The bread-and-butter language-modelling
   objective, but grammar-aware.
2. **Backward reconstruction** — predict the previous symbol from the
   current one.  Forces the model to learn reversible representations,
   echoing the reversibility of many grammatical transformations.
3. **Grammar induction** — given example derivations, learn to predict
   which rule was applied.  This teaches the model the *grammar itself*
   rather than just surface patterns.
4. **Cross-domain alignment** — given parallel examples from two
   substrates, learn embeddings that place isomorphic structures nearby.
   This is the fugue objective: same theme, different instruments.
5. **Strange-loop detection** — learn to identify self-referential
   patterns in sequences (e.g., quines, palindromes, fixed points of
   grammar rules).

Gradients are computed via finite differences (no autograd dependency).
This keeps the implementation dependency-free while remaining
functionally correct.  The trade-off is speed — fine for a small model.

All maths use only the Python standard library.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .glm import GrammarLanguageModel, GLMConfig
from .attention import _softmax

# ═══════════════════════════════════════════════════════════════════════
# Training configuration
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class TrainingConfig:
    """Configuration for GLM training.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent.  Kept small for stability with
        finite-difference gradients.
    epochs : int
        Number of full passes over the training data.
    batch_size : int
        Number of examples per gradient update.
    grad_epsilon : float
        Perturbation size for finite-difference gradient estimation.
        Smaller = more accurate but more numerically fragile.
    loss_weight_forward : float
        Weight for the forward-prediction loss.
    loss_weight_backward : float
        Weight for the backward-reconstruction loss.
    loss_weight_grammar : float
        Weight for the grammar-induction loss.
    loss_weight_alignment : float
        Weight for the cross-domain alignment loss.
    loss_weight_loop : float
        Weight for the strange-loop detection loss.
    clip_grad_norm : float
        Maximum gradient norm (for stability).
    log_interval : int
        Print training stats every this many steps.
    """

    learning_rate: float = 1e-3
    epochs: int = 10
    batch_size: int = 8
    grad_epsilon: float = 1e-4
    loss_weight_forward: float = 1.0
    loss_weight_backward: float = 0.5
    loss_weight_grammar: float = 0.3
    loss_weight_alignment: float = 0.2
    loss_weight_loop: float = 0.1
    clip_grad_norm: float = 5.0
    log_interval: int = 10


# ═══════════════════════════════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════════════════════════════


def cross_entropy_loss(logits: List[float], target: int) -> float:
    """Cross-entropy loss for a single prediction.

    Parameters
    ----------
    logits : list[float]
        Raw (un-normalised) scores over the vocabulary.
    target : int
        Index of the correct symbol.

    Returns
    -------
    float
        -log P(target), where P = softmax(logits).
    """
    probs = _softmax(logits)
    p = probs[target]
    return -math.log(max(p, 1e-12))


def alignment_loss(
    embeddings_a: List[List[float]],
    embeddings_b: List[List[float]],
) -> float:
    """Cross-domain alignment loss.

    Given paired embeddings from two substrates that should be
    isomorphic, push them towards cosine similarity = 1.

    Uses (1 - cosine_similarity) averaged over pairs.
    """
    if not embeddings_a or not embeddings_b:
        return 0.0

    n = min(len(embeddings_a), len(embeddings_b))
    total = 0.0
    for i in range(n):
        a, b = embeddings_a[i], embeddings_b[i]
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na > 1e-12 and nb > 1e-12:
            cos = dot / (na * nb)
        else:
            cos = 0.0
        total += 1.0 - cos

    return total / n


def loop_detection_loss(
    loop_info: Dict[str, Any],
    has_loop: bool,
) -> float:
    """Strange-loop detection loss.

    Binary cross-entropy: the model should predict high
    ``total_loop_strength`` when the input genuinely contains a
    self-referential pattern, and low otherwise.

    Parameters
    ----------
    loop_info : dict
        Output of ``model.find_strange_loops()``.
    has_loop : bool
        Ground-truth label.
    """
    # Use total_loop_strength as the model's "prediction"
    # Clamp to (0, 1) via sigmoid-like mapping
    raw = loop_info.get("total_loop_strength", 0.0)
    pred = 1.0 / (1.0 + math.exp(-raw + 0.5))

    target = 1.0 if has_loop else 0.0
    # Binary cross-entropy
    eps = 1e-7
    loss = -(
        target * math.log(pred + eps)
        + (1.0 - target) * math.log(1.0 - pred + eps)
    )
    return loss


# ═══════════════════════════════════════════════════════════════════════
# Training data types
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class TrainingExample:
    """A single training example for the GLM.

    Fields
    ------
    input_ids : list[int]
        Input symbol sequence.
    target_ids : list[int], optional
        Target symbol sequence for forward prediction (shifted by 1).
    prev_ids : list[int], optional
        Previous symbols for backward reconstruction.
    rule_lhs : int, optional
        LHS symbol of the grammar rule that produced this example.
    rule_body : list[int], optional
        Body of the rule.
    substrate_id : int
        Which substrate this example belongs to.
    has_strange_loop : bool
        Whether this example contains a self-referential pattern.
    aligned_example : 'TrainingExample', optional
        A parallel example from a different substrate for alignment.
    """

    input_ids: List[int] = field(default_factory=list)
    target_ids: Optional[List[int]] = None
    prev_ids: Optional[List[int]] = None
    rule_lhs: Optional[int] = None
    rule_body: Optional[List[int]] = None
    substrate_id: int = 0
    has_strange_loop: bool = False
    aligned_example: Optional["TrainingExample"] = None


# ═══════════════════════════════════════════════════════════════════════
# GLMTrainer
# ═══════════════════════════════════════════════════════════════════════


class GLMTrainer:
    """Trains the Grammar Language Model.

    Uses finite-difference gradient estimation: for each trainable
    scalar parameter *p*, the gradient is approximated as:

        ∂L/∂p ≈ (L(p + ε) - L(p - ε)) / (2ε)

    This is O(2 * num_params) forward passes per gradient step, which
    is expensive — but for a *small* model (the GLM's design
    philosophy), it is tractable and keeps the code dependency-free.

    To make training practical, we use **parameter sampling**: at each
    step, only a random subset of parameters are perturbed, giving a
    stochastic approximation of the full gradient.  This is akin to
    coordinate descent / SPSA (Simultaneous Perturbation Stochastic
    Approximation).

    Parameters
    ----------
    model : GrammarLanguageModel
        The model to train.
    config : TrainingConfig
        Training hyperparameters.
    """

    def __init__(
        self,
        model: GrammarLanguageModel,
        config: Optional[TrainingConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        self.step_count: int = 0
        self.loss_history: List[float] = []

        # Track per-objective losses
        self.objective_history: Dict[str, List[float]] = {
            "forward": [],
            "backward": [],
            "grammar": [],
            "alignment": [],
            "loop": [],
        }

    # ─── Main training loop ──────────────────────────────────────────

    def train(
        self,
        data: List[TrainingExample],
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> Dict[str, Any]:
        """Run the full training loop.

        Parameters
        ----------
        data : list[TrainingExample]
            Training dataset.
        callback : callable, optional
            Called after each step with ``(step, loss)``.

        Returns
        -------
        dict
            ``final_loss``: loss at the last step.
            ``loss_history``: list of losses per step.
            ``objective_history``: dict of per-objective loss histories.
            ``total_steps``: number of gradient steps taken.
            ``elapsed_seconds``: wall-clock training time.
        """
        cfg = self.config
        start_time = time.time()

        for epoch in range(cfg.epochs):
            # Shuffle data each epoch
            indices = list(range(len(data)))
            random.shuffle(indices)

            for batch_start in range(0, len(data), cfg.batch_size):
                batch_indices = indices[
                    batch_start : batch_start + cfg.batch_size
                ]
                batch = [data[i] for i in batch_indices]

                loss = self._train_step(batch)
                self.loss_history.append(loss)
                self.step_count += 1

                if callback:
                    callback(self.step_count, loss)

                if (
                    cfg.log_interval > 0
                    and self.step_count % cfg.log_interval == 0
                ):
                    elapsed = time.time() - start_time
                    print(
                        f"[step {self.step_count:5d}] "
                        f"loss={loss:.4f}  "
                        f"elapsed={elapsed:.1f}s"
                    )

        elapsed = time.time() - start_time
        return {
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
            "loss_history": list(self.loss_history),
            "objective_history": {
                k: list(v) for k, v in self.objective_history.items()
            },
            "total_steps": self.step_count,
            "elapsed_seconds": elapsed,
        }

    # ─── Single training step ────────────────────────────────────────

    def _train_step(self, batch: List[TrainingExample]) -> float:
        """One gradient-descent step on a mini-batch.

        1. Compute the multi-objective loss.
        2. Estimate gradients via finite differences (SPSA).
        3. Update parameters.
        """
        cfg = self.config

        # Current loss
        loss_center = self._compute_batch_loss(batch)

        # Gradient estimation via Simultaneous Perturbation (SPSA)
        # More efficient than per-parameter finite differences:
        # only 2 forward passes regardless of parameter count.
        params = self.model.parameters
        total_scalars = sum(len(p) for p in params)

        # Generate a random perturbation direction
        delta: List[List[float]] = []
        for p in params:
            d = []
            for _ in range(len(p)):
                d.append(1.0 if random.random() > 0.5 else -1.0)
            delta.append(d)

        # Perturb positively
        self._perturb_params(params, delta, cfg.grad_epsilon)
        loss_plus = self._compute_batch_loss(batch)

        # Perturb negatively (undo +ε, apply -ε)
        self._perturb_params(params, delta, -2.0 * cfg.grad_epsilon)
        loss_minus = self._compute_batch_loss(batch)

        # Restore original params
        self._perturb_params(params, delta, cfg.grad_epsilon)

        # SPSA gradient estimate
        grad_scale = (loss_plus - loss_minus) / (2.0 * cfg.grad_epsilon)

        # Compute gradient norm for clipping
        grad_norm = abs(grad_scale) * math.sqrt(total_scalars)
        if grad_norm > cfg.clip_grad_norm:
            grad_scale *= cfg.clip_grad_norm / grad_norm

        # Update parameters: p -= lr * grad_estimate / delta
        for p_vec, d_vec in zip(params, delta):
            for k in range(len(p_vec)):
                p_vec[k] -= cfg.learning_rate * grad_scale / d_vec[k]

        # Write updated parameter values back into the model
        self._write_params_back(params)

        return loss_center

    # ─── Loss computation ────────────────────────────────────────────

    def _compute_batch_loss(
        self, batch: List[TrainingExample]
    ) -> float:
        """Compute the weighted multi-objective loss over a batch."""
        cfg = self.config
        total_loss = 0.0
        n = len(batch)

        fwd_sum = 0.0
        bwd_sum = 0.0
        gram_sum = 0.0
        align_sum = 0.0
        loop_sum = 0.0

        for ex in batch:
            if not ex.input_ids:
                continue

            result = self.model.forward(ex.input_ids)

            # ── 1. Forward prediction loss ────────────────────────
            if ex.target_ids and cfg.loss_weight_forward > 0:
                fwd_logits = result["forward_logits"]
                fwd_loss = 0.0
                count = 0
                for i, target in enumerate(ex.target_ids):
                    if i < len(fwd_logits) and 0 <= target < len(fwd_logits[i]):
                        fwd_loss += cross_entropy_loss(fwd_logits[i], target)
                        count += 1
                if count > 0:
                    fwd_loss /= count
                fwd_sum += fwd_loss
                total_loss += cfg.loss_weight_forward * fwd_loss

            # ── 2. Backward reconstruction loss ──────────────────
            if ex.prev_ids and cfg.loss_weight_backward > 0:
                bwd_logits = result["backward_logits"]
                bwd_loss = 0.0
                count = 0
                for i, prev_target in enumerate(ex.prev_ids):
                    if i < len(bwd_logits) and 0 <= prev_target < len(bwd_logits[i]):
                        bwd_loss += cross_entropy_loss(bwd_logits[i], prev_target)
                        count += 1
                if count > 0:
                    bwd_loss /= count
                bwd_sum += bwd_loss
                total_loss += cfg.loss_weight_backward * bwd_loss

            # ── 3. Grammar induction loss ────────────────────────
            if (
                ex.rule_lhs is not None
                and ex.rule_body
                and cfg.loss_weight_grammar > 0
            ):
                # The grammar-induction objective: given the rule body
                # as input, the forward head at position 0 should
                # predict the LHS.
                body_result = self.model.forward(ex.rule_body)
                if body_result["forward_logits"]:
                    lhs_logits = body_result["forward_logits"][0]
                    if 0 <= ex.rule_lhs < len(lhs_logits):
                        gram_loss = cross_entropy_loss(lhs_logits, ex.rule_lhs)
                        gram_sum += gram_loss
                        total_loss += cfg.loss_weight_grammar * gram_loss

            # ── 4. Cross-domain alignment loss ───────────────────
            if ex.aligned_example and cfg.loss_weight_alignment > 0:
                hidden_a = result["hidden"]
                aligned_result = self.model.forward(
                    ex.aligned_example.input_ids
                )
                hidden_b = aligned_result["hidden"]
                align_loss = alignment_loss(hidden_a, hidden_b)
                align_sum += align_loss
                total_loss += cfg.loss_weight_alignment * align_loss

            # ── 5. Strange-loop detection loss ───────────────────
            if cfg.loss_weight_loop > 0:
                loop_info = self.model.find_strange_loops()
                l_loss = loop_detection_loss(loop_info, ex.has_strange_loop)
                loop_sum += l_loss
                total_loss += cfg.loss_weight_loop * l_loss

        # Average over batch
        if n > 0:
            total_loss /= n

        # Record per-objective losses
        self.objective_history["forward"].append(fwd_sum / max(n, 1))
        self.objective_history["backward"].append(bwd_sum / max(n, 1))
        self.objective_history["grammar"].append(gram_sum / max(n, 1))
        self.objective_history["alignment"].append(align_sum / max(n, 1))
        self.objective_history["loop"].append(loop_sum / max(n, 1))

        return total_loss

    # ─── Parameter manipulation ──────────────────────────────────────

    def _perturb_params(
        self,
        params: List[List[float]],
        delta: List[List[float]],
        epsilon: float,
    ) -> None:
        """Apply a perturbation to all parameter vectors in-place.

        params[i][j] += epsilon * delta[i][j]
        """
        for p_vec, d_vec in zip(params, delta):
            for k in range(len(p_vec)):
                p_vec[k] += epsilon * d_vec[k]

    def _write_params_back(self, params: List[List[float]]) -> None:
        """Write parameter vectors back into the model's actual storage.

        Because ``model.parameters`` returns references to the actual
        lists stored in the model (not copies), modifications made via
        ``_perturb_params`` and the update step already affect the model
        directly.  This method is a no-op but exists as a hook for
        future implementations that might copy parameters.
        """
        # In the current implementation, params are references to the
        # model's internal lists, so they are already updated in-place.
        # This method serves as a documentation anchor.
        pass

    # ─── Utilities ───────────────────────────────────────────────────

    def generate_synthetic_data(
        self,
        rules: List[Tuple[int, List[int]]],
        num_examples: int = 100,
        seq_len_range: Tuple[int, int] = (3, 12),
    ) -> List[TrainingExample]:
        """Generate synthetic training data from grammar rules.

        For each example:
        * Pick a random rule and apply it to generate a sequence.
        * Set target_ids = input_ids shifted by 1.
        * Set prev_ids = input_ids shifted by -1.
        * Randomly flag some examples as containing strange loops
          (palindromes, repeated motifs).

        Parameters
        ----------
        rules : list[tuple[int, list[int]]]
            Grammar rules as ``(lhs, body)`` pairs.
        num_examples : int
            How many examples to generate.
        seq_len_range : tuple[int, int]
            Min and max sequence length.

        Returns
        -------
        list[TrainingExample]
            Generated training data.
        """
        if not rules:
            return []

        vocab_size = self.model.config.vocab_size
        examples: List[TrainingExample] = []

        for _ in range(num_examples):
            # Pick a random starting rule and expand
            lhs, body = random.choice(rules)
            seq = list(body)

            # Optionally expand further using random rules
            target_len = random.randint(*seq_len_range)
            attempts = 0
            while len(seq) < target_len and attempts < 50:
                # Find a non-terminal in seq and expand it
                expandable = [
                    (i, s)
                    for i, s in enumerate(seq)
                    if any(r[0] == s for r in rules)
                ]
                if not expandable:
                    break
                idx, sym = random.choice(expandable)
                matching = [r for r in rules if r[0] == sym]
                _, expansion = random.choice(matching)
                seq = seq[:idx] + list(expansion) + seq[idx + 1 :]
                attempts += 1

            # Truncate if too long
            seq = seq[:target_len]
            if len(seq) < 2:
                seq = list(body)[:target_len] if len(body) >= 2 else body + [0]

            # Clamp symbol ids to vocab range
            seq = [s % vocab_size for s in seq]

            # Build targets
            target_ids = seq[1:] + [seq[0]]  # shifted right, wrap
            prev_ids = [seq[-1]] + seq[:-1]  # shifted left, wrap

            # Strange loop: randomly make some palindromes
            has_loop = random.random() < 0.15
            if has_loop:
                half = seq[: len(seq) // 2]
                seq = half + half[::-1]
                seq = seq[:target_len]
                target_ids = seq[1:] + [seq[0]]
                prev_ids = [seq[-1]] + seq[:-1]

            ex = TrainingExample(
                input_ids=seq,
                target_ids=target_ids,
                prev_ids=prev_ids,
                rule_lhs=lhs,
                rule_body=list(body),
                has_strange_loop=has_loop,
            )
            examples.append(ex)

        return examples

    def evaluate(
        self, data: List[TrainingExample]
    ) -> Dict[str, float]:
        """Evaluate the model on a dataset without updating parameters.

        Returns
        -------
        dict
            Per-objective and total losses.
        """
        if not data:
            return {"total": 0.0}

        # Save and restore objective history
        saved = {k: list(v) for k, v in self.objective_history.items()}

        total = self._compute_batch_loss(data)

        # Get the just-recorded values
        result: Dict[str, float] = {"total": total}
        for key in self.objective_history:
            if self.objective_history[key]:
                result[key] = self.objective_history[key][-1]

        # Restore
        self.objective_history = saved

        return result
