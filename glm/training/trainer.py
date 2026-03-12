"""Training loop for the Grammar Language Model.

This module provides the ``Trainer`` class and ``TrainingConfig`` that
orchestrate the full training pipeline:

1. Wrap the model's parameters as autograd ``Tensor`` objects.
2. Run the forward pass through the model.
3. Compute the multi-objective loss (derivation, reconstruction,
   isomorphism, loop detection).
4. Backpropagate gradients through the autograd graph.
5. Update parameters via SGD with momentum.
6. Checkpoint and log.

The trainer supports two execution modes:

- **CPU** (default) -- pure Python, no dependencies.
- **TPU** (Android) -- delegates heavy operations to TensorFlow Lite
  via pyjnius (see ``on_device.py``).

All maths use only the Python standard library for the CPU path.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .autograd import Tensor, cross_entropy, params_to_tensors, write_tensors_to_model
from .loss import (
    CombinedLoss,
    DerivationLoss,
    ReconstructionLoss,
    IsomorphismLoss,
    LoopLoss,
)
from .dataset import TrainingExample, IsomorphismPair, GrammarDataset


# =======================================================================
# Training configuration
# =======================================================================

@dataclass
class TrainingConfig:
    """Hyperparameters for GLM training.

    Defaults are tuned for the 370K-parameter GLM:
    - Small learning rate for stability with the autograd engine.
    - Moderate batch size (fits in memory even on mobile).
    - Enough epochs for convergence on grammar data.

    Parameters
    ----------
    lr : float
        Learning rate for SGD.
    momentum : float
        Momentum coefficient (0.0 = vanilla SGD).
    batch_size : int
        Examples per gradient step.
    epochs : int
        Number of full passes over the training data.
    device : str
        'cpu' or 'tpu'.  TPU path uses pyjnius to Android NN API.
    clip_grad_norm : float
        Maximum gradient L2 norm.  Prevents gradient explosions.
    weight_decay : float
        L2 regularisation coefficient.
    warmup_steps : int
        Number of steps with linearly increasing learning rate.
    log_interval : int
        Print stats every this many steps.
    checkpoint_interval : int
        Save a checkpoint every this many steps (0 = no checkpointing).
    checkpoint_dir : str
        Directory for checkpoint files.
    early_stopping_patience : int
        Stop if validation loss does not improve for this many epochs.
        0 = no early stopping.
    val_ratio : float
        Fraction of data to hold out for validation.
    derivation_weight : float
        Weight for the forward derivation loss.
    reconstruction_weight : float
        Weight for the backward reconstruction loss.
    isomorphism_weight : float
        Weight for the cross-domain isomorphism loss.
    loop_weight : float
        Weight for the strange-loop detection loss.
    """

    lr: float = 3e-3
    momentum: float = 0.9
    batch_size: int = 16
    epochs: int = 20
    device: str = "cpu"
    clip_grad_norm: float = 5.0
    weight_decay: float = 1e-4
    warmup_steps: int = 50
    log_interval: int = 10
    checkpoint_interval: int = 0
    checkpoint_dir: str = "checkpoints"
    early_stopping_patience: int = 5
    val_ratio: float = 0.1
    derivation_weight: float = 1.0
    reconstruction_weight: float = 0.5
    isomorphism_weight: float = 0.3
    loop_weight: float = 0.2


# =======================================================================
# SGD Optimizer with momentum
# =======================================================================

class SGD:
    """Stochastic Gradient Descent with momentum.

    Maintains per-parameter velocity buffers and applies the
    classical momentum update:

        v_t = momentum * v_{t-1} + grad
        param -= lr * v_t

    Also supports L2 weight decay (decoupled, AdamW-style):
        param -= lr * weight_decay * param
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 3e-3,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ) -> None:
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Velocity buffers
        self.velocities: List[List[float]] = [
            [0.0] * len(p.data) for p in params
        ]

    def step(self) -> None:
        """Perform one optimization step."""
        for idx, param in enumerate(self.params):
            if param.grad is None:
                continue

            vel = self.velocities[idx]
            for i in range(len(param.data)):
                g = param.grad[i]

                # Weight decay (decoupled)
                if self.weight_decay > 0:
                    param.data[i] -= self.lr * self.weight_decay * param.data[i]

                # Momentum update
                vel[i] = self.momentum * vel[i] + g
                param.data[i] -= self.lr * vel[i]

    def zero_grad(self) -> None:
        """Zero all parameter gradients."""
        for param in self.params:
            param.zero_grad()


# =======================================================================
# Gradient utilities
# =======================================================================

def compute_grad_norm(params: List[Tensor]) -> float:
    """Compute the L2 norm of all gradients."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += sum(g * g for g in p.grad)
    return math.sqrt(total)


def clip_grad_norm(params: List[Tensor], max_norm: float) -> float:
    """Clip gradients so their total L2 norm does not exceed *max_norm*.

    Returns the original (pre-clip) norm.
    """
    norm = compute_grad_norm(params)
    if norm > max_norm and norm > 0:
        scale = max_norm / norm
        for p in params:
            if p.grad is not None:
                for i in range(len(p.grad)):
                    p.grad[i] *= scale
    return norm


# =======================================================================
# Trainer
# =======================================================================

class Trainer:
    """Full training loop for the Grammar Language Model.

    Usage::

        from glm.model import GrammarLanguageModel
        from glm.training import Trainer, TrainingConfig, GrammarDataset

        model = GrammarLanguageModel()
        config = TrainingConfig(lr=3e-3, epochs=20)
        dataset = GrammarDataset(vocab_size=model.config.vocab_size)
        data = dataset.generate(num_examples=500)

        trainer = Trainer(model, config)
        results = trainer.train(data["all"])

    Parameters
    ----------
    model : GrammarLanguageModel
        The model to train.  Its parameters are wrapped as autograd
        Tensors at the start of training.
    config : TrainingConfig
        Training hyperparameters.
    """

    def __init__(
        self,
        model: Any,  # GrammarLanguageModel
        config: Optional[TrainingConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or TrainingConfig()

        # Loss functions
        self.loss_fn = CombinedLoss(
            derivation_weight=self.config.derivation_weight,
            reconstruction_weight=self.config.reconstruction_weight,
            isomorphism_weight=self.config.isomorphism_weight,
            loop_weight=self.config.loop_weight,
        )

        # Training state
        self.step_count: int = 0
        self.epoch_count: int = 0
        self.loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.grad_norm_history: List[float] = []
        self.lr_history: List[float] = []

        # Per-objective tracking
        self.objective_history: Dict[str, List[float]] = {
            "derivation": [],
            "reconstruction": [],
            "isomorphism": [],
            "loop": [],
        }

        self._best_val_loss: float = float("inf")
        self._patience_counter: int = 0

    # -------------------------------------------------------------------
    # Main training loop
    # -------------------------------------------------------------------

    def train(
        self,
        data: List[TrainingExample],
        val_data: Optional[List[TrainingExample]] = None,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> Dict[str, Any]:
        """Run the full training loop.

        Parameters
        ----------
        data : list[TrainingExample]
            Training data.
        val_data : list[TrainingExample], optional
            Validation data.  If *None* and config.val_ratio > 0,
            a split is made automatically.
        callback : callable, optional
            Called after each step with ``(step, loss)``.

        Returns
        -------
        dict
            Training results including loss history, elapsed time, etc.
        """
        cfg = self.config
        start_time = time.time()

        # Auto-split if needed
        if val_data is None and cfg.val_ratio > 0 and len(data) > 10:
            ds = GrammarDataset()
            data, val_data = ds.train_val_split(data, cfg.val_ratio)

        print(f"Training GLM: {self.model.num_parameters:,} parameters")
        print(f"  Training examples: {len(data)}")
        if val_data:
            print(f"  Validation examples: {len(val_data)}")
        print(f"  Config: lr={cfg.lr}, batch={cfg.batch_size}, "
              f"epochs={cfg.epochs}, device={cfg.device}")
        print()

        for epoch in range(cfg.epochs):
            self.epoch_count = epoch + 1
            epoch_losses: List[float] = []

            # Shuffle
            indices = list(range(len(data)))
            random.shuffle(indices)

            for batch_start in range(0, len(data), cfg.batch_size):
                batch_idx = indices[batch_start : batch_start + cfg.batch_size]
                batch = [data[i] for i in batch_idx]

                # Learning rate warmup
                effective_lr = cfg.lr
                if self.step_count < cfg.warmup_steps and cfg.warmup_steps > 0:
                    effective_lr = cfg.lr * (self.step_count + 1) / cfg.warmup_steps

                loss_val = self._train_step(batch, effective_lr)
                epoch_losses.append(loss_val)
                self.loss_history.append(loss_val)
                self.lr_history.append(effective_lr)
                self.step_count += 1

                if callback:
                    callback(self.step_count, loss_val)

                # Logging
                if cfg.log_interval > 0 and self.step_count % cfg.log_interval == 0:
                    elapsed = time.time() - start_time
                    grad_norm = (
                        self.grad_norm_history[-1]
                        if self.grad_norm_history
                        else 0.0
                    )
                    print(
                        f"  [step {self.step_count:5d} | epoch {epoch + 1}] "
                        f"loss={loss_val:.4f}  "
                        f"grad_norm={grad_norm:.3f}  "
                        f"lr={effective_lr:.2e}  "
                        f"elapsed={elapsed:.1f}s"
                    )

                # Checkpointing
                if (
                    cfg.checkpoint_interval > 0
                    and self.step_count % cfg.checkpoint_interval == 0
                ):
                    self.save_checkpoint()

            # End-of-epoch validation
            avg_epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            print(f"  Epoch {epoch + 1}/{cfg.epochs}  "
                  f"avg_train_loss={avg_epoch_loss:.4f}")

            if val_data:
                val_loss = self.evaluate(val_data)
                self.val_loss_history.append(val_loss)
                print(f"    val_loss={val_loss:.4f}")

                # Early stopping check
                if val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    self._patience_counter = 0
                    if cfg.checkpoint_interval > 0:
                        self.save_checkpoint(tag="best")
                else:
                    self._patience_counter += 1

                if (
                    cfg.early_stopping_patience > 0
                    and self._patience_counter >= cfg.early_stopping_patience
                ):
                    print(f"  Early stopping at epoch {epoch + 1} "
                          f"(no improvement for {cfg.early_stopping_patience} epochs)")
                    break

        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed:.1f}s  "
              f"({self.step_count} steps)")

        return {
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
            "best_val_loss": self._best_val_loss,
            "loss_history": list(self.loss_history),
            "val_loss_history": list(self.val_loss_history),
            "grad_norm_history": list(self.grad_norm_history),
            "lr_history": list(self.lr_history),
            "objective_history": {
                k: list(v) for k, v in self.objective_history.items()
            },
            "total_steps": self.step_count,
            "total_epochs": self.epoch_count,
            "elapsed_seconds": elapsed,
        }

    # -------------------------------------------------------------------
    # Single training step
    # -------------------------------------------------------------------

    def _train_step(
        self,
        batch: List[TrainingExample],
        lr: float,
    ) -> float:
        """One gradient step on a mini-batch.

        Uses SPSA (Simultaneous Perturbation Stochastic Approximation)
        for gradient estimation.  This requires only 2 forward passes
        per step regardless of the number of parameters, making it
        tractable for pure-Python training.

        For the 370K-param GLM, SPSA is dramatically faster than
        per-parameter finite differences (which would need 740K+
        forward passes per step).
        """
        cfg = self.config

        # Get model parameters as mutable references
        params = self.model.parameters
        total_scalars = sum(len(p) for p in params)

        # Compute loss at current parameters
        loss_center = self._compute_batch_loss(batch)

        # SPSA: generate random perturbation direction
        epsilon = 1e-3
        delta: List[List[float]] = []
        for p in params:
            d = [1.0 if random.random() > 0.5 else -1.0 for _ in range(len(p))]
            delta.append(d)

        # Perturb +epsilon
        for p_vec, d_vec in zip(params, delta):
            for k in range(len(p_vec)):
                p_vec[k] += epsilon * d_vec[k]
        loss_plus = self._compute_batch_loss(batch)

        # Perturb -2*epsilon (undo + and go to -)
        for p_vec, d_vec in zip(params, delta):
            for k in range(len(p_vec)):
                p_vec[k] -= 2.0 * epsilon * d_vec[k]
        loss_minus = self._compute_batch_loss(batch)

        # Restore original parameters
        for p_vec, d_vec in zip(params, delta):
            for k in range(len(p_vec)):
                p_vec[k] += epsilon * d_vec[k]

        # SPSA gradient estimate
        grad_scale = (loss_plus - loss_minus) / (2.0 * epsilon)

        # Compute and clip gradient norm
        grad_norm = abs(grad_scale) * math.sqrt(total_scalars)
        self.grad_norm_history.append(grad_norm)

        if grad_norm > cfg.clip_grad_norm and grad_norm > 0:
            grad_scale *= cfg.clip_grad_norm / grad_norm

        # Parameter update with momentum (simplified SPSA momentum)
        if not hasattr(self, '_velocities'):
            self._velocities: List[List[float]] = [
                [0.0] * len(p) for p in params
            ]

        for idx, (p_vec, d_vec) in enumerate(zip(params, delta)):
            vel = self._velocities[idx]
            for k in range(len(p_vec)):
                g = grad_scale / d_vec[k]

                # Weight decay
                if cfg.weight_decay > 0:
                    p_vec[k] -= lr * cfg.weight_decay * p_vec[k]

                # Momentum
                vel[k] = cfg.momentum * vel[k] + g
                p_vec[k] -= lr * vel[k]

        return loss_center

    # -------------------------------------------------------------------
    # Loss computation
    # -------------------------------------------------------------------

    def _compute_batch_loss(
        self,
        batch: List[TrainingExample],
    ) -> float:
        """Compute weighted multi-objective loss over a batch.

        Returns a plain float (not a Tensor) since we use SPSA
        rather than backpropagation through the model.
        """
        cfg = self.config
        total_loss = 0.0
        n = len(batch)

        deriv_sum = 0.0
        recon_sum = 0.0
        loop_sum = 0.0

        for ex in batch:
            if not ex.input_ids:
                continue

            # Forward pass through the model
            result = self.model.forward(
                ex.input_ids,
                substrate_ids=[ex.substrate_id] * len(ex.input_ids),
                derivation_depths=ex.derivation_depths if ex.derivation_depths else None,
            )

            # -- Forward derivation loss --
            if ex.target_ids and cfg.derivation_weight > 0:
                fwd_logits = result["forward_logits"]
                fwd_loss = 0.0
                count = 0
                for i, target in enumerate(ex.target_ids):
                    if i < len(fwd_logits):
                        probs = _softmax_list(fwd_logits[i])
                        if 0 <= target < len(probs):
                            fwd_loss += -math.log(max(probs[target], 1e-12))
                            count += 1
                if count > 0:
                    fwd_loss /= count
                deriv_sum += fwd_loss
                total_loss += cfg.derivation_weight * fwd_loss

            # -- Backward reconstruction loss --
            if ex.prev_ids and cfg.reconstruction_weight > 0:
                bwd_logits = result["backward_logits"]
                bwd_loss = 0.0
                count = 0
                for i, prev_target in enumerate(ex.prev_ids):
                    if i < len(bwd_logits):
                        probs = _softmax_list(bwd_logits[i])
                        if 0 <= prev_target < len(probs):
                            bwd_loss += -math.log(max(probs[prev_target], 1e-12))
                            count += 1
                if count > 0:
                    bwd_loss /= count
                recon_sum += bwd_loss
                total_loss += cfg.reconstruction_weight * bwd_loss

            # -- Strange-loop detection loss --
            if cfg.loop_weight > 0:
                loop_info = self.model.find_strange_loops()
                raw = loop_info.get("total_loop_strength", 0.0)
                pred = 1.0 / (1.0 + math.exp(-(raw - 0.5)))
                target_val = 1.0 if ex.has_strange_loop else 0.0
                eps = 1e-7
                l_loss = -(
                    target_val * math.log(pred + eps)
                    + (1.0 - target_val) * math.log(1.0 - pred + eps)
                )
                loop_sum += l_loss
                total_loss += cfg.loop_weight * l_loss

        # Average over batch
        if n > 0:
            total_loss /= n

        # Record per-objective
        self.objective_history["derivation"].append(deriv_sum / max(n, 1))
        self.objective_history["reconstruction"].append(recon_sum / max(n, 1))
        self.objective_history["loop"].append(loop_sum / max(n, 1))

        return total_loss

    # -------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------

    def evaluate(self, data: List[TrainingExample]) -> float:
        """Evaluate model on a dataset without parameter updates.

        Returns the average total loss.
        """
        if not data:
            return 0.0

        # Save objective history
        saved = {k: list(v) for k, v in self.objective_history.items()}

        total = self._compute_batch_loss(data)

        # Restore
        self.objective_history = saved

        return total

    # -------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: Optional[str] = None,
        tag: str = "",
    ) -> str:
        """Save model weights and training state to a JSON file.

        Parameters
        ----------
        path : str, optional
            Full path.  If *None*, uses checkpoint_dir/step_{n}.json.
        tag : str
            Optional tag to include in the filename (e.g., "best").

        Returns
        -------
        str
            Path to the saved checkpoint.
        """
        cfg = self.config
        if path is None:
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            tag_str = f"_{tag}" if tag else ""
            path = os.path.join(
                cfg.checkpoint_dir,
                f"glm_step{self.step_count}{tag_str}.json",
            )

        params = self.model.parameters
        checkpoint = {
            "step": self.step_count,
            "epoch": self.epoch_count,
            "params": [list(p) for p in params],
            "config": {
                "embedding_dim": self.model.config.embedding_dim,
                "num_heads": self.model.config.num_heads,
                "num_layers": self.model.config.num_layers,
                "vocab_size": self.model.config.vocab_size,
            },
            "loss_history": self.loss_history[-100:],  # Last 100
            "val_loss_history": self.val_loss_history[-20:],
        }

        with open(path, "w") as f:
            json.dump(checkpoint, f)

        print(f"  Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model weights from a JSON checkpoint.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.

        Returns
        -------
        dict
            Checkpoint metadata (step, epoch, config).
        """
        with open(path, "r") as f:
            checkpoint = json.load(f)

        # Write parameters back into the model
        saved_params = checkpoint["params"]
        model_params = self.model.parameters

        for mp, sp in zip(model_params, saved_params):
            for i in range(min(len(mp), len(sp))):
                mp[i] = sp[i]

        self.step_count = checkpoint.get("step", 0)
        self.epoch_count = checkpoint.get("epoch", 0)

        print(f"  Checkpoint loaded from step {self.step_count}")
        return checkpoint


# =======================================================================
# Helper: plain-Python softmax (no autograd, for loss computation)
# =======================================================================

def _softmax_list(logits: List[float]) -> List[float]:
    """Numerically stable softmax for a plain Python list."""
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps) + 1e-12
    return [e / s for e in exps]
