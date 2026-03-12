"""GLM Training Infrastructure.

Provides everything needed to train the Grammar Language Model
from scratch on CPU or mobile TPU (Android / Pixel 10 Pro XL).

Quick start::

    from glm.model import GrammarLanguageModel, GLMConfig
    from glm.training import Trainer, TrainingConfig, GrammarDataset

    model = GrammarLanguageModel(GLMConfig())
    dataset = GrammarDataset(vocab_size=model.config.vocab_size)
    examples = dataset.generate(n=500)

    config = TrainingConfig(epochs=10, lr=3e-3)
    trainer = Trainer(model, config)
    history = trainer.train(examples)

Modules
-------
autograd
    Minimal reverse-mode automatic differentiation engine.
loss
    Grammar-specific loss functions (derivation, reconstruction,
    isomorphism, loop detection).
dataset
    Training data generation from grammar derivations.
trainer
    Training loop with SPSA gradient estimation, SGD + momentum,
    checkpointing, and early stopping.
on_device
    Device detection, CPU/TPU bridges, INT8 quantization helpers.
"""

from __future__ import annotations

# -- Core training API -------------------------------------------------------
from .trainer import Trainer, TrainingConfig, SGD

# -- Dataset ------------------------------------------------------------------
from .dataset import GrammarDataset, TrainingExample, IsomorphismPair

# -- Loss functions -----------------------------------------------------------
from .loss import (
    CombinedLoss,
    DerivationLoss,
    ReconstructionLoss,
    IsomorphismLoss,
    LoopLoss,
)

# -- Autograd engine ----------------------------------------------------------
from .autograd import (
    Tensor,
    cross_entropy,
    mse_loss,
    params_to_tensors,
    tensors_to_params,
    write_tensors_to_model,
)

# -- On-device utilities ------------------------------------------------------
from .on_device import (
    detect_device,
    device_info,
    CPUTrainer,
    AndroidTPUBridge,
    MobileGradientAccumulator,
    quantize_model,
    save_quantized_model,
    load_quantized_model,
    dequantize_model_params,
)

__all__ = [
    # Trainer
    "Trainer",
    "TrainingConfig",
    "SGD",
    # Dataset
    "GrammarDataset",
    "TrainingExample",
    "IsomorphismPair",
    # Loss
    "CombinedLoss",
    "DerivationLoss",
    "ReconstructionLoss",
    "IsomorphismLoss",
    "LoopLoss",
    # Autograd
    "Tensor",
    "cross_entropy",
    "mse_loss",
    "params_to_tensors",
    "tensors_to_params",
    "write_tensors_to_model",
    # On-device
    "detect_device",
    "device_info",
    "CPUTrainer",
    "AndroidTPUBridge",
    "MobileGradientAccumulator",
    "quantize_model",
    "save_quantized_model",
    "load_quantized_model",
    "dequantize_model_params",
]
