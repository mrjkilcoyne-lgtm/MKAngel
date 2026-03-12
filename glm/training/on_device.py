"""Device-specific optimizations for GLM training.

This module provides the bridge between the pure-Python training loop
and device-specific hardware:

* **CPU** (default) -- all operations stay in Python.  No dependencies.
* **Android TPU** -- uses pyjnius to call the Android Neural Networks
  API (NNAPI) or TensorFlow Lite delegate for hardware-accelerated
  inference during training.  The Pixel 10 Pro XL (Tensor G5) has a
  dedicated Edge TPU that can accelerate int8 matrix operations.

Also provides quantization helpers for converting the model from
FP32 (training precision) to INT8 (mobile inference precision).

Key design principles:
- CPU path must work without *any* imports beyond stdlib.
- TPU path gracefully degrades to CPU if pyjnius is unavailable.
- Quantization is *post-training* (PTQ) -- no quantization-aware
  training needed for a 370K-param model.
"""

from __future__ import annotations

import json
import math
import os
import struct
import time
from typing import Any, Dict, List, Optional, Tuple


# =======================================================================
# Device detection
# =======================================================================

def detect_device() -> str:
    """Detect the best available compute device.

    Returns 'tpu' if running on Android with NNAPI access,
    otherwise 'cpu'.
    """
    try:
        from jnius import autoclass  # type: ignore
        # Check if we can access the Android NeuralNetworks API
        _NNModel = autoclass("android.ml.Model")
        return "tpu"
    except Exception:
        pass

    return "cpu"


def device_info() -> Dict[str, Any]:
    """Return information about the current device."""
    info: Dict[str, Any] = {
        "device": detect_device(),
        "python_only": True,
    }

    try:
        from jnius import autoclass  # type: ignore
        Build = autoclass("android.os.Build")
        info["android_model"] = Build.MODEL
        info["android_device"] = Build.DEVICE
        info["android_sdk"] = Build.VERSION.SDK_INT
        info["python_only"] = False
    except Exception:
        pass

    return info


# =======================================================================
# CPU training helpers
# =======================================================================

class CPUTrainer:
    """CPU-specific training optimizations.

    For a 370K-param pure-Python model, the bottleneck is matrix
    multiplication in the attention layers.  This class provides
    optimized routines that reduce Python overhead:

    - Blocked matrix multiply (better cache locality in CPython).
    - Pre-allocated buffers to reduce garbage collection pressure.
    - Gradient accumulation across micro-batches for large effective
      batch sizes without memory blowup.
    """

    def __init__(self, accumulation_steps: int = 1) -> None:
        self.accumulation_steps = accumulation_steps
        self._grad_buffer: Optional[List[List[float]]] = None
        self._micro_step: int = 0

    def init_grad_buffer(self, params: List[List[float]]) -> None:
        """Pre-allocate gradient accumulation buffer."""
        self._grad_buffer = [[0.0] * len(p) for p in params]
        self._micro_step = 0

    def accumulate_gradients(
        self,
        params: List[List[float]],
        grad_estimates: List[List[float]],
    ) -> bool:
        """Accumulate gradients and return True when ready to update.

        Parameters
        ----------
        params : model parameters (unused, for API compatibility).
        grad_estimates : estimated gradients for each parameter vector.

        Returns
        -------
        bool
            True if accumulation_steps have been reached and an
            update should be performed.
        """
        if self._grad_buffer is None:
            self.init_grad_buffer(params)

        for buf, grad in zip(self._grad_buffer, grad_estimates):
            for i in range(len(buf)):
                buf[i] += grad[i]

        self._micro_step += 1

        if self._micro_step >= self.accumulation_steps:
            # Average the accumulated gradients
            scale = 1.0 / self.accumulation_steps
            for buf in self._grad_buffer:
                for i in range(len(buf)):
                    buf[i] *= scale
            return True

        return False

    def get_accumulated_gradients(self) -> List[List[float]]:
        """Return the accumulated (averaged) gradients and reset."""
        grads = self._grad_buffer or []
        # Reset
        if self._grad_buffer is not None:
            for buf in self._grad_buffer:
                for i in range(len(buf)):
                    buf[i] = 0.0
        self._micro_step = 0
        return grads

    @staticmethod
    def blocked_matvec(
        mat: List[List[float]],
        vec: List[float],
        block_size: int = 16,
    ) -> List[float]:
        """Blocked matrix-vector multiply for better cache behaviour.

        Processes rows in blocks of *block_size* to reduce overhead
        from Python for-loop dispatch.
        """
        rows = len(mat)
        cols = len(vec)
        result = [0.0] * rows

        for r_start in range(0, rows, block_size):
            r_end = min(r_start + block_size, rows)
            for r in range(r_start, r_end):
                row = mat[r]
                s = 0.0
                for c in range(cols):
                    s += row[c] * vec[c]
                result[r] = s

        return result


# =======================================================================
# Android TPU training bridge
# =======================================================================

class AndroidTPUBridge:
    """Bridge to Android's Neural Networks API for hardware acceleration.

    On Pixel 10 Pro XL (Tensor G5), the Edge TPU can accelerate:
    - INT8 matrix multiplication (primary use case)
    - Convolutions (not used in GLM)
    - Element-wise operations

    The bridge works by:
    1. Quantizing model weights to INT8.
    2. Creating a TFLite model in-memory.
    3. Running inference through the NNAPI delegate.
    4. De-quantizing outputs back to FP32 for gradient computation.

    Gradient computation stays in Python (CPU) since the TPU only
    accelerates the forward pass.  For a 370K-param model this is
    fine -- the forward pass is the bottleneck.
    """

    def __init__(self) -> None:
        self._available = False
        self._interpreter = None
        self._delegate = None

        try:
            from jnius import autoclass  # type: ignore
            self._TFLiteInterpreter = autoclass(
                "org.tensorflow.lite.Interpreter"
            )
            self._NNAPIDelegate = autoclass(
                "org.tensorflow.lite.nnapi.NnApiDelegate"
            )
            self._available = True
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._available

    def accelerate_forward(
        self,
        weights: List[List[float]],
        input_vec: List[float],
        quantized: bool = True,
    ) -> List[float]:
        """Run a matrix-vector multiply on the TPU.

        Falls back to CPU if TPU is unavailable.

        Parameters
        ----------
        weights : list of list of float
            Weight matrix (rows x cols).
        input_vec : list of float
            Input vector (cols).
        quantized : bool
            If True, quantize to INT8 before TPU execution.

        Returns
        -------
        list of float
            Result vector (rows).
        """
        if not self._available:
            return _cpu_matvec(weights, input_vec)

        try:
            if quantized:
                q_weights, w_scale, w_zero = quantize_matrix(weights)
                q_input, i_scale, i_zero = quantize_vector(input_vec)
                # Run on TPU (simplified -- real implementation would
                # create a TFLite flatbuffer)
                q_result = self._run_quantized_matvec(
                    q_weights, q_input, w_scale, w_zero, i_scale, i_zero
                )
                return dequantize_vector(
                    q_result, w_scale * i_scale, 0
                )
            else:
                return _cpu_matvec(weights, input_vec)
        except Exception:
            return _cpu_matvec(weights, input_vec)

    def _run_quantized_matvec(
        self,
        q_weights: List[List[int]],
        q_input: List[int],
        w_scale: float,
        w_zero: int,
        i_scale: float,
        i_zero: int,
    ) -> List[int]:
        """Run INT8 matrix-vector multiply.

        On real hardware this would go through NNAPI.  For now it
        simulates INT8 arithmetic in Python as a correctness check.
        """
        rows = len(q_weights)
        cols = len(q_input)
        result: List[int] = []

        for r in range(rows):
            acc = 0
            for c in range(cols):
                acc += (q_weights[r][c] - w_zero) * (q_input[c] - i_zero)
            # Clamp to INT8 range
            result.append(max(-128, min(127, acc >> 8)))

        return result


# =======================================================================
# Quantization helpers
# =======================================================================

def quantize_matrix(
    mat: List[List[float]],
) -> Tuple[List[List[int]], float, int]:
    """Quantize a FP32 matrix to INT8.

    Uses affine quantization:
        q = round(x / scale) + zero_point

    where scale and zero_point are chosen to map the [min, max] range
    of the matrix to [-128, 127].

    Returns (quantized_matrix, scale, zero_point).
    """
    # Find global min/max
    flat = [x for row in mat for x in row]
    if not flat:
        return [], 1.0, 0

    x_min = min(flat)
    x_max = max(flat)

    # Compute scale and zero point
    if x_max == x_min:
        scale = 1.0
        zero_point = 0
    else:
        scale = (x_max - x_min) / 255.0
        zero_point = round(-128 - x_min / scale)
        zero_point = max(-128, min(127, zero_point))

    # Quantize
    q_mat: List[List[int]] = []
    for row in mat:
        q_row = [
            max(-128, min(127, round(x / scale) + zero_point))
            for x in row
        ]
        q_mat.append(q_row)

    return q_mat, scale, zero_point


def quantize_vector(
    vec: List[float],
) -> Tuple[List[int], float, int]:
    """Quantize a FP32 vector to INT8."""
    if not vec:
        return [], 1.0, 0

    x_min = min(vec)
    x_max = max(vec)

    if x_max == x_min:
        scale = 1.0
        zero_point = 0
    else:
        scale = (x_max - x_min) / 255.0
        zero_point = round(-128 - x_min / scale)
        zero_point = max(-128, min(127, zero_point))

    q_vec = [
        max(-128, min(127, round(x / scale) + zero_point))
        for x in vec
    ]

    return q_vec, scale, zero_point


def dequantize_vector(
    q_vec: List[int],
    scale: float,
    zero_point: int,
) -> List[float]:
    """Dequantize an INT8 vector back to FP32."""
    return [scale * (q - zero_point) for q in q_vec]


def quantize_model(
    model: Any,
) -> Dict[str, Any]:
    """Quantize an entire GLM model from FP32 to INT8.

    Returns a dictionary containing:
    - ``quantized_params``: list of (quantized_values, scale, zero_point)
      for each parameter vector.
    - ``size_fp32``: original model size in bytes.
    - ``size_int8``: quantized model size in bytes.
    - ``compression_ratio``: fp32_size / int8_size.

    Parameters
    ----------
    model : GrammarLanguageModel
        The model to quantize.

    Returns
    -------
    dict
        Quantization results.
    """
    params = model.parameters
    total_fp32 = 0
    total_int8 = 0
    quantized_params: List[Tuple[List[int], float, int]] = []

    for p in params:
        fp32_bytes = len(p) * 4  # 4 bytes per float32
        int8_bytes = len(p) * 1  # 1 byte per int8 + 8 bytes for scale/zero
        total_fp32 += fp32_bytes
        total_int8 += int8_bytes + 8  # +8 for scale (float64) + zero (int)

        q_vec, scale, zero = quantize_vector(p)
        quantized_params.append((q_vec, scale, zero))

    return {
        "quantized_params": quantized_params,
        "size_fp32": total_fp32,
        "size_int8": total_int8,
        "compression_ratio": total_fp32 / max(total_int8, 1),
        "num_params": sum(len(p) for p in params),
    }


def save_quantized_model(
    quantized: Dict[str, Any],
    path: str,
) -> None:
    """Save a quantized model to a compact binary format.

    Format:
    - 4 bytes: magic number (0x474C4D51 = "GLMQ")
    - 4 bytes: number of parameter vectors
    - For each vector:
      - 4 bytes: vector length
      - 8 bytes: scale (float64)
      - 4 bytes: zero_point (int32)
      - N bytes: quantized values (int8)
    """
    with open(path, "wb") as f:
        # Magic
        f.write(b"GLMQ")
        # Number of param vectors
        q_params = quantized["quantized_params"]
        f.write(struct.pack("<I", len(q_params)))

        for q_vec, scale, zero in q_params:
            f.write(struct.pack("<I", len(q_vec)))
            f.write(struct.pack("<d", scale))
            f.write(struct.pack("<i", zero))
            f.write(bytes([v & 0xFF for v in q_vec]))


def load_quantized_model(
    path: str,
) -> Dict[str, Any]:
    """Load a quantized model from the compact binary format."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GLMQ":
            raise ValueError(f"Invalid magic: {magic!r}")

        n_vectors = struct.unpack("<I", f.read(4))[0]
        quantized_params = []

        for _ in range(n_vectors):
            vec_len = struct.unpack("<I", f.read(4))[0]
            scale = struct.unpack("<d", f.read(8))[0]
            zero = struct.unpack("<i", f.read(4))[0]
            raw = f.read(vec_len)
            # Convert unsigned bytes back to signed int8
            q_vec = [b if b < 128 else b - 256 for b in raw]
            quantized_params.append((q_vec, scale, zero))

    return {"quantized_params": quantized_params}


def dequantize_model_params(
    quantized: Dict[str, Any],
) -> List[List[float]]:
    """Dequantize all model parameters back to FP32 lists."""
    params: List[List[float]] = []
    for q_vec, scale, zero in quantized["quantized_params"]:
        params.append(dequantize_vector(q_vec, scale, zero))
    return params


# =======================================================================
# Memory-efficient gradient accumulation for mobile
# =======================================================================

class MobileGradientAccumulator:
    """Memory-efficient gradient accumulation for mobile training.

    On mobile devices (especially Android), memory is limited.
    This accumulator:

    1. Processes one example at a time (micro-batch of 1).
    2. Accumulates gradients in a fixed-size buffer.
    3. Applies the update only after accumulating the full batch.

    This achieves the same mathematical result as a large batch
    but with peak memory usage proportional to a single example.

    Parameters
    ----------
    target_batch_size : int
        Effective batch size.  Gradients are accumulated over this
        many examples before updating.
    """

    def __init__(self, target_batch_size: int = 16) -> None:
        self.target_batch_size = target_batch_size
        self._buffer: Optional[List[List[float]]] = None
        self._count: int = 0

    def reset(self, param_shapes: List[int]) -> None:
        """Reset the accumulator for a new batch."""
        self._buffer = [[0.0] * s for s in param_shapes]
        self._count = 0

    def add(self, grad_scale: float, delta: List[List[float]]) -> None:
        """Add a single-example gradient estimate."""
        if self._buffer is None:
            raise RuntimeError("Call reset() before add()")

        for buf, d in zip(self._buffer, delta):
            for i in range(len(buf)):
                buf[i] += grad_scale / d[i]

        self._count += 1

    @property
    def ready(self) -> bool:
        """True if target_batch_size examples have been accumulated."""
        return self._count >= self.target_batch_size

    def get_average_gradient(self) -> List[List[float]]:
        """Return averaged gradients and reset for the next batch."""
        if self._buffer is None or self._count == 0:
            return []

        scale = 1.0 / self._count
        result = [
            [g * scale for g in buf]
            for buf in self._buffer
        ]

        # Reset
        for buf in self._buffer:
            for i in range(len(buf)):
                buf[i] = 0.0
        self._count = 0

        return result


# =======================================================================
# Internal helpers
# =======================================================================

def _cpu_matvec(mat: List[List[float]], vec: List[float]) -> List[float]:
    """Plain CPU matrix-vector product."""
    return [sum(r * v for r, v in zip(row, vec)) for row in mat]
