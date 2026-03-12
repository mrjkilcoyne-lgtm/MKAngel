"""Minimal autograd engine for the Grammar Language Model.

Inspired by Andrej Karpathy's micrograd and George Hotz's tinygrad,
this module provides a ``Tensor`` class that wraps Python lists (nested
for 2-D) and tracks the computational graph so that gradients can be
computed via reverse-mode automatic differentiation (backpropagation).

Key design decisions:

* **Pure Python** -- no numpy, no torch, no C extensions.  This keeps
  the GLM dependency-free and deployable on any platform including
  bare Android via pyjnius.
* **Eager execution** -- every operation immediately computes its
  result *and* records a backward function on the graph.  Calling
  ``backward()`` on the final loss tensor propagates gradients to all
  leaf tensors that had ``requires_grad=True``.
* **Flat storage** -- internally all data is stored as a 1-D Python
  list with an explicit ``shape`` tuple.  2-D indexing is computed
  from the shape.  This simplifies the backward pass and avoids
  nested-list overhead.

Supported operations (forward + backward):
    add, sub, mul (element-wise), matmul, relu, gelu, softmax,
    layer_norm, log, neg, sum, mean, transpose, reshape, slice/index.
"""

from __future__ import annotations

import math
import random
from typing import Callable, List, Optional, Sequence, Tuple, Union

# -----------------------------------------------------------------------
# Tensor
# -----------------------------------------------------------------------

class Tensor:
    """A multi-dimensional array with automatic differentiation.

    Parameters
    ----------
    data : flat list of floats, or nested list for convenience.
    shape : explicit shape tuple.  If *None*, inferred from *data*.
    requires_grad : whether to track gradients for this tensor.
    """

    def __init__(
        self,
        data: Union[List[float], List[List[float]], "Tensor"],
        shape: Optional[Tuple[int, ...]] = None,
        requires_grad: bool = False,
    ) -> None:
        if isinstance(data, Tensor):
            self.data = list(data.data)
            self.shape = data.shape
        elif shape is not None:
            # Flat data with explicit shape
            self.data = [float(x) for x in data]
            self.shape = shape
        elif data and isinstance(data[0], (list, tuple)):
            # Nested list -- flatten
            rows = len(data)
            cols = len(data[0])
            self.data = [float(x) for row in data for x in row]
            self.shape = (rows, cols)
        else:
            self.data = [float(x) for x in data]
            self.shape = (len(self.data),)

        self.requires_grad = requires_grad
        self.grad: Optional[List[float]] = None
        if requires_grad:
            self.grad = [0.0] * len(self.data)

        # Autograd graph
        self._backward: Callable[[], None] = lambda: None
        self._prev: Tuple[Tensor, ...] = ()

    # -- properties --------------------------------------------------------

    @property
    def numel(self) -> int:
        """Total number of elements."""
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def ndim(self) -> int:
        return len(self.shape)

    # -- factory methods ---------------------------------------------------

    @staticmethod
    def zeros(shape: Tuple[int, ...], requires_grad: bool = False) -> "Tensor":
        n = 1
        for s in shape:
            n *= s
        return Tensor([0.0] * n, shape=shape, requires_grad=requires_grad)

    @staticmethod
    def ones(shape: Tuple[int, ...], requires_grad: bool = False) -> "Tensor":
        n = 1
        for s in shape:
            n *= s
        return Tensor([1.0] * n, shape=shape, requires_grad=requires_grad)

    @staticmethod
    def randn(
        shape: Tuple[int, ...],
        requires_grad: bool = False,
        scale: float = 0.02,
    ) -> "Tensor":
        n = 1
        for s in shape:
            n *= s
        data = [random.gauss(0.0, scale) for _ in range(n)]
        return Tensor(data, shape=shape, requires_grad=requires_grad)

    @staticmethod
    def from_lists(nested: List[List[float]], requires_grad: bool = False) -> "Tensor":
        """Create a 2-D tensor from a list of lists (the model's native format)."""
        return Tensor(nested, requires_grad=requires_grad)

    @staticmethod
    def from_flat(data: List[float], requires_grad: bool = False) -> "Tensor":
        """Create a 1-D tensor from a flat list."""
        return Tensor(data, shape=(len(data),), requires_grad=requires_grad)

    # -- 2-D access helpers ------------------------------------------------

    def _idx(self, r: int, c: int) -> int:
        """Flat index for row r, col c in a 2-D tensor."""
        return r * self.shape[1] + c

    def get2d(self, r: int, c: int) -> float:
        return self.data[self._idx(r, c)]

    def row(self, r: int) -> List[float]:
        """Return row *r* as a plain Python list."""
        if self.ndim < 2:
            raise ValueError("row() requires a 2-D tensor")
        cols = self.shape[1]
        start = r * cols
        return self.data[start : start + cols]

    def to_lists(self) -> List[List[float]]:
        """Convert a 2-D tensor back to nested lists."""
        if self.ndim != 2:
            raise ValueError("to_lists() requires a 2-D tensor")
        rows, cols = self.shape
        return [self.data[r * cols : (r + 1) * cols] for r in range(rows)]

    def to_flat(self) -> List[float]:
        """Return a copy of the flat data."""
        return list(self.data)

    # -- backward / autograd -----------------------------------------------

    def backward(self) -> None:
        """Reverse-mode autodiff: propagate gradients from this tensor."""
        # Build topological order
        topo: List[Tensor] = []
        visited: set = set()

        def _build(t: Tensor) -> None:
            if id(t) not in visited:
                visited.add(id(t))
                for p in t._prev:
                    _build(p)
                topo.append(t)

        _build(self)

        # Seed gradient
        if self.grad is None:
            self.grad = [0.0] * len(self.data)
        for i in range(len(self.grad)):
            self.grad[i] = 1.0

        # Backpropagate
        for t in reversed(topo):
            t._backward()

    def zero_grad(self) -> None:
        if self.grad is not None:
            for i in range(len(self.grad)):
                self.grad[i] = 0.0

    # -- element-wise operations -------------------------------------------

    def __add__(self, other: "Tensor") -> "Tensor":
        return _add(self, other)

    def __sub__(self, other: "Tensor") -> "Tensor":
        return _sub(self, other)

    def __mul__(self, other: Union["Tensor", float]) -> "Tensor":
        if isinstance(other, (int, float)):
            return _scale(self, float(other))
        return _mul(self, other)

    def __rmul__(self, other: float) -> "Tensor":
        return _scale(self, float(other))

    def __neg__(self) -> "Tensor":
        return _scale(self, -1.0)

    # -- reduction operations ----------------------------------------------

    def sum(self) -> "Tensor":
        return _sum(self)

    def mean(self) -> "Tensor":
        return _mean(self)

    # -- activation / transform operations ---------------------------------

    def relu(self) -> "Tensor":
        return _relu(self)

    def gelu(self) -> "Tensor":
        return _gelu(self)

    def softmax(self, axis: int = -1) -> "Tensor":
        return _softmax(self, axis=axis)

    def layer_norm(self, eps: float = 1e-5) -> "Tensor":
        return _layer_norm(self, eps=eps)

    def log(self) -> "Tensor":
        return _log(self)

    # -- linear algebra ----------------------------------------------------

    def matmul(self, other: "Tensor") -> "Tensor":
        return _matmul(self, other)

    def transpose(self) -> "Tensor":
        return _transpose(self)

    # -- indexing ----------------------------------------------------------

    def __getitem__(self, idx: int) -> "Tensor":
        """Row-slice a 2-D tensor, returning a 1-D tensor."""
        if self.ndim == 2:
            row_data = self.row(idx)
            return _slice_row(self, idx)
        elif self.ndim == 1:
            return _slice_element(self, idx)
        raise IndexError("Only 1-D and 2-D tensors support indexing")

    # -- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        preview = self.data[:6]
        suffix = ", ..." if len(self.data) > 6 else ""
        return f"Tensor(shape={self.shape}, data=[{', '.join(f'{x:.4f}' for x in preview)}{suffix}])"


# =======================================================================
# Autograd operations (functional style)
# =======================================================================


def _add(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise addition with broadcasting for scalar b."""
    if len(a.data) != len(b.data):
        # Simple broadcast: if b is scalar (1 element), broadcast to a's shape
        if len(b.data) == 1:
            out_data = [x + b.data[0] for x in a.data]
        elif len(a.data) == 1:
            out_data = [a.data[0] + x for x in b.data]
        else:
            raise ValueError(
                f"Cannot add tensors of sizes {len(a.data)} and {len(b.data)}"
            )
    else:
        out_data = [x + y for x, y in zip(a.data, b.data)]

    out = Tensor(out_data, shape=a.shape if len(a.data) >= len(b.data) else b.shape)
    out._prev = (a, b)

    def _backward() -> None:
        if a.grad is not None:
            if len(a.data) == len(out.data):
                for i in range(len(a.grad)):
                    a.grad[i] += out.grad[i]
            else:
                # a was broadcast scalar
                a.grad[0] += sum(out.grad)
        if b.grad is not None:
            if len(b.data) == len(out.data):
                for i in range(len(b.grad)):
                    b.grad[i] += out.grad[i]
            else:
                b.grad[0] += sum(out.grad)

    out._backward = _backward
    return out


def _sub(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise subtraction."""
    out_data = [x - y for x, y in zip(a.data, b.data)]
    out = Tensor(out_data, shape=a.shape)
    out._prev = (a, b)

    def _backward() -> None:
        if a.grad is not None:
            for i in range(len(a.grad)):
                a.grad[i] += out.grad[i]
        if b.grad is not None:
            for i in range(len(b.grad)):
                b.grad[i] -= out.grad[i]

    out._backward = _backward
    return out


def _mul(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise multiplication (Hadamard product)."""
    out_data = [x * y for x, y in zip(a.data, b.data)]
    out = Tensor(out_data, shape=a.shape)
    out._prev = (a, b)

    def _backward() -> None:
        if a.grad is not None:
            for i in range(len(a.grad)):
                a.grad[i] += b.data[i] * out.grad[i]
        if b.grad is not None:
            for i in range(len(b.grad)):
                b.grad[i] += a.data[i] * out.grad[i]

    out._backward = _backward
    return out


def _scale(a: Tensor, s: float) -> Tensor:
    """Scale a tensor by a scalar."""
    out_data = [x * s for x in a.data]
    out = Tensor(out_data, shape=a.shape)
    out._prev = (a,)

    def _backward() -> None:
        if a.grad is not None:
            for i in range(len(a.grad)):
                a.grad[i] += s * out.grad[i]

    out._backward = _backward
    return out


def _matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication: a @ b.

    Shapes: (M, K) @ (K, N) -> (M, N)
    Also supports: (M, K) @ (K,) -> (M,) [matvec]
    """
    if a.ndim == 2 and b.ndim == 2:
        M, K = a.shape
        K2, N = b.shape
        assert K == K2, f"matmul shape mismatch: ({M},{K}) @ ({K2},{N})"

        out_data = [0.0] * (M * N)
        for i in range(M):
            for j in range(N):
                s = 0.0
                for k in range(K):
                    s += a.data[i * K + k] * b.data[k * N + j]
                out_data[i * N + j] = s

        out = Tensor(out_data, shape=(M, N))
        out._prev = (a, b)

        def _backward() -> None:
            if a.grad is not None:
                # dL/dA = dL/dC @ B^T
                for i in range(M):
                    for k in range(K):
                        s = 0.0
                        for j in range(N):
                            s += out.grad[i * N + j] * b.data[k * N + j]
                        a.grad[i * K + k] += s
            if b.grad is not None:
                # dL/dB = A^T @ dL/dC
                for k in range(K):
                    for j in range(N):
                        s = 0.0
                        for i in range(M):
                            s += a.data[i * K + k] * out.grad[i * N + j]
                        b.grad[k * N + j] += s

        out._backward = _backward
        return out

    elif a.ndim == 2 and b.ndim == 1:
        # Matrix-vector product: (M, K) @ (K,) -> (M,)
        M, K = a.shape
        assert b.shape[0] == K, f"matvec shape mismatch: ({M},{K}) @ ({K},)"

        out_data = [0.0] * M
        for i in range(M):
            s = 0.0
            for k in range(K):
                s += a.data[i * K + k] * b.data[k]
            out_data[i] = s

        out = Tensor(out_data, shape=(M,))
        out._prev = (a, b)

        def _backward() -> None:
            if a.grad is not None:
                for i in range(M):
                    for k in range(K):
                        a.grad[i * K + k] += out.grad[i] * b.data[k]
            if b.grad is not None:
                for k in range(K):
                    s = 0.0
                    for i in range(M):
                        s += a.data[i * K + k] * out.grad[i]
                    b.grad[k] += s

        out._backward = _backward
        return out

    else:
        raise ValueError(f"matmul not supported for shapes {a.shape} and {b.shape}")


def _transpose(a: Tensor) -> Tensor:
    """Transpose a 2-D tensor."""
    assert a.ndim == 2, "transpose requires 2-D tensor"
    rows, cols = a.shape
    out_data = [0.0] * (rows * cols)
    for r in range(rows):
        for c in range(cols):
            out_data[c * rows + r] = a.data[r * cols + c]

    out = Tensor(out_data, shape=(cols, rows))
    out._prev = (a,)

    def _backward() -> None:
        if a.grad is not None:
            for r in range(rows):
                for c in range(cols):
                    a.grad[r * cols + c] += out.grad[c * rows + r]

    out._backward = _backward
    return out


def _relu(a: Tensor) -> Tensor:
    """ReLU activation."""
    out_data = [max(0.0, x) for x in a.data]
    out = Tensor(out_data, shape=a.shape)
    out._prev = (a,)

    def _backward() -> None:
        if a.grad is not None:
            for i in range(len(a.grad)):
                a.grad[i] += out.grad[i] * (1.0 if a.data[i] > 0 else 0.0)

    out._backward = _backward
    return out


def _gelu(a: Tensor) -> Tensor:
    """GELU activation (sigmoid approximation: x * sigma(1.702 * x))."""
    out_data = []
    for x in a.data:
        sig = 1.0 / (1.0 + math.exp(-1.702 * x)) if abs(x) < 500 else (1.0 if x > 0 else 0.0)
        out_data.append(x * sig)

    out = Tensor(out_data, shape=a.shape)
    out._prev = (a,)

    def _backward() -> None:
        if a.grad is not None:
            for i in range(len(a.grad)):
                x = a.data[i]
                sig = 1.0 / (1.0 + math.exp(-1.702 * x)) if abs(x) < 500 else (1.0 if x > 0 else 0.0)
                # d/dx [x * sig(1.702x)] = sig + x * 1.702 * sig * (1 - sig)
                dsig = 1.702 * sig * (1.0 - sig)
                grad_val = sig + x * dsig
                a.grad[i] += out.grad[i] * grad_val

    out._backward = _backward
    return out


def _softmax(a: Tensor, axis: int = -1) -> Tensor:
    """Softmax along the last axis.

    For a 1-D tensor, applies over all elements.
    For a 2-D tensor, applies per row.
    """
    if a.ndim == 1:
        m = max(a.data)
        exps = [math.exp(x - m) for x in a.data]
        s = sum(exps) + 1e-12
        out_data = [e / s for e in exps]
        out = Tensor(out_data, shape=a.shape)
        out._prev = (a,)

        def _backward_1d() -> None:
            if a.grad is not None:
                # Jacobian of softmax: diag(p) - p p^T
                n = len(out_data)
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            a.grad[i] += out.grad[j] * out_data[i] * (1.0 - out_data[i])
                        else:
                            a.grad[i] -= out.grad[j] * out_data[j] * out_data[i]

        out._backward = _backward_1d
        return out

    elif a.ndim == 2:
        rows, cols = a.shape
        out_data = [0.0] * (rows * cols)
        row_probs: List[List[float]] = []

        for r in range(rows):
            row = a.data[r * cols : (r + 1) * cols]
            m = max(row)
            exps = [math.exp(x - m) for x in row]
            s = sum(exps) + 1e-12
            probs = [e / s for e in exps]
            row_probs.append(probs)
            for c in range(cols):
                out_data[r * cols + c] = probs[c]

        out = Tensor(out_data, shape=a.shape)
        out._prev = (a,)

        def _backward_2d() -> None:
            if a.grad is not None:
                for r in range(rows):
                    p = row_probs[r]
                    for i in range(cols):
                        for j in range(cols):
                            if i == j:
                                a.grad[r * cols + i] += (
                                    out.grad[r * cols + j] * p[i] * (1.0 - p[i])
                                )
                            else:
                                a.grad[r * cols + i] -= (
                                    out.grad[r * cols + j] * p[j] * p[i]
                                )

        out._backward = _backward_2d
        return out

    raise ValueError(f"softmax not supported for ndim={a.ndim}")


def _layer_norm(a: Tensor, eps: float = 1e-5) -> Tensor:
    """Layer normalisation over the last axis.

    For 1-D: normalise the whole vector.
    For 2-D: normalise each row independently.
    """
    if a.ndim == 1:
        n = len(a.data)
        mu = sum(a.data) / n
        var = sum((x - mu) ** 2 for x in a.data) / n
        inv_std = 1.0 / math.sqrt(var + eps)
        out_data = [(x - mu) * inv_std for x in a.data]

        out = Tensor(out_data, shape=a.shape)
        out._prev = (a,)

        def _backward_1d() -> None:
            if a.grad is not None:
                # Simplified layer norm backward
                for i in range(n):
                    # d(LN)/dx_i involves the full Jacobian
                    # Use the standard formula
                    dx = out.grad[i] * inv_std
                    dx -= inv_std / n * sum(out.grad)
                    dx -= out_data[i] / n * sum(
                        out.grad[j] * out_data[j] for j in range(n)
                    )
                    a.grad[i] += dx

        out._backward = _backward_1d
        return out

    elif a.ndim == 2:
        rows, cols = a.shape
        out_data = [0.0] * (rows * cols)
        row_stats: List[Tuple[float, float, List[float]]] = []  # mu, inv_std, normed

        for r in range(rows):
            row = a.data[r * cols : (r + 1) * cols]
            mu = sum(row) / cols
            var = sum((x - mu) ** 2 for x in row) / cols
            inv_s = 1.0 / math.sqrt(var + eps)
            normed = [(x - mu) * inv_s for x in row]
            row_stats.append((mu, inv_s, normed))
            for c in range(cols):
                out_data[r * cols + c] = normed[c]

        out = Tensor(out_data, shape=a.shape)
        out._prev = (a,)

        def _backward_2d() -> None:
            if a.grad is not None:
                for r in range(rows):
                    mu_r, inv_s_r, normed_r = row_stats[r]
                    og = out.grad[r * cols : (r + 1) * cols]
                    sum_og = sum(og)
                    sum_og_n = sum(og[j] * normed_r[j] for j in range(cols))
                    for c in range(cols):
                        dx = og[c] * inv_s_r
                        dx -= inv_s_r / cols * sum_og
                        dx -= normed_r[c] / cols * sum_og_n
                        a.grad[r * cols + c] += dx

        out._backward = _backward_2d
        return out

    raise ValueError(f"layer_norm not supported for ndim={a.ndim}")


def _log(a: Tensor) -> Tensor:
    """Element-wise natural logarithm (clamped for numerical safety)."""
    out_data = [math.log(max(x, 1e-12)) for x in a.data]
    out = Tensor(out_data, shape=a.shape)
    out._prev = (a,)

    def _backward() -> None:
        if a.grad is not None:
            for i in range(len(a.grad)):
                a.grad[i] += out.grad[i] / max(a.data[i], 1e-12)

    out._backward = _backward
    return out


def _sum(a: Tensor) -> Tensor:
    """Sum all elements to a scalar tensor."""
    s = sum(a.data)
    out = Tensor([s], shape=(1,))
    out._prev = (a,)

    def _backward() -> None:
        if a.grad is not None:
            for i in range(len(a.grad)):
                a.grad[i] += out.grad[0]

    out._backward = _backward
    return out


def _mean(a: Tensor) -> Tensor:
    """Mean of all elements to a scalar tensor."""
    n = len(a.data)
    m = sum(a.data) / max(n, 1)
    out = Tensor([m], shape=(1,))
    out._prev = (a,)

    def _backward() -> None:
        if a.grad is not None:
            for i in range(len(a.grad)):
                a.grad[i] += out.grad[0] / n

    out._backward = _backward
    return out


def _slice_row(a: Tensor, idx: int) -> Tensor:
    """Extract row *idx* from a 2-D tensor as a 1-D tensor."""
    cols = a.shape[1]
    start = idx * cols
    out_data = a.data[start : start + cols]
    out = Tensor(out_data, shape=(cols,))
    out._prev = (a,)

    def _backward() -> None:
        if a.grad is not None:
            for c in range(cols):
                a.grad[start + c] += out.grad[c]

    out._backward = _backward
    return out


def _slice_element(a: Tensor, idx: int) -> Tensor:
    """Extract a single element from a 1-D tensor as a scalar tensor."""
    out = Tensor([a.data[idx]], shape=(1,))
    out._prev = (a,)

    def _backward() -> None:
        if a.grad is not None:
            a.grad[idx] += out.grad[0]

    out._backward = _backward
    return out


# =======================================================================
# Convenience: cross-entropy loss (differentiable)
# =======================================================================

def cross_entropy(logits: Tensor, target: int) -> Tensor:
    """Cross-entropy loss for a single prediction.

    Parameters
    ----------
    logits : Tensor, shape (V,)
        Raw un-normalised scores over vocabulary.
    target : int
        Index of the correct class.

    Returns
    -------
    Tensor
        Scalar loss value with gradient support.
    """
    probs = logits.softmax()
    target_prob = probs[target]
    loss = -(target_prob.log())
    return loss


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error loss."""
    diff = pred - target
    sq = diff * diff
    return sq.mean()


# =======================================================================
# Utility: wrap / unwrap model parameters
# =======================================================================

def params_to_tensors(
    param_lists: List[List[float]],
) -> List[Tensor]:
    """Wrap the model's List[List[float]] parameters as grad-tracking Tensors.

    Each List[float] becomes a 1-D Tensor with ``requires_grad=True``.
    """
    return [
        Tensor(list(p), shape=(len(p),), requires_grad=True)
        for p in param_lists
    ]


def tensors_to_params(tensors: List[Tensor]) -> List[List[float]]:
    """Unwrap Tensors back to plain lists."""
    return [list(t.data) for t in tensors]


def write_tensors_to_model(
    model_params: List[List[float]],
    tensors: List[Tensor],
) -> None:
    """Write Tensor data back into the model's parameter lists in-place.

    The model stores parameters as mutable ``List[float]`` objects.
    This function copies the Tensor data back into those lists so the
    model's forward pass sees the updated weights.
    """
    for p_list, t in zip(model_params, tensors):
        for i in range(len(p_list)):
            p_list[i] = t.data[i]
