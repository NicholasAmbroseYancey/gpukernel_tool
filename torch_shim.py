"""Lightweight shim of a tiny subset of PyTorch API used by the tests.

This is a local substitute for CI/local runs where installing full PyTorch
is impractical. It uses NumPy under the hood and implements only the methods
required by the test suite.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Tuple

import numpy as _np


class Tensor:
    def __init__(self, arr: Any):
        # Preserve incoming dtype (important for integer indices from topk)
        self._arr = _np.array(arr)

    def clone(self):
        return Tensor(self._arr.copy())

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._arr + other._arr)
        return Tensor(self._arr + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._arr - other._arr)
        return Tensor(self._arr - other)

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(other._arr - self._arr)
        return Tensor(other - self._arr)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._arr * other._arr)
        return Tensor(self._arr * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._arr / other._arr)
        return Tensor(self._arr / other)

    def __neg__(self):
        return Tensor(-self._arr)

    def abs(self):
        return Tensor(_np.abs(self._arr))

    def sum(self):
        return Tensor(_np.sum(self._arr))

    def mean(self):
        return Tensor(_np.mean(self._arr))

    def max(self):
        return Tensor(_np.max(self._arr))

    def numel(self):
        return int(self._arr.size)

    def flatten(self):
        return Tensor(self._arr.flatten())

    def tolist(self):
        return self._arr.tolist()

    def item(self):
        return float(self._arr.item())

    def __gt__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._arr > other._arr)
        return Tensor(self._arr > other)

    def __iter__(self):
        for v in self._arr:
            yield Tensor(v)

    def __getitem__(self, idx):
        return Tensor(self._arr[idx])

    def __setitem__(self, idx, value):
        if isinstance(value, Tensor):
            self._arr[idx] = value._arr
        else:
            self._arr[idx] = value

    def topk(self, k: int) -> Tuple["Tensor", "Tensor"]:
        flat = self._arr.flatten()
        if k <= 0:
            return Tensor(_np.array([])), Tensor(_np.array([], dtype=int))
        idx = _np.argpartition(-flat, k - 1)[:k]
        top_vals = flat[idx]
        order = _np.argsort(-top_vals)
        idx = idx[order]
        return Tensor(flat[idx]), Tensor(idx)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Tensor({self._arr!r})"


# Module-level helper functions to mimic torch API used in tests
def randn(n: int) -> Tensor:
    return Tensor(_np.random.randn(n))


def tensor(x: Iterable[float]) -> Tensor:
    return Tensor(_np.array(x, dtype=float))


def sin(x: Tensor) -> Tensor:
    return Tensor(_np.sin(x._arr))


def abs(x: Tensor) -> Tensor:
    return Tensor(_np.abs(x._arr))


def cos(x: Tensor) -> Tensor:
    return Tensor(_np.cos(x._arr))


def tan(x: Tensor) -> Tensor:
    return Tensor(_np.tan(x._arr))


def exp(x: Tensor) -> Tensor:
    return Tensor(_np.exp(x._arr))


def log(x: Tensor) -> Tensor:
    return Tensor(_np.log(x._arr))


def sqrt(x: Tensor) -> Tensor:
    return Tensor(_np.sqrt(x._arr))


def tanh(x: Tensor) -> Tensor:
    return Tensor(_np.tanh(x._arr))


def sigmoid(x: Tensor) -> Tensor:
    return Tensor(1.0 / (1.0 + _np.exp(-x._arr)))


def relu(x: Tensor) -> Tensor:
    return Tensor(_np.maximum(0.0, x._arr))


def allclose(a: Tensor, b: Tensor, atol: float = 1e-8) -> bool:
    return _np.allclose(a._arr, b._arr, atol=atol)


def topk(t: Tensor, k: int):
    return t.topk(k)


def argmax(t: Tensor):
    return int(_np.argmax(t._arr))


def full_like(sample: Tensor, value: float) -> Tensor:
    return Tensor(_np.full_like(sample._arr, value, dtype=float))


# Expose commonly used names
__all__ = [
    "Tensor",
    "randn",
    "tensor",
    "sin",
    "allclose",
    "topk",
]
