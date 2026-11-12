"""Operator and function mapping: Python AST → Triton / PyTorch."""

import math
import struct

import torch

ALLOWED_VARS = frozenset({"x", "y"})

ALLOWED_FUNCS = frozenset({
    "sin", "cos", "tan", "exp", "log", "sqrt", "abs", "tanh",
    "sigmoid", "relu", "max", "min",
    "asin", "acos", "atan", "atan2",
})

FUNC_ARITY = {
    "sin": 1,
    "cos": 1,
    "tan": 1,
    "exp": 1,
    "log": 1,
    "sqrt": 1,
    "abs": 1,
    "tanh": 1,
    "sigmoid": 1,
    "relu": 1,
    "max": 2,
    "min": 2,
    "asin": 1,
    "acos": 1,
    "atan": 1,
    "atan2": 2,
}

REDUCTIONS = frozenset({"sum", "max", "min"})

REDUCTION_IDENTITY = {
    "sum": 0.0,
    "max": -math.inf,
    "min": math.inf,
}

FUNC_ALIASES = {
    "ln": "log",
    "tg": "tan",
    "arcsin": "asin",
    "arccos": "acos",
    "arctan": "atan",
    "arctan2": "atan2",
}

BINOP_MAP = {
    "Add": "+",
    "Sub": "-",
    "Mult": "*",
    "Div": "/",
    "Pow": "**",
}

UNARYOP_MAP = {
    "UAdd": "+",
    "USub": "-",
}

TRITON_BINOPS = {
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
}

TRITON_FUNCS = {
    "sin": "tl.sin",
    "cos": "tl.cos",
    # triton.language has no tan, so build it from ops it does have.
    "tan": "(tl.sin({0}) / tl.cos({0}))",
    # tl.exp is up to ~55 ulp off on a GPU, and tl.sigmoid uses it; see triton_helpers.
    "exp": "_gk_exp",
    "log": "tl.log",
    "sqrt": "tl.sqrt",
    "abs": "tl.abs",
    "tanh": "_gk_tanh",
    "sigmoid": "_gk_sigmoid",
    # PropagateNan.ALL matches torch: NaN in, NaN out (the default drops NaN on GPU).
    "relu": "tl.maximum(0.0, {0}, propagate_nan=tl.PropagateNan.ALL)",
    "max": "tl.maximum({0}, {1}, propagate_nan=tl.PropagateNan.ALL)",
    "min": "tl.minimum({0}, {1}, propagate_nan=tl.PropagateNan.ALL)",
    "asin": "_gk_asin",
    "acos": "_gk_acos",
    "atan": "_gk_atan",
    "atan2": "_gk_atan2",
}

TRITON_REDUCTIONS = {
    "sum": ("tl.sum", "tl.atomic_add"),
    "max": ("tl.max", "tl.atomic_max"),
    "min": ("tl.min", "tl.atomic_min"),
}

TORCH_FUNCS = {
    "sin": torch.sin,
    "cos": torch.cos,
    "tan": torch.tan,
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "abs": torch.abs,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "relu": torch.relu,
    "max": torch.maximum,
    "min": torch.minimum,
    "asin": torch.asin,
    "acos": torch.acos,
    "atan": torch.atan,
    "atan2": torch.atan2,
}

TORCH_REDUCTIONS = {
    "sum": torch.sum,
    "max": torch.amax,
    "min": torch.amin,
}

MATH_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "abs": abs,
    "tanh": math.tanh,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
}


def to_float32(value: float) -> float:
    """value rounded to float32, as torch.full(..., value, dtype=float32) would
    store it, except that out-of-range values become +-inf instead of raising."""
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def normalize_func(name: str) -> str:
    return FUNC_ALIASES.get(name, name)


def is_allowed_func(name: str) -> bool:
    return normalize_func(name) in ALLOWED_FUNCS


def is_reduction_call(name: str, arg_count: int) -> bool:
    return normalize_func(name) in REDUCTIONS and arg_count == 1


def is_allowed_var(name: str) -> bool:
    return name in ALLOWED_VARS


def triton_func_call(func: str, *args: str) -> str:
    func = normalize_func(func)
    mapping = TRITON_FUNCS[func]
    if "{0}" in mapping:
        return mapping.format(*args)
    return f"{mapping}({', '.join(args)})"
