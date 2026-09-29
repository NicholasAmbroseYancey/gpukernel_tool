"""Operator and function mapping: Python AST → Triton / PyTorch."""

import math

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
    # triton.language has no tan/tanh, so build them from ops it does have.
    "tan": "(tl.sin({0}) / tl.cos({0}))",
    "exp": "tl.exp",
    "log": "tl.log",
    "sqrt": "tl.sqrt",
    "abs": "tl.abs",
    "tanh": "(2.0 * tl.sigmoid(2.0 * ({0})) - 1.0)",
    "sigmoid": "tl.sigmoid",
    "relu": "tl.maximum(0.0, {0})",
    "max": "tl.maximum({0}, {1})",
    "min": "tl.minimum({0}, {1})",
    "asin": "_gk_asin",
    "acos": "_gk_acos",
    "atan": "_gk_atan",
    "atan2": "_gk_atan2",
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


def normalize_func(name: str) -> str:
    return FUNC_ALIASES.get(name, name)


def is_allowed_func(name: str) -> bool:
    return normalize_func(name) in ALLOWED_FUNCS


def is_allowed_var(name: str) -> bool:
    return name in ALLOWED_VARS


def triton_func_call(func: str, *args: str) -> str:
    func = normalize_func(func)
    mapping = TRITON_FUNCS[func]
    if "{0}" in mapping:
        return mapping.format(*args)
    return f"{mapping}({', '.join(args)})"
