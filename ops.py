"""Operator and function mapping: Python AST → Triton / PyTorch."""

import math

import torch

ALLOWED_VARS = frozenset({"x", "y"})

ALLOWED_FUNCS = frozenset({
    "sin", "cos", "tan", "exp", "log", "sqrt", "abs", "tanh",
    "sigmoid", "relu",
})

FUNC_ALIASES = {
    "ln": "log",
    "tg": "tan",
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
    "tan": "tl.tan",
    "exp": "tl.exp",
    "log": "tl.log",
    "sqrt": "tl.sqrt",
    "abs": "tl.abs",
    "tanh": "tl.tanh",
    "sigmoid": "tl.sigmoid",
    "relu": "tl.maximum(0.0, {0})",
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
}


def normalize_func(name: str) -> str:
    return FUNC_ALIASES.get(name, name)


def is_allowed_func(name: str) -> bool:
    return normalize_func(name) in ALLOWED_FUNCS


def is_allowed_var(name: str) -> bool:
    return name in ALLOWED_VARS


def triton_func_call(func: str, arg: str) -> str:
    func = normalize_func(func)
    mapping = TRITON_FUNCS[func]
    if "{0}" in mapping:
        return mapping.format(arg)
    return f"{mapping}({arg})"
