"""Evaluate IR / AST expressions against PyTorch tensors."""

from __future__ import annotations

import ast

import torch

from ir import (
    IRAssignment,
    IRBinOp,
    IRCall,
    IRConst,
    IRExpr,
    IRMultiProgram,
    IRProgram,
    IRReduce,
    IRUnaryOp,
    IRVar,
    ast_to_ir,
)
from ops import TORCH_FUNCS, TORCH_REDUCTIONS, to_float32


def evaluate_ir(program: IRProgram, env: dict[str, torch.Tensor]) -> torch.Tensor:
    missing = program.inputs - env.keys()
    if missing:
        raise ValueError(f"Missing inputs: {sorted(missing)}")
    return _eval_expr(program.output, env)


def evaluate_multi(program: IRMultiProgram, env: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    missing = program.inputs - env.keys()
    if missing:
        raise ValueError(f"Missing inputs: {sorted(missing)}")

    values: dict[str, torch.Tensor] = dict(env)
    for assignment in program.temps:
        values[assignment.name] = _eval_expr(assignment.expr, values)
    for assignment in program.outputs:
        values[assignment.name] = _eval_expr(assignment.expr, values)
    return {assignment.name: values[assignment.name] for assignment in program.outputs}


def evaluate_ast(tree: ast.Expression, env: dict[str, torch.Tensor]) -> torch.Tensor:
    return evaluate_ir(ast_to_ir(tree), env)


def constant_value(expr: IRExpr) -> float | None:
    """The float32 value of an expression with no inputs, such as atan2(1, 0),
    computed as the reference computes it; None if it uses an input."""
    if not _is_constant(expr):
        return None
    return float(_eval_expr(expr, {"x": torch.zeros((), dtype=torch.float32)}))


def _is_constant(expr: IRExpr) -> bool:
    match expr:
        case IRConst():
            return True
        case IRBinOp(left=left, right=right):
            return _is_constant(left) and _is_constant(right)
        case IRUnaryOp(operand=operand):
            return _is_constant(operand)
        case IRCall(args=args):
            return all(_is_constant(arg) for arg in args)
    return False  # IRVar (inputs and fused temps) and IRReduce


def _eval_expr(expr: IRExpr, env: dict[str, torch.Tensor]) -> torch.Tensor:
    match expr:
        case IRVar(name=name):
            if name.startswith("_t") and name in env:
                return env[name]
            return env[name]
        case IRConst(value=value):
            sample = env["x"] if "x" in env else env["y"]
            # full_like raises for 1e39 in float32; the kernel sees inf.
            if sample.dtype == torch.float32:
                value = to_float32(value)
            return torch.full_like(sample, value)
        case IRBinOp(op=op, left=left, right=right):
            left_val = _eval_expr(left, env)
            right_val = _eval_expr(right, env)
            if op == "+":
                return left_val + right_val
            if op == "-":
                return left_val - right_val
            if op == "*":
                return left_val * right_val
            if op == "/":
                return left_val / right_val
            if op == "**":
                return left_val ** right_val
            raise ValueError(f"Unsupported binary op: {op}")
        case IRUnaryOp(op=op, operand=operand):
            val = _eval_expr(operand, env)
            if op == "+":
                return val
            if op == "-":
                return -val
            raise ValueError(f"Unsupported unary op: {op}")
        case IRCall(func=func, args=args):
            fn = TORCH_FUNCS[func]
            return fn(*(_eval_expr(arg, env) for arg in args))
        case IRReduce(op=op, arg=arg):
            return TORCH_REDUCTIONS[op](_eval_expr(arg, env)).reshape(1)
