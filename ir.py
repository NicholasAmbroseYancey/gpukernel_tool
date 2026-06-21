"""Structured IR and AST → IR lowering."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ops import ALLOWED_VARS, BINOP_MAP, UNARYOP_MAP


@dataclass(frozen=True)
class IRVar:
    name: str


@dataclass(frozen=True)
class IRConst:
    value: float


@dataclass(frozen=True)
class IRBinOp:
    op: str
    left: IRExpr
    right: IRExpr


@dataclass(frozen=True)
class IRUnaryOp:
    op: str
    operand: IRExpr


@dataclass(frozen=True)
class IRCall:
    func: str
    args: tuple[IRExpr, ...]


IRExpr = IRVar | IRConst | IRBinOp | IRUnaryOp | IRCall


@dataclass(frozen=True)
class IRAssignment:
    name: str
    expr: IRExpr


@dataclass(frozen=True)
class IRProgram:
    output: IRExpr
    inputs: frozenset[str]


@dataclass(frozen=True)
class IRMultiProgram:
    outputs: tuple[IRAssignment, ...]
    temps: tuple[IRAssignment, ...]
    inputs: frozenset[str]


def expr_to_ir(tree: ast.Expression) -> IRExpr:
    return _lower(tree.body)


def ast_to_ir(tree: ast.Expression) -> IRProgram:
    output = expr_to_ir(tree)
    inputs = _collect_vars(output)
    return IRProgram(output=output, inputs=frozenset(inputs))


def build_multi_program(
    assignments: list[tuple[str, IRExpr]],
    temps: list[tuple[str, IRExpr]],
) -> IRMultiProgram:
    outputs = tuple(IRAssignment(name, expr) for name, expr in assignments)
    temp_nodes = tuple(IRAssignment(name, expr) for name, expr in temps)
    inputs: set[str] = set()
    for assignment in outputs:
        inputs |= _collect_vars(assignment.expr)
    for assignment in temp_nodes:
        inputs |= _collect_vars(assignment.expr)
    return IRMultiProgram(outputs=outputs, temps=temp_nodes, inputs=frozenset(inputs))


def _lower(node: ast.AST) -> IRExpr:
    if isinstance(node, ast.Name):
        return IRVar(node.id)

    if isinstance(node, ast.Constant):
        return IRConst(float(node.value))

    if isinstance(node, ast.BinOp):
        op = BINOP_MAP[type(node.op).__name__]
        return IRBinOp(op, _lower(node.left), _lower(node.right))

    if isinstance(node, ast.UnaryOp):
        op = UNARYOP_MAP[type(node.op).__name__]
        return IRUnaryOp(op, _lower(node.operand))

    if isinstance(node, ast.Call):
        func = node.func.id
        args = tuple(_lower(arg) for arg in node.args)
        return IRCall(func, args)

    raise ValueError(f"Cannot lower AST node: {type(node).__name__}")


def _collect_vars(expr: IRExpr) -> set[str]:
    match expr:
        case IRVar(name=name):
            if name in ALLOWED_VARS:
                return {name}
            return set()
        case IRConst():
            return set()
        case IRBinOp(left=left, right=right):
            return _collect_vars(left) | _collect_vars(right)
        case IRUnaryOp(operand=operand):
            return _collect_vars(operand)
        case IRCall(args=args):
            result: set[str] = set()
            for arg in args:
                result |= _collect_vars(arg)
            return result


def _depth(expr: IRExpr) -> int:
    match expr:
        case IRVar() | IRConst():
            return 0
        case IRBinOp(left=left, right=right):
            return 1 + max(_depth(left), _depth(right))
        case IRUnaryOp(operand=operand):
            return 1 + _depth(operand)
        case IRCall(args=args):
            return 1 + max((_depth(arg) for arg in args), default=0)
