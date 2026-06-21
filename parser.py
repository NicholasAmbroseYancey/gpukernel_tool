"""Parse expression strings into validated Python AST."""

from __future__ import annotations

import ast
import re

from ops import ALLOWED_FUNCS, ALLOWED_VARS, BINOP_MAP, FUNC_ALIASES, UNARYOP_MAP, is_allowed_func, normalize_func
from rewrite import preprocess_source, rewrite_ast


class ParseError(Exception):
    pass


def parse_expression(source: str, *, rewrite: bool = True) -> ast.Expression:
    source = preprocess_source(source)
    if not source:
        raise ParseError("Empty expression")

    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as e:
        raise ParseError(f"Syntax error: {e}") from e

    if rewrite:
        tree = rewrite_ast(tree)
    _validate_node(tree.body)
    return tree


def parse_program(source: str) -> list[tuple[str, str]]:
    """Parse multi-output program into named expression strings."""
    source = source.strip()
    if not source:
        raise ParseError("Empty program")

    chunks: list[str] = []
    if ";" in source:
        chunks = [part.strip() for part in source.split(";") if part.strip()]
    else:
        chunks = [line.strip() for line in source.splitlines() if line.strip()]

    if not chunks:
        raise ParseError("Empty program")

    if len(chunks) == 1 and "=" not in chunks[0]:
        return [("out", chunks[0])]

    assignments: list[tuple[str, str]] = []
    auto_idx = 0
    assign_pattern = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<expr>.+)$")

    for chunk in chunks:
        match = assign_pattern.match(chunk)
        if match:
            name = match.group("name")
            expr = match.group("expr").strip()
        else:
            name = f"out{auto_idx}"
            expr = chunk
            auto_idx += 1
        assignments.append((name, expr))

    if len(assignments) == 1 and assignments[0][0] == "out":
        return assignments

    for name, _ in assignments:
        if not re.fullmatch(r"out\d+", name):
            raise ParseError(f"Multi-output name must be outN, got {name!r}")

    return assignments


def _validate_node(node: ast.AST) -> None:
    if isinstance(node, ast.Name):
        if node.id not in ALLOWED_VARS:
            raise ParseError(f"Unknown variable: {node.id!r} (allowed: {sorted(ALLOWED_VARS)})")
        return

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ParseError(f"Unsupported constant type: {type(node.value).__name__}")
        return

    if isinstance(node, ast.BinOp):
        op_name = type(node.op).__name__
        if op_name not in BINOP_MAP:
            raise ParseError(f"Unsupported binary operator: {op_name}")
        _validate_node(node.left)
        _validate_node(node.right)
        return

    if isinstance(node, ast.UnaryOp):
        op_name = type(node.op).__name__
        if op_name not in UNARYOP_MAP:
            raise ParseError(f"Unsupported unary operator: {op_name}")
        _validate_node(node.operand)
        return

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ParseError("Only simple function calls are supported (e.g. sin(x))")
        func = normalize_func(node.func.id)
        if not is_allowed_func(node.func.id):
            raise ParseError(
                f"Unknown function: {node.func.id!r} "
                f"(allowed: {sorted(ALLOWED_FUNCS | set(FUNC_ALIASES))})"
            )
        node.func.id = func
        if node.keywords:
            raise ParseError("Keyword arguments are not supported")
        if len(node.args) != 1:
            raise ParseError(f"{func}() expects exactly 1 argument")
        _validate_node(node.args[0])
        return

    raise ParseError(f"Unsupported expression node: {type(node).__name__}")
