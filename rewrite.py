"""Tensor-friendly expression preprocessing and AST rewriting."""

from __future__ import annotations

import ast
import re

from ops import ALLOWED_FUNCS, FUNC_ALIASES, normalize_func


def preprocess_source(source: str) -> str:
    text = source.strip()
    text = text.replace("^", "**")
    for alias, canonical in FUNC_ALIASES.items():
        text = re.sub(rf"\b{alias}\s*\(", f"{canonical}(", text)
    return text


def rewrite_ast(tree: ast.Expression) -> ast.Expression:
    body = _AliasRewriter().visit(tree.body)
    body = _AlgebraicRewriter().visit(body)
    ast.fix_missing_locations(tree)
    return ast.Expression(body=body)


class _AliasRewriter(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name):
            return node

        name = normalize_func(node.func.id)
        if name == "pow" and len(node.args) == 2:
            return ast.BinOp(
                left=node.args[0],
                op=ast.Pow(),
                right=node.args[1],
            )

        if name in ALLOWED_FUNCS:
            node.func.id = name
        return node


class _AlgebraicRewriter(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or len(node.args) != 1:
            return node

        func = node.func.id
        arg = node.args[0]

        if func == "log" and isinstance(arg, ast.Call):
            inner = arg
            if isinstance(inner.func, ast.Name) and inner.func.id == "exp" and len(inner.args) == 1:
                return inner.args[0]

        if func == "exp" and isinstance(arg, ast.Call):
            inner = arg
            if isinstance(inner.func, ast.Name) and inner.func.id == "log" and len(inner.args) == 1:
                return inner.args[0]

        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Sub) and _is_zero(node.right):
            return node.left
        if isinstance(node.op, ast.Add) and _is_zero(node.right):
            return node.left
        if isinstance(node.op, ast.Add) and _is_zero(node.left):
            return node.right
        if isinstance(node.op, ast.Mult) and _is_one(node.right):
            return node.left
        if isinstance(node.op, ast.Mult) and _is_one(node.left):
            return node.right
        return node


def _is_zero(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == 0


def _is_one(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == 1
