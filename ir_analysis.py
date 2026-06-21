"""IR analysis helpers for the AI optimizer."""

from __future__ import annotations

from collections import Counter

from compiler import compile_expression, compile_program, is_multi_output
from ir import IRBinOp, IRCall, IRConst, IRExpr, IRUnaryOp, IRVar, expr_to_ir
from parser import parse_expression


def analyze_source(source: str) -> str:
    multi = is_multi_output(source)
    if multi:
        _, program = compile_program(source)
        exprs = [(a.name, a.expr) for a in program.outputs]
        temp_count = len(program.temps)
    else:
        tree = parse_expression(source)
        exprs = [("out", expr_to_ir(tree))]
        temp_count = 0

    op_counts: Counter[str] = Counter()
    func_counts: Counter[str] = Counter()
    max_depth = 0

    for _, expr in exprs:
        max_depth = max(max_depth, _depth(expr))
        _count_ops(expr, op_counts, func_counts)

    lines = [
        f"mode={'multi-output' if multi else 'single-output'}",
        f"outputs={len(exprs)}",
        f"shared_temps={temp_count}",
        f"max_depth={max_depth}",
        f"binops={dict(op_counts)}",
        f"funcs={dict(func_counts)}",
    ]
    if multi:
        lines.append("outputs:")
        for name, expr in exprs:
            lines.append(f"  {name}: {_brief(expr)}")
    else:
        lines.append(f"expr: {_brief(exprs[0][1])}")
    return "\n".join(lines)


def _count_ops(expr: IRExpr, op_counts: Counter[str], func_counts: Counter[str]) -> None:
    match expr:
        case IRBinOp(op=op, left=left, right=right):
            op_counts[op] += 1
            _count_ops(left, op_counts, func_counts)
            _count_ops(right, op_counts, func_counts)
        case IRUnaryOp(operand=operand):
            _count_ops(operand, op_counts, func_counts)
        case IRCall(func=func, args=args):
            func_counts[func] += 1
            for arg in args:
                _count_ops(arg, op_counts, func_counts)
        case _:
            return


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


def _brief(expr: IRExpr) -> str:
    match expr:
        case IRVar(name=name):
            return name
        case IRConst(value=value):
            return str(value)
        case IRBinOp(op=op, left=left, right=right):
            return f"({_brief(left)} {op} {_brief(right)})"
        case IRUnaryOp(op=op, operand=operand):
            return f"({op}{_brief(operand)})"
        case IRCall(func=func, args=args):
            return f"{func}({_brief(args[0])})"
