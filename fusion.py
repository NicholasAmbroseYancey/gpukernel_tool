"""Common subexpression elimination for fused multi-output kernels."""

from __future__ import annotations

from collections import Counter

from ir import IRExpr, IRBinOp, IRCall, IRConst, IRUnaryOp, IRVar, _depth


def fuse_outputs(assignments: list[tuple[str, IRExpr]]) -> tuple[list[tuple[str, IRExpr]], list[tuple[str, IRExpr]]]:
    counts: Counter[IRExpr] = Counter()

    def count_subexprs(expr: IRExpr) -> None:
        if isinstance(expr, (IRVar, IRConst)):
            return
        counts[expr] += 1
        match expr:
            case IRBinOp(left=left, right=right):
                count_subexprs(left)
                count_subexprs(right)
            case IRUnaryOp(operand=operand):
                count_subexprs(operand)
            case IRCall(args=args):
                for arg in args:
                    count_subexprs(arg)

    for _, expr in assignments:
        count_subexprs(expr)

    shared = [expr for expr, freq in counts.items() if freq >= 2]
    shared.sort(key=lambda expr: (_depth(expr), repr(expr)), reverse=True)

    temp_map: dict[IRExpr, str] = {}
    temps: list[tuple[str, IRExpr]] = []
    for index, expr in enumerate(shared):
        name = f"_t{index}"
        temp_map[expr] = name
        temps.append((name, expr))

    def substitute(expr: IRExpr) -> IRExpr:
        if expr in temp_map:
            return IRVar(temp_map[expr])
        match expr:
            case IRBinOp(op=op, left=left, right=right):
                return IRBinOp(op, substitute(left), substitute(right))
            case IRUnaryOp(op=op, operand=operand):
                return IRUnaryOp(op, substitute(operand))
            case IRCall(func=func, args=args):
                return IRCall(func, tuple(substitute(arg) for arg in args))
            case _:
                return expr

    fused_outputs = [(name, substitute(expr)) for name, expr in assignments]
    return temps, fused_outputs
