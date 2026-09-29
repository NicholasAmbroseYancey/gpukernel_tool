"""Deterministic IR → Triton kernel generation."""

from ir import IRAssignment, IRBinOp, IRCall, IRConst, IRExpr, IRMultiProgram, IRProgram, IRUnaryOp, IRVar
from ops import triton_func_call
from triton_helpers import helpers_for

MAX_EXPANDED_POWER = 16


SINGLE_KERNEL_TEMPLATE = """import triton
import triton.language as tl
{helpers}
@triton.jit
def kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    out = {output_expr}

    tl.store(out_ptr + offsets, out, mask=mask)
"""


def generate_kernel(program: IRProgram) -> str:
    output_expr = emit_triton(program.output)
    return generate_kernel_from_expr(output_expr)


def generate_kernel_from_expr(output_expr: str) -> str:
    return SINGLE_KERNEL_TEMPLATE.format(output_expr=output_expr, helpers=helpers_for(output_expr))


def generate_multi_kernel(program: IRMultiProgram) -> str:
    ptr_params = ", ".join(f"{item.name}_ptr" for item in program.outputs)
    body_lines = _emit_body(program.temps, program.outputs)
    store_lines = [
        f"    tl.store({item.name}_ptr + offsets, {item.name}, mask=mask)"
        for item in program.outputs
    ]
    helpers = helpers_for("\n".join(body_lines))

    return f"""import triton
import triton.language as tl
{helpers}
@triton.jit
def kernel(x_ptr, y_ptr, {ptr_params}, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

{chr(10).join(body_lines)}

{chr(10).join(store_lines)}
"""


def _emit_body(
    temps: tuple[IRAssignment, ...],
    outputs: tuple[IRAssignment, ...],
) -> list[str]:
    lines: list[str] = []
    for temp in temps:
        lines.append(f"    {temp.name} = {emit_triton(temp.expr)}")
    for output in outputs:
        lines.append(f"    {output.name} = {emit_triton(output.expr)}")
    return lines


def emit_triton(expr: IRExpr) -> str:
    match expr:
        case IRVar(name=name):
            return name
        case IRConst(value=value):
            if value == int(value):
                return str(int(value))
            return repr(float(value))
        case IRBinOp(op=op, left=left, right=right):
            left_s = emit_triton(left)
            right_s = emit_triton(right)
            if op == "**":
                return _emit_pow(left_s, right, right_s)
            return f"({left_s} {op} {right_s})"
        case IRUnaryOp(op=op, operand=operand):
            val = emit_triton(operand)
            if op == "+":
                return f"(+{val})"
            return f"(-{val})"
        case IRCall(func=func, args=args):
            arg_s = [emit_triton(arg) for arg in args]
            return triton_func_call(func, *arg_s)


def _emit_pow(base: str, exponent: IRExpr, exponent_s: str) -> str:
    value = _const_value(exponent)
    if value is not None:
        if value == 0.5:
            return f"tl.sqrt({base})"
        if value == -0.5:
            return f"tl.rsqrt({base})"
        if value.is_integer() and abs(value) <= MAX_EXPANDED_POWER:
            return _emit_int_pow(base, int(value))
    return f"_gk_pow({base}, {exponent_s})"


def _emit_int_pow(base: str, n: int) -> str:
    if n == 0:
        return "1.0"
    if n < 0:
        return f"(1.0 / {_emit_int_pow(base, -n)})"
    return "(" + " * ".join([base] * n) + ")"


def _const_value(expr: IRExpr) -> float | None:
    match expr:
        case IRConst(value=value):
            return value
        case IRUnaryOp(op=op, operand=operand):
            inner = _const_value(operand)
            if inner is None:
                return None
            return -inner if op == "-" else inner
    return None
