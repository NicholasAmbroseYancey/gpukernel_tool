"""End-to-end compiler: expression → AST → IR → Triton kernel."""

from fusion import fuse_outputs
from ir import IRMultiProgram, IRProgram, ast_to_ir, build_multi_program, expr_to_ir
from kernel_gen import generate_kernel, generate_multi_kernel
from parser import parse_expression, parse_program


class CompileError(Exception):
    pass


def compile_expression(source: str) -> tuple[str, IRProgram]:
    tree = parse_expression(source)
    program = ast_to_ir(tree)
    code = generate_kernel(program)
    return code, program


def compile_program(source: str) -> tuple[str, IRMultiProgram]:
    assignments = parse_program(source)
    lowered: list[tuple[str, object]] = []
    for name, expr_source in assignments:
        tree = parse_expression(expr_source)
        lowered.append((name, expr_to_ir(tree)))

    expr_assignments = [(name, expr) for name, expr in lowered]
    temps, fused = fuse_outputs(expr_assignments)
    program = build_multi_program(fused, temps)
    code = generate_multi_kernel(program)
    return code, program


def is_multi_output(source: str) -> bool:
    source = source.strip()
    if ";" in source:
        return len([part for part in source.split(";") if part.strip()]) > 1
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    if len(lines) > 1:
        return True
    if len(lines) == 1 and "=" in lines[0]:
        return parse_program(source)[0][0] != "out"
    return False
