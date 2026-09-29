import importlib.util

import torch

from compiler import compile_program, is_multi_output
from feedback import DiffStats, FailureFeedback, LintIssue
from kernel_lint import extract_output_expr, lint_kernel
from verify import check, check_multi, compute_diff, compute_multi_diff


class RunResult:
    __slots__ = ("success", "stage", "message", "diff", "exception_type", "lint_issues")

    def __init__(
        self,
        success: bool,
        *,
        stage: str | None = None,
        message: str = "",
        diff: DiffStats | None = None,
        exception_type: str | None = None,
        lint_issues: list[LintIssue] | None = None,
    ):
        self.success = success
        self.stage = stage
        self.message = message
        self.diff = diff
        self.exception_type = exception_type
        self.lint_issues = lint_issues


def lint_code(code: str) -> RunResult | None:
    issues = lint_kernel(code)
    if not issues:
        return None
    return RunResult(
        success=False,
        stage="lint",
        message="Kernel lint failed",
        lint_issues=issues,
    )


def load_kernel():
    spec = importlib.util.spec_from_file_location(
        "kernel_module",
        "kernels/kernel.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.kernel


def run(expression, *, kernel_code: str | None = None):
    if is_multi_output(expression):
        return run_program(expression, kernel_code=kernel_code)

    if kernel_code is not None:
        lint_result = lint_code(kernel_code)
        if lint_result is not None:
            return lint_result

    if not torch.cuda.is_available():
        return RunResult(
            success=False,
            stage="runtime",
            message="CUDA is not available",
            exception_type="RuntimeError",
        )

    import triton

    n = 1024
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.zeros(n, device="cuda", dtype=torch.float32)

    block_size = 256
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    try:
        kernel = load_kernel()
        kernel[grid](x, y, out, n, block_size)
    except Exception as e:
        return RunResult(
            success=False,
            stage="runtime",
            message=str(e),
            exception_type=type(e).__name__,
        )

    try:
        if check(x, y, out, expression):
            return RunResult(success=True, message="OK")
    except Exception as e:
        return RunResult(
            success=False,
            stage="runtime",
            message=str(e),
            exception_type=type(e).__name__,
        )

    diff = compute_diff(x, y, out, expression)
    return RunResult(
        success=False,
        stage="verify",
        message="Output mismatch with PyTorch reference",
        diff=diff,
    )


def run_program(source: str, *, kernel_code: str | None = None):
    if kernel_code is not None:
        lint_result = lint_code(kernel_code)
        if lint_result is not None:
            return lint_result

    if not torch.cuda.is_available():
        return RunResult(
            success=False,
            stage="runtime",
            message="CUDA is not available",
            exception_type="RuntimeError",
        )

    import triton

    _, program = compile_program(source)
    n = 1024
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    outputs = {
        assignment.name: torch.zeros(n, device="cuda", dtype=torch.float32)
        for assignment in program.outputs
    }

    block_size = 256
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    kernel_args = [x, y] + [outputs[assignment.name] for assignment in program.outputs] + [n, block_size]

    try:
        kernel = load_kernel()
        kernel[grid](*kernel_args)
    except Exception as e:
        return RunResult(
            success=False,
            stage="runtime",
            message=str(e),
            exception_type=type(e).__name__,
        )

    try:
        if check_multi(x, y, outputs, source):
            return RunResult(success=True, message="OK")
    except Exception as e:
        return RunResult(
            success=False,
            stage="runtime",
            message=str(e),
            exception_type=type(e).__name__,
        )

    diff = compute_multi_diff(x, y, outputs, source)
    return RunResult(
        success=False,
        stage="verify",
        message="Output mismatch with PyTorch reference",
        diff=diff,
    )


def build_feedback(expression: str, kernel_code: str, result: RunResult) -> FailureFeedback:
    return FailureFeedback(
        stage=result.stage or "unknown",
        message=result.message,
        expression=expression,
        kernel_code=kernel_code,
        output_expr=extract_output_expr(kernel_code),
        lint_issues=result.lint_issues,
        diff=result.diff,
        exception_type=result.exception_type,
    )
