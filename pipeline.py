"""Self-correcting compiler loop: compile → lint → run → LLM fix → retry."""

from __future__ import annotations

import re

from compiler import compile_expression, compile_program, is_multi_output
from feedback import FailureFeedback
from kernel_gen import generate_kernel_from_expr
from kernel_lint import extract_output_expr
from kernel_writer import clean_output, save_kernel_source
from ollama_client import generate
from parser import ParseError, parse_expression
from prompt_builder import fix_prompt
from run_kernel import build_feedback, lint_code, run


class PipelineResult:
    __slots__ = ("success", "expression", "kernel_code", "attempts", "last_feedback", "multi")

    def __init__(
        self,
        success: bool,
        expression: str,
        kernel_code: str,
        attempts: int,
        last_feedback: FailureFeedback | None = None,
        multi: bool = False,
    ):
        self.success = success
        self.expression = expression
        self.kernel_code = kernel_code
        self.attempts = attempts
        self.last_feedback = last_feedback
        self.multi = multi


def parse_llm_fix(response: str) -> tuple[str, str]:
    text = clean_output(response).strip()
    first_line = text.splitlines()[0].strip() if text else ""
    if first_line.upper().startswith("TRITON:"):
        return "triton", first_line.split(":", 1)[1].strip()
    return "math", first_line


def compile_from_source(source: str) -> tuple[str, str]:
    if is_multi_output(source):
        code, _ = compile_program(source)
        return source, code
    code, _ = compile_expression(source)
    return source, code


def compile_from_triton_expr(triton_expr: str, expression: str) -> tuple[str, str]:
    if is_multi_output(expression):
        raise ValueError("Triton expression override is not supported for multi-output programs")
    if re.search(r"[;\n]|import |def |@", triton_expr):
        raise ValueError("Triton fix must be a single expression")
    banned = ["numpy", "torch", "eval(", "exec("]
    if any(token in triton_expr for token in banned):
        raise ValueError("Triton fix contains banned tokens")
    return expression, generate_kernel_from_expr(triton_expr)


def build_kernel(expression: str, triton_expr: str | None = None) -> tuple[str, str]:
    if triton_expr is None:
        return compile_from_source(expression)
    return compile_from_triton_expr(triton_expr, expression)


def execute_attempt(expression: str, kernel_code: str) -> tuple[bool, FailureFeedback | None]:
    lint_result = lint_code(kernel_code)
    if lint_result is not None:
        return False, build_feedback(expression, kernel_code, lint_result)

    save_kernel_source(kernel_code)
    run_result = run(expression, kernel_code=kernel_code)
    if run_result.success:
        return True, None

    return False, build_feedback(expression, kernel_code, run_result)


def run_pipeline(source: str, *, max_attempts: int = 3, use_llm: bool = True) -> PipelineResult:
    multi = is_multi_output(source)
    if multi:
        use_llm = False

    current_expression = source.strip()
    triton_override: str | None = None
    kernel_code = ""
    last_feedback: FailureFeedback | None = None

    for attempt in range(1, max_attempts + 1):
        if triton_override is None:
            try:
                current_expression, kernel_code = build_kernel(current_expression)
            except (ParseError, ValueError) as e:
                last_feedback = FailureFeedback(
                    stage="compile",
                    message=str(e),
                    expression=current_expression,
                    kernel_code=kernel_code,
                )
                if not use_llm or attempt == max_attempts:
                    break
                current_expression, triton_override = _llm_fix(last_feedback)
                continue
        else:
            try:
                current_expression, kernel_code = build_kernel(
                    current_expression,
                    triton_override,
                )
            except ValueError as e:
                last_feedback = FailureFeedback(
                    stage="compile",
                    message=str(e),
                    expression=current_expression,
                    kernel_code=kernel_code,
                    output_expr=triton_override,
                )
                if not use_llm or attempt == max_attempts:
                    break
                current_expression, triton_override = _llm_fix(last_feedback)
                continue

        ok, last_feedback = execute_attempt(current_expression, kernel_code)
        triton_override = None
        if ok:
            return PipelineResult(
                success=True,
                expression=current_expression,
                kernel_code=kernel_code,
                attempts=attempt,
                multi=multi,
            )

        print(f"Attempt {attempt} failed: {last_feedback.brief()}")
        if not use_llm or attempt == max_attempts:
            break

        current_expression, triton_override = _llm_fix(last_feedback)

    return PipelineResult(
        success=False,
        expression=current_expression,
        kernel_code=kernel_code,
        attempts=max_attempts,
        last_feedback=last_feedback,
        multi=multi,
    )


def _llm_fix(feedback: FailureFeedback) -> tuple[str, str | None]:
    response = generate(fix_prompt(feedback))
    kind, value = parse_llm_fix(response)
    print(f"LLM fix ({kind}): {value}")

    if kind == "triton":
        return feedback.expression, value

    parse_expression(value)
    return value, None
