"""End-to-end kernel tests: compile → launch Triton → verify vs PyTorch.

Runs on a CUDA GPU, or on the CPU with TRITON_INTERPRET=1 (as in CI).
Benchmark tests are marked ``gpu`` and need a real GPU.
"""

import pytest

from benchmark import run_benchmark, verify_correctness
from kernel_gen import generate_kernel_from_expr
from kernel_writer import save_kernel_source
from pipeline import compile_from_source, run_pipeline
from run_kernel import run

pytestmark = pytest.mark.kernel

# Inputs are randn, so log/sqrt get abs(...) to stay in-domain.
SINGLE_EXPRESSIONS = [
    "x + y",
    "x - y",
    "x * y",
    "x / (abs(y) + 1)",
    "-x + 2 * y",
    "x * y + sin(x)",
    "cos(x) * exp(y)",
    "tan(x)",
    "tg(x / 2)",
    "log(abs(x) + 1)",
    "ln(abs(y) + 1)",
    "sqrt(abs(x))",
    "tanh(x) + sigmoid(y)",
    "tanh(50 * x)",
    "relu(x - y)",
    "(x + y) * (x - y)",
    "log(exp(x))",
    "max(x, y)",
    "min(x, 1) + max(sin(x), y)",
    "x ** 2",
    "x ^ 3 - y",
    "(abs(x) + 1) ** -2",
    "abs(x) ** 0.5",
    "(abs(x) + 1) ** -0.5",
    "pow(abs(x) + 1, 1.5)",
    "(abs(x) + 0.5) ** y",
    "x ** (y * 0 + 3)",
]

MULTI_PROGRAMS = [
    "out0 = x * y; out1 = x + y",
    "out0 = x * y\nout1 = x + y\nout2 = x * y + sin(x)",
    "x * y; relu(x); exp(y) - 1",
    "out0 = x ** 2; out1 = (abs(x) + 1) ** y",
]


@pytest.mark.parametrize("expression", SINGLE_EXPRESSIONS)
def test_single_output_kernel_matches_pytorch(expression):
    result = run_pipeline(expression, max_attempts=1, use_llm=False)
    assert result.success, result.last_feedback and result.last_feedback.to_prompt()


@pytest.mark.parametrize("source", MULTI_PROGRAMS)
def test_multi_output_kernel_matches_pytorch(source):
    result = run_pipeline(source, max_attempts=1, use_llm=False)
    assert result.success, result.last_feedback and result.last_feedback.to_prompt()


def test_wrong_kernel_is_caught_by_verify():
    # An LLM "fix" that compiles and runs but computes the wrong thing.
    save_kernel_source(generate_kernel_from_expr("x - y"))
    result = run("x + y")
    assert not result.success
    assert result.stage == "verify"
    assert result.diff is not None and result.diff.mismatch_count > 0


def test_broken_kernel_is_a_runtime_failure():
    save_kernel_source(generate_kernel_from_expr("tl.not_a_function(x)"))
    result = run("x + y")
    assert not result.success
    assert result.stage == "runtime"
    assert "not_a_function" in result.message


@pytest.mark.parametrize("block_size", [64, 256, 1024])
def test_verify_correctness_across_block_sizes(block_size):
    # n not a multiple of block_size exercises the tail mask.
    _, code = compile_from_source("x * y + sin(x)")
    save_kernel_source(code)
    assert verify_correctness("x * y + sin(x)", n=1000, block_size=block_size)


@pytest.mark.gpu
@pytest.mark.parametrize("source", ["x * y + sin(x)", "out0 = x * y; out1 = x + y"])
def test_benchmark_produces_sane_report(source):
    _, code = compile_from_source(source)
    save_kernel_source(code)
    report = run_benchmark(source, n=1 << 16, autotune=True)
    assert report.triton_ms > 0
    assert report.pytorch_ms > 0
    assert report.speedup > 0
    assert report.block_size in {item.block_size for item in report.block_size_sweep}
    assert report.memory.peak_bytes >= report.memory.io_bytes
