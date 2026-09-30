"""Tests for the LLM Optimization Engine's accept/reject logic.

Owner: Bryson Wingate (LLM Optimization Engine).

These test classify_candidate() directly with plain floats — no GPU,
no Ollama, no real benchmark run required. The goal is to pin down the
noise-margin boundary deterministically, since real benchmark numbers
won't reliably land in the narrow "faster but not by enough" band on
any given run.
"""

from ai_optimizer import NOISE_MARGIN, classify_candidate


def test_default_margin_is_two_percent():
    assert NOISE_MARGIN == 0.02


def test_clear_improvement_is_accepted():
    # 10% faster than best: comfortably past the 2% margin.
    assert classify_candidate(0.90, 1.00) == "improved"


def test_exactly_at_margin_is_not_improved():
    # Exactly 2% faster == the threshold itself; strict "<" means this
    # boundary value does NOT count as improved.
    assert classify_candidate(0.98, 1.00) == "within_margin"


def test_just_inside_margin_is_improved():
    # Just over 2% faster: should cross the strict "<" threshold.
    assert classify_candidate(0.979, 1.00) == "improved"


def test_faster_but_within_margin_is_flagged_not_accepted():
    # 1% faster than best: numerically better, but inside the 2% margin,
    # so it should NOT be accepted as an improvement.
    assert classify_candidate(0.99, 1.00) == "within_margin"


def test_slower_candidate_is_no_improvement():
    assert classify_candidate(1.05, 1.00) == "no_improvement"


def test_equal_candidate_is_no_improvement():
    assert classify_candidate(1.00, 1.00) == "no_improvement"


def test_custom_margin_is_respected():
    # With a 10% margin, a 5% improvement should NOT count.
    assert classify_candidate(0.95, 1.00, margin=0.10) == "within_margin"
    # But a 15% improvement should.
    assert classify_candidate(0.85, 1.00, margin=0.10) == "improved"


# --- Repeated-plan skipping and early stopping ---------------------------
# The LLM often re-proposes the configuration that is already the best (or
# one it already tried). Re-benchmarking an identical kernel only measures
# jitter, so the loop must skip repeats and stop after MAX_REPEATED_PLANS
# repeats in a row. Benchmark, verification, and the LLM are mocked.

from unittest.mock import patch

import pytest

from ai_optimizer import MAX_REPEATED_PLANS, run_ai_compiler
from benchmark import BenchmarkReport, estimate_memory
from compiler import compile_expression
from pipeline import PipelineResult

BASE_SOURCE = "x * y + x"


def _bench(triton_ms, block_size=256):
    return BenchmarkReport(
        source=BASE_SOURCE,
        n=1024,
        block_size=block_size,
        triton_ms=triton_ms,
        pytorch_ms=1.0,
        speedup=1.0 / triton_ms,
        memory=estimate_memory(1024),
    )


@pytest.fixture
def mocks():
    # Baseline uses the real generated code so a plan that re-proposes the
    # starting expression compiles to an identical kernel.
    code, _ = compile_expression(BASE_SOURCE)
    compiled = PipelineResult(success=True, expression=BASE_SOURCE, kernel_code=code, attempts=1)
    with patch("ai_optimizer.run_pipeline", return_value=compiled), \
         patch("ai_optimizer.run_benchmark") as bench, \
         patch("ai_optimizer.verify_correctness", return_value=True) as verify, \
         patch("ai_optimizer.generate") as llm:
        yield {"bench": bench, "verify": verify, "llm": llm}


def _run(mocks, replies, timings, max_rounds=None):
    mocks["llm"].side_effect = replies
    mocks["bench"].side_effect = [_bench(ms, bs) for ms, bs in timings]
    return run_ai_compiler(BASE_SOURCE, n=1024, max_rounds=max_rounds or len(replies))


def test_replaying_the_baseline_is_skipped(mocks):
    result = _run(mocks, [f"EXPRESSION: {BASE_SOURCE}"], [(1.0, 256)])
    assert mocks["bench"].call_count == 1  # baseline only
    mocks["verify"].assert_not_called()
    assert result.benchmark.triton_ms == 1.0


def test_same_plan_twice_is_benchmarked_once(mocks):
    replies = ["BLOCK_SIZE: 512", "BLOCK_SIZE: 512"]
    _run(mocks, replies, [(1.0, 256), (0.9, 512)])
    assert mocks["bench"].call_count == 2  # baseline + first 512 attempt


def test_spelling_differences_count_as_repeats(mocks):
    # Same expression, different whitespace -> identical generated kernel.
    replies = ["EXPRESSION: x * (y + 1)", "EXPRESSION: x*(y+1)"]
    _run(mocks, replies, [(1.0, 256), (0.9, 256)])
    assert mocks["bench"].call_count == 2


def test_stops_after_consecutive_repeats(mocks):
    replies = ["BLOCK_SIZE: 512"] * 5
    result = _run(mocks, replies, [(1.0, 256), (0.9, 512)])
    # Round 1 is new; rounds 2..(1 + MAX_REPEATED_PLANS) repeat, then stop.
    assert mocks["llm"].call_count == 1 + MAX_REPEATED_PLANS
    assert len(result.plans) == 1 + MAX_REPEATED_PLANS


def test_new_plan_resets_repeat_counter(mocks):
    replies = [
        "BLOCK_SIZE: 512",
        "BLOCK_SIZE: 512",   # repeat 1
        "BLOCK_SIZE: 1024",  # new -> counter resets
        "BLOCK_SIZE: 512",   # repeat 1 again, not 2
        "BLOCK_SIZE: 128",   # still running
    ]
    _run(mocks, replies, [(1.0, 256), (0.9, 512), (0.95, 1024), (0.97, 128)])
    assert mocks["llm"].call_count == 5
    assert mocks["bench"].call_count == 4


def test_autotuned_block_size_is_remembered(mocks):
    # First plan autotunes and lands on 512; an explicit 512 later is a repeat.
    replies = ["EXPRESSION: x * (y + 1)", "EXPRESSION: x * (y + 1)\nBLOCK_SIZE: 512"]
    _run(mocks, replies, [(1.0, 256), (0.9, 512)])
    assert mocks["bench"].call_count == 2