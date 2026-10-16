"""Tests for feeding plan outcomes back into the optimizer prompt (no GPU or Ollama)."""

import unittest
from unittest.mock import patch

from ai_optimizer import run_ai_compiler
from benchmark import BenchmarkReport, estimate_memory
from pipeline import PipelineResult
from prompt_builder import MAX_HISTORY, OUTCOME_REASONS, PlanAttempt, optimize_prompt


def _bench(triton_ms=1.0, block_size=256):
    return BenchmarkReport(
        source="x * y + x",
        n=1024,
        block_size=block_size,
        triton_ms=triton_ms,
        pytorch_ms=1.0,
        speedup=1.0 / triton_ms,
        memory=estimate_memory(1024),
    )


def _prompt(history=()):
    return optimize_prompt(
        source="x * y + x",
        benchmark=_bench(),
        ir_summary="(ir)",
        round_idx=2,
        history=history,
    )


class TestOptimizePromptHistory(unittest.TestCase):
    def test_no_history_has_no_section(self):
        self.assertNotIn("Previous attempts", _prompt())

    def test_every_outcome_has_a_reason(self):
        for outcome in ["invalid", "changes_math", "repeat", "verify_failed",
                        "within_margin", "no_improvement", "improved"]:
            self.assertIn(outcome, OUTCOME_REASONS)

    def test_history_lists_plan_and_reason(self):
        prompt = _prompt([
            PlanAttempt(1, "expression='x * y'", "changes_math"),
            PlanAttempt(2, "block_size=512", "within_margin", "0.0450 ms vs best 0.0455 ms"),
        ])
        self.assertIn("Previous attempts this run:", prompt)
        self.assertIn("round 1: expression='x * y' -> rejected: changes what the kernel computes", prompt)
        self.assertIn("round 2: block_size=512 -> rejected: faster, but within", prompt)
        self.assertIn("(0.0450 ms vs best 0.0455 ms)", prompt)
        self.assertIn("Do NOT propose any plan listed above again", prompt)

    def test_only_recent_attempts_are_shown(self):
        history = [PlanAttempt(i, f"block_size={i}", "no_improvement") for i in range(1, MAX_HISTORY + 3)]
        prompt = _prompt(history)
        self.assertNotIn("round 1:", prompt)
        self.assertNotIn("round 2:", prompt)
        self.assertIn(f"round {MAX_HISTORY + 2}:", prompt)

    def test_long_plans_are_truncated(self):
        line = PlanAttempt(1, "x" * 500, "invalid").to_prompt_line()
        self.assertLess(len(line), 300)
        self.assertIn("...", line)


class TestLoopFeedsHistory(unittest.TestCase):
    """The optimizer loop passes each round's outcome into the next prompt."""

    def setUp(self):
        patches = {
            "pipeline": patch(
                "ai_optimizer.run_pipeline",
                return_value=PipelineResult(success=True, expression="x * y + x", kernel_code="# kernel", attempts=1),
            ),
            "bench": patch("ai_optimizer.run_benchmark"),
            "verify": patch("ai_optimizer.verify_correctness", return_value=True),
            "llm": patch("ai_optimizer.generate"),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def prompts(self, replies, timings):
        self.mocks["llm"].side_effect = replies
        self.mocks["bench"].side_effect = [_bench(ms) for ms in timings]
        run_ai_compiler("x * y + x", n=1024, max_rounds=len(replies))
        return [call.args[0] for call in self.mocks["llm"].call_args_list]

    def test_first_round_has_no_history(self):
        prompts = self.prompts(["EXPRESSION: x * (y + 1)"], [1.0, 0.5])
        self.assertNotIn("Previous attempts", prompts[0])

    def test_math_changing_plan_is_reported(self):
        prompts = self.prompts(["EXPRESSION: x * y", "EXPRESSION: x * (y + 1)"], [1.0, 0.5])
        # summary() also lists fuse/reason, so check the plan and the reason separately.
        line = next(l for l in prompts[1].splitlines() if l.strip().startswith("round 1:"))
        self.assertIn("expression='x * y'", line)
        self.assertIn("rejected: changes what the kernel computes", line)

    def test_failed_verification_is_reported(self):
        self.mocks["verify"].return_value = False
        prompts = self.prompts(["EXPRESSION: x * (y + 1)", "EXPRESSION: y * x + x"], [1.0])
        self.assertIn("rejected: kernel output did not match PyTorch", prompts[1])

    def test_benchmark_verdict_is_reported_with_timings(self):
        prompts = self.prompts(["EXPRESSION: x * (y + 1)", "BLOCK_SIZE: 512"], [1.0, 2.0, 2.0])
        self.assertIn("rejected: not faster than the current best (2.0000 ms vs best 1.0000 ms", prompts[1])


if __name__ == "__main__":
    unittest.main()
