"""Mocked tests for the LLM optimization loop (no GPU or Ollama required).

Benchmarking, verification, and the LLM are all patched, so these tests pin
down the loop's decision logic: a candidate is kept only if it verifies and
is strictly faster, and a bad LLM reply never crashes the loop.
"""

import unittest
from unittest.mock import patch

from ai_optimizer import run_ai_compiler
from benchmark import BenchmarkReport, estimate_memory
from ollama_client import OllamaError
from pipeline import PipelineResult


def _bench(source, triton_ms, block_size=256):
    return BenchmarkReport(
        source=source,
        n=1024,
        block_size=block_size,
        triton_ms=triton_ms,
        pytorch_ms=1.0,
        speedup=1.0 / triton_ms,
        memory=estimate_memory(1024),
    )


def _compiled(source="x * y + x"):
    return PipelineResult(success=True, expression=source, kernel_code="# kernel", attempts=1)


class OptimizerLoopTest(unittest.TestCase):
    def setUp(self):
        patches = {
            "pipeline": patch("ai_optimizer.run_pipeline", return_value=_compiled()),
            "bench": patch("ai_optimizer.run_benchmark"),
            "verify": patch("ai_optimizer.verify_correctness", return_value=True),
            "llm": patch("ai_optimizer.generate"),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def run_loop(self, replies, timings, **kwargs):
        self.mocks["llm"].side_effect = replies
        self.mocks["bench"].side_effect = [
            _bench(f"src{i}", ms) for i, ms in enumerate(timings)
        ]
        return run_ai_compiler("x * y + x", n=1024, max_rounds=len(replies), **kwargs)

    def test_faster_candidate_is_accepted(self):
        result = self.run_loop(["EXPRESSION: x * (y + 1)"], [1.0, 0.5])
        self.assertTrue(result.success)
        self.assertEqual(result.source, "x * (y + 1)")
        self.assertEqual(result.benchmark.triton_ms, 0.5)

    def test_slower_candidate_is_rejected(self):
        result = self.run_loop(["EXPRESSION: x * (y + 1)"], [1.0, 2.0])
        self.assertEqual(result.source, "x * y + x")
        self.assertEqual(result.benchmark.triton_ms, 1.0)

    def test_equal_time_candidate_is_rejected(self):
        result = self.run_loop(["EXPRESSION: x * (y + 1)"], [1.0, 1.0])
        self.assertEqual(result.source, "x * y + x")

    def test_unverified_candidate_is_never_benchmarked(self):
        self.mocks["verify"].return_value = False
        result = self.run_loop(["EXPRESSION: x - y"], [1.0])
        self.assertEqual(result.source, "x * y + x")
        self.assertEqual(self.mocks["bench"].call_count, 1)

    def test_garbage_reply_does_not_crash(self):
        replies = [
            "I think you should use shared memory!",
            "EXPRESSION: x ** ** y",
            "BLOCK_SIZE: banana",
            "",
        ]
        result = self.run_loop(replies, [1.0, 1.1, 1.2])
        self.assertTrue(result.success)
        self.assertEqual(len(result.plans), 4)

    def test_ollama_down_keeps_best_so_far(self):
        replies = ["EXPRESSION: x * (y + 1)", OllamaError("server down")]
        result = self.run_loop(replies, [1.0, 0.5])
        self.assertTrue(result.success)
        self.assertEqual(result.source, "x * (y + 1)")
        self.assertEqual(len(result.plans), 1)

    def test_block_size_plan_disables_autotune(self):
        self.run_loop(["BLOCK_SIZE: 512"], [1.0, 0.9])
        call = self.mocks["bench"].call_args_list[1]
        self.assertEqual(call.kwargs["block_size"], 512)
        self.assertFalse(call.kwargs["autotune"])

    def test_best_of_several_rounds_is_kept(self):
        replies = ["EXPRESSION: x * (y + 1)", "EXPRESSION: x + x * y", "EXPRESSION: x*y+x"]
        result = self.run_loop(replies, [1.0, 0.8, 0.9, 0.3])
        self.assertEqual(result.source, "x*y+x")
        self.assertEqual(result.benchmark.triton_ms, 0.3)

    def test_no_llm_skips_generate(self):
        result = self.run_loop([], [1.0], use_llm=False)
        self.assertTrue(result.success)
        self.mocks["llm"].assert_not_called()

    def test_compile_failure_short_circuits(self):
        self.mocks["pipeline"].return_value = PipelineResult(
            success=False, expression="z", kernel_code="", attempts=3
        )
        result = run_ai_compiler("z", use_llm=True)
        self.assertFalse(result.success)
        self.mocks["bench"].assert_not_called()
        self.mocks["llm"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
