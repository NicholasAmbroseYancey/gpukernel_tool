"""Mocked tests for the compile → run → LLM-fix loop and the no-CUDA paths."""

import unittest
from unittest.mock import patch

from benchmark import run_benchmark, verify_correctness
from feedback import FailureFeedback
from ollama_client import OllamaError
from pipeline import compile_from_triton_expr, run_pipeline
from run_kernel import run


def _verify_failure(expression):
    return False, FailureFeedback(
        stage="verify", message="mismatch", expression=expression, kernel_code=""
    )


class TestLLMFixLoop(unittest.TestCase):
    @patch("pipeline.generate", return_value="x * y")
    @patch("pipeline.execute_attempt")
    def test_math_fix_is_retried(self, execute, generate):
        execute.side_effect = [_verify_failure("x * y + 1"), (True, None)]
        result = run_pipeline("x * y + 1", max_attempts=3, use_llm=True)
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.expression, "x * y")

    @patch("pipeline.generate", return_value="TRITON: x * y + tl.sin(x)")
    @patch("pipeline.execute_attempt")
    def test_triton_fix_is_compiled_into_kernel(self, execute, generate):
        execute.side_effect = [_verify_failure("x * y + sin(x)"), (True, None)]
        result = run_pipeline("x * y + sin(x)", max_attempts=3, use_llm=True)
        self.assertTrue(result.success)
        self.assertIn("out = x * y + tl.sin(x)", result.kernel_code)

    @patch("pipeline.generate", return_value="TRITON: torch.sin(x)")
    @patch("pipeline.execute_attempt")
    def test_banned_triton_fix_is_rejected(self, execute, generate):
        execute.side_effect = [_verify_failure("sin(x)"), (True, None)]
        result = run_pipeline("sin(x)", max_attempts=2, use_llm=True)
        self.assertFalse(result.success)
        self.assertEqual(result.last_feedback.stage, "compile")

    @patch("pipeline.generate")
    @patch("pipeline.execute_attempt")
    def test_llm_not_called_when_disabled(self, execute, generate):
        execute.side_effect = [_verify_failure("x + y")]
        result = run_pipeline("x + y", max_attempts=3, use_llm=False)
        self.assertFalse(result.success)
        generate.assert_not_called()

    @patch("pipeline.generate")
    @patch("pipeline.execute_attempt")
    def test_multi_output_never_calls_llm(self, execute, generate):
        execute.side_effect = [_verify_failure("out0 = x; out1 = y")]
        run_pipeline("out0 = x; out1 = y", max_attempts=3, use_llm=True)
        generate.assert_not_called()

    @patch("pipeline.execute_attempt")
    def test_attempt_budget_is_respected(self, execute):
        execute.side_effect = lambda expr, code: _verify_failure(expr)
        with patch("pipeline.generate", return_value="x + y") as generate:
            result = run_pipeline("x + y", max_attempts=3, use_llm=True)
        self.assertFalse(result.success)
        self.assertEqual(execute.call_count, 3)
        self.assertEqual(generate.call_count, 2)


class TestTritonFixValidation(unittest.TestCase):
    def test_rejects_statements(self):
        for bad in ["x; import os", "def f(): pass", "@triton.jit", "x\ny"]:
            with self.assertRaises(ValueError, msg=bad):
                compile_from_triton_expr(bad, "x + y")

    def test_rejects_multi_output(self):
        with self.assertRaises(ValueError):
            compile_from_triton_expr("x", "out0 = x; out1 = y")


class TestLLMIsAdversarial(unittest.TestCase):
    """A bad LLM reply or an unreachable Ollama must never crash the loop."""

    @patch("pipeline.generate", return_value="Sure! Here is the fixed expression:")
    @patch("pipeline.execute_attempt", side_effect=lambda expr, code: _verify_failure(expr))
    def test_chatty_reply_does_not_crash(self, execute, generate):
        result = run_pipeline("x + y", max_attempts=3, use_llm=True)
        self.assertFalse(result.success)
        self.assertEqual(result.expression, "x + y")
        # Rejected replies aren't rebuilt: one real run, then two LLM asks.
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(generate.call_count, 2)
        self.assertIn("previous_fix_rejected", result.last_feedback.message)

    @patch("pipeline.execute_attempt")
    def test_recovers_after_rejected_reply(self, execute):
        execute.side_effect = [_verify_failure("x * y + 1"), (True, None)]
        with patch("pipeline.generate", side_effect=["", "x * y"]) as generate:
            result = run_pipeline("x * y + 1", max_attempts=3, use_llm=True)
        self.assertTrue(result.success)
        self.assertEqual(result.expression, "x * y")
        # The second prompt tells the LLM its first reply was rejected.
        self.assertIn("previous_fix_rejected", generate.call_args_list[1].args[0])

    @patch("pipeline.generate", return_value="TRITON:")
    @patch("pipeline.execute_attempt", side_effect=lambda expr, code: _verify_failure(expr))
    def test_empty_triton_fix_is_rejected(self, execute, generate):
        result = run_pipeline("x + y", max_attempts=2, use_llm=True)
        self.assertFalse(result.success)
        self.assertEqual(execute.call_count, 1)

    @patch("pipeline.generate", side_effect=OllamaError("server down"))
    @patch("pipeline.execute_attempt", side_effect=lambda expr, code: _verify_failure(expr))
    def test_ollama_down_stops_without_crashing(self, execute, generate):
        result = run_pipeline("x + y", max_attempts=3, use_llm=True)
        self.assertFalse(result.success)
        self.assertEqual(result.last_feedback.stage, "verify")
        self.assertEqual(generate.call_count, 1)

    @patch("pipeline.generate", side_effect=OllamaError("server down"))
    def test_ollama_down_on_compile_error(self, generate):
        result = run_pipeline("z + 1", max_attempts=3, use_llm=True)
        self.assertFalse(result.success)
        self.assertEqual(result.last_feedback.stage, "compile")


@patch("torch.cuda.is_available", return_value=False)
class TestWithoutCuda(unittest.TestCase):
    def test_run_reports_runtime_failure(self, _):
        result = run("x + y")
        self.assertFalse(result.success)
        self.assertEqual(result.stage, "runtime")
        self.assertIn("CUDA", result.message)

    def test_run_program_reports_runtime_failure(self, _):
        result = run("out0 = x; out1 = y")
        self.assertFalse(result.success)
        self.assertEqual(result.stage, "runtime")

    def test_lint_runs_before_cuda_check(self, _):
        result = run("x + y", kernel_code="not a kernel")
        self.assertEqual(result.stage, "lint")

    def test_run_benchmark_raises(self, _):
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            run_benchmark("x + y", n=1024)

    def test_verify_correctness_is_false(self, _):
        self.assertFalse(verify_correctness("x + y"))


if __name__ == "__main__":
    unittest.main()
