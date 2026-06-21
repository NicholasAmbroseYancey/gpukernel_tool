"""Tests for Level 4 feedback, linting, and pipeline helpers."""

import unittest
from unittest.mock import patch

import torch

from compiler import compile_expression
from feedback import DiffStats, FailureFeedback
from kernel_lint import extract_output_expr, is_valid, lint_kernel
from pipeline import parse_llm_fix, run_pipeline
from run_kernel import build_feedback, lint_code
from verify import compute_diff


class TestKernelLint(unittest.TestCase):
    def test_valid_kernel_passes(self):
        _, program = compile_expression("x + y")
        from kernel_gen import generate_kernel

        code = generate_kernel(program)
        self.assertTrue(is_valid(code))
        self.assertEqual(lint_kernel(code), [])

    def test_missing_store_fails(self):
        code = "out = x + y"
        issues = lint_kernel(code)
        kinds = {issue.kind for issue in issues}
        self.assertIn("missing_required", kinds)

    def test_banned_torch_fails(self):
        code = """
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + torch.sin(y)
    tl.store(out_ptr + offsets, out, mask=mask)
"""
        issues = lint_kernel(code)
        self.assertTrue(any(issue.kind == "banned_token" for issue in issues))

    def test_extract_output_expr(self):
        code = "    out = ((x * y) + tl.sin(x))\n"
        self.assertEqual(extract_output_expr(code), "((x * y) + tl.sin(x))")


class TestDiffFeedback(unittest.TestCase):
    def test_compute_diff_on_mismatch(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        ref = x * y + 1
        out = ref.clone()
        out[1] += 0.5

        diff = compute_diff(x, y, out, "x * y + 1", atol=1e-3)
        self.assertIsNotNone(diff)
        assert diff is not None
        self.assertGreater(diff.max_abs_diff, 0.4)
        self.assertGreater(diff.mismatch_count, 0)
        self.assertEqual(len(diff.expected_samples), len(diff.actual_samples))

    def test_feedback_prompt_contains_diff(self):
        feedback = FailureFeedback(
            stage="verify",
            message="Output mismatch",
            expression="x * y",
            kernel_code="out = x * y",
            diff=DiffStats(
                max_abs_diff=0.25,
                mean_abs_diff=0.01,
                mismatch_count=3,
                total_elements=1024,
                worst_indices=[1, 2],
                expected_samples=[1.0, 2.0],
                actual_samples=[1.25, 2.1],
            ),
        )
        prompt = feedback.to_prompt()
        self.assertIn("max_abs_diff=0.25", prompt)
        self.assertIn("target_expression: x * y", prompt)

    def test_build_feedback_from_lint(self):
        code = "bad kernel"
        result = lint_code(code)
        assert result is not None
        feedback = build_feedback("x + y", code, result)
        self.assertEqual(feedback.stage, "lint")
        self.assertTrue(feedback.lint_issues)


class TestLLMFixParsing(unittest.TestCase):
    def test_parse_math_fix(self):
        kind, value = parse_llm_fix("x * y + sin(x)")
        self.assertEqual(kind, "math")
        self.assertEqual(value, "x * y + sin(x)")

    def test_parse_triton_fix(self):
        kind, value = parse_llm_fix("TRITON: tl.sin(x) + x * y")
        self.assertEqual(kind, "triton")
        self.assertEqual(value, "tl.sin(x) + x * y")


class TestPipeline(unittest.TestCase):
    def test_compile_failure_without_llm(self):
        result = run_pipeline("z + 1", max_attempts=1, use_llm=False)
        self.assertFalse(result.success)
        assert result.last_feedback is not None
        self.assertEqual(result.last_feedback.stage, "compile")

    @patch("pipeline.execute_attempt")
    def test_pipeline_success_on_first_attempt(self, mock_execute):
        code, _ = compile_expression("x + y")
        mock_execute.return_value = (True, None)
        result = run_pipeline("x + y", max_attempts=3, use_llm=False)
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 1)


if __name__ == "__main__":
    unittest.main()
