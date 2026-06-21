"""Tests for Level 6 benchmarking and Level 7 optimizer planning."""

import unittest

from benchmark import BlockSizeResult, BenchmarkReport, MemoryStats, estimate_memory
from ir_analysis import analyze_source
from optimizer_plan import OptimizerPlan, parse_optimizer_plan
from report import save_text_report


class TestMemoryAnalysis(unittest.TestCase):
    def test_estimate_single_output(self):
        mem = estimate_memory(1_000_000, num_outputs=1)
        self.assertEqual(mem.input_bytes, 8_000_000)
        self.assertEqual(mem.output_bytes, 4_000_000)
        self.assertEqual(mem.io_bytes, 12_000_000)

    def test_estimate_multi_output(self):
        mem = estimate_memory(1024, num_outputs=3)
        self.assertEqual(mem.output_bytes, 1024 * 4 * 3)


class TestOptimizerPlan(unittest.TestCase):
    def test_parse_expression_and_block_size(self):
        plan = parse_optimizer_plan(
            "EXPRESSION: x * y + sin(x)\nBLOCK_SIZE: 512\nFUSE: yes\nREASON: fewer ops"
        )
        self.assertEqual(plan.expression, "x * y + sin(x)")
        self.assertEqual(plan.block_size, 512)
        self.assertTrue(plan.fuse)
        self.assertEqual(plan.reason, "fewer ops")

    def test_parse_program_block(self):
        plan = parse_optimizer_plan(
            "PROGRAM:\nout0 = x * y\nout1 = x + y\nBLOCK_SIZE: 256\nREASON: fusion"
        )
        self.assertIn("out0 = x * y", plan.program or "")
        self.assertIn("out1 = x + y", plan.program or "")
        self.assertEqual(plan.block_size, 256)

    def test_resolved_source_prefers_program(self):
        plan = OptimizerPlan(expression="x + y", program="out0 = x * y")
        self.assertEqual(plan.resolved_source("fallback"), "out0 = x * y")


class TestIRAnalysis(unittest.TestCase):
    def test_single_expression_analysis(self):
        summary = analyze_source("x * y + sin(x)")
        self.assertIn("mode=single-output", summary)
        self.assertIn("sin", summary)

    def test_multi_output_analysis(self):
        source = "out0 = x * y\nout1 = x + y\nout2 = x * y + sin(x)"
        summary = analyze_source(source)
        self.assertIn("mode=multi-output", summary)
        self.assertIn("shared_temps=1", summary)


class TestReport(unittest.TestCase):
    def test_save_text_report(self):
        report = BenchmarkReport(
            source="x * y",
            n=1024,
            block_size=256,
            triton_ms=0.5,
            pytorch_ms=1.0,
            speedup=2.0,
            memory=MemoryStats(io_bytes=100, peak_bytes=200, input_bytes=80, output_bytes=20),
            block_size_sweep=[
                BlockSizeResult(block_size=128, median_ms=0.6, throughput_gels=1.7),
                BlockSizeResult(block_size=256, median_ms=0.5, throughput_gels=2.0),
            ],
        )
        path = save_text_report(report, directory="reports_test")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("speedup=2.00x", text)
        self.assertIn("block_size=256", text)


if __name__ == "__main__":
    unittest.main()
