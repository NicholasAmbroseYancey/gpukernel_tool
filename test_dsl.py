"""Tests for Level 3 DSL rewriting and Level 5 multi-output fusion."""

import ast
import unittest

import torch

from compiler import compile_expression, compile_program, is_multi_output
from evaluator import evaluate_multi
from fusion import fuse_outputs
from ir import IRBinOp, IRCall, IRVar, ast_to_ir, expr_to_ir
from kernel_gen import generate_multi_kernel
from parser import parse_expression
from rewrite import preprocess_source, rewrite_ast


class TestRewrite(unittest.TestCase):
    def test_ln_alias(self):
        tree = parse_expression("ln(x) + y")
        self.assertIsInstance(tree.body, ast.BinOp)
        self.assertIsInstance(tree.body.left, ast.Call)
        self.assertEqual(tree.body.left.func.id, "log")

    def test_pow_function_to_binop(self):
        tree = parse_expression("pow(x, 2)")
        self.assertIsInstance(tree.body, ast.BinOp)
        self.assertIsInstance(tree.body.op, ast.Pow)

    def test_caret_power(self):
        tree = parse_expression("x^2 + y")
        self.assertIsInstance(tree.body, ast.BinOp)

    def test_log_exp_cancels(self):
        tree = rewrite_ast(parse_expression("log(exp(x))", rewrite=False))
        self.assertIsInstance(tree.body, ast.Name)
        self.assertEqual(tree.body.id, "x")

    def test_exp_log_cancels(self):
        tree = rewrite_ast(parse_expression("exp(log(y))", rewrite=False))
        self.assertIsInstance(tree.body, ast.Name)
        self.assertEqual(tree.body.id, "y")

    def test_preprocess_source(self):
        self.assertEqual(preprocess_source("ln(x)^2"), "log(x)**2")


class TestFusion(unittest.TestCase):
    def test_shared_subexpression(self):
        a = IRBinOp("*", IRVar("x"), IRVar("y"))
        assignments = [("out0", a), ("out2", IRBinOp("+", a, IRCall("sin", (IRVar("x"),))))]
        temps, outputs = fuse_outputs(assignments)
        self.assertEqual(len(temps), 1)
        self.assertEqual(temps[0][0], "_t0")
        self.assertIsInstance(outputs[0][1], IRVar)
        self.assertEqual(outputs[0][1].name, "_t0")

    def test_fused_kernel_emits_temps(self):
        source = "\n".join([
            "out0 = x * y",
            "out1 = x + y",
            "out2 = x * y + sin(x)",
        ])
        code, program = compile_program(source)
        self.assertEqual(len(program.temps), 1)
        self.assertIn("_t0 =", code)
        self.assertIn("out0 = _t0", code)
        self.assertIn("out2 = (_t0 + tl.sin(x))", code)
        self.assertIn("out0_ptr", code)
        self.assertIn("tl.store(out1_ptr", code)


class TestMultiOutput(unittest.TestCase):
    def test_semicolon_program(self):
        self.assertTrue(is_multi_output("out0 = x * y; out1 = x + y"))

    def test_single_expression_not_multi(self):
        self.assertFalse(is_multi_output("x * y + sin(x)"))

    def test_evaluate_multi_matches_pytorch(self):
        source = "out0 = x * y; out1 = x + y; out2 = x * y + sin(x)"
        _, program = compile_program(source)
        x = torch.randn(32)
        y = torch.randn(32)
        outputs = evaluate_multi(program, {"x": x, "y": y})
        self.assertTrue(torch.allclose(outputs["out0"], x * y))
        self.assertTrue(torch.allclose(outputs["out1"], x + y))
        self.assertTrue(torch.allclose(outputs["out2"], x * y + torch.sin(x)))

    def test_auto_output_names(self):
        code, program = compile_program("x * y; x + y")
        self.assertEqual([item.name for item in program.outputs], ["out0", "out1"])
        self.assertIn("out0_ptr", generate_multi_kernel(program))


class TestLevel3Ops(unittest.TestCase):
    def test_exp_log_sin(self):
        code, _ = compile_expression("exp(x) + log(y)")
        self.assertIn("tl.exp(x)", code)
        self.assertIn("tl.log(y)", code)

    def test_complex_dsl(self):
        code, _ = compile_expression("(x + y) * (x - y) + sin(x)")
        self.assertIn("tl.sin(x)", code)


if __name__ == "__main__":
    unittest.main()
