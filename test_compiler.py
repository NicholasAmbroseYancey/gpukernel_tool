"""Unit tests for the Level 2 compiler pipeline (no GPU required)."""

import ast
import unittest

from compiler import compile_expression
from evaluator import evaluate_ast, evaluate_ir
from ir import IRBinOp, IRCall, IRConst, IRVar, ast_to_ir
from kernel_gen import emit_triton, generate_kernel
from parser import ParseError, parse_expression

import torch


class TestParser(unittest.TestCase):
    def test_simple_mul(self):
        tree = parse_expression("x * y")
        self.assertIsInstance(tree.body, ast.BinOp)

    def test_function_call(self):
        tree = parse_expression("x * y + sin(x)")
        self.assertIsInstance(tree.body, ast.BinOp)

    def test_ln_alias_parsed(self):
        tree = parse_expression("ln(x) + y")
        self.assertEqual(tree.body.left.func.id, "log")

    def test_rejects_unknown_var(self):
        with self.assertRaises(ParseError):
            parse_expression("z + 1")

    def test_rejects_unknown_func(self):
        with self.assertRaises(ParseError):
            parse_expression("foobar(x)")

    def test_multi_arg_call_parsed(self):
        tree = parse_expression("max(x, y)")
        self.assertEqual(len(tree.body.args), 2)

    def test_rejects_wrong_arity(self):
        for source in ["max(x)", "sin(x, y)", "min(x, y, 1)"]:
            with self.subTest(source=source), self.assertRaises(ParseError):
                parse_expression(source)

    def test_validates_every_argument(self):
        with self.assertRaises(ParseError):
            parse_expression("max(x, z)")


class TestIR(unittest.TestCase):
    def test_lowering(self):
        program = ast_to_ir(parse_expression("x * y + sin(x)"))
        self.assertEqual(program.inputs, frozenset({"x", "y"}))
        self.assertIsInstance(program.output, IRBinOp)
        self.assertIsInstance(program.output.right, IRCall)

    def test_nested_expression(self):
        program = ast_to_ir(parse_expression("(x + y) * (x - y)"))
        self.assertEqual(program.inputs, frozenset({"x", "y"}))


class TestKernelGen(unittest.TestCase):
    def test_emits_triton_ops(self):
        _, program = compile_expression("exp(x) + log(y)")
        code = generate_kernel(program)
        self.assertIn("tl.exp(x)", code)
        self.assertIn("tl.log(y)", code)
        self.assertIn("tl.load", code)
        self.assertIn("tl.store", code)

    def test_emit_triton_sin(self):
        expr = IRCall("sin", (IRVar("x"),))
        self.assertEqual(emit_triton(expr), "tl.sin(x)")

    def test_small_integer_powers_expand_to_multiplication(self):
        self.assertEqual(emit_triton(ast_to_ir(parse_expression("x ** 3")).output), "(x * x * x)")
        self.assertEqual(emit_triton(ast_to_ir(parse_expression("x ** -2")).output), "(1.0 / (x * x))")
        self.assertEqual(emit_triton(ast_to_ir(parse_expression("x ** 0")).output), "1.0")

    def test_half_powers_use_sqrt(self):
        self.assertEqual(emit_triton(ast_to_ir(parse_expression("x ** 0.5")).output), "tl.sqrt(x)")
        self.assertEqual(emit_triton(ast_to_ir(parse_expression("x ** -0.5")).output), "tl.rsqrt(x)")

    def test_general_power_includes_helper(self):
        code, _ = compile_expression("x ** y")
        self.assertIn("out = _gk_pow(x, y)", code)
        self.assertIn("def _gk_pow(b, e):", code)
        self.assertNotIn("tl.math.pow", code)

    def test_helpers_only_included_when_called(self):
        code, _ = compile_expression("x * y")
        self.assertNotIn("_gk_", code)

    def test_emit_triton_multi_arg(self):
        expr = IRCall("max", (IRVar("x"), IRConst(1.0)))
        self.assertEqual(emit_triton(expr), "tl.maximum(x, 1)")


class TestEvaluator(unittest.TestCase):
    def test_matches_pytorch(self):
        x = torch.randn(128)
        y = torch.randn(128)
        tree = parse_expression("x * y + sin(x)")
        program = ast_to_ir(tree)
        ref = evaluate_ir(program, {"x": x, "y": y})
        expected = x * y + torch.sin(x)
        self.assertTrue(torch.allclose(ref, expected))

    def test_fused_expression(self):
        x = torch.randn(64)
        y = torch.randn(64)
        tree = parse_expression("(x + y) * (x - y)")
        ref = evaluate_ast(tree, {"x": x, "y": y})
        expected = (x + y) * (x - y)
        self.assertTrue(torch.allclose(ref, expected))

    def test_multi_arg_matches_pytorch(self):
        x = torch.randn(64)
        y = torch.randn(64)
        ref = evaluate_ast(parse_expression("max(x, y) - min(x, 0)"), {"x": x, "y": y})
        expected = torch.maximum(x, y) - torch.minimum(x, torch.zeros_like(x))
        self.assertTrue(torch.allclose(ref, expected))


class TestCompiler(unittest.TestCase):
    def test_end_to_end_codegen(self):
        code, program = compile_expression("x * y - 2")
        self.assertIn("((x * y) - 2)", code)
        self.assertEqual(program.inputs, frozenset({"x", "y"}))


if __name__ == "__main__":
    unittest.main()
