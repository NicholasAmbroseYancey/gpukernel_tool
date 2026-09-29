"""Unit tests for the Level 2 compiler pipeline (no GPU required)."""

import ast
import unittest

from compiler import compile_expression, compile_program
from evaluator import evaluate_ast, evaluate_ir
from ir import IRBinOp, IRCall, IRConst, IRReduce, IRVar, ast_to_ir
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
        for source in ["atan2(x)", "sin(x, y)", "min(x, y, 1)"]:
            with self.subTest(source=source), self.assertRaises(ParseError):
                parse_expression(source)

    def test_validates_every_argument(self):
        with self.assertRaises(ParseError):
            parse_expression("max(x, z)")

    def test_reduction_must_be_whole_output(self):
        for source in ["sum(x) + 1", "sin(max(x))", "sum(sum(x))", "sum(x, y)"]:
            with self.subTest(source=source), self.assertRaises(ParseError):
                parse_expression(source)


class TestIR(unittest.TestCase):
    def test_lowering(self):
        program = ast_to_ir(parse_expression("x * y + sin(x)"))
        self.assertEqual(program.inputs, frozenset({"x", "y"}))
        self.assertIsInstance(program.output, IRBinOp)
        self.assertIsInstance(program.output.right, IRCall)

    def test_single_arg_max_lowers_to_reduction(self):
        program = ast_to_ir(parse_expression("max(x * y)"))
        self.assertEqual(program.output, IRReduce("max", IRBinOp("*", IRVar("x"), IRVar("y"))))
        self.assertIsInstance(ast_to_ir(parse_expression("max(x, y)")).output, IRCall)

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

    def test_inverse_trig_includes_dependent_helpers(self):
        code, _ = compile_expression("asin(x) + atan2(y, x)")
        for helper in ["_gk_atan(v)", "_gk_atan2(y, x)", "_gk_asin(v)"]:
            self.assertEqual(code.count(f"def {helper}:"), 1, helper)
        self.assertLess(code.index("def _gk_atan("), code.index("def _gk_atan2("))
        self.assertNotIn("_gk_acos", code)

    def test_arc_aliases(self):
        tree = parse_expression("arcsin(x) + arctan2(y, x)")
        self.assertEqual(tree.body.left.func.id, "asin")
        self.assertEqual(tree.body.right.func.id, "atan2")

    def test_reduction_uses_atomic_instead_of_store(self):
        code, _ = compile_expression("sum(x * y)")
        self.assertIn("out = (x * y)", code)
        self.assertIn("tl.atomic_add(out_ptr, tl.sum(tl.where(mask, out, 0.0), axis=0))", code)
        self.assertNotIn("tl.store", code)
        code, _ = compile_expression("min(x)")
        self.assertIn('tl.atomic_min(out_ptr, tl.min(tl.where(mask, out, float("inf")), axis=0))', code)

    def test_multi_output_mixes_store_and_reduction(self):
        code, _ = compile_program("out0 = x * y; out1 = max(x * y)")
        self.assertIn("tl.store(out0_ptr + offsets, out0, mask=mask)", code)
        self.assertIn("tl.atomic_max(out1_ptr, tl.max(", code)

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


    def test_reductions_match_pytorch(self):
        x = torch.randn(64)
        y = torch.randn(64)
        for source, expected in [
            ("sum(x * y)", torch.sum(x * y)),
            ("max(x - y)", torch.max(x - y)),
            ("min(sin(x))", torch.min(torch.sin(x))),
        ]:
            with self.subTest(source=source):
                ref = evaluate_ast(parse_expression(source), {"x": x, "y": y})
                self.assertEqual(ref.shape, (1,))
                self.assertTrue(torch.allclose(ref, expected.reshape(1)))


class TestCompiler(unittest.TestCase):
    def test_end_to_end_codegen(self):
        code, program = compile_expression("x * y - 2")
        self.assertIn("((x * y) - 2)", code)
        self.assertEqual(program.inputs, frozenset({"x", "y"}))


if __name__ == "__main__":
    unittest.main()
