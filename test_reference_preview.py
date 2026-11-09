import pytest

from parser import ParseError
from reference_preview import build_reference_preview


def test_reference_multiplication():
    result = build_reference_preview("x * y")

    assert result["mode"] == "pytorch_cpu_reference"
    assert result["kernel_executed"] is False
    assert result["performance_measured"] is False
    assert result["outputs"]["out"] == [0.5, 1.5, 4.0, 10.0]


def test_reference_multi_output():
    result = build_reference_preview("out0 = x + y; out1 = x * y")

    assert result["outputs"]["out0"] == [2.25, 3.5, 5.0, 7.0]
    assert result["outputs"]["out1"] == [0.5, 1.5, 4.0, 10.0]


def test_invalid_expression_rejected():
    with pytest.raises(ParseError):
        build_reference_preview("unknown_function(x)")


def test_empty_expression_rejected():
    with pytest.raises(ValueError, match="An expression is required"):
        build_reference_preview("")


def test_expression_over_length_limit_rejected():
    with pytest.raises(ValueError, match="Expression is too long"):
        build_reference_preview("x" * 401)
