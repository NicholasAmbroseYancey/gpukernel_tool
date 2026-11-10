"""CPU-only PyTorch reference preview for supported expressions."""

import math

import torch

from compiler import is_multi_output
from verify import reference, reference_program


def build_reference_preview(expression: str) -> dict[str, object]:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("An expression is required.")

    if len(expression) > 400:
        raise ValueError("Expression is too long for preview.")

    x = torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=torch.float32)
    y = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float32)

    with torch.no_grad():
        if is_multi_output(expression):
            raw_outputs = reference_program(expression, x, y)
        else:
            raw_outputs = {"out": reference(expression, x, y)}

    outputs = {}
    for name, tensor in raw_outputs.items():
        values = tensor.detach().cpu().flatten().tolist()
        outputs[name] = [
            round(float(value), 6) if math.isfinite(float(value)) else None
            for value in values
        ]

    return {
        "mode": "pytorch_cpu_reference",
        "kernel_executed": False,
        "performance_measured": False,
        "inputs": {"x": x.tolist(), "y": y.tolist()},
        "outputs": outputs,
    }
