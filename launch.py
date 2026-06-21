"""GPU kernel launch helpers with configurable tensor size and block size."""

from __future__ import annotations

import importlib.util

import torch

from compiler import compile_program, is_multi_output
from config import DEFAULT_BLOCK_SIZE, DEFAULT_N, VERIFY_N


def load_kernel():
    spec = importlib.util.spec_from_file_location(
        "kernel_module",
        "kernels/kernel.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.kernel


def make_single_tensors(n: int = DEFAULT_N, *, device: str = "cuda"):
    x = torch.randn(n, device=device, dtype=torch.float32)
    y = torch.randn(n, device=device, dtype=torch.float32)
    out = torch.zeros(n, device=device, dtype=torch.float32)
    return x, y, out


def make_multi_tensors(source: str, n: int = DEFAULT_N, *, device: str = "cuda"):
    _, program = compile_program(source)
    x = torch.randn(n, device=device, dtype=torch.float32)
    y = torch.randn(n, device=device, dtype=torch.float32)
    outputs = {
        assignment.name: torch.zeros(n, device=device, dtype=torch.float32)
        for assignment in program.outputs
    }
    return x, y, outputs, program


def launch_single(
    kernel,
    x: torch.Tensor,
    y: torch.Tensor,
    out: torch.Tensor,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> None:
    import triton

    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    kernel[grid](x, y, out, n, block_size)


def launch_multi(
    kernel,
    x: torch.Tensor,
    y: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    program,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> None:
    import triton

    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    args = [x, y] + [outputs[assignment.name] for assignment in program.outputs]
    args.extend([n, block_size])
    kernel[grid](*args)


def launch_expression(
    source: str,
    *,
    n: int = VERIFY_N,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    kernel = load_kernel()
    if is_multi_output(source):
        x, y, outputs, program = make_multi_tensors(source, n=n)
        launch_multi(kernel, x, y, outputs, program, block_size=block_size)
        return

    x, y, out = make_single_tensors(n=n)
    launch_single(kernel, x, y, out, block_size=block_size)
