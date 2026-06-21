"""Level 6: kernel benchmarking, memory analysis, and BLOCK_SIZE autotune."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import torch

from compiler import compile_program, is_multi_output
from config import (
    BENCHMARK_REPEATS,
    BENCHMARK_WARMUP,
    BLOCK_SIZE_CANDIDATES,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_N,
)
from evaluator import evaluate_ast, evaluate_multi
from launch import (
    launch_multi,
    launch_single,
    load_kernel,
    make_multi_tensors,
    make_single_tensors,
)
from parser import parse_expression
from kernel_writer import save_kernel_source


@dataclass
class MemoryStats:
    io_bytes: int
    peak_bytes: int
    input_bytes: int
    output_bytes: int

    def summary(self) -> str:
        return (
            f"io={self.io_bytes / 1e6:.2f} MB, "
            f"peak={self.peak_bytes / 1e6:.2f} MB, "
            f"read={self.input_bytes / 1e6:.2f} MB, "
            f"write={self.output_bytes / 1e6:.2f} MB"
        )


@dataclass
class BlockSizeResult:
    block_size: int
    median_ms: float
    throughput_gels: float


@dataclass
class BenchmarkReport:
    source: str
    n: int
    block_size: int
    triton_ms: float
    pytorch_ms: float
    speedup: float
    memory: MemoryStats
    block_size_sweep: list[BlockSizeResult] = field(default_factory=list)
    multi: bool = False

    def summary(self) -> str:
        lines = [
            f"n={self.n:,}  block_size={self.block_size}",
            f"triton={self.triton_ms:.4f} ms  pytorch={self.pytorch_ms:.4f} ms  speedup={self.speedup:.2f}x",
            f"memory: {self.memory.summary()}",
            f"throughput: {self.n / (self.triton_ms / 1000) / 1e6:.2f} Melem/s",
        ]
        if self.block_size_sweep:
            best = min(self.block_size_sweep, key=lambda item: item.median_ms)
            lines.append(f"autotune best: block_size={best.block_size} ({best.median_ms:.4f} ms)")
        return "\n".join(lines)


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _median_ms(fn, *, warmup: int = BENCHMARK_WARMUP, repeats: int = BENCHMARK_REPEATS) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    return statistics.median(timings)


def estimate_memory(n: int, *, num_inputs: int = 2, num_outputs: int = 1, elem_size: int = 4) -> MemoryStats:
    input_bytes = n * elem_size * num_inputs
    output_bytes = n * elem_size * num_outputs
    return MemoryStats(
        io_bytes=input_bytes + output_bytes,
        peak_bytes=input_bytes + output_bytes,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
    )


def _measure_peak(fn) -> int:
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def benchmark_pytorch(source: str, x: torch.Tensor, y: torch.Tensor) -> float:
    multi = is_multi_output(source)

    def run_ref():
        if multi:
            evaluate_multi(compile_program(source)[1], {"x": x, "y": y})
        else:
            tree = parse_expression(source)
            evaluate_ast(tree, {"x": x, "y": y})

    return _median_ms(run_ref)


def benchmark_triton(
    source: str,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    outputs=None,
    program=None,
) -> float:
    kernel = load_kernel()
    multi = is_multi_output(source)

    if multi:
        if outputs is None or program is None:
            x, y, outputs, program = make_multi_tensors(source, n=x.numel())

        def run_kernel():
            launch_multi(kernel, x, y, outputs, program, block_size=block_size)
    else:
        out = torch.zeros_like(x)

        def run_kernel():
            launch_single(kernel, x, y, out, block_size=block_size)

    return _median_ms(run_kernel)


def autotune_block_size(
    source: str,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    candidates: list[int] | None = None,
) -> tuple[int, list[BlockSizeResult]]:
    candidates = candidates or BLOCK_SIZE_CANDIDATES
    multi = is_multi_output(source)
    outputs = program = None
    if multi:
        _, _, outputs, program = make_multi_tensors(source, n=x.numel())

    sweep: list[BlockSizeResult] = []
    for block_size in candidates:
        try:
            ms = benchmark_triton(
                source,
                x,
                y,
                block_size=block_size,
                outputs=outputs,
                program=program,
            )
        except Exception:
            continue
        sweep.append(
            BlockSizeResult(
                block_size=block_size,
                median_ms=ms,
                throughput_gels=x.numel() / (ms / 1000) / 1e6,
            )
        )

    if not sweep:
        return DEFAULT_BLOCK_SIZE, sweep

    best = min(sweep, key=lambda item: item.median_ms)
    return best.block_size, sweep


def run_benchmark(
    source: str,
    *,
    n: int = DEFAULT_N,
    autotune: bool = True,
    block_size: int | None = None,
) -> BenchmarkReport:
    if not _cuda_available():
        raise RuntimeError("CUDA is not available")

    multi = is_multi_output(source)
    if multi:
        x, y, outputs, program = make_multi_tensors(source, n=n)
        num_outputs = len(program.outputs)
    else:
        x, y, _ = make_single_tensors(n=n)
        outputs = program = None
        num_outputs = 1

    mem = estimate_memory(n, num_outputs=num_outputs)

    sweep: list[BlockSizeResult] = []
    if autotune and block_size is None:
        block_size, sweep = autotune_block_size(source, x, y)
    else:
        block_size = block_size or DEFAULT_BLOCK_SIZE

    def peak_launch():
        kernel = load_kernel()
        if multi:
            launch_multi(kernel, x, y, outputs, program, block_size=block_size)
        else:
            out = torch.zeros_like(x)
            launch_single(kernel, x, y, out, block_size=block_size)

    mem.peak_bytes = max(mem.peak_bytes, _measure_peak(peak_launch))

    triton_ms = benchmark_triton(
        source,
        x,
        y,
        block_size=block_size,
        outputs=outputs,
        program=program,
    )
    pytorch_ms = benchmark_pytorch(source, x, y)
    speedup = pytorch_ms / triton_ms if triton_ms > 0 else 0.0

    if autotune and not sweep:
        _, sweep = autotune_block_size(source, x, y)

    return BenchmarkReport(
        source=source,
        n=n,
        block_size=block_size,
        triton_ms=triton_ms,
        pytorch_ms=pytorch_ms,
        speedup=speedup,
        memory=mem,
        block_size_sweep=sweep,
        multi=multi,
    )


def benchmark_triton_code(
    code: str,
    source: str,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> float:
    """Save the provided Triton kernel source, load it, and benchmark it.

    This lets you benchmark hand-written or LLM-generated kernels by saving
    them to `kernels/kernel.py` (the same path `load_kernel()` uses).
    """
    save_kernel_source(code)

    # reuse benchmark_triton which loads `kernels/kernel.py`
    return benchmark_triton(source, x, y, block_size=block_size)


@dataclass
class ComparativeBenchmark:
    source: str
    n: int
    block_size: int
    pytorch_ms: float
    triton_generated_ms: float | None
    triton_hand_ms: float | None
    llm_ms: float | None

    def summary(self) -> str:
        lines = [f"n={self.n:,}  block_size={self.block_size}", f"pytorch={self.pytorch_ms:.4f} ms"]
        if self.triton_generated_ms is not None:
            lines.append(f"triton_generated={self.triton_generated_ms:.4f} ms  speedup={self.pytorch_ms / self.triton_generated_ms:.2f}x")
        if self.triton_hand_ms is not None:
            lines.append(f"triton_hand={self.triton_hand_ms:.4f} ms  speedup={self.pytorch_ms / self.triton_hand_ms:.2f}x")
        if self.llm_ms is not None:
            lines.append(f"llm_kernel={self.llm_ms:.4f} ms  speedup={self.pytorch_ms / self.llm_ms:.2f}x")
        return " \n".join(lines)


def run_comparative_benchmark(
    source: str,
    *,
    n: int = DEFAULT_N,
    block_size: int | None = None,
    triton_hand_code: str | None = None,
    llm_kernel_code: str | None = None,
) -> ComparativeBenchmark:
    """Run PyTorch baseline, compiled Triton (from compiler), optional hand-written Triton, and optional LLM kernel.

    Returns timings (ms) and allows direct comparison and speedup calculations.
    """
    if not _cuda_available():
        raise RuntimeError("CUDA is not available")

    multi = is_multi_output(source)
    if multi:
        x, y, outputs, program = make_multi_tensors(source, n=n)
    else:
        x, y, _ = make_single_tensors(n=n)
        outputs = program = None

    chosen_block = block_size or DEFAULT_BLOCK_SIZE

    # baseline pytorch
    pytorch_ms = benchmark_pytorch(source, x, y)

    # compiled triton (from compiler/save kernel used by load_kernel)
    triton_generated_ms = None
    try:
        # assume kernels/kernel.py currently contains the generated kernel
        triton_generated_ms = benchmark_triton(source, x, y, block_size=chosen_block, outputs=outputs, program=program)
    except Exception:
        triton_generated_ms = None

    # hand-written triton code
    triton_hand_ms = None
    if triton_hand_code is not None:
        try:
            triton_hand_ms = benchmark_triton_code(triton_hand_code, source, x, y, block_size=chosen_block)
        except Exception:
            triton_hand_ms = None

    # LLM-generated kernel code
    llm_ms = None
    if llm_kernel_code is not None:
        try:
            llm_ms = benchmark_triton_code(llm_kernel_code, source, x, y, block_size=chosen_block)
        except Exception:
            llm_ms = None

    return ComparativeBenchmark(
        source=source,
        n=n,
        block_size=chosen_block,
        pytorch_ms=pytorch_ms,
        triton_generated_ms=triton_generated_ms,
        triton_hand_ms=triton_hand_ms,
        llm_ms=llm_ms,
    )


def verify_correctness(source: str, *, n: int = 1024, block_size: int = DEFAULT_BLOCK_SIZE) -> bool:
    if not _cuda_available():
        return False

    from verify import check, check_multi

    multi = is_multi_output(source)
    kernel = load_kernel()
    if multi:
        x, y, outputs, program = make_multi_tensors(source, n=n)
        launch_multi(kernel, x, y, outputs, program, block_size=block_size)
        return check_multi(x, y, outputs, source)

    x, y, out = make_single_tensors(n=n)
    launch_single(kernel, x, y, out, block_size=block_size)
    return check(x, y, out, source)
