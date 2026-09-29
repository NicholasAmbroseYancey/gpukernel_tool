"""Level 7: LLM-driven kernel optimization backend."""

from __future__ import annotations

from dataclasses import dataclass, field

from benchmark import BenchmarkReport, run_benchmark, verify_correctness
from compiler import compile_expression, compile_program, is_multi_output
from ir_analysis import analyze_source
from kernel_writer import save_kernel_source
from ollama_client import generate
from optimizer_plan import OptimizerPlan, parse_optimizer_plan
from parser import ParseError, parse_expression, parse_program
from pipeline import run_pipeline
from prompt_builder import optimize_prompt
from report import save_speedup_graph, save_text_report


@dataclass
class AICompilerResult:
    success: bool
    source: str
    kernel_code: str
    benchmark: BenchmarkReport | None
    block_size: int
    plans: list[OptimizerPlan] = field(default_factory=list)
    report_path: str | None = None
    graph_path: str | None = None
    multi: bool = False


def apply_plan(source: str, plan: OptimizerPlan) -> tuple[str, str]:
    resolved = plan.resolved_source(source)
    if plan.program or is_multi_output(resolved):
        code, _ = compile_program(resolved)
    else:
        code, _ = compile_expression(resolved)
    return resolved, code


def run_ai_compiler(
    source: str,
    *,
    n: int = 1 << 20,
    max_rounds: int = 3,
    use_llm: bool = True,
    autotune: bool = True,
) -> AICompilerResult:
    compile_result = run_pipeline(source, use_llm=use_llm)
    if not compile_result.success:
        return AICompilerResult(
            success=False,
            source=source,
            kernel_code=compile_result.kernel_code,
            benchmark=None,
            block_size=256,
            multi=compile_result.multi,
        )

    current_source = compile_result.expression
    current_code = compile_result.kernel_code
    save_kernel_source(current_code)

    benchmark = run_benchmark(current_source, n=n, autotune=autotune)
    best_source = current_source
    best_code = current_code
    best_benchmark = benchmark
    best_block_size = benchmark.block_size
    plans: list[OptimizerPlan] = []

    if not use_llm:
        return _finalize(best_source, best_code, best_benchmark, best_block_size, plans, compile_result.multi)

    for round_idx in range(1, max_rounds + 1):
        ir_summary = analyze_source(best_source)
        prompt = optimize_prompt(
            source=best_source,
            benchmark=best_benchmark,
            ir_summary=ir_summary,
            round_idx=round_idx,
        )
        response = generate(prompt)
        plan = parse_optimizer_plan(response)
        plans.append(plan)
        print(f"Optimizer round {round_idx}: {plan.summary()}")

        try:
            if plan.program:
                parse_program(plan.resolved_source(best_source))
            elif plan.expression:
                parse_expression(plan.resolved_source(best_source))
            candidate_source, candidate_code = apply_plan(best_source, plan)
        except (ParseError, ValueError) as error:
            print(f"  skipped invalid plan: {error}")
            continue

        save_kernel_source(candidate_code)
        block_size = plan.block_size or best_block_size
        if not verify_correctness(candidate_source, block_size=block_size):
            print("  skipped: verification failed")
            continue

        candidate_bench = run_benchmark(
            candidate_source,
            n=n,
            autotune=autotune and plan.block_size is None,
            block_size=plan.block_size,
        )
        if candidate_bench.triton_ms < best_benchmark.triton_ms:
            print(
                f"  improved: {best_benchmark.triton_ms:.4f} ms -> "
                f"{candidate_bench.triton_ms:.4f} ms "
                f"({candidate_bench.speedup:.2f}x vs PyTorch)"
            )
            best_source = candidate_source
            best_code = candidate_code
            best_benchmark = candidate_bench
            best_block_size = candidate_bench.block_size
        else:
            print(
                f"  no improvement: {candidate_bench.triton_ms:.4f} ms "
                f"(best {best_benchmark.triton_ms:.4f} ms)"
            )

    save_kernel_source(best_code)
    return _finalize(best_source, best_code, best_benchmark, best_block_size, plans, compile_result.multi)


def _finalize(
    source: str,
    code: str,
    benchmark: BenchmarkReport,
    block_size: int,
    plans: list[OptimizerPlan],
    multi: bool,
) -> AICompilerResult:
    report_path = save_text_report(benchmark)
    graph_path = save_speedup_graph(benchmark)
    return AICompilerResult(
        success=True,
        source=source,
        kernel_code=code,
        benchmark=benchmark,
        block_size=block_size,
        plans=plans,
        report_path=report_path,
        graph_path=graph_path,
        multi=multi,
    )
