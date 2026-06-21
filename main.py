import argparse

from ai_optimizer import run_ai_compiler
from benchmark import run_benchmark
from config import DEFAULT_N
from pipeline import run_pipeline
from report import save_speedup_graph, save_text_report


def read_input() -> str:
    print("Enter expression or multi-output program.")
    print("  Single: x * y + sin(x)")
    print("  Multi:  out0 = x * y; out1 = x + y; out2 = x * y + sin(x)")
    print("  (paste lines, then a blank line to finish)")
    lines: list[str] = []
    while True:
        line = input("> " if not lines else "> ")
        if not line.strip():
            break
        lines.append(line)
    if not lines:
        return input("Enter expression: ").strip()
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI GPU expression compiler")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Compile, verify, then benchmark vs PyTorch with BLOCK_SIZE autotune",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Full AI compiler: compile, benchmark, LLM optimize, re-benchmark",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help=f"Benchmark tensor size (default {DEFAULT_N:,})",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM fix/optimize steps",
    )
    args = parser.parse_args()

    source = read_input()

    if args.optimize:
        result = run_ai_compiler(
            source,
            n=args.n,
            use_llm=not args.no_llm,
        )
        mode = "multi-output" if result.multi else "single-output"
        print(f"\nMode: {mode}")
        print(f"Source: {result.source}")
        if not result.success:
            print("Compilation failed")
            exit(1)
        if result.benchmark:
            print("\nBenchmark:")
            print(result.benchmark.summary())
        if result.report_path:
            print(f"Report: {result.report_path}")
        if result.graph_path:
            print(f"Graph: {result.graph_path}")
        if result.plans:
            print(f"Optimizer rounds: {len(result.plans)}")
        print(f"Best BLOCK_SIZE: {result.block_size}")
        print("Saved optimized kernel")
        return

    result = run_pipeline(source, use_llm=not args.no_llm)
    mode = "multi-output" if result.multi else "single-output"
    print(f"\nMode: {mode}")
    print(f"Source: {result.expression}")
    print(f"Attempts: {result.attempts}")

    if not result.success:
        if result.last_feedback:
            print(f"Failed: {result.last_feedback.brief()}")
            print("\nStructured feedback:")
            print(result.last_feedback.to_prompt())
        exit(1)

    print("Kernel correct")
    print("Saved kernel")

    if args.benchmark:
        try:
            bench = run_benchmark(result.expression, n=args.n, autotune=True)
            print("\nBenchmark:")
            print(bench.summary())
            print(f"Report: {save_text_report(bench)}")
            graph = save_speedup_graph(bench)
            if graph:
                print(f"Graph: {graph}")
        except RuntimeError as error:
            print(f"\nBenchmark skipped: {error}")


if __name__ == "__main__":
    main()
