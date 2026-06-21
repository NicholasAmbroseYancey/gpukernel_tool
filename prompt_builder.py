from ops import ALLOWED_FUNCS, FUNC_ALIASES
from feedback import FailureFeedback


def _allowed_func_list() -> str:
    names = sorted(ALLOWED_FUNCS | set(FUNC_ALIASES))
    return ", ".join(names)


def fix_prompt(feedback: FailureFeedback) -> str:
    return f"""
You are the debugging backend for a GPU expression compiler.

The compiler generates a fixed Triton kernel skeleton and fills in one compute line:
    out = <expression>

{feedback.to_prompt()}

Rules for your fix:
- Prefer returning a corrected math expression (example: x * y + sin(x))
- Use only variables x, y and functions: {_allowed_func_list()}
- Aliases ln/tg are rewritten to log/tan automatically
- If the failure is Triton-specific, prefix with TRITON: and return only the rhs
- Do NOT return a full kernel, markdown, or explanation
"""


def optimize_prompt(
    *,
    source: str,
    benchmark,
    ir_summary: str,
    round_idx: int,
) -> str:
    sweep_lines = []
    for item in benchmark.block_size_sweep:
        sweep_lines.append(
            f"  block_size={item.block_size}: {item.median_ms:.4f} ms "
            f"({item.throughput_gels:.2f} Melem/s)"
        )
    sweep_text = "\n".join(sweep_lines) if sweep_lines else "  (no sweep data)"

    return f"""
You are the optimization backend for an AI GPU compiler.

Goal: improve Triton kernel performance while preserving mathematical correctness.

Current source:
{source}

IR analysis:
{ir_summary}

Benchmark (n={benchmark.n:,}):
  triton={benchmark.triton_ms:.4f} ms
  pytorch={benchmark.pytorch_ms:.4f} ms
  speedup={benchmark.speedup:.2f}x
  best_block_size={benchmark.block_size}
  memory: {benchmark.memory.summary()}

BLOCK_SIZE sweep:
{sweep_text}

Optimization round: {round_idx}

Return a plan using EXACTLY this format (omit lines to keep current value):
EXPRESSION: <single math expression, or leave empty>
PROGRAM:
out0 = ...
out1 = ...
BLOCK_SIZE: <64|128|256|512|1024>
FUSE: yes
REASON: <one line why this helps>

Rules:
- Use only variables x, y and functions: {_allowed_func_list()}
- Prefer mathematically equivalent rewrites that reduce redundant work
- Suggest PROGRAM (multi-output fusion) when subexpressions repeat
- Examples: fuse x*y reused across outputs; rewrite x*(y+1) as x*y+x
- BLOCK_SIZE should target better occupancy/memory coalescing
- Do NOT return Triton code or markdown fences
"""

