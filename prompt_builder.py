from collections.abc import Sequence
from dataclasses import dataclass

from ops import ALLOWED_FUNCS, FUNC_ALIASES, REDUCTIONS
from feedback import FailureFeedback


# Why the optimizer loop rejected (or accepted) a plan, phrased for the LLM.
OUTCOME_REASONS = {
    "invalid": "rejected: plan did not parse or compile",
    "changes_math": "rejected: changes what the kernel computes",
    "repeat": "rejected: same kernel and BLOCK_SIZE as an earlier attempt",
    "verify_failed": "rejected: kernel output did not match PyTorch",
    "within_margin": "rejected: faster, but within the benchmark noise margin",
    "no_improvement": "rejected: not faster than the current best",
    "improved": "accepted as the new best",
}

# Only the most recent attempts go in the prompt, to keep it short for a 7B model.
MAX_HISTORY = 8
_MAX_PLAN_CHARS = 160


@dataclass(frozen=True)
class PlanAttempt:
    """One optimizer round: the plan the LLM proposed and what happened to it."""

    round_idx: int
    plan: str  # OptimizerPlan.summary()
    outcome: str  # a key of OUTCOME_REASONS
    detail: str = ""

    def to_prompt_line(self) -> str:
        plan = self.plan if len(self.plan) <= _MAX_PLAN_CHARS else self.plan[: _MAX_PLAN_CHARS - 3] + "..."
        reason = OUTCOME_REASONS.get(self.outcome, self.outcome)
        line = f"  round {self.round_idx}: {plan or '(empty plan)'} -> {reason}"
        if self.detail:
            line += f" ({self.detail})"
        return line


def _history_section(history: Sequence[PlanAttempt]) -> str:
    if not history:
        return ""
    recent = list(history)[-MAX_HISTORY:]
    lines = "\n".join(attempt.to_prompt_line() for attempt in recent)
    return f"""
Previous attempts this run:
{lines}

Do NOT propose any plan listed above again, including the same rewrite with a
different spelling. If a rewrite was rejected for changing the math, it was not
equivalent; do not try variations of it. Try a different idea instead.
"""


def _allowed_func_list() -> str:
    names = sorted(ALLOWED_FUNCS | REDUCTIONS | set(FUNC_ALIASES))
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
- sum/max/min with one argument reduce to a single value and must be the whole output (example: sum(x * y))
- If the failure is Triton-specific, prefix with TRITON: and return only the rhs
- Do NOT return a full kernel, markdown, or explanation
"""


def optimize_prompt(
    *,
    source: str,
    benchmark,
    ir_summary: str,
    round_idx: int,
    history: Sequence[PlanAttempt] = (),
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
{_history_section(history)}
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
- sum/max/min with one argument reduce to a single value and must be the whole output (example: sum(x * y))
- Suggest PROGRAM (multi-output fusion) when subexpressions repeat
- Examples: fuse x*y reused across outputs; rewrite x*(y+1) as x*y+x
- BLOCK_SIZE should target better occupancy/memory coalescing
- Do NOT return Triton code or markdown fences
"""

