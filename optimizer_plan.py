"""Structured optimization plans for the Level 7 AI compiler backend.

Owner: Bryson Wingate (LLM Optimization Engine).

Parsing here must be *total*: an LLM can return malformed, chatty, or
out-of-spec plans, and a single bad reply must never crash the optimize
loop. Unparseable fields are dropped (left at their defaults) rather than
raising.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from kernel_writer import clean_output

# Single source of truth for valid block sizes lives in config (infra-owned).
# Fall back to a local copy so this module has no hard import dependency.
try:  # pragma: no cover - trivial import guard
    from config import BLOCK_SIZE_CANDIDATES as _BLOCK_SIZE_CANDIDATES
except Exception:  # pragma: no cover
    _BLOCK_SIZE_CANDIDATES = [64, 128, 256, 512, 1024]

ALLOWED_BLOCK_SIZES = set(_BLOCK_SIZE_CANDIDATES)


def _parse_block_size(value: str) -> int | None:
    """Extract a valid BLOCK_SIZE from an LLM line, or None.

    Tolerates trailing prose ("256 (better occupancy)") and non-numeric
    values ("auto", "keep"). Returns the integer only if it is one of the
    allowed candidates; otherwise None so callers fall back to autotune.
    """
    match = re.search(r"\d+", value)
    if not match:
        return None
    size = int(match.group())
    return size if size in ALLOWED_BLOCK_SIZES else None


@dataclass
class OptimizerPlan:
    expression: str | None = None
    program: str | None = None
    block_size: int | None = None
    fuse: bool = True
    reason: str = ""

    def resolved_source(self, fallback: str) -> str:
        if self.program:
            return self.program.strip()
        if self.expression:
            return self.expression.strip()
        return fallback.strip()

    def summary(self) -> str:
        parts = []
        if self.program:
            parts.append(f"program={self.program!r}")
        elif self.expression:
            parts.append(f"expression={self.expression!r}")
        if self.block_size is not None:
            parts.append(f"block_size={self.block_size}")
        parts.append(f"fuse={'yes' if self.fuse else 'no'}")
        if self.reason:
            parts.append(f"reason={self.reason}")
        return ", ".join(parts)


def parse_optimizer_plan(response: str) -> OptimizerPlan:
    text = clean_output(response).strip()
    plan = OptimizerPlan()

    program_lines: list[str] = []
    in_program = False
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("PROGRAM:"):
            in_program = True
            remainder = stripped.split(":", 1)[1].strip()
            if remainder:
                program_lines.append(remainder)
            continue

        if in_program:
            if re.match(r"^[A-Z_]+:", stripped):
                in_program = False
            else:
                program_lines.append(stripped)
                continue

        if upper.startswith("EXPRESSION:"):
            plan.expression = stripped.split(":", 1)[1].strip() or None
        elif upper.startswith("BLOCK_SIZE:"):
            plan.block_size = _parse_block_size(stripped.split(":", 1)[1])
        elif upper.startswith("FUSE:"):
            plan.fuse = stripped.split(":", 1)[1].strip().lower() in {"1", "true", "yes", "on"}
        elif upper.startswith("REASON:"):
            plan.reason = stripped.split(":", 1)[1].strip()

    if program_lines:
        plan.program = "\n".join(program_lines)

    if not plan.expression and not plan.program:
        first = text.splitlines()[0].strip() if text else ""
        if first and not re.match(r"^[A-Z_]+:", first):
            plan.expression = first

    return plan
