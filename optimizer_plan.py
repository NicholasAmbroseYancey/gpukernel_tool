"""Structured optimization plans for the Level 7 AI compiler backend."""

from __future__ import annotations

import re
from dataclasses import dataclass

from kernel_writer import clean_output


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
            plan.expression = stripped.split(":", 1)[1].strip()
        elif upper.startswith("BLOCK_SIZE:"):
            value = stripped.split(":", 1)[1].strip()
            plan.block_size = int(value)
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
