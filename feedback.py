"""Structured error feedback for the self-correcting compiler loop."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DiffStats:
    max_abs_diff: float
    mean_abs_diff: float
    mismatch_count: int
    total_elements: int
    worst_indices: list[int] = field(default_factory=list)
    expected_samples: list[float] = field(default_factory=list)
    actual_samples: list[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"max_abs_diff={self.max_abs_diff:.6g}",
            f"mean_abs_diff={self.mean_abs_diff:.6g}",
            f"mismatch_count={self.mismatch_count}/{self.total_elements}",
        ]
        if self.worst_indices:
            lines.append(f"worst_indices={self.worst_indices[:5]}")
            lines.append(f"expected={self.expected_samples[:5]}")
            lines.append(f"actual={self.actual_samples[:5]}")
        return "\n".join(lines)


@dataclass(frozen=True)
class LintIssue:
    kind: str
    detail: str


@dataclass
class FailureFeedback:
    stage: str
    message: str
    expression: str
    kernel_code: str
    output_expr: str | None = None
    lint_issues: list[LintIssue] | None = None
    diff: DiffStats | None = None
    exception_type: str | None = None

    def to_prompt(self) -> str:
        lines = [
            "KERNEL DEBUG REPORT",
            f"stage: {self.stage}",
            f"message: {self.message}",
            f"target_expression: {self.expression}",
        ]
        if self.output_expr:
            lines.append(f"generated_output_expr: {self.output_expr}")
        if self.exception_type:
            lines.append(f"exception_type: {self.exception_type}")
        if self.lint_issues:
            lines.append("lint_issues:")
            for issue in self.lint_issues:
                lines.append(f"  - {issue.kind}: {issue.detail}")
        if self.diff:
            lines.append("verify_diff:")
            lines.append(self.diff.summary())
        lines.append("")
        lines.append("Return ONLY one of:")
        lines.append("1) A corrected math expression using x, y and whitelisted funcs")
        lines.append("2) TRITON: <triton rhs expression> if the bug is Triton-specific")
        return "\n".join(lines)

    def brief(self) -> str:
        parts = [f"[{self.stage}] {self.message}"]
        if self.diff:
            parts.append(
                f"max_diff={self.diff.max_abs_diff:.4g}, "
                f"mismatches={self.diff.mismatch_count}/{self.diff.total_elements}"
            )
        if self.lint_issues:
            parts.append(f"lint={len(self.lint_issues)} issue(s)")
        return " | ".join(parts)
