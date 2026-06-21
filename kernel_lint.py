"""Kernel linting before GPU execution."""

import re

from feedback import LintIssue

BASE_REQUIRED_TOKENS = [
    "tl.load",
    "tl.store",
    "@triton.jit",
    "program_id",
    "offsets",
    "mask",
    "x_ptr",
    "y_ptr",
]

BANNED_TOKENS = [
    "numpy",
    "torch",
    "tl.tensor",
    "import os",
    "eval(",
    "exec(",
    "__import__",
]


def is_multi_kernel(code: str) -> bool:
    return bool(re.search(r"out\d+_ptr", code)) or code.count("tl.store") > 1


def lint_kernel(code: str) -> list[LintIssue]:
    issues: list[LintIssue] = []
    multi = is_multi_kernel(code)

    required = list(BASE_REQUIRED_TOKENS)
    if multi:
        if not re.search(r"out\d+_ptr", code):
            issues.append(LintIssue("missing_required", "outN_ptr"))
    else:
        required.append("out_ptr")

    for token in required:
        if token not in code:
            issues.append(LintIssue("missing_required", token))

    for token in BANNED_TOKENS:
        if token in code:
            issues.append(LintIssue("banned_token", token))

    if multi:
        if not re.search(r"out\d+\s*=", code):
            issues.append(LintIssue("missing_assign", "outN = ..."))
    elif not re.search(r"^\s*out\s*=", code, re.MULTILINE):
        issues.append(LintIssue("missing_assign", "out = ..."))

    return issues


def is_valid(code: str) -> bool:
    return not lint_kernel(code)


def extract_output_expr(code: str) -> str | None:
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("out ="):
            return stripped[len("out =") :].strip()
    return None
