import torch

from compiler import compile_program
from evaluator import evaluate_ast, evaluate_multi
from feedback import DiffStats
from ir import IRMultiProgram
from parser import parse_expression, parse_program


def reference(expression, x, y):
    tree = parse_expression(expression)
    return evaluate_ast(tree, {"x": x, "y": y})


def reference_program(source: str, x, y) -> dict[str, torch.Tensor]:
    _, program = compile_program(source)
    return evaluate_multi(program, {"x": x, "y": y})


def check(x, y, out, expression, *, atol=1e-3):
    ref = reference(expression, x, y)
    return torch.allclose(out, ref, atol=atol)


def check_multi(x, y, outputs: dict[str, torch.Tensor], source: str, *, atol=1e-3) -> bool:
    refs = reference_program(source, x, y)
    for name, ref in refs.items():
        if name not in outputs:
            return False
        if not torch.allclose(outputs[name], ref, atol=atol):
            return False
    return True


def compute_diff(x, y, out, expression, *, atol=1e-3, sample_count=5) -> DiffStats | None:
    ref = reference(expression, x, y)
    return _tensor_diff(out, ref, atol=atol, sample_count=sample_count)


def compute_multi_diff(
    x,
    y,
    outputs: dict[str, torch.Tensor],
    source: str,
    *,
    atol=1e-3,
    sample_count=5,
) -> DiffStats | None:
    refs = reference_program(source, x, y)
    worst: DiffStats | None = None
    for name, ref in refs.items():
        diff = _tensor_diff(outputs[name], ref, atol=atol, sample_count=sample_count)
        if diff is None:
            continue
        if worst is None or diff.max_abs_diff > worst.max_abs_diff:
            worst = DiffStats(
                max_abs_diff=diff.max_abs_diff,
                mean_abs_diff=diff.mean_abs_diff,
                mismatch_count=diff.mismatch_count,
                total_elements=diff.total_elements,
                worst_indices=diff.worst_indices,
                expected_samples=diff.expected_samples,
                actual_samples=diff.actual_samples,
            )
    return worst


def _tensor_diff(out, ref, *, atol=1e-3, sample_count=5) -> DiffStats | None:
    if torch.allclose(out, ref, atol=atol):
        return None

    diff = (out - ref).abs()
    mismatch_mask = diff > atol
    mismatch_count = int(mismatch_mask.sum().item())
    total = out.numel()

    worst_indices: list[int] = []
    expected_samples: list[float] = []
    actual_samples: list[float] = []

    if mismatch_count:
        flat_diff = diff.flatten()
        flat_ref = ref.flatten()
        flat_out = out.flatten()
        k = min(sample_count, mismatch_count)
        _, worst = torch.topk(flat_diff, k)
        for idx in worst.tolist():
            worst_indices.append(int(idx))
            expected_samples.append(float(flat_ref[idx].item()))
            actual_samples.append(float(flat_out[idx].item()))

    return DiffStats(
        max_abs_diff=float(diff.max().item()),
        mean_abs_diff=float(diff.mean().item()),
        mismatch_count=mismatch_count,
        total_elements=total,
        worst_indices=worst_indices,
        expected_samples=expected_samples,
        actual_samples=actual_samples,
    )
