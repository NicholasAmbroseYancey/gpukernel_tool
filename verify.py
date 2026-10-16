import torch

from compiler import compile_program, is_multi_output
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


def _close(out, ref, *, atol):
    # NaN where the reference is NaN is correct: log(x), sqrt(x), asin(x),
    # x ** 2.5 ... are NaN for part of the random inputs.
    return torch.allclose(out, ref, atol=atol, equal_nan=True)


def check(x, y, out, expression, *, atol=1e-3):
    ref = reference(expression, x, y)
    return _close(out, ref, atol=atol)


def check_multi(x, y, outputs: dict[str, torch.Tensor], source: str, *, atol=1e-3) -> bool:
    refs = reference_program(source, x, y)
    for name, ref in refs.items():
        if name not in outputs:
            return False
        if not _close(outputs[name], ref, atol=atol):
            return False
    return True


def same_outputs(source_a: str, source_b: str, *, n: int = 1024, atol=1e-3) -> bool:
    """Whether two sources compute the same outputs, checked in PyTorch on
    random inputs. Catches LLM rewrites that change the math."""
    x = torch.randn(n)
    y = torch.randn(n)
    multi = is_multi_output(source_a)
    if multi != is_multi_output(source_b):
        return False
    if multi:
        refs_a = reference_program(source_a, x, y)
        refs_b = reference_program(source_b, x, y)
        if refs_a.keys() != refs_b.keys():
            return False
        pairs = [(refs_a[name], refs_b[name]) for name in refs_a]
    else:
        pairs = [(reference(source_a, x, y), reference(source_b, x, y))]
    return all(a.shape == b.shape and _close(a, b, atol=atol) for a, b in pairs)


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
    if _close(out, ref, atol=atol):
        return None

    mismatch_mask = ~torch.isclose(out, ref, atol=atol, equal_nan=True)
    # NaN on both sides is a match; NaN on one side is an infinite difference.
    diff = (out - ref).abs()
    diff = torch.where(diff.isnan(), torch.where(mismatch_mask, float("inf"), 0.0), diff)
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
