"""@triton.jit helpers that generated kernels include when they call them."""

from __future__ import annotations

import os
import re

# On a GPU, pow uses CUDA's powf, as PyTorch does: within 1 ulp, IEEE 754 special
# cases included, and as fast as the old float32 helper. The float64 helper
# above was 8x slower on an RTX 4070, so it is only used under the interpreter.
GPU_POW = '''
from triton.language.extra import libdevice


@triton.jit
def _gk_pow(b, e):
    return libdevice.pow(tl.cast(b, tl.float32), tl.cast(e, tl.float32))
'''

# Built from tl primitives because libdevice doesn't run under TRITON_INTERPRET=1.
HELPERS = {
    "_gk_pow": '''
@triton.jit
def _gk_pow(b, e):
    # exp2 and log2 in float64: in float32, rounding e * log2(|b|) cost up to
    # ~65 ulp. The interpreter runs on the CPU, where float64 is cheap.
    r = tl.exp2(tl.log2(tl.abs(b).to(tl.float64)) * e).to(tl.float32)
    e_int = tl.floor(e) == e
    e_odd = tl.floor(e * 0.5) * 2.0 != e
    # The sign bit, so (-0.0) ** -1 is -inf. A negative base needs an integer e,
    # except -inf: (-inf) ** 0.5 is inf. r * -1.0, not -r, which turns -0.0 into 0.0.
    b_neg = tl.where(b == 0, tl.div_rn(1.0, b), b) < 0
    non_int = tl.where(b < 0, tl.where(b == float("-inf"), r, float("nan")), r)
    neg = tl.where(e_int, tl.where(e_odd, r * -1.0, r), non_int)
    r = tl.where(b_neg, neg, r)
    # IEEE 754 pow: 1 ** e and (-1) ** +-inf are 1, even for a NaN e.
    # Nested tl.where, not & and |: the interpreter can't combine a constant e's bools.
    r = tl.where(b == -1.0, tl.where(tl.abs(e) == float("inf"), 1.0, r), r)
    r = tl.where(b == 1.0, 1.0, r)
    return tl.where(e == 0, 1.0, r)
''',
    # tl.exp lowers to a fast exp2(v * log2(e)) on a GPU. Rounding that product
    # costs up to ~55 ulp for large |v|, which fails verification for exp(exp(x)).
    # Cody-Waite reduction: v = n * ln2 + r, so exp2 only sees a small argument.
    "_gk_exp": '''
@triton.jit
def _gk_exp(v):
    # exp is inf above 89 and 0 below -104 in float32. The clamp also keeps
    # +-inf and huge |v| from turning r into inf - inf = NaN.
    v = tl.minimum(tl.maximum(v, -104.0, propagate_nan=tl.PropagateNan.ALL), 89.0, propagate_nan=tl.PropagateNan.ALL)
    n = tl.floor(v * 1.4426950408889634 + 0.5)
    r = (v - n * 0.693145751953125) - n * 1.428606765330187e-06
    # 2 ** n in two halves: 2 ** 128 overflows although exp(88.7) doesn't.
    h = tl.floor(n * 0.5)
    return tl.exp2(r * 1.4426950408889634) * tl.exp2(h) * tl.exp2(n - h)
''',
    "_gk_sigmoid": '''
@triton.jit
def _gk_sigmoid(v):
    return tl.div_rn(1.0, 1.0 + _gk_exp(-v))
''',
    # 2 * sigmoid(2v) - 1 cancels near 0: it gave tanh(1e-8) = 0.
    # Cephes tanhf: a polynomial for |v| < 0.625, 1 - 2 / (exp(2|v|) + 1) above.
    "_gk_tanh": '''
@triton.jit
def _gk_tanh(v):
    a = tl.abs(v)
    z = v * v
    small = ((((-5.70498872745e-3 * z + 2.06390887954e-2) * z - 5.37397155531e-2) * z + 1.33314422036e-1) * z - 3.33332819422e-1) * z * v + v
    big = 1.0 - 2.0 / (_gk_exp(2.0 * a) + 1.0)
    return tl.where(a < 0.625, small, tl.where(v < 0, -big, big))
''',
    "_gk_atan": '''
@triton.jit
def _gk_atan(v):
    a = tl.abs(v)
    big = a > 2.414213562373095
    mid = a > 0.4142135623730951
    t = tl.where(big, -1.0 / a, tl.where(mid, (a - 1.0) / (a + 1.0), a))
    base = tl.where(big, 1.5707963267948966, tl.where(mid, 0.7853981633974483, 0.0))
    z = t * t
    p = (((8.05374449538e-2 * z - 1.38776856032e-1) * z + 1.99777106478e-1) * z - 3.33329491539e-1) * z * t + t
    r = base + p
    # atan(+-0) is +-0: v itself, since base + p loses the sign of -0.0.
    return tl.where(a == 0, v, tl.where(v < 0, -r, r))
''',
    "_gk_atan2": '''
@triton.jit
def _gk_atan2(y, x):
    # Sign bits, not < 0: IEEE atan2(-0.0, -1) is -pi and atan2(0.0, -0.0) is pi.
    y_neg = tl.where(y == 0, tl.div_rn(1.0, y), y) < 0
    x_neg = tl.where(x == 0, tl.div_rn(1.0, x), x) < 0
    # inf / inf is NaN, but atan2(+-inf, +-inf) is +-pi/4 or +-3pi/4.
    both_inf = tl.where(tl.abs(x) == float("inf"), tl.abs(y) == float("inf"), False)
    q = tl.where(both_inf, tl.where(y_neg == x_neg, 1.0, -1.0), y / x)
    r = _gk_atan(q)
    r = tl.where(x_neg, tl.where(y_neg, r - 3.141592653589793, r + 3.141592653589793), r)
    # y / x is NaN for a NaN y, but that branch is replaced when x is 0.
    on_y_axis = tl.where(y != y, y, tl.where(y_neg, -1.5707963267948966, 1.5707963267948966))
    at_origin = tl.where(x_neg, tl.where(y_neg, -3.141592653589793, 3.141592653589793), y)
    return tl.where(x == 0, tl.where(y == 0, at_origin, on_y_axis), r)
''',
    "_gk_asin": '''
@triton.jit
def _gk_asin(v):
    return _gk_atan2(v, tl.sqrt((1.0 - v) * (1.0 + v)))
''',
    "_gk_acos": '''
@triton.jit
def _gk_acos(v):
    return _gk_atan2(tl.sqrt((1.0 - v) * (1.0 + v)), v)
''',
}

HELPER_DEPS = {
    "_gk_sigmoid": ("_gk_exp",),
    "_gk_tanh": ("_gk_exp",),
    "_gk_atan2": ("_gk_atan",),
    "_gk_asin": ("_gk_atan2",),
    "_gk_acos": ("_gk_atan2",),
}


def helpers_for(code: str) -> str:
    needed: list[str] = []

    def add(name: str) -> None:
        if name in needed:
            return
        for dep in HELPER_DEPS.get(name, ()):
            add(dep)
        needed.append(name)

    for name in HELPERS:
        if re.search(rf"\b{name}\(", code):
            add(name)
    return "".join(_helper_source(name) for name in needed)


def _helper_source(name: str) -> str:
    if name == "_gk_pow" and os.getenv("TRITON_INTERPRET") != "1":
        return GPU_POW
    return HELPERS[name]
