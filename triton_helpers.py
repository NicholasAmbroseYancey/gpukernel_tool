"""@triton.jit helpers that generated kernels include when they call them."""

from __future__ import annotations

import re

# Built from tl primitives because libdevice doesn't run under TRITON_INTERPRET=1.
HELPERS = {
    "_gk_pow": '''
@triton.jit
def _gk_pow(b, e):
    r = tl.exp2(e * tl.log2(tl.abs(b)))
    e_int = tl.floor(e) == e
    e_odd = tl.floor(e * 0.5) * 2.0 != e
    neg = tl.where(e_int, tl.where(e_odd, -r, r), float("nan"))
    r = tl.where(b < 0, neg, r)
    return tl.where(e == 0, 1.0, r)
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
    return tl.where(v < 0, -r, r)
''',
    "_gk_atan2": '''
@triton.jit
def _gk_atan2(y, x):
    r = _gk_atan(y / x)
    r = tl.where(x < 0, tl.where(y < 0, r - 3.141592653589793, r + 3.141592653589793), r)
    on_axis = tl.where(y > 0, 1.5707963267948966, tl.where(y < 0, -1.5707963267948966, 0.0))
    return tl.where(x == 0, on_axis, r)
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
    return "".join(HELPERS[name] for name in needed)
