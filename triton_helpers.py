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
}

HELPER_DEPS: dict[str, tuple[str, ...]] = {}


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
