import importlib.util
import os
import re

from kernel_lint import is_valid as _is_valid


def clean_output(text):
    # remove any ```language or ```
    text = re.sub(r"```[a-zA-Z]*", "", text)
    text = text.replace("```", "")
    return text.strip()


def validate_kernel(code):
    required = ["tl.load", "tl.store", "program_id"]
    banned = ["numpy", "torch", "tl.tensor"]

    if not all(r in code for r in required):
        return False

    if any(b in code for b in banned):
        return False

    return True

def is_valid(code):
    return _is_valid(code)

def save_kernel(code):
    os.makedirs("kernels", exist_ok=True)

    _write_kernel(clean_output(code))


def save_kernel_source(code):
    """Save compiler-generated kernel source without LLM cleanup."""
    os.makedirs("kernels", exist_ok=True)
    _write_kernel(code)


def _write_kernel(source):
    path = "kernels/kernel.py"
    with open(path, "w") as f:
        f.write(source)
    # A rewrite with the same size and mtime would otherwise load the old .pyc.
    try:
        os.remove(importlib.util.cache_from_source(path))
    except FileNotFoundError:
        pass
