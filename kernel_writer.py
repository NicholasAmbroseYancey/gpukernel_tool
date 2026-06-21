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

    cleaned = clean_output(code)

    with open("kernels/kernel.py", "w") as f:
        f.write(cleaned)


def save_kernel_source(code):
    """Save compiler-generated kernel source without LLM cleanup."""
    os.makedirs("kernels", exist_ok=True)
    with open("kernels/kernel.py", "w") as f:
        f.write(code)
