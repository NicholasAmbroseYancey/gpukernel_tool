"""Shared pytest setup.

Test tiers:
- unmarked: mocked, run anywhere (no GPU, Triton, or Ollama).
- ``kernel``: compile and run real Triton kernels, checked against PyTorch.
  They run on a CUDA GPU, or on the CPU with ``TRITON_INTERPRET=1`` (what CI
  does); otherwise they're skipped.
- ``gpu``: benchmarks, which need a real CUDA GPU and are skipped otherwise.

Every test also runs in its own temp working directory, because the pipeline
writes ``kernels/kernel.py`` and ``reports/`` relative to the cwd, and the
Ollama client's network calls are disabled unless a test mocks them, so a
missing mock fails fast instead of hanging on retries.
"""

from __future__ import annotations

import os

import pytest


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def pytest_collection_modifyitems(config, items):
    interpret = os.getenv("TRITON_INTERPRET") == "1"
    real_gpu = _cuda_available() and not interpret
    can_run_kernels = interpret or real_gpu
    for item in items:
        if "gpu" in item.keywords and not real_gpu:
            item.add_marker(pytest.mark.skip(reason="needs a real CUDA GPU"))
        elif "kernel" in item.keywords and not can_run_kernels:
            item.add_marker(pytest.mark.skip(reason="no CUDA; set TRITON_INTERPRET=1 to run on CPU"))


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    import ollama_client

    def _blocked(*args, **kwargs):
        raise RuntimeError("Network access is disabled in tests; mock ollama_client.requests")

    if ollama_client.requests is not None:
        monkeypatch.setattr(ollama_client.requests, "post", _blocked)
        monkeypatch.setattr(ollama_client.requests, "get", _blocked)
