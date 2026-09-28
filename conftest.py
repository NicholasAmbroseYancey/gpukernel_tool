"""Shared pytest setup for the fast (mocked) and GPU test tiers.

- Tests marked ``gpu`` are skipped when CUDA is unavailable. Set
  ``REQUIRE_GPU=1`` (the GPU CI job does) to abort the run instead, so a
  broken driver can't turn the GPU job green by skipping everything.
- Every test runs in its own temp working directory, because the pipeline
  writes ``kernels/kernel.py`` and ``reports/`` relative to the cwd.
- The Ollama client's network calls are disabled unless a test mocks them,
  so a missing mock fails fast instead of hanging on retries.
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
    gpu_items = [item for item in items if "gpu" in item.keywords]
    if not gpu_items or _cuda_available():
        return
    if os.getenv("REQUIRE_GPU") == "1":
        pytest.exit("REQUIRE_GPU=1 but torch.cuda.is_available() is False", returncode=1)
    skip = pytest.mark.skip(reason="CUDA not available")
    for item in gpu_items:
        item.add_marker(skip)


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
