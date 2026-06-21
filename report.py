"""Benchmark reports and speedup graphs."""

from __future__ import annotations

import os
import re
from datetime import datetime

from benchmark import BenchmarkReport


def save_text_report(report: BenchmarkReport, directory: str = "reports") -> str:
    os.makedirs(directory, exist_ok=True)
    slug = _slugify(report.source)
    path = os.path.join(directory, f"benchmark_{slug}.txt")
    lines = [
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Source: {report.source}",
        f"Mode: {'multi-output' if report.multi else 'single-output'}",
        report.summary(),
        "",
        "BLOCK_SIZE sweep:",
    ]
    for item in report.block_size_sweep:
        lines.append(
            f"  block_size={item.block_size:4d}  "
            f"median={item.median_ms:.4f} ms  "
            f"throughput={item.throughput_gels:.2f} Melem/s"
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def save_speedup_graph(report: BenchmarkReport, directory: str = "reports") -> str | None:
    if not report.block_size_sweep:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    os.makedirs(directory, exist_ok=True)
    slug = _slugify(report.source)
    path = os.path.join(directory, f"speedup_{slug}.png")

    sizes = [item.block_size for item in report.block_size_sweep]
    triton_ms = [item.median_ms for item in report.block_size_sweep]
    speedups = [report.pytorch_ms / ms if ms > 0 else 0 for ms in triton_ms]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(sizes, triton_ms, marker="o")
    axes[0].axhline(report.pytorch_ms, color="r", linestyle="--", label="PyTorch")
    axes[0].set_xlabel("BLOCK_SIZE")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Latency vs BLOCK_SIZE")
    axes[0].legend()

    axes[1].plot(sizes, speedups, marker="o", color="green")
    axes[1].set_xlabel("BLOCK_SIZE")
    axes[1].set_ylabel("Speedup vs PyTorch")
    axes[1].set_title("Speedup vs BLOCK_SIZE")

    fig.suptitle(_short_source(report.source))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _slugify(source: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", source)[:48].strip("_")
    return slug or "kernel"


def _short_source(source: str) -> str:
    return source.replace("\n", "; ")[:80]
