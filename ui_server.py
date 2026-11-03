from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(ROOT, "ui")

# Benchmarks are timing-sensitive, so only one optimize run may use the GPU
# at a time. A second request is rejected instead of queued.
_OPTIMIZE_LOCK = threading.Lock()


def _run_pytest(mode: str) -> dict[str, object]:
    env = os.environ.copy()
    if mode == "kernel":
        env["TRITON_INTERPRET"] = "1"

    if mode == "unit":
        cmd = [sys.executable, "-m", "pytest", "-m", "not kernel and not gpu", "-q"]
    elif mode == "kernel":
        cmd = [sys.executable, "-m", "pytest", "-m", "kernel", "-q"]
    elif mode == "gpu":
        cmd = [sys.executable, "-m", "pytest", "-m", "gpu", "-q"]
    else:
        raise ValueError(f"Unsupported mode: {mode!r}")

    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    return {
        "mode": mode,
        "returncode": completed.returncode,
        "output": output.strip() or "No output captured.",
    }


def _num(value: object) -> float | None:
    """A JSON-safe float: None for missing, NaN or infinite values."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _benchmark_payload(bench) -> dict[str, object]:
    triton_ms = _num(bench.triton_ms)
    throughput = None
    if triton_ms and triton_ms > 0:
        throughput = _num(bench.n / (triton_ms / 1000) / 1e6)

    try:
        memory = bench.memory.summary()
    except Exception:  # pragma: no cover - memory summary is optional
        memory = None

    return {
        "n": bench.n,
        "block_size": bench.block_size,
        "triton_ms": triton_ms,
        "pytorch_ms": _num(bench.pytorch_ms),
        "speedup": _num(bench.speedup),
        "throughput_melem_s": throughput,
        "memory": memory,
        "sweep": [
            {"block_size": item.block_size, "median_ms": _num(item.median_ms)}
            for item in (bench.block_size_sweep or [])
        ],
    }


def _run_optimize(expression: str, *, use_llm: bool) -> dict[str, object]:
    from ai_optimizer import run_ai_compiler
    from benchmark import verify_correctness

    started = time.perf_counter()
    result = run_ai_compiler(expression, use_llm=use_llm)
    elapsed = time.perf_counter() - started

    if not result.success:
        return {
            "success": False,
            "error": "Compilation failed, so there was nothing to optimize.",
            "kernel": result.kernel_code,
        }

    # Independent correctness check on the final source. None means the check
    # could not run (for example no GPU), which is different from "wrong".
    try:
        correct: bool | None = bool(
            verify_correctness(result.source, block_size=result.block_size)
        )
    except Exception:
        correct = None

    return {
        "success": True,
        "source": expression,
        "optimized_source": result.source,
        "kernel": result.kernel_code,
        "block_size": result.block_size,
        "multi": result.multi,
        "correct": correct,
        "elapsed_s": round(elapsed, 2),
        "use_llm": use_llm,
        "benchmark": _benchmark_payload(result.benchmark) if result.benchmark else None,
        "plans": [
            {
                "summary": plan.summary(),
                "block_size": plan.block_size,
                "fuse": plan.fuse,
                "reason": plan.reason,
            }
            for plan in result.plans
        ],
    }


def _ui_path(relative_name: str) -> str | None:
    """Absolute path of a file under ui/, or None if it would escape ui/."""
    path = os.path.realpath(os.path.join(UI_DIR, relative_name))
    if os.path.commonpath([path, os.path.realpath(UI_DIR)]) != os.path.realpath(UI_DIR):
        return None
    return path


class UiRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_file("index.html")
            return

        if parsed.path.startswith("/api/tests"):
            mode = parse_qs(parsed.query).get("mode", ["unit"])[0]
            try:
                payload = _run_pytest(mode)
                self._send_json(200, payload)
            except Exception as exc:  # pragma: no cover - UI-only handler
                self._send_json(500, {"error": str(exc)})
            return

        static_path = parsed.path.lstrip("/")
        if static_path and static_path not in {"index.html"} and os.path.isfile(_ui_path(static_path) or ""):
            self._serve_file(static_path)
            return

        self._serve_file("index.html")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/compile":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8")
                payload = json.loads(raw or "{}")
            except Exception as exc:  # pragma: no cover - UI-only handler
                self._send_json(400, {"error": f"Invalid JSON: {exc}"})
                return

            expression = str(payload.get("expression", "")).strip()
            if not expression:
                self._send_json(400, {"error": "Expression is required."})
                return

            try:
                from pipeline import compile_from_source

                source, kernel = compile_from_source(expression)
                self._send_json(
                    200,
                    {
                        "success": True,
                        "expression": source,
                        "kernel": kernel,
                        "preview": kernel.splitlines()[:12],
                    },
                )
            except Exception as exc:  # pragma: no cover - UI-only handler
                self._send_json(500, {"error": str(exc)})
            return

        if parsed.path == "/api/optimize":
            self._handle_optimize()
            return

        self._send_json(404, {"error": "Not found"})

    def _handle_optimize(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw or "{}")
        except Exception as exc:  # pragma: no cover - UI-only handler
            self._send_json(400, {"error": f"Invalid JSON: {exc}"})
            return

        expression = str(payload.get("expression", "")).strip()
        if not expression:
            self._send_json(400, {"error": "Expression is required."})
            return
        use_llm = bool(payload.get("use_llm", True))

        if not _OPTIMIZE_LOCK.acquire(blocking=False):
            self._send_json(409, {"error": "Another optimize run is already in progress."})
            return

        try:
            result = _run_optimize(expression, use_llm=use_llm)
            self._send_json(200 if result.get("success") else 422, result)
        except Exception as exc:  # pragma: no cover - UI-only handler
            self._send_json(500, {"error": str(exc)})
        finally:
            _OPTIMIZE_LOCK.release()

    def _serve_file(self, relative_name: str):
        file_path = _ui_path(relative_name)
        if file_path is None or not os.path.isfile(file_path):
            self.send_response(404)
            self.end_headers()
            return

        with open(file_path, "rb") as handle:
            data = handle.read()

        content_type = "text/html"
        if relative_name.endswith(".css"):
            content_type = "text/css"
        elif relative_name.endswith(".js"):
            content_type = "application/javascript"

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status_code: int, payload: dict[str, object]):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args):
        return


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    server = ThreadingHTTPServer((host, port), UiRequestHandler)
    print(f"UI server running at http://{host}:{port}")
    print("Use Ctrl+C to stop.")
    server.serve_forever()
