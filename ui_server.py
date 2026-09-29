from __future__ import annotations

import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(ROOT, "ui")


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


class UiRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_file("index.html")
            return

        if parsed.path.startswith("/api/tests"):
            mode = parsed.query.split("=", 1)[1] if "mode=" in parsed.query else "unit"
            try:
                payload = _run_pytest(mode)
                self._send_json(200, payload)
            except Exception as exc:  # pragma: no cover - UI-only handler
                self._send_json(500, {"error": str(exc)})
            return

        static_path = parsed.path.lstrip("/")
        if static_path and static_path not in {"index.html"}:
            safe_path = os.path.normpath(os.path.join(UI_DIR, static_path)).replace("\\", "/")
            if safe_path.startswith((UI_DIR.replace("\\", "/"), "/")):
                if os.path.isfile(safe_path):
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

        self._send_json(404, {"error": "Not found"})

    def _serve_file(self, relative_name: str):
        file_path = os.path.join(UI_DIR, relative_name)
        if not os.path.isfile(file_path):
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
