"""Ollama LLM client for the AI GPU compiler.

Owner: Bryson Wingate (LLM Optimization Engine).

Configuration is read from environment variables with sensible defaults so
this module stays self-contained and does not depend on config.py (owned by
infra). Override any value via the OLLAMA_* environment variables below.
"""

from __future__ import annotations

import os
import time

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None


# --- Configuration (env-overridable, no config.py dependency) ---------------
def _env_str(name: str, default: str) -> str:
    """os.getenv but an unset OR empty/whitespace value falls back to default."""
    value = os.getenv(name)
    return value if value and value.strip() else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


OLLAMA_URL = _env_str("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = _env_str("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_TIMEOUT = _env_float("OLLAMA_TIMEOUT", 120.0)      # seconds
OLLAMA_TEMPERATURE = _env_float("OLLAMA_TEMPERATURE", 0)  # 0 = deterministic
OLLAMA_RETRIES = _env_int("OLLAMA_RETRIES", 2)            # extra attempts after first
OLLAMA_BACKOFF = _env_float("OLLAMA_BACKOFF", 1.5)        # seconds, grows linearly


class OllamaError(RuntimeError):
    """Raised when the Ollama backend is unreachable or returns a bad response."""


def _require_requests() -> None:
    if requests is None:
        raise OllamaError(
            "requests package is not installed. "
            "Install requirements.txt to enable Ollama LLM features."
        )


def _extract_content(payload: dict) -> str:
    """Pull the assistant message text out of an /api/chat response.

    Raises OllamaError with a helpful message instead of a raw KeyError when
    the response shape is not what we expect (e.g. an Ollama error payload).
    """
    if not isinstance(payload, dict):
        raise OllamaError(f"Unexpected Ollama response (not an object): {payload!r}")

    if "error" in payload:
        raise OllamaError(f"Ollama returned an error: {payload['error']}")

    message = payload.get("message")
    if not isinstance(message, dict) or "content" not in message:
        raise OllamaError(
            f"Ollama response missing message.content. Got keys: {sorted(payload)}"
        )

    content = message["content"]
    if not isinstance(content, str) or not content.strip():
        raise OllamaError("Ollama returned an empty message.")
    return content


def generate(
    prompt: str,
    model: str | None = None,
    *,
    temperature: float | None = None,
    timeout: float | None = None,
    retries: int | None = None,
) -> str:
    """Send a single-turn chat prompt to Ollama and return the reply text.

    Deterministic by default (temperature=0). Retries transient network and
    5xx failures with a linear backoff, then raises OllamaError with context.
    """
    _require_requests()

    model = model or OLLAMA_MODEL
    temperature = OLLAMA_TEMPERATURE if temperature is None else temperature
    timeout = OLLAMA_TIMEOUT if timeout is None else timeout
    attempts = (OLLAMA_RETRIES if retries is None else retries) + 1

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
    }

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(OLLAMA_URL, json=body, timeout=timeout)
            response.raise_for_status()
            return _extract_content(response.json())
        except OllamaError:
            # Bad payload shape / model error — retrying won't help.
            raise
        except requests.exceptions.HTTPError as error:  # type: ignore[union-attr]
            status = getattr(error.response, "status_code", None)
            # Retry only on server-side (5xx); client errors are fatal.
            if status is not None and status < 500:
                raise OllamaError(
                    f"Ollama HTTP {status} for model {model!r}: {error}"
                ) from error
            last_error = error
        except requests.exceptions.RequestException as error:  # type: ignore[union-attr]
            # Timeouts, connection refused, etc.
            last_error = error
        except ValueError as error:  # response.json() failed to parse
            last_error = error

        if attempt < attempts:
            time.sleep(OLLAMA_BACKOFF * attempt)

    raise OllamaError(
        f"Ollama request to {OLLAMA_URL} failed after {attempts} attempt(s). "
        f"Is the server running and model {model!r} pulled? "
        f"Last error: {last_error}"
    ) from last_error


def is_available(timeout: float = 2.0) -> bool:
    """Best-effort check that an Ollama server is reachable. Never raises."""
    if requests is None:
        return False
    base = OLLAMA_URL.rsplit("/api/", 1)[0]
    try:
        resp = requests.get(base, timeout=timeout)
        return resp.status_code < 500
    except Exception:
        return False
