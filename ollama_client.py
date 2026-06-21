try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


def generate(prompt, model="qwen2.5-coder:7b"):
    if requests is None:
        raise RuntimeError(
            "requests package is not installed. Install requirements.txt to enable Ollama LLM features."
        )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
    )

    response.raise_for_status()

    return response.json()["message"]["content"]
