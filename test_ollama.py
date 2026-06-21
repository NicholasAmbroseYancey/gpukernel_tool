import requests


def _example_call():
    res = requests.post(
        "http://127.0.0.1:11434/api/chat",
        json={
            "model": "qwen2.5-coder:7b",
            "messages": [
                {"role": "user", "content": "Write a python function that adds two numbers"}
            ],
            "stream": False,
        },
    )

    print(res.status_code)
    print(res.json())


if __name__ == "__main__":
    _example_call()
