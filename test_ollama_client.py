"""Mocked tests for the Ollama client (no server or GPU required)."""

import unittest
from unittest.mock import MagicMock, patch

import requests

import ollama_client
from ollama_client import OllamaError, generate, is_available


def _response(payload=None, status=200, json_error=None):
    resp = MagicMock()
    resp.status_code = status
    if status >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp)
    if json_error is not None:
        resp.json.side_effect = json_error
    else:
        resp.json.return_value = payload
    return resp


def _ok(content="x * y"):
    return _response({"message": {"role": "assistant", "content": content}})


@patch("ollama_client.time.sleep", lambda *_: None)
class TestGenerate(unittest.TestCase):
    @patch("ollama_client.requests.post")
    def test_returns_content(self, post):
        post.return_value = _ok("x + y")
        self.assertEqual(generate("prompt"), "x + y")

    @patch("ollama_client.requests.post")
    def test_request_is_deterministic(self, post):
        post.return_value = _ok()
        generate("prompt")
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["options"]["temperature"], 0)
        self.assertFalse(body["stream"])
        self.assertEqual(body["messages"], [{"role": "user", "content": "prompt"}])

    @patch("ollama_client.requests.post")
    def test_error_payload_raises_without_retry(self, post):
        post.return_value = _response({"error": "model not found"})
        with self.assertRaisesRegex(OllamaError, "model not found"):
            generate("prompt", retries=3)
        self.assertEqual(post.call_count, 1)

    @patch("ollama_client.requests.post")
    def test_empty_message_raises(self, post):
        post.return_value = _ok("   ")
        with self.assertRaisesRegex(OllamaError, "empty"):
            generate("prompt")

    @patch("ollama_client.requests.post")
    def test_missing_message_raises(self, post):
        post.return_value = _response({"done": True})
        with self.assertRaisesRegex(OllamaError, "missing message.content"):
            generate("prompt")

    @patch("ollama_client.requests.post")
    def test_non_dict_payload_raises(self, post):
        post.return_value = _response(["not", "a", "dict"])
        with self.assertRaises(OllamaError):
            generate("prompt")

    @patch("ollama_client.requests.post")
    def test_client_error_is_not_retried(self, post):
        post.return_value = _response(status=404)
        with self.assertRaisesRegex(OllamaError, "HTTP 404"):
            generate("prompt", retries=3)
        self.assertEqual(post.call_count, 1)

    @patch("ollama_client.requests.post")
    def test_server_error_is_retried(self, post):
        post.side_effect = [_response(status=503), _ok("x")]
        self.assertEqual(generate("prompt", retries=2), "x")
        self.assertEqual(post.call_count, 2)

    @patch("ollama_client.requests.post")
    def test_connection_error_exhausts_retries(self, post):
        post.side_effect = requests.exceptions.ConnectionError("refused")
        with self.assertRaisesRegex(OllamaError, "after 3 attempt"):
            generate("prompt", retries=2)
        self.assertEqual(post.call_count, 3)

    @patch("ollama_client.requests.post")
    def test_bad_json_is_retried(self, post):
        post.side_effect = [_response(json_error=ValueError("bad json")), _ok("y")]
        self.assertEqual(generate("prompt", retries=1), "y")

    def test_missing_requests_package(self):
        with patch.object(ollama_client, "requests", None):
            with self.assertRaisesRegex(OllamaError, "requests package"):
                generate("prompt")


class TestIsAvailable(unittest.TestCase):
    @patch("ollama_client.requests.get")
    def test_reachable(self, get):
        get.return_value = _response(status=200)
        self.assertTrue(is_available())
        self.assertNotIn("/api/", get.call_args.args[0])

    @patch("ollama_client.requests.get")
    def test_unreachable_never_raises(self, get):
        get.side_effect = requests.exceptions.ConnectionError("refused")
        self.assertFalse(is_available())


if __name__ == "__main__":
    unittest.main()
