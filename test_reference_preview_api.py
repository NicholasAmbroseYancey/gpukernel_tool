import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from ui_server import UiRequestHandler


@pytest.fixture
def reference_api_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), UiRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def post_reference(url: str, body: bytes) -> tuple[int, dict[str, object]]:
    request = Request(
        f"{url}/api/reference",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        response = urlopen(request)
    except HTTPError as error:
        response = error

    with response:
        return response.status, json.loads(response.read())


def test_reference_api_single_output(reference_api_url):
    status, payload = post_reference(
        reference_api_url,
        json.dumps({"expression": "x * y"}).encode("utf-8"),
    )

    assert status == 200
    assert payload["success"] is True
    assert payload["expression"] == "x * y"
    assert payload["mode"] == "pytorch_cpu_reference"
    assert payload["kernel_executed"] is False
    assert payload["performance_measured"] is False
    assert payload["inputs"] == {
        "x": [0.25, 0.5, 1.0, 2.0],
        "y": [2.0, 3.0, 4.0, 5.0],
    }
    assert payload["outputs"] == {"out": [0.5, 1.5, 4.0, 10.0]}


def test_reference_api_multi_output(reference_api_url):
    status, payload = post_reference(
        reference_api_url,
        json.dumps({"expression": "out0 = x + y; out1 = x * y"}).encode("utf-8"),
    )

    assert status == 200
    assert payload["outputs"] == {
        "out0": [2.25, 3.5, 5.0, 7.0],
        "out1": [0.5, 1.5, 4.0, 10.0],
    }


@pytest.mark.parametrize(
    ("body", "error"),
    [
        (b'{"expression":"unknown_function(x)"}', "Unknown function"),
        (b"{}", "Expression is required"),
        (b'{"expression":""}', "Expression is required"),
        (b'{"expression":42}', "Expression is required"),
        (b"not-json", "Invalid JSON"),
        (b"[]", "JSON object"),
    ],
)
def test_reference_api_rejects_invalid_requests(reference_api_url, body, error):
    status, payload = post_reference(reference_api_url, body)

    assert status == 400
    assert payload["success"] is False
    assert error in payload["error"]
