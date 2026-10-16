"""Tests for the UI server's static file handling."""

import os

from ui_server import UI_DIR, _ui_path


def test_files_under_ui_resolve():
    assert _ui_path("app.js") == os.path.realpath(os.path.join(UI_DIR, "app.js"))


def test_paths_outside_ui_are_rejected():
    for path in ["../config.py", "../../etc/passwd", "/etc/passwd", "sub/../../config.py"]:
        assert _ui_path(path) is None, path
