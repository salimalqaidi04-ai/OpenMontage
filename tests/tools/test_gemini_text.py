"""Offline tests for the text-only Gemini tool (gemini_text).

No live API calls: subprocess.run is monkeypatched for the request paths, and
the real Node bridge is only exercised on paths that exit before any network
call (missing key, bad JSON, empty prompt). A fake key is used throughout.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.base_tool import ToolRuntime, ToolStatus
from tools.text import gemini_text as mod
from tools.text.gemini_text import GeminiText
from tools.tool_registry import ToolRegistry

FAKE_KEY = "fake-test-key-AIzaDoNotUse-123"


@pytest.fixture
def keyed_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("GOOGLE_API_KEY", "other-google-key")
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/node")
    monkeypatch.setattr(GeminiText, "check_dependencies", lambda self: None)


def _fake_run(payload, calls, returncode=0):
    def run(cmd, **kwargs):
        calls.append({"cmd": cmd, **kwargs})
        return subprocess.CompletedProcess(cmd, returncode, json.dumps(payload), "")

    return run


def test_contract_and_registry_discovery():
    tool = GeminiText()
    assert tool.name == "gemini_text"
    assert tool.provider == "google"
    assert tool.runtime == ToolRuntime.API
    assert "env:GEMINI_API_KEY" in tool.dependencies
    assert tool.input_schema["required"] == ["prompt"]

    reg = ToolRegistry()
    reg.discover()
    assert reg.get("gemini_text") is not None


def test_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert GeminiText().get_status() == ToolStatus.UNAVAILABLE


def test_success_keeps_key_out_of_argv_stdin_and_result(keyed_env, monkeypatch):
    calls = []
    payload = {
        "ok": True,
        "text": "مرحبا",
        "model": "gemini-3-flash-preview",
        "model_version": "v1",
        "finish_reason": "STOP",
        "usage": {"total_tokens": 7},
    }
    monkeypatch.setattr(mod.subprocess, "run", _fake_run(payload, calls))

    result = GeminiText().execute({"prompt": "قل مرحبا", "temperature": 0.2})

    assert result.success, result.error
    assert result.data["text"] == "مرحبا"
    assert result.data["usage"] == {"total_tokens": 7}

    call = calls[0]
    assert FAKE_KEY not in " ".join(call["cmd"])
    request = json.loads(call["input"])
    assert FAKE_KEY not in call["input"]
    assert request["prompt"] == "قل مرحبا"
    assert request["temperature"] == 0.2
    # Key reaches the child only via its environment; competing creds removed.
    assert call["env"]["GEMINI_API_KEY"] == FAKE_KEY
    assert "GOOGLE_API_KEY" not in call["env"]
    assert FAKE_KEY not in repr(result)


def test_error_message_redacts_key(keyed_env, monkeypatch):
    calls = []
    payload = {"ok": False, "error": f"API key {FAKE_KEY} not valid", "status": 400}
    monkeypatch.setattr(mod.subprocess, "run", _fake_run(payload, calls, returncode=1))

    result = GeminiText().execute({"prompt": "hi"})

    assert not result.success
    assert FAKE_KEY not in result.error
    assert "***" in result.error and "HTTP 400" in result.error


def test_invalid_bridge_output_redacts_key(keyed_env, monkeypatch):
    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", f"crash with {FAKE_KEY}")

    monkeypatch.setattr(mod.subprocess, "run", run)
    result = GeminiText().execute({"prompt": "hi"})
    assert not result.success
    assert FAKE_KEY not in result.error


def test_output_path_writes_text_only(keyed_env, monkeypatch, tmp_path):
    calls = []
    payload = {"ok": True, "text": "generated text", "model": "m"}
    monkeypatch.setattr(mod.subprocess, "run", _fake_run(payload, calls))

    out = tmp_path / "out" / "text.md"
    result = GeminiText().execute({"prompt": "hi", "output_path": str(out)})

    assert result.success
    assert out.read_text(encoding="utf-8") == "generated text"
    assert result.artifacts == [str(out)]


def test_empty_prompt_rejected_without_subprocess(keyed_env, monkeypatch):
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: pytest.fail("ran node"))
    result = GeminiText().execute({"prompt": "   "})
    assert not result.success


# ---- Real Node bridge, offline-only paths ---------------------------------

needs_bridge = pytest.mark.skipif(
    shutil.which("node") is None or not mod.SDK_MARKER.is_file(),
    reason="node or scripts/gemini_text/node_modules not installed",
)


def _run_bridge(stdin: str, key: str | None):
    env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
    if key is not None:
        env["GEMINI_API_KEY"] = key
    return subprocess.run(
        [shutil.which("node"), str(mod.SCRIPT_PATH)],
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=str(mod.SCRIPT_DIR),
        timeout=30,
    )


@needs_bridge
def test_bridge_missing_key_exits_before_network():
    proc = _run_bridge(json.dumps({"prompt": "hi"}), key=None)
    assert proc.returncode == 2
    assert json.loads(proc.stdout) == {"ok": False, "error": "GEMINI_API_KEY is not set."}


@needs_bridge
@pytest.mark.parametrize("stdin", ["not json", json.dumps({"prompt": "  "})])
def test_bridge_rejects_bad_request_without_leaking_key(stdin):
    proc = _run_bridge(stdin, key=FAKE_KEY)
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["ok"] is False
    assert FAKE_KEY not in proc.stdout + proc.stderr
