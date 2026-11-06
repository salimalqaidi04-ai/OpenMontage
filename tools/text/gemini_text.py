"""Text-only Gemini generation through the official @google/genai Node SDK.

The Python side is a thin BaseTool wrapper. It sends the request as JSON on
stdin to ``scripts/gemini_text/generate.mjs`` and reads one JSON response from
stdout.

Key handling: the key is read only from ``GEMINI_API_KEY`` and reaches the
Node process through its environment, never through argv, stdin, files, or
the ToolResult. Any occurrence of the key value in an error is redacted.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    DependencyError,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolTier,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = _REPO_ROOT / "scripts" / "gemini_text"
SCRIPT_PATH = SCRIPT_DIR / "generate.mjs"
SDK_MARKER = SCRIPT_DIR / "node_modules" / "@google" / "genai" / "package.json"

DEFAULT_MODEL = "gemini-3-flash-preview"
KEY_ENV = "GEMINI_API_KEY"

# Variables that could make the SDK pick a different credential or backend.
# Stripped from the child environment so only GEMINI_API_KEY is ever used.
_COMPETING_ENV = (
    "GOOGLE_API_KEY",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_APPLICATION_CREDENTIALS",
)


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text


class GeminiText(BaseTool):
    name = "gemini_text"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = ["env:GEMINI_API_KEY", "cmd:node"]
    install_instructions = (
        "Set GEMINI_API_KEY in .env (Google AI Studio key), install Node.js >= 22, "
        "then run `npm install` inside scripts/gemini_text/."
    )

    capability = "text_generation"
    provider = "google"
    capabilities = ["text_generation"]
    best_for = [
        "drafting or rewriting short text (titles, captions, descriptions, script lines)",
        "text-only Gemini calls through the official JavaScript SDK",
    ]
    not_good_for = ["image, audio, or video output", "multimodal input"]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "minLength": 1},
            "model": {"type": "string", "default": DEFAULT_MODEL},
            "system_instruction": {"type": "string"},
            "temperature": {"type": "number", "minimum": 0, "maximum": 2},
            "max_output_tokens": {"type": "integer", "minimum": 1},
            "timeout_seconds": {"type": "number", "minimum": 1, "default": 120},
            "output_path": {
                "type": "string",
                "description": "Optional file to write the generated text to.",
            },
        },
    }
    output_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "model": {"type": "string"},
            "model_version": {"type": ["string", "null"]},
            "finish_reason": {"type": ["string", "null"]},
            "usage": {"type": "object"},
            "output": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(network_required=True)
    retry_policy = RetryPolicy(max_retries=0)
    side_effects = ["calls the Gemini API (billed per token on paid tiers)"]

    def check_dependencies(self) -> None:
        super().check_dependencies()
        if not SDK_MARKER.is_file():
            raise DependencyError(
                f"@google/genai is not installed in {SCRIPT_DIR}. {self.install_instructions}"
            )

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        # Token pricing varies by model; usage is reported but not priced here.
        return 0.0

    def _child_env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in _COMPETING_ENV}
        return env

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        start = time.time()
        key = os.environ.get(KEY_ENV, "")

        prompt = inputs.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return ToolResult(success=False, error="'prompt' must be a non-empty string.")

        try:
            self.check_dependencies()
        except DependencyError as exc:
            return ToolResult(success=False, error=_redact(str(exc), key))

        model = inputs.get("model") or DEFAULT_MODEL
        timeout_s = float(inputs.get("timeout_seconds") or 120)
        request = {
            "prompt": prompt,
            "model": model,
            "system_instruction": inputs.get("system_instruction"),
            "temperature": inputs.get("temperature"),
            "max_output_tokens": inputs.get("max_output_tokens"),
            "timeout_ms": int(timeout_s * 1000),
        }

        node = shutil.which("node") or "node"
        try:
            proc = subprocess.run(
                [node, str(SCRIPT_PATH)],
                input=json.dumps(request, ensure_ascii=False),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(SCRIPT_DIR),
                env=self._child_env(),
                timeout=timeout_s + 15,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Gemini text generation timed out after {timeout_s:.0f}s.",
                duration_seconds=time.time() - start,
            )
        except OSError as exc:
            return ToolResult(success=False, error=_redact(f"Could not start node: {exc}", key))

        try:
            payload = json.loads(proc.stdout or "")
        except json.JSONDecodeError:
            detail = (proc.stderr or proc.stdout or "no output").strip()[-500:]
            return ToolResult(
                success=False,
                error=_redact(f"Gemini bridge returned invalid output: {detail}", key),
                duration_seconds=time.time() - start,
            )

        if not payload.get("ok"):
            status = payload.get("status")
            msg = payload.get("error") or "Gemini request failed."
            if status:
                msg = f"[HTTP {status}] {msg}"
            return ToolResult(
                success=False,
                error=_redact(msg, key),
                model=model,
                duration_seconds=time.time() - start,
            )

        text = payload.get("text", "")
        data: dict[str, Any] = {
            "text": text,
            "provider": self.provider,
            "model": payload.get("model", model),
            "model_version": payload.get("model_version"),
            "finish_reason": payload.get("finish_reason"),
            "usage": payload.get("usage") or {},
            "cost_status": "unquoted",
        }
        artifacts: list[str] = []
        if inputs.get("output_path"):
            out = Path(inputs["output_path"])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            data["output"] = str(out)
            artifacts.append(str(out))

        return ToolResult(
            success=True,
            data=data,
            artifacts=artifacts,
            model=data["model"],
            cost_usd=None,
            duration_seconds=time.time() - start,
        )
