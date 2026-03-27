"""
NVIDIA NIM (OpenAI-compatible) client for Surgeon Mind phases.

API key: set NVIDIA_API_KEY or NVAPI_KEY (never commit keys to the repo).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from openai import OpenAI


def _load_project_dotenv() -> None:
    """Load repo-root .env into os.environ if keys are not already set (no extra deps)."""
    try:
        root = Path(__file__).resolve().parent.parent
        path = root / ".env"
        if not path.is_file():
            return
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if val[:1] == val[-1:] and val[:1] in '"\'':
                val = val[1:-1]
            if key and key not in os.environ:
                os.environ[key] = val
    except OSError:
        pass


_load_project_dotenv()

_DEFAULT_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", _DEFAULT_BASE).rstrip("/")
DEEPSEEK_MODEL = os.environ.get(
    "NVIDIA_CHAT_MODEL", "deepseek-ai/deepseek-v3.2"
)


def nvidia_api_key() -> str:
    key = (
        os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("NVAPI_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not key or not str(key).strip():
        raise RuntimeError(
            "Missing API key: set NVIDIA_API_KEY (or NVAPI_KEY) for NVIDIA NIM access."
        )
    return str(key).strip()


def nvidia_openai_client(**kwargs: Any) -> OpenAI:
    """OpenAI-compatible client pointed at NVIDIA integrate API."""
    base = NVIDIA_BASE_URL
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return OpenAI(base_url=base, api_key=nvidia_api_key(), **kwargs)


def chat_extra_body_thinking() -> dict:
    """Matches NVIDIA DeepSeek chat template (extended thinking)."""
    return {"chat_template_kwargs": {"thinking": True}}


def default_chat_params_stream() -> dict:
    # Keep this helper strictly for template / thinking flags.
    # All generation size params (max_tokens/temperature/top_p) must be set at call-sites
    # to avoid accidental huge outputs during connectivity issues.
    return {"extra_body": chat_extra_body_thinking()}


def stream_delta_reasoning_and_content(chunk) -> tuple[str | None, str | None]:
    """Extract (reasoning_content, content) from one streaming chunk, if any."""
    if not getattr(chunk, "choices", None):
        return None, None
    delta = chunk.choices[0].delta
    reasoning = getattr(delta, "reasoning_content", None)
    if reasoning is None or reasoning == "":
        reasoning = getattr(delta, "reasoning", None)
    if reasoning is None or reasoning == "":
        reasoning = getattr(delta, "thought", None)

    content = getattr(delta, "content", None)
    if content is None or content == "":
        content = getattr(delta, "text", None)

    return reasoning, content
