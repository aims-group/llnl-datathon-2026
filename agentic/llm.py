"""
Centralized LLM configuration for agentic scientific workflows.

PoC / prototyping version:
- Supports Ollama (OpenAI-compatible endpoint) and OpenAI API
- Backend/model/temperature configured via environment variables (.env)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

# -----------------------------------------------------------------------------
# Configuration via environment variables
# -----------------------------------------------------------------------------
# .env example:
#   LLM_BACKEND=ollama          # "ollama" or "openai"
#   LLM_MODEL=qwen2.5:7b        # e.g., "qwen2.5:7b" or "gpt-4o-mini"
#   LLM_TEMPERATURE=0.1
#   LLM_MAX_TOKENS=2048
#   OLLAMA_BASE_URL=http://localhost:11434/v1
#   OPENAI_API_KEY=...          # required only for LLM_BACKEND=openai
#   OPENAI_BASE_URL=...         # optional (usually unset)

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").strip().lower()
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b-instruct").strip()
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()  # usually empty

print(f"[LLM] backend={LLM_BACKEND}, model={DEFAULT_MODEL}, temp={DEFAULT_TEMPERATURE}")


def _make_client() -> OpenAI:
    """
    Create an OpenAI client for the configured backend.
    - ollama: uses OpenAI-compatible API at OLLAMA_BASE_URL; api_key is required but unused.
    - openai: uses standard OpenAI API (requires OPENAI_API_KEY in environment).
    """
    if LLM_BACKEND == "ollama":
        return OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",  # required by the SDK; ignored by Ollama
        )

    if LLM_BACKEND == "openai":
        # OPENAI_API_KEY is read automatically from the environment by the SDK.
        # Optionally support OPENAI_BASE_URL for non-default routing/proxies.
        kwargs: Dict[str, Any] = {}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL

        return OpenAI(**kwargs)

    raise ValueError(
        f"Unsupported LLM_BACKEND={LLM_BACKEND!r}. Use 'ollama' or 'openai'."
    )


_client = _make_client()

# -----------------------------------------------------------------------------
# System prompt
# -----------------------------------------------------------------------------

SCIENTIFIC_AGENT_SYSTEM_PROMPT = """\
You are a scientific analysis agent assisting with Earth system model (ESM)
post-processing and diagnostics.

Guidelines:
- Be explicit about assumptions and uncertainty.
- Prefer step-by-step reasoning over conclusions.
- Do not guess values; request or compute them via tools.
- When comparing datasets or plots, explain why differences matter.
- If evidence is insufficient, say so clearly.

You do NOT perform heavy computation yourself.
You rely on tools (xarray, numpy, image analysis) for data inspection.
Your role is planning, interpretation, and explanation.
"""


def system_message() -> Dict[str, str]:
    return {"role": "system", "content": SCIENTIFIC_AGENT_SYSTEM_PROMPT}


# -----------------------------------------------------------------------------
# LLM call wrapper
# -----------------------------------------------------------------------------


def call_llm(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Standard LLM call wrapper.

    Defaults come from environment variables:
      - LLM_MODEL
      - LLM_TEMPERATURE
      - LLM_MAX_TOKENS
    """
    model = model or DEFAULT_MODEL
    temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
    max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens

    response = _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return {
        "content": response.choices[0].message.content,
        "model": model,
        "backend": LLM_BACKEND,
        "usage": response.usage,
    }
