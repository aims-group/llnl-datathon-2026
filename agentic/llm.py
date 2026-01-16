"""
Centralized LLM configuration for agentic scientific workflows.

PoC / prototyping version:
- Supports Ollama (local LLaMA 3.1)
- Supports LLNL LivAI (Claude / GPT)
- Backend/model/temperature configured via environment variables (.env)
"""

from __future__ import annotations

import os
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------------------------------------------------------
# Load environment variables from .env
# -----------------------------------------------------------------------------

load_dotenv()

# -----------------------------------------------------------------------------
# Configuration via environment variables
# -----------------------------------------------------------------------------
# .env example:
#   LLM_BACKEND=ollama            # "ollama" or "livai"
#   LLM_MODEL=llama3.1:8b         # or "gpt-5", "claude-sonnet-3.7"
#   LLM_TEMPERATURE=0.1
#   LLM_MAX_TOKENS=2048
#   OLLAMA_BASE_URL=http://localhost:11434/v1
#   LIVAI_API_KEY=...
#   LIVAI_BASE_URL=https://api.livai.llnl.gov/v1

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").strip().lower()
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b").strip()
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").strip()
LIVAI_BASE_URL = os.getenv("LIVAI_BASE_URL", "https://api.livai.llnl.gov/v1").strip()

print(
    f"[LLM] backend={LLM_BACKEND}, model={DEFAULT_MODEL}, "
    f"temp={DEFAULT_TEMPERATURE}, max_tokens={DEFAULT_MAX_TOKENS}"
)

# -----------------------------------------------------------------------------
# Client factory
# -----------------------------------------------------------------------------


def _make_client() -> OpenAI:
    """
    Create an OpenAI-compatible client for the configured backend.

    Backends:
    - ollama : local LLaMA models via Ollama
    - livai  : LLNL LivAI gateway (Claude / GPT)
    """
    if LLM_BACKEND == "ollama":
        return OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",  # required by SDK; ignored by Ollama
        )

    if LLM_BACKEND == "livai":
        if not os.getenv("LIVAI_API_KEY"):
            raise RuntimeError("LIVAI_API_KEY is required when LLM_BACKEND=livai")

        return OpenAI(
            api_key=os.environ["LIVAI_API_KEY"],
            base_url=LIVAI_BASE_URL,
        )

    raise ValueError(
        f"Unsupported LLM_BACKEND={LLM_BACKEND!r}. Supported values: 'ollama', 'livai'."
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

You rely on tools (xarray, numpy, image analysis) for computation.
Your role is planning, interpretation, and explanation.
"""


def system_message() -> dict[str, str]:
    return {"role": "system", "content": SCIENTIFIC_AGENT_SYSTEM_PROMPT}


# -----------------------------------------------------------------------------
# LLM call wrapper
# -----------------------------------------------------------------------------


def call_llm(
    messages: list[dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """
    Standard LLM call wrapper.

    Defaults come from environment variables:
      - LLM_MODEL
      - LLM_TEMPERATURE
      - LLM_MAX_TOKENS
    """
    chosen_model = model or DEFAULT_MODEL
    chosen_temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
    chosen_max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens

    request_kwargs = dict(
        model=chosen_model,
        messages=messages,
        max_tokens=chosen_max_tokens,
    )

    if chosen_model.startswith("gpt-5"):
        # gpt-5 (LivAI / Azure) has restricted parameters:
        # - no temperature control
        # - no reasoning_effort='none'
        # Rely on prompt instructions instead.
        pass
    else:
        request_kwargs["temperature"] = chosen_temperature

    response = _client.chat.completions.create(**request_kwargs)

    return {
        "content": response.choices[0].message.content,
        "model": chosen_model,
        "backend": LLM_BACKEND,
        "usage": getattr(response, "usage", None),
    }
