"""
Centralized LLM configuration for agentic scientific workflows.

PoC / prototyping version:
- Supports Ollama (local LLaMA 3.1)
- Supports LLNL LivAI (Claude / GPT)
- Backend/model/temperature configured via environment variables (.env)
"""

from __future__ import annotations

from typing import Any, Optional

from agentic.config import (
    LLM_BACKEND,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    client,
)


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

    response = client.chat.completions.create(**request_kwargs)

    # return response

    return {
        "content": response.choices[0].message.content,
        "model": chosen_model,
        "backend": LLM_BACKEND,
        "usage": getattr(response, "usage", None),
    }
