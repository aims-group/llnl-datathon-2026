"""
Centralized LLM configuration for agentic scientific workflows.

PoC / prototyping version:
- Uses a local Qwen 2.5 model via Ollama
- OpenAI-compatible API for simplicity and debuggability
"""

from typing import Any, List, Dict
from openai import OpenAI

_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required but unused
)

DEFAULT_MODEL = "qwen2.5:14b"  # or qwen2.5:32b if you have resources

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


def call_llm(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """
    Standard LLM call wrapper.

    Returns raw content + metadata for logging and analysis.
    """
    response = _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return {
        "content": response.choices[0].message.content,
        "model": model,
        "usage": response.usage,
    }
