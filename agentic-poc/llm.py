"""Centralized LLM configuration for agentic scientific workflows.

Designed for Datathon / PoC use with GPT-4.1 or GPT-4o.
"""

from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage


# -----------------------------------------------------------------------------
# Model selection
# -----------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-4.1"  # or "gpt-4o"


def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> ChatOpenAI:
    """
    Return a configured ChatOpenAI instance.

    Low temperature is intentional:
    - More deterministic
    - Better for scientific reasoning
    - Fewer hallucinations
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


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
- When comparing datasets or plots, explain *why* differences matter.
- If evidence is insufficient, say so clearly.

You do NOT perform heavy computation yourself.
You rely on tools (xarray, numpy, image analysis) for data inspection.
Your role is planning, interpretation, and explanation.
"""


def get_system_message() -> SystemMessage:
    return SystemMessage(content=SCIENTIFIC_AGENT_SYSTEM_PROMPT)


# -----------------------------------------------------------------------------
# Helper for LangGraph nodes
# -----------------------------------------------------------------------------


def call_llm(
    llm: ChatOpenAI,
    messages: list,
) -> dict[str, Any]:
    """
    Standard LLM call wrapper for LangGraph nodes.
    Returns a dict so it can be merged into graph state.
    """
    response = llm.invoke(messages)
    return {"messages": [response]}
