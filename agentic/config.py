import os

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


client = _make_client()


print(
    f"[LLM] backend={LLM_BACKEND}, model={DEFAULT_MODEL}, "
    f"temp={DEFAULT_TEMPERATURE}, max_tokens={DEFAULT_MAX_TOKENS}"
)
