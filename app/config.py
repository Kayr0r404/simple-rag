"""Settings + Langfuse client setup. Import this BEFORE anything that uses Langfuse."""
import os
import re

from dotenv import load_dotenv

load_dotenv()

from langfuse import Langfuse  # noqa: E402

MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)
ENV = os.getenv("APP_ENV", "dev")
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "500"))
PROMPT_NAME = "ask-the-docs-system"
DATASET_NAME = "ask-the-docs-qa"

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def mask_pii(data, **kwargs):
    """STAGE 6: Langfuse calls this on every input/output before sending it."""
    if isinstance(data, str):
        return EMAIL_RE.sub("[EMAIL]", data)
    if isinstance(data, dict):
        return {k: mask_pii(v) for k, v in data.items()}
    if isinstance(data, list):
        return [mask_pii(v) for v in data]
    return data


# Creating Langfuse(...) once registers it as the global client,
# so get_client() everywhere else returns this configured instance.
langfuse = Langfuse(
    mask=mask_pii,
    sample_rate=float(os.getenv("LANGFUSE_SAMPLE_RATE", "1.0")),
)


def build_llm_client():
    """Return a Gemini-backed chat client that still reports through Langfuse.

    Gemini is called over its OpenAI-compatible endpoint, so the Langfuse
    OpenAI wrapper keeps recording tokens, cost, latency and prompt linkage.
    """
    from langfuse.openai import OpenAI

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Create one at https://aistudio.google.com/apikey"
        )
    return OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
