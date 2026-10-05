"""STAGE 3: create prompt versions in Langfuse.
  python -m scripts.seed_prompt        -> v1 labelled 'production'
  python -m scripts.seed_prompt --v2   -> v2 labelled 'staging' (promote it in the UI when happy)
"""
import sys

from app.config import PROMPT_NAME, langfuse

V1 = (
    "You are a documentation assistant for Pebble. Answer using ONLY the context below. "
    "If the answer is not in the context, reply exactly: I don't know based on the docs.\n\n"
    "Context:\n{{context}}"
)
V2 = V1 + "\n\nBe concise (max 3 sentences) and cite the source file in brackets, e.g. [pricing.md]."

if "--v2" in sys.argv:
    langfuse.create_prompt(name=PROMPT_NAME, prompt=V2, labels=["staging"], type="text")
    print("created v2 with label 'staging'")
else:
    langfuse.create_prompt(name=PROMPT_NAME, prompt=V1, labels=["production"], type="text")
    print("created v1 with label 'production'")
langfuse.flush()
