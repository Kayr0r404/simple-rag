"""The RAG pipeline. The RAG itself is deliberately tiny; the Langfuse wiring is the lesson."""
from pathlib import Path

import chromadb
from langfuse import get_client, observe

from app.config import (
    DEFAULT_CHUNK_SIZE,
    ENV,
    MODEL,
    PROMPT_NAME,
    build_llm_client,
)

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
chroma = chromadb.PersistentClient(path=str(ROOT / ".chroma"))
llm = build_llm_client()

# STAGE 3: used only if Langfuse is unreachable or the prompt hasn't been seeded yet.
FALLBACK_PROMPT = (
    "Answer using ONLY the context below. If the answer is not in the context, "
    "reply exactly: I don't know based on the docs.\n\nContext:\n{{context}}"
)


# ---------- indexing (not traced; it's offline work) ----------
def chunk_text(text: str, chunk_size: int) -> list[str]:
    chunks, current = [], ""
    for para in text.split("\n\n"):
        if current and len(current) + len(para) > chunk_size:
            chunks.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


def get_collection(chunk_size: int):
    return chroma.get_or_create_collection(f"docs_{chunk_size}")


def ingest(chunk_size: int = DEFAULT_CHUNK_SIZE) -> int:
    col = get_collection(chunk_size)
    ids, docs, metas = [], [], []
    for path in sorted(DOCS_DIR.glob("*.md")):
        for i, chunk in enumerate(chunk_text(path.read_text(), chunk_size)):
            ids.append(f"{path.name}-{i}")
            docs.append(chunk)
            metas.append({"source": path.name})
    col.upsert(ids=ids, documents=docs, metadatas=metas)
    return len(ids)


def ensure_indexed(chunk_size: int) -> None:
    if get_collection(chunk_size).count() == 0:
        ingest(chunk_size)


# ---------- pipeline ----------
@observe(name="retrieve")  # STAGE 1: each step becomes a span nested under the trace
def retrieve(question: str, chunk_size: int, k: int = 3) -> list[dict]:
    ensure_indexed(chunk_size)
    res = get_collection(chunk_size).query(query_texts=[question], n_results=k)
    return [
        {"text": d, "source": m["source"], "distance": dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


@observe(name="build_prompt")
def build_prompt(chunks: list[dict], prompt_label: str):
    """STAGE 3: fetch the prompt from Langfuse by name + label."""
    prompt = get_client().get_prompt(PROMPT_NAME, label=prompt_label, fallback=FALLBACK_PROMPT)
    context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
    return prompt, prompt.compile(context=context)


@observe(name="generate")
def generate(question: str, system: str, prompt, history: list[dict]) -> str:
    resp = llm.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[{"role": "system", "content": system}, *history, {"role": "user", "content": question}],
        name="llm-answer",
        langfuse_prompt=prompt,  # STAGE 3: links this generation to the prompt version used
    )
    return resp.choices[0].message.content


@observe(name="ask-the-docs")  # STAGE 1: the root = one trace per request
def ask(
    question: str,
    user_id: str = "anon",
    session_id: str | None = None,
    history: list[dict] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    prompt_label: str = "production",
) -> dict:
    lf = get_client()
    # STAGE 2 + 6: who, which conversation, and filterable tags
    lf.update_current_trace(
        user_id=user_id,
        session_id=session_id,
        tags=[f"env:{ENV}", f"chunk:{chunk_size}", f"prompt:{prompt_label}"],
        metadata={"chunk_size": chunk_size, "model": MODEL},
    )
    chunks = retrieve(question, chunk_size)
    prompt, system = build_prompt(chunks, prompt_label)
    answer = generate(question, system, prompt, history or [])
    return {
        "answer": answer,
        "sources": sorted({c["source"] for c in chunks}),
        "contexts": [c["text"] for c in chunks],
        "trace_id": lf.get_current_trace_id(),
    }
