from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from app.config import DEFAULT_CHUNK_SIZE, langfuse  # config must load first
from app.evals import judge_faithfulness
from app.rag import ask

SESSIONS: dict[str, list[dict]] = {}  # in-memory chat history, keyed by session_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    langfuse.shutdown()  # flush buffered events on exit


app = FastAPI(title="Ask the Docs", lifespan=lifespan)


class AskIn(BaseModel):
    question: str
    user_id: str = "anon"
    session_id: str | None = None
    chunk_size: int = DEFAULT_CHUNK_SIZE
    prompt_label: str = "production"


class FeedbackIn(BaseModel):
    trace_id: str
    thumbs_up: bool
    comment: str | None = None


def score_faithfulness(trace_id: str, question: str, answer: str, contexts: list[str]):
    result = judge_faithfulness(question, answer, contexts)
    langfuse.create_score(
        trace_id=trace_id, name="faithfulness", value=result["score"],
        data_type="NUMERIC", comment=result["reason"],
    )


@app.post("/ask")
def ask_endpoint(body: AskIn, bg: BackgroundTasks):
    session_id = body.session_id or str(uuid4())
    history = SESSIONS.setdefault(session_id, [])  # STAGE 2: follow-ups share a session
    result = ask(
        body.question, user_id=body.user_id, session_id=session_id, history=list(history),
        chunk_size=body.chunk_size, prompt_label=body.prompt_label,
    )
    history += [{"role": "user", "content": body.question},
                {"role": "assistant", "content": result["answer"]}]
    # STAGE 4: LLM-as-judge runs after the response so users don't wait for it
    bg.add_task(score_faithfulness, result["trace_id"], body.question, result["answer"], result["contexts"])
    return {"answer": result["answer"], "sources": result["sources"],
            "trace_id": result["trace_id"], "session_id": session_id}


@app.post("/feedback")
def feedback(body: FeedbackIn):
    """STAGE 4: user feedback attaches to the exact trace that produced the answer."""
    langfuse.create_score(
        trace_id=body.trace_id, name="user-feedback", value=1 if body.thumbs_up else 0,
        data_type="BOOLEAN", comment=body.comment,
    )
    return {"ok": True}
