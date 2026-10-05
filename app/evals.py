"""STAGE 4/5: scoring helpers."""
import json

from langfuse import observe

from app.config import MODEL, build_llm_client

judge_llm = build_llm_client()

JUDGE_PROMPT = """You check whether an ANSWER is fully supported by the CONTEXT.
Score 1 if every claim in the answer is supported by the context, or if the answer
honestly says it doesn't know. Score 0 if it states anything not in the context.
Reply as JSON: {{"score": 0 or 1, "reason": "<one sentence>"}}

CONTEXT:
{context}

QUESTION: {question}
ANSWER: {answer}"""


@observe(name="judge-faithfulness")
def judge_faithfulness(question: str, answer: str, contexts: list[str]) -> dict:
    resp = judge_llm.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(context="\n---\n".join(contexts), question=question, answer=answer),
        }],
        name="judge-llm",
    )
    try:
        out = json.loads(resp.choices[0].message.content)
        return {"score": float(out["score"]), "reason": out.get("reason", "")}
    except Exception:
        return {"score": 0.0, "reason": "judge returned unparseable output"}


def correctness(answer: str, expected: dict) -> float:
    """Cheap deterministic check against the dataset's expected_output."""
    a = answer.lower()
    if expected.get("answerable") is False:
        return 1.0 if ("don't know" in a or "do not know" in a) else 0.0
    kws = expected["keywords"]
    return sum(k.lower() in a for k in kws) / len(kws)
