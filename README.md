# Ask the Docs: learn Langfuse by building a RAG app

A tiny RAG app over a fake product's docs ("Pebble", a note-taking CLI). The RAG is
deliberately trivial; **each stage teaches one Langfuse feature**. Code is marked
with `# STAGE n` comments so you can find the relevant lines.

Stack: FastAPI, Chroma (local embeddings), Gemini for generation (via its
OpenAI-compatible endpoint), Langfuse Python SDK v3.

## Setup

```bash
# 1. Run Langfuse locally (needs Docker)
git clone https://github.com/langfuse/langfuse.git && cd langfuse
docker compose up -d          # UI at http://localhost:3000
# In the UI: create account -> new project -> Settings -> API Keys

# 2. This project
cd ../ask-the-docs
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # paste your Langfuse + Gemini keys
python -m scripts.ingest      # index the docs
uvicorn app.main:app --reload
```

Try it:

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"How much is the Plus plan?","user_id":"alice"}'
```

---

## Stage 1: Tracing

**Read:** `app/rag.py` (`@observe` decorators, `build_llm_client` in `app/config.py`).
**Do:** ask 5 questions, open **Tracing** in the UI.
**Look for:** one trace per request, with `retrieve`, `build_prompt`, `generate` as nested spans; the generation shows model, tokens, cost, latency.
**Exercises:**

1. Which chunks were retrieved for "What is the API rate limit?" Were they the right ones?
2. Ask something off-topic. What did retrieval return anyway?
3. Add a new `@observe` span (e.g. `rerank`) between retrieve and generate.

## Stage 2: Sessions and users

**Read:** `update_current_trace(user_id, session_id)` in `ask()`; session history in `main.py`.
**Do:** ask a question, then a follow-up reusing the returned `session_id` ("and for the Team plan?"). Repeat with a second `user_id`.
**Look for:** the **Sessions** tab replays the conversation; filter Tracing by user.
**Exercise:** does the follow-up retrieve good chunks? (Hint: retrieval only sees the last message. How would you fix that?)

## Stage 3: Prompt management

**Read:** `build_prompt()`, `scripts/seed_prompt.py`.
**Do:** `python -m scripts.seed_prompt`, then `python -m scripts.seed_prompt --v2`. Ask the same question with `"prompt_label":"staging"` vs default.
**Look for:** **Prompts** → version history; each generation links to the exact prompt version; metrics per version.
**Exercises:**

1. Edit the prompt in the UI, save, and promote it to `production` without restarting the app.
2. Stop Langfuse (`docker compose stop`) and ask a question. What does the fallback do?

## Stage 4: Scores and feedback

**Read:** `/feedback` and `score_faithfulness` in `main.py`, `app/evals.py`.
**Do:** ask questions, then `curl localhost:8000/feedback -H 'content-type: application/json' -d '{"trace_id":"<id>","thumbs_up":false,"comment":"too vague"}'`.
**Look for:** `user-feedback` and `faithfulness` scores on traces. Filter by `faithfulness = 0` or thumbs down.
**Exercises:**

1. Find your worst trace and diagnose it: retrieval, prompt, or model?
2. Break the docs on purpose (edit a number in `pricing.md`, re-ingest). Does the judge notice?

## Stage 5: Datasets and experiments

**Read:** `data/dataset.json`, `scripts/seed_dataset.py`, `scripts/run_experiment.py`.
**Do:**

```bash
python -m scripts.seed_dataset
python -m scripts.run_experiment --chunk-size 150
python -m scripts.run_experiment --chunk-size 800
python -m scripts.run_experiment --chunk-size 500 --prompt-label staging
```

**Look for:** **Datasets** → run comparison table: correctness, faithfulness, cost, latency per run.
**Exercises:**

1. Which config wins? Is it worth the extra cost/latency?
2. Which items fail across *every* run? What does that say about the docs vs the pipeline?
3. Add 5 new items, including 2 unanswerable ones.

## Stage 6 (stretch): Production habits

- **Masking:** `mask_pii` in `config.py` redacts emails before they leave your app. Ask "email me at <bob@x.com>" and check the trace.
- **Sampling:** set `LANGFUSE_SAMPLE_RATE=0.5` and watch roughly half the traces disappear.
- **Tags/environments:** `env:*`, `chunk:*`, `prompt:*` tags are set in `ask()`. Build a filtered view for `env:ci`.
- **Regression tests:** `pytest -q` runs the dataset against the pipeline; `.github/workflows/regression.yml` runs it on PRs.

## Layout

```
app/config.py    env + Langfuse client + PII masking
app/rag.py       chunking, retrieval, prompt fetch, generation (traced)
app/evals.py     LLM-as-judge faithfulness + deterministic correctness
app/main.py      FastAPI: /ask, /feedback
scripts/         ingest, seed_prompt, seed_dataset, run_experiment
data/dataset.json  16 Q&A items (3 unanswerable)
tests/           CI regression test
```

## Notes

- Built against `langfuse>=3,<4`. SDK method names move between majors; if something errors, check the Langfuse Python SDK docs for your version.
- Tracing flushes in the background; short scripts call `langfuse.flush()` before exiting.

# simple-rag
