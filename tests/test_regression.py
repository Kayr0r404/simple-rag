"""STAGE 6: CI regression test. Needs real API keys; run with `pytest -q`."""
import json
from pathlib import Path

import pytest

from app.config import langfuse
from app.evals import correctness
from app.rag import ask

ITEMS = json.loads((Path(__file__).resolve().parent.parent / "data" / "dataset.json").read_text())


@pytest.mark.parametrize("item", ITEMS, ids=[i["question"][:40] for i in ITEMS])
def test_answer_quality(item):
    result = ask(item["question"], user_id="ci", session_id="ci-regression")
    assert correctness(result["answer"], item["expected"]) == 1.0, result["answer"]


def teardown_module(_):
    langfuse.flush()
