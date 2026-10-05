"""STAGE 5: upload data/dataset.json as a Langfuse dataset."""
import json
from pathlib import Path

from app.config import DATASET_NAME, langfuse

items = json.loads((Path(__file__).resolve().parent.parent / "data" / "dataset.json").read_text())
try:
    langfuse.create_dataset(name=DATASET_NAME, description="Pebble docs Q&A (incl. unanswerable)")
except Exception:
    pass  # already exists
for it in items:
    langfuse.create_dataset_item(
        dataset_name=DATASET_NAME, input={"question": it["question"]}, expected_output=it["expected"],
    )
langfuse.flush()
print(f"uploaded {len(items)} items to '{DATASET_NAME}' (re-running adds duplicates; delete in UI first)")
