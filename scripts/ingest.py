"""Index docs/ into Chroma. Usage: python -m scripts.ingest 300 800"""
import sys

from app.config import DEFAULT_CHUNK_SIZE
from app.rag import ingest

for size in [int(s) for s in sys.argv[1:]] or [DEFAULT_CHUNK_SIZE]:
    print(f"chunk_size={size}: {ingest(size)} chunks")
