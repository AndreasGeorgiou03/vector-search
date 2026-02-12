import os
import json
import numpy as np
from pipelines.chunk import chunk_text
from core.embedder import Embedder
from core.index import VectorIndex

DATA_DIR = "data"
STORAGE_DIR = "storage"
INDEX_PATH = os.path.join(STORAGE_DIR, "index.faiss")
META_PATH = os.path.join(STORAGE_DIR, "meta.json")

os.makedirs(STORAGE_DIR, exist_ok=True)

embedder = Embedder()

all_chunks = []
metadata = []

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".txt"):
        path = os.path.join(DATA_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)

        for chunk in chunks:
            metadata.append({
                "source": filename,
                "text": chunk
            })
            all_chunks.append(chunk)

embeddings = embedder.encode(all_chunks)
dim = embeddings.shape[1]

index = VectorIndex(dim)
index.add(embeddings)
index.save(INDEX_PATH)

with open(META_PATH, "w") as f:
    json.dump(metadata, f)

print("Ingestion complete.")
