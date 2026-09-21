"""Embed all chunks with a local model and load them into Qdrant.

Runs fully offline (no API calls, no rate limits) using sentence-transformers.
Indexes into a separate collection from the Gemini-embedded one so the two don't
collide - see COLLECTION below vs embed_and_index.py's fastapi_corpus.

BGE models want an instruction prefix on the *query* side only (not on documents),
which retriever.py applies when EMBED_BACKEND=local.
"""
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.chunk import chunk_corpus  # noqa: E402

load_dotenv()

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
BATCH_SIZE = 64

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION_LOCAL", "fastapi_corpus_local")


def ensure_collection(qdrant: QdrantClient) -> None:
    if qdrant.collection_exists(COLLECTION):
        qdrant.delete_collection(COLLECTION)
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[2] / "data" / "raw"
    chunks = chunk_corpus(root / "docs", root / "code")
    print(f"Loaded {len(chunks)} chunks to index")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    qdrant_client = QdrantClient(url=QDRANT_URL)
    ensure_collection(qdrant_client)

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding + upserting (local)"):
        batch = chunks[i : i + BATCH_SIZE]
        vectors = model.encode([c.text for c in batch], normalize_embeddings=True)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec.tolist(),
                payload={
                    "text": c.text,
                    "source": c.source,
                    "kind": c.kind,
                    "heading_path": c.heading_path,
                },
            )
            for c, vec in zip(batch, vectors)
        ]
        qdrant_client.upsert(collection_name=COLLECTION, points=points)

    count = qdrant_client.count(COLLECTION).count
    print(f"Indexed {count} points into Qdrant collection '{COLLECTION}'")


if __name__ == "__main__":
    main()
