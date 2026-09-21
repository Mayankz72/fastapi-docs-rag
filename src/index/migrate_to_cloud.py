"""One-off migration: copy an existing local Qdrant collection's points to a
Qdrant Cloud cluster, for deploying the app somewhere the local Docker Qdrant
isn't reachable from. Reads the already-embedded points directly - no
re-embedding, so this costs no Gemini API quota.

Usage:
    python -m src.index.migrate_to_cloud --collection fastapi_corpus_contextual

Reads the destination cluster's URL/API key from QDRANT_CLOUD_URL and
QDRANT_CLOUD_API_KEY (kept separate from QDRANT_URL/QDRANT_API_KEY, which the
app itself reads at request time, so this script's source is always "whatever
local Qdrant docker-compose is running" regardless of what the app is
currently pointed at).

Point IDs are the corpus chunk's absolute index (0..total-1) - see
embed_and_index.py - which is what makes the payload-reconstruction fallback
below possible: chunk_corpus() deterministically regenerates the exact same
chunk text/source/kind/heading_path from data/raw, so a corrupted payload can
be rebuilt from scratch rather than lost.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.chunk import chunk_corpus  # noqa: E402

load_dotenv()

LOCAL_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
CLOUD_URL = os.environ.get("QDRANT_CLOUD_URL")
CLOUD_API_KEY = os.environ.get("QDRANT_CLOUD_API_KEY")
CONTEXT_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "chunk_contexts.json"

BATCH_SIZE = 100
RETRIEVE_RETRIES = 5


class ReconstructPayload:
    """Lazily regenerates chunk text/source/kind/heading_path + context from
    data/raw + chunk_contexts.json, matching embed_and_index.py's payload
    schema exactly - used only for points whose stored payload is corrupted
    server-side (the payload panics on every fetch, even alone, while the
    vector fetches fine)."""

    def __init__(self) -> None:
        self._chunks = None
        self._contexts = None

    def _ensure_loaded(self) -> None:
        if self._chunks is not None:
            return
        root = Path(__file__).resolve().parents[2] / "data" / "raw"
        self._chunks = chunk_corpus(root / "docs", root / "code")
        self._contexts = json.loads(CONTEXT_FILE.read_text(encoding="utf-8")) if CONTEXT_FILE.exists() else {}

    def get(self, index: int) -> dict:
        self._ensure_loaded()
        c = self._chunks[index]
        return {
            "text": c.text,
            "source": c.source,
            "kind": c.kind,
            "heading_path": c.heading_path,
            "context": self._contexts.get(str(index), ""),
        }


def fetch_point(source: QdrantClient, collection: str, point_id: int, reconstruct: ReconstructPayload) -> PointStruct:
    """Fetch one point with retries; if its payload is corrupted server-side
    (fetching with payload panics even alone, but vector-only succeeds),
    rebuild the payload from source instead of giving up or dropping it."""
    for attempt in range(RETRIEVE_RETRIES + 1):
        try:
            (record,) = source.retrieve(collection_name=collection, ids=[point_id], with_payload=True, with_vectors=True)
            return PointStruct(id=record.id, vector=record.vector, payload=record.payload)
        except UnexpectedResponse as e:
            if e.status_code != 500:
                raise
            if attempt < RETRIEVE_RETRIES:
                time.sleep(2 * (attempt + 1))
                continue
            print(f"  point {point_id}: payload fetch panics server-side, reconstructing payload from source instead")
            (record,) = source.retrieve(collection_name=collection, ids=[point_id], with_payload=False, with_vectors=True)
            return PointStruct(id=record.id, vector=record.vector, payload=reconstruct.get(point_id))


def fetch_batch(source: QdrantClient, collection: str, ids: list[int], reconstruct: ReconstructPayload) -> list[PointStruct]:
    """Try the whole batch at once (fast path); if the server panics on it
    (qdrant_client.http.exceptions.UnexpectedResponse, a known Qdrant
    "OffsetZero" bug on corrupted payloads), fall back to fetching this
    batch one point at a time so a single corrupted point doesn't block the
    other 99 in its batch."""
    for attempt in range(RETRIEVE_RETRIES + 1):
        try:
            records = source.retrieve(collection_name=collection, ids=ids, with_payload=True, with_vectors=True)
            return [PointStruct(id=r.id, vector=r.vector, payload=r.payload) for r in records]
        except UnexpectedResponse as e:
            if e.status_code != 500:
                raise
            if attempt < RETRIEVE_RETRIES:
                time.sleep(2 * (attempt + 1))
                continue
            print(f"  batch {ids[0]}-{ids[-1]} still failing after retries, falling back to per-point fetch...")
            return [fetch_point(source, collection, i, reconstruct) for i in ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, help="Collection name, same on both source and destination")
    args = parser.parse_args()

    if not CLOUD_URL or not CLOUD_API_KEY:
        raise SystemExit("Set QDRANT_CLOUD_URL and QDRANT_CLOUD_API_KEY in .env first.")

    source = QdrantClient(url=LOCAL_URL)
    dest = QdrantClient(url=CLOUD_URL, api_key=CLOUD_API_KEY)
    reconstruct = ReconstructPayload()

    if not source.collection_exists(args.collection):
        raise SystemExit(f"Source collection '{args.collection}' not found at {LOCAL_URL}")

    info = source.get_collection(args.collection)
    total = source.count(args.collection).count
    print(f"Source: {args.collection} ({total} points, dim={info.config.params.vectors.size})")

    if not dest.collection_exists(args.collection):
        dest.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(
                size=info.config.params.vectors.size,
                distance=info.config.params.vectors.distance,
            ),
        )
        print(f"Created destination collection '{args.collection}' on {CLOUD_URL}")

    already = dest.count(args.collection).count
    if already >= total:
        print(f"Destination already has {already}/{total} points - nothing to do.")
        return

    # upsert() is idempotent by ID, so re-sending already-migrated batches on a
    # rerun is harmless - not worth tracking a resume point for a collection
    # this size.
    migrated = 0
    for start in range(0, total, BATCH_SIZE):
        batch_ids = list(range(start, min(start + BATCH_SIZE, total)))
        points = fetch_batch(source, args.collection, batch_ids, reconstruct)
        dest.upsert(collection_name=args.collection, points=points)
        migrated += len(points)
        print(f"  migrated {migrated}/{total}")

    final = dest.count(args.collection).count
    print(f"Done. Destination now has {final} points.")


if __name__ == "__main__":
    main()
