"""Embed corpus chunks and load them into Qdrant.

Resumable: progress is checkpointed after every successful batch, so re-running
after a rate-limit/quota error picks up where it left off instead of re-embedding
(and re-spending free-tier quota on) chunks already indexed.

Batches are sized by character budget, not a fixed chunk count: the free tier's real
constraint (confirmed empirically) is ~1000 *tokens* per minute for this model, not
request count, and chunk lengths vary a lot (some are 20 chars, some 2000+), so a
fixed-count batch can randomly land on several long chunks and blow the budget. At
this rate, indexing the full corpus takes several hours - that's the accepted
tradeoff for a $0 pipeline.

Supports running multiple workers in parallel, each on a different chunk range and
a different account's API key, to split the backlog and finish faster - e.g.:

    python src/index/embed_and_index.py --start 1341 --end 2086 \
        --checkpoint data/processed/embed_checkpoint_a.json --api-key-env GEMINI_API_KEY_EMBED
    python src/index/embed_and_index.py --start 2086 --end 2832 \
        --checkpoint data/processed/embed_checkpoint_b.json --api-key-env GEMINI_API_KEY_EMBED_2

Each worker's checkpoint tracks its own range independently; they upsert into the
same Qdrant collection (safe - point IDs are the chunk's absolute corpus index, so
ranges never collide). With no arguments, behaves as a single full-range worker
using the default checkpoint file, same as before.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from google.genai.types import HttpOptions, HttpRetryOptions
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.chunk import chunk_corpus  # noqa: E402

load_dotenv()

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072
CHAR_BUDGET_PER_BATCH = 2400  # ~600 tokens, extra margin under the ~1000 tokens/min cap
REQUEST_PACING_SECONDS = 75.0
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF_SECONDS = 70

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "fastapi_corpus")
DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "embed_checkpoint.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="First chunk index to process (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="Last chunk index to process (exclusive); default = end of corpus")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH), help="Path to this worker's checkpoint file")
    parser.add_argument("--api-key-env", type=str, default="GEMINI_API_KEY_EMBED", help="Env var name holding the API key to use")
    parser.add_argument(
        "--context-file", type=str, default=None,
        help="Path to a chunk_contexts.json (from contextualize_chunks.py) - when given, each chunk's "
             "LLM-generated context is prepended before embedding (Anthropic's Contextual Retrieval). "
             "Point payloads still store the original, uncontextualized text.",
    )
    return parser.parse_args()


def get_clients(api_key_env: str) -> tuple[genai.Client, QdrantClient]:
    # attempts=1 disables the SDK's own silent internal retry-on-429, so every
    # HTTP request we make is visible and accounted for in our own pacing/backoff.
    genai_client = genai.Client(
        api_key=os.environ.get(api_key_env, os.environ["GEMINI_API_KEY"]),
        http_options=HttpOptions(retry_options=HttpRetryOptions(attempts=1)),
    )
    qdrant_client = QdrantClient(url=QDRANT_URL)
    return genai_client, qdrant_client


def load_checkpoint(checkpoint_path: Path, range_start: int) -> int:
    if checkpoint_path.exists():
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))["next_index"]
    return range_start


def save_checkpoint(checkpoint_path: Path, next_index: int) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps({"next_index": next_index}), encoding="utf-8")


def ensure_collection(qdrant: QdrantClient, want_fresh_start: bool) -> None:
    exists = qdrant.collection_exists(COLLECTION)
    current_count = qdrant.count(COLLECTION).count if exists else 0
    # Only ever wipe a collection that's actually empty - protects against a
    # parallel worker's fresh checkpoint wiping data another worker already wrote.
    if want_fresh_start and exists and current_count == 0:
        qdrant.delete_collection(COLLECTION)
        exists = False
    if not exists:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )


def make_batches(embed_texts: list[str], start_index: int, end_index: int) -> list[tuple[int, list[int]]]:
    """Group indices [start_index, end_index) into (start_offset, batch_of_indices)
    pairs, each batch capped at CHAR_BUDGET_PER_BATCH total characters of embed_texts
    (at least one chunk per batch, even if that single chunk alone exceeds the
    budget). Batching by embed_texts length (not the original chunk text) so a
    contextualized chunk's extra prepended text still counts toward the budget."""
    batches = []
    i = start_index
    while i < end_index:
        batch = [i]
        total_chars = len(embed_texts[i])
        j = i + 1
        while j < end_index and total_chars + len(embed_texts[j]) <= CHAR_BUDGET_PER_BATCH:
            batch.append(j)
            total_chars += len(embed_texts[j])
            j += 1
        batches.append((i, batch))
        i = j
    return batches


def embed_batch(genai_client: genai.Client, texts: list[str]) -> list[list[float]]:
    """One request = one batchEmbedContents call. On 429, sleep past the quota
    window and retry a few times rather than hammering it with exponential backoff
    (which just burns more of the same per-minute quota). Also retries on transient
    network/DNS failures (httpx.TransportError - covers ConnectError, ConnectTimeout,
    ReadTimeout, etc.) and transient server-side errors (google.genai.errors.ServerError,
    e.g. 503 UNAVAILABLE) - an internet blip or a brief model outage shouldn't kill
    a multi-hour unattended job."""
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            resp = genai_client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            return [e.values for e in resp.embeddings]
        except ClientError as e:
            if e.code == 429 and attempt < RATE_LIMIT_RETRIES:
                print(f"\nRate limited, sleeping {RATE_LIMIT_BACKOFF_SECONDS}s before retry...")
                time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                continue
            raise
        except ServerError as e:
            if attempt < RATE_LIMIT_RETRIES:
                print(f"\nServer error ({e.code}), sleeping 30s before retry...")
                time.sleep(30)
                continue
            raise
        except httpx.TransportError as e:
            if attempt < RATE_LIMIT_RETRIES:
                print(f"\nNetwork error ({e}), sleeping 30s before retry...")
                time.sleep(30)
                continue
            raise


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)

    root = Path(__file__).resolve().parents[2] / "data" / "raw"
    chunks = chunk_corpus(root / "docs", root / "code")
    range_end = args.end if args.end is not None else len(chunks)
    print(f"Loaded {len(chunks)} chunks total; this worker handles [{args.start}, {range_end})")

    contexts: dict[str, str] = {}
    if args.context_file:
        contexts = json.loads(Path(args.context_file).read_text(encoding="utf-8"))
        missing = sum(1 for i in range(len(chunks)) if str(i) not in contexts)
        print(f"Loaded {len(contexts)} chunk contexts from {args.context_file} ({missing} chunks have no context yet)")
    embed_texts = [
        f"{contexts[str(i)]}\n\n{c.text}" if str(i) in contexts else c.text
        for i, c in enumerate(chunks)
    ]

    start_index = load_checkpoint(checkpoint_path, args.start)
    genai_client, qdrant_client = get_clients(args.api_key_env)
    ensure_collection(qdrant_client, want_fresh_start=(args.start == 0 and start_index == 0))

    if start_index > args.start:
        print(f"Resuming from chunk {start_index} (checkpoint found)")

    if start_index >= range_end:
        print("This worker's range is already fully indexed - nothing to do.")
        return

    batches = make_batches(embed_texts, start_index, range_end)

    try:
        for offset, indices in tqdm(batches, desc="Embedding + upserting"):
            vectors = embed_batch(genai_client, [embed_texts[i] for i in indices])
            points = [
                PointStruct(
                    id=i,
                    vector=vec,
                    payload={
                        "text": chunks[i].text,
                        "source": chunks[i].source,
                        "kind": chunks[i].kind,
                        "heading_path": chunks[i].heading_path,
                        "context": contexts.get(str(i), ""),
                    },
                )
                for i, vec in zip(indices, vectors)
            ]
            qdrant_client.upsert(collection_name=COLLECTION, points=points)
            save_checkpoint(checkpoint_path, offset + len(indices))
            time.sleep(REQUEST_PACING_SECONDS)
    except ClientError as e:
        if e.code == 429:
            print(
                f"\nStill rate-limited after {RATE_LIMIT_RETRIES} retries. "
                f"Progress is checkpointed — just re-run this script later to resume."
            )
            sys.exit(1)
        raise

    count = qdrant_client.count(COLLECTION).count
    print(f"This worker's range done. Collection '{COLLECTION}' now has {count} points total.")
    checkpoint_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
