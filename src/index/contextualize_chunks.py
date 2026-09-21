"""Generate short LLM-written context for each chunk (Anthropic's Contextual
Retrieval): a 1-2 sentence blurb describing where the chunk sits in its source
document, meant to be prepended to the chunk before embedding so an isolated
chunk retains document-level context it would otherwise lose. Reported to cut
failed retrievals ~49% in Anthropic's own benchmark - this is the fix aimed at
this project's measured weak point (Contextual Precision/Recall).

Anthropic's reference design is one LLM call per chunk (made cheap there via prompt
caching the whole document). Our free-tier quota is capped by *call count*
(20-500 generate_content calls/day/model, not a caching-eligible product), so instead
this batches every chunk belonging to one document into as few calls as possible:
each call gets the (possibly truncated) full document plus a numbered list of its
chunks, and returns a JSON array of one context string per chunk. A handful of
outlier files (release-notes.md alone is 37% of the corpus) still need multiple
calls, capped at MAX_CHUNKS_PER_CALL chunks each - full document text is still
attached to every sub-batch call so each one has independent document context.

Resumable: writes to data/processed/chunk_contexts.json (absolute chunk index ->
context string) after every successful call, so a rerun skips chunks already done.
Chains across accounts the same way run_deepeval.py does (--api-key-env), and rotates
across GEN_MODEL_CANDIDATES the same way generator.py does when a model's quota runs out.
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
from google.genai.errors import APIError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.chunk import Chunk, chunk_corpus  # noqa: E402
from rag.generator import GEN_MODEL_CANDIDATES  # noqa: E402

load_dotenv()

RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"
CONTEXTS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "chunk_contexts.json"

MAX_DOC_CHARS = 12_000  # cap on document text sent per call - bounds prompt size/cost
MAX_CHUNKS_PER_CALL = 25  # sub-batch large documents instead of one giant call
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF_SECONDS = 65
REQUEST_PACING_SECONDS = 20.0

SYSTEM_PROMPT = """You are helping build a search index over technical documentation \
and source code. You will be given a full document and a numbered list of chunks \
extracted from it. For each chunk, write a short (1 sentence, max ~25 words) context \
that situates it within the overall document, so the chunk remains understandable \
retrieved in isolation. Do not repeat the chunk's content verbatim - describe where \
it fits and what it's part of.

Respond with ONLY a JSON array of strings, one per chunk, in the same order as the \
numbered list. No other text."""


def load_source_text(source: str, kind: str) -> str:
    base = RAW_ROOT / ("docs" if kind == "doc" else "code")
    return (base / source).read_text(encoding="utf-8", errors="ignore")


def group_into_call_batches(chunks: list[Chunk]) -> list[list[int]]:
    """Group absolute chunk indices into per-call batches: all chunks from the same
    source document, sub-batched at MAX_CHUNKS_PER_CALL. Preserves corpus order."""
    batches: list[list[int]] = []
    current_source = None
    current: list[int] = []
    for i, c in enumerate(chunks):
        if c.source != current_source or len(current) >= MAX_CHUNKS_PER_CALL:
            if current:
                batches.append(current)
            current = []
            current_source = c.source
        current.append(i)
    if current:
        batches.append(current)
    return batches


def load_contexts() -> dict[str, str]:
    if CONTEXTS_PATH.exists():
        return json.loads(CONTEXTS_PATH.read_text(encoding="utf-8"))
    return {}


def save_contexts(contexts: dict[str, str]) -> None:
    CONTEXTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTEXTS_PATH.write_text(json.dumps(contexts, indent=2, ensure_ascii=False), encoding="utf-8")


def build_prompt(chunks: list[Chunk], indices: list[int]) -> str:
    doc = load_source_text(chunks[indices[0]].source, chunks[indices[0]].kind)
    if len(doc) > MAX_DOC_CHARS:
        doc = doc[:MAX_DOC_CHARS] + "\n...[document truncated]"
    numbered = "\n\n".join(f"[{n + 1}] {chunks[i].text}" for n, i in enumerate(indices))
    return (
        f"<document source=\"{chunks[indices[0]].source}\">\n{doc}\n</document>\n\n"
        f"Chunks:\n{numbered}\n\n"
        f"Return a JSON array of exactly {len(indices)} context strings, one per chunk above, in order."
    )


class ContextGenerator:
    def __init__(self, api_key_env: str) -> None:
        self.client = genai.Client(api_key=os.environ.get(api_key_env, os.environ["GEMINI_API_KEY"]))
        self.remaining_candidates = list(GEN_MODEL_CANDIDATES)
        self.model = None

    def _pick_model(self) -> str:
        while self.remaining_candidates:
            candidate = self.remaining_candidates[0]
            try:
                self.client.models.generate_content(model=candidate, contents="OK")
                return candidate
            except APIError as e:
                if e.code in (429, 404, 503):
                    print(f"  generation model {candidate} unavailable ({e.code}), trying next...")
                    self.remaining_candidates.pop(0)
                    continue
                raise
        raise RuntimeError("All candidate generation models exhausted or unavailable")

    def generate_batch(self, chunks: list[Chunk], indices: list[int]) -> list[str]:
        if self.model is None:
            self.model = self._pick_model()

        prompt = build_prompt(chunks, indices)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT, temperature=0.1, response_mime_type="application/json"
        )
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                resp = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
                contexts = json.loads(resp.text)
                if not isinstance(contexts, list) or len(contexts) != len(indices):
                    raise ValueError(f"expected {len(indices)} contexts, got {contexts!r}")
                return [str(c) for c in contexts]
            except APIError as e:
                if e.code in (429, 503):
                    print(f"  {e.code} error, switching model..." if e.code == 429 else f"  {e.code} error, retrying...")
                    if e.code == 429:
                        self.remaining_candidates.pop(0)
                        self.model = self._pick_model()
                    elif attempt < RATE_LIMIT_RETRIES:
                        time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                    continue
                raise
            except httpx.TransportError as e:
                if attempt < RATE_LIMIT_RETRIES:
                    print(f"  network error ({e}), sleeping 30s before retry...")
                    time.sleep(30)
                    continue
                raise
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < RATE_LIMIT_RETRIES:
                    print(f"  malformed response ({e}), retrying...")
                    continue
                raise
        raise RuntimeError("unreachable")


def main(max_calls: int, api_key_env: str) -> None:
    chunks = chunk_corpus(RAW_ROOT / "docs", RAW_ROOT / "code")
    call_batches = group_into_call_batches(chunks)
    contexts = load_contexts()

    pending = [b for b in call_batches if not all(str(i) in contexts for i in b)]
    if not pending:
        print(f"All {len(chunks)} chunks across {len(call_batches)} calls already have contexts - nothing to do.")
        return

    print(f"{len(contexts)}/{len(chunks)} chunks contextualized so far "
          f"({len(pending)}/{len(call_batches)} calls remaining). Running up to {max_calls} more calls...")

    gen = ContextGenerator(api_key_env)
    made = 0
    for indices in pending:
        if made >= max_calls:
            break
        source = chunks[indices[0]].source
        print(f"Call {made + 1}/{max_calls}: {source} ({len(indices)} chunks)")
        try:
            new_contexts = gen.generate_batch(chunks, indices)
        except RuntimeError:
            print("All candidate generation models exhausted or unavailable for today.")
            break
        except Exception as e:
            # Not a quota/exhaustion signal - some other persistent failure (e.g.
            # the model kept returning the wrong count after every retry). Skip
            # this batch rather than crashing the whole run; it's not marked done,
            # so a later invocation will simply try it again.
            print(f"  giving up on this batch after retries ({e}), skipping...")
            made += 1
            time.sleep(REQUEST_PACING_SECONDS)
            continue
        for i, ctx in zip(indices, new_contexts):
            contexts[str(i)] = ctx
        save_contexts(contexts)
        made += 1
        time.sleep(REQUEST_PACING_SECONDS)

    print(f"\nSaved -> {CONTEXTS_PATH} ({len(contexts)}/{len(chunks)} chunks contextualized, {made} calls made this run)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("max_calls", type=int, nargs="?", default=10, help="Max LLM calls to make this run")
    parser.add_argument("--api-key-env", type=str, default="GEMINI_API_KEY", help="Env var name holding the API key to use")
    args = parser.parse_args()
    main(max_calls=args.max_calls, api_key_env=args.api_key_env)
