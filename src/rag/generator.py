"""Generation step: turn retrieved chunks + a question into a grounded answer."""
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

load_dotenv()

# Each free-tier Gemini model has its own daily generate_content quota (discovered
# empirically to vary wildly, 20-500/day depending on model).
# Rather than hardcoding one model and failing once it runs dry, try these in order
# and stick with whichever one currently works.
GEN_MODEL_CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3-flash-preview",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
]

SYSTEM_PROMPT = """You are a documentation assistant for the FastAPI Python framework.
Answer the user's question using ONLY the provided context chunks. Each chunk is
labeled with its source file. Cite the source file(s) you used in square brackets,
e.g. [tutorial/path-params.md].

If the context doesn't contain enough information to answer, say so explicitly instead
of guessing or using outside knowledge."""


def reorder_lost_in_middle(chunks: list[dict]) -> list[dict]:
    """Chunks arrive ranked most-relevant-first from retrieval. LLMs attend best to
    the start and end of their context and worst to the middle (the "lost in the
    middle" effect), so interleave from both ends inward instead:
    highest-ranked chunk first, second-highest last, third-highest second, etc.
    Only reorders what the model sees for generation - callers that need retrieval
    ranking order (citations, eval's contextual-precision/recall metrics) should
    keep using the original chunks list."""
    reordered = [None] * len(chunks)
    left, right = 0, len(chunks) - 1
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            reordered[left] = chunk
            left += 1
        else:
            reordered[right] = chunk
            right -= 1
    return reordered


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in reorder_lost_in_middle(chunks):
        parts.append(f"[{c['source']}] ({c['heading_path']})\n{c['text']}")
    return "\n\n---\n\n".join(parts)


class Generator:
    def __init__(self, api_key_env: str = "GEMINI_API_KEY") -> None:
        self.client = genai.Client(api_key=os.environ.get(api_key_env, os.environ["GEMINI_API_KEY"]))
        self.remaining_candidates = list(GEN_MODEL_CANDIDATES)
        self.model = None  # picked lazily on first generate() call

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
        raise RuntimeError(f"All candidate generation models exhausted or unavailable: {GEN_MODEL_CANDIDATES}")

    def generate(self, question: str, chunks: list[dict]) -> str:
        if self.model is None:
            self.model = self._pick_model()

        context = build_context(chunks)
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.1)

        while True:
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=user_prompt, config=config
                )
                return resp.text
            except APIError as e:
                if e.code in (429, 503):
                    print(f"  generation model {self.model} exhausted ({e.code}), switching...")
                    self.remaining_candidates.pop(0)
                    self.model = self._pick_model()
                    continue
                raise
