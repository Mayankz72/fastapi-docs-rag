"""Retrieval over the Qdrant collection.

Supports two embedding backends, selected by the EMBED_BACKEND env var:
- "gemini" (default): Gemini API embeddings, collection QDRANT_COLLECTION
- "local": offline sentence-transformers embeddings, collection QDRANT_COLLECTION_LOCAL

Use "local" while the Gemini free-tier indexing job (slow, rate-limited) is still
catching up, or to avoid API calls entirely.

Optional cross-encoder reranking stage, enabled via USE_RERANKER=true: over-fetches
RERANK_FETCH_K candidates by embedding similarity, then rescores each (query, chunk)
pair with a small local cross-encoder and returns the top_k by that score instead.
Cross-encoders attend to the query and chunk jointly (unlike the bi-encoder embedding
similarity used for the initial fetch), which is more accurate but too slow to run
over the whole corpus - hence retrieve-then-rerank rather than rerank-everything.
Local/free (no API quota spent) since Contextual Precision/Recall were still the
weak point after Contextual Retrieval, and this doesn't compete with the
free-tier Gemini quota the rest of the pipeline is bottlenecked on.
"""
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "gemini")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")  # required for Qdrant Cloud, unset for local Docker

GEMINI_MODEL = "gemini-embedding-001"
GEMINI_COLLECTION = os.environ.get("QDRANT_COLLECTION", "fastapi_corpus")

LOCAL_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LOCAL_COLLECTION = os.environ.get("QDRANT_COLLECTION_LOCAL", "fastapi_corpus_local")
# BGE models are trained to embed queries with this instruction prefix; documents
# get no prefix (see embed_and_index_local.py).
LOCAL_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

USE_RERANKER = os.environ.get("USE_RERANKER", "false").lower() == "true"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_FETCH_K = int(os.environ.get("RERANK_FETCH_K", "20"))


class Retriever:
    def __init__(self) -> None:
        self.qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        if EMBED_BACKEND == "local":
            from sentence_transformers import SentenceTransformer

            self.collection = LOCAL_COLLECTION
            self.local_model = SentenceTransformer(LOCAL_MODEL_NAME)
        else:
            from google import genai
            from google.genai import types

            self.collection = GEMINI_COLLECTION
            self.genai = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY_EMBED", os.environ["GEMINI_API_KEY"])
            )
            self._embed_config = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")

        self.reranker = None
        if USE_RERANKER:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(RERANK_MODEL_NAME)

    def embed_query(self, query: str) -> list[float]:
        if EMBED_BACKEND == "local":
            vec = self.local_model.encode(LOCAL_QUERY_PREFIX + query, normalize_embeddings=True)
            return vec.tolist()
        resp = self.genai.models.embed_content(
            model=GEMINI_MODEL, contents=query, config=self._embed_config
        )
        return resp.embeddings[0].values

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        fetch_k = RERANK_FETCH_K if self.reranker else top_k
        vector = self.embed_query(query)
        hits = self.qdrant.query_points(
            collection_name=self.collection, query=vector, limit=fetch_k
        ).points
        candidates = [
            {
                "text": h.payload["text"],
                "source": h.payload["source"],
                "kind": h.payload["kind"],
                "heading_path": h.payload["heading_path"],
                "score": h.score,
            }
            for h in hits
        ]
        if self.reranker and candidates:
            pairs = [(query, c["text"]) for c in candidates]
            rerank_scores = self.reranker.predict(pairs)
            for c, s in zip(candidates, rerank_scores):
                c["score"] = float(s)
            candidates.sort(key=lambda c: c["score"], reverse=True)
            candidates = candidates[:top_k]
        return candidates
