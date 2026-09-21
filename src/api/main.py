from typing import Annotated

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, StringConstraints

from ..eval.phoenix_tracing import start_tracing
from ..rag.pipeline import RAGPipeline

start_tracing()

app = FastAPI(title="FastAPI Docs RAG")
pipeline = RAGPipeline()


class QueryRequest(BaseModel):
    # Rejected at validation (422) instead of reaching Gemini, which errors on empty
    # input and would surface as a 500; max_length bounds per-request embedding cost.
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    # Passed per call, not set on the shared pipeline, so concurrent requests
    # with different top_k can't overwrite each other.
    result = pipeline.run(req.question, top_k=req.top_k)
    return QueryResponse(answer=result.answer, sources=result.contexts)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
