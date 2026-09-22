# FastAPI Docs RAG

A RAG system that answers questions about FastAPI (docs + source code), built
around an automated evaluation harness: retrieval and generation quality are
scored against a golden dataset of real, answered GitHub Discussions, so
changes to the pipeline are measured with actual numbers instead of guesswork.

**Stack:** Qdrant (vector store) · Google Gemini (`gemini-embedding-001` embeddings +
`gemini-3.5-flash-lite` generation) · DeepEval (metrics/CI) ·
Arize Phoenix (tracing) · FastAPI (serving)

## Live demo

**https://fastapi-docs-rag.onrender.com** redirects to an interactive Swagger UI —
expand **POST /query**, click **"Try it out"**, edit the request body, click
**"Execute"**.

```bash
curl -X POST https://fastapi-docs-rag.onrender.com/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I add a query parameter with a default value?"}'
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env          # fill in GEMINI_API_KEY and (optionally) GITHUB_TOKEN
docker compose up -d          # starts Qdrant on localhost:6333
```

## Pipeline

```bash
# 1. Pull the corpus (docs + source) and the Q&A golden-dataset source
python src/ingest/fetch_docs.py
python src/ingest/fetch_discussions.py      # needs GITHUB_TOKEN in .env

# 2. Chunk + embed + index into Qdrant
python src/index/embed_and_index.py

# 3. Build the golden eval set from fetched discussions
python src/eval/build_golden_dataset.py

# 4. Ask a question
python src/rag/pipeline.py "How do I define a path parameter with a type?"

# 5. Run the eval harness (scores 30 golden examples by default)
python src/eval/run_deepeval.py 30

# 6. (optional) Trace calls in Phoenix while querying
python src/eval/phoenix_tracing.py    # UI on localhost, port 6006

# 7. Serve the API
uvicorn src.api.main:app --reload
```
