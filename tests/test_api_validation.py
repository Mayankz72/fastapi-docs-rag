"""Request validation for POST /query.
"""
import sys
import types

import pytest
from fastapi.testclient import TestClient


class _StubPipeline:
    calls: list[tuple[str, int]] = []

    def run(self, question: str, top_k: int | None = None):
        self.calls.append((question, top_k))
        return types.SimpleNamespace(answer="stub answer", contexts=[])


@pytest.fixture()
def client(monkeypatch):
    # Replace the heavy modules (google-genai, qdrant, phoenix) wholesale so importing
    # the app needs none of them installed.
    monkeypatch.setitem(
        sys.modules, "src.rag.pipeline", types.SimpleNamespace(RAGPipeline=_StubPipeline)
    )
    monkeypatch.setitem(
        sys.modules, "src.eval.phoenix_tracing", types.SimpleNamespace(start_tracing=lambda: None)
    )
    monkeypatch.delitem(sys.modules, "src.api.main", raising=False)
    _StubPipeline.calls = []

    from src.api.main import app

    return TestClient(app)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 2001},
        {"question": "ok", "top_k": 0},
        {"question": "ok", "top_k": -3},
        {"question": "ok", "top_k": 21},
    ],
)
def test_invalid_requests_are_422_and_never_reach_pipeline(client, body):
    resp = client.post("/query", json=body)
    assert resp.status_code == 422
    assert _StubPipeline.calls == []


def test_valid_request_passes_stripped_question_and_top_k_per_call(client):
    resp = client.post("/query", json={"question": "  how do I x?  ", "top_k": 3})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "stub answer"
    assert _StubPipeline.calls == [("how do I x?", 3)]


def test_top_k_defaults_to_5(client):
    client.post("/query", json={"question": "q"})
    assert _StubPipeline.calls == [("q", 5)]
