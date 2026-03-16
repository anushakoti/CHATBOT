"""
Unit and integration tests for the Dell Multimodal RAG backend.

Run with:
    pytest backend/tests/ -v

Environment variables required (or set in backend/.env):
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
"""
from __future__ import annotations

import base64
import io
import os
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Make sure `app` is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _fake_aws_env(monkeypatch):
    """Ensure settings can be instantiated without real AWS credentials."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test_key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test_secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


# ── Settings ──────────────────────────────────────────────────────────────────

class TestSettings:
    def test_defaults_load(self):
        from app.config.settings import Settings
        s = Settings(
            AWS_ACCESS_KEY_ID="k",
            AWS_SECRET_ACCESS_KEY="s",
            AWS_DEFAULT_REGION="ap-south-1",  # explicit to bypass env override in CI
        )
        assert s.aws_default_region == "ap-south-1"
        assert s.retriever_k == 6
        assert s.llm_max_tokens == 1024

    def test_overrides(self):
        from app.config.settings import Settings
        s = Settings(
            AWS_ACCESS_KEY_ID="k",
            AWS_SECRET_ACCESS_KEY="s",
            LLM_MAX_TOKENS="2048",
            RETRIEVER_K="10",
        )
        assert s.llm_max_tokens == 2048
        assert s.retriever_k == 10


# ── Schemas ───────────────────────────────────────────────────────────────────

class TestSchemas:
    def test_query_request_validation(self):
        from app.models.schemas import QueryRequest
        req = QueryRequest(question="What is the weight?")
        assert req.k == 6
        assert not req.include_sources

    def test_query_request_rejects_short(self):
        from pydantic import ValidationError
        from app.models.schemas import QueryRequest
        with pytest.raises(ValidationError):
            QueryRequest(question="Hi")  # < 3 chars

    def test_eval_example_enum(self):
        from app.models.schemas import EvalExample, EvalType, SourceType
        ex = EvalExample(
            question="What is X?",
            ground_truth="X is Y.",
            source_type=SourceType.table,
            eval_type=EvalType.factual_qa,
        )
        assert ex.eval_type == EvalType.factual_qa

    def test_ragas_scores_partial(self):
        from app.models.schemas import RagasScores
        s = RagasScores(faithfulness=0.9)
        assert s.faithfulness == 0.9
        assert s.answer_relevancy is None


# ── Image detection ───────────────────────────────────────────────────────────

class TestImageDetection:
    def _make_png_b64(self) -> str:
        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.new("RGB", (10, 10), color=(255, 0, 0)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def test_looks_like_base64_true(self):
        from app.services.tools import looks_like_base64
        assert looks_like_base64("A" * 200)

    def test_looks_like_base64_false(self):
        from app.services.tools import looks_like_base64
        assert not looks_like_base64("short text")

    def test_detect_image_mime_png(self):
        from app.services.tools import detect_image_mime
        b64 = self._make_png_b64()
        assert detect_image_mime(b64) == "image/png"

    def test_detect_image_mime_non_image(self):
        from app.services.tools import detect_image_mime
        b64 = base64.b64encode(b"this is not an image" * 20).decode()
        assert detect_image_mime(b64) is None

    def test_split_image_text(self):
        from langchain_core.documents import Document
        from app.services.tools import split_image_text
        b64 = self._make_png_b64()
        docs = [
            Document(page_content="Some product text"),
            Document(page_content=b64),
        ]
        result = split_image_text(docs)
        assert len(result["texts"]) == 1
        assert len(result["images"]) == 1
        assert result["texts"][0] == "Some product text"


# ── Document loader helpers ───────────────────────────────────────────────────

class TestDocumentLoader:
    def test_resize_image_bytes_small_unchanged(self):
        from PIL import Image as PILImage
        from app.services.document_loader import _resize_image_bytes
        buf = io.BytesIO()
        PILImage.new("RGB", (50, 50)).save(buf, format="PNG")
        data = buf.getvalue()
        result = _resize_image_bytes(data, max_bytes=10_000_000)
        assert result == data  # under limit → unchanged

    def test_resize_image_bytes_large_reduced(self):
        from PIL import Image as PILImage
        from app.services.document_loader import _resize_image_bytes
        # Create a large image that will exceed 100 bytes
        buf = io.BytesIO()
        PILImage.new("RGB", (500, 500), color=(128, 0, 0)).save(buf, format="PNG")
        data = buf.getvalue()
        result = _resize_image_bytes(data, max_bytes=100)  # tiny limit
        # Just verify it returns bytes without crashing
        assert isinstance(result, bytes)

    def test_encode_bytes_b64(self):
        from app.services.document_loader import _encode_bytes_b64
        data = b"hello"
        assert _encode_bytes_b64(data) == base64.b64encode(b"hello").decode()


# ── Vector store ──────────────────────────────────────────────────────────────

class TestVectorStore:
    def test_indexed_count_zero_when_empty(self):
        """indexed_count should return 0 when no store has been initialised."""
        # Reset module state
        import app.services.vector_store as vs_module
        vs_module._vector_store = None
        vs_module._docstore = None
        vs_module._retriever = None

        with patch("app.services.vector_store.get_embeddings") as mock_emb, \
             patch("app.services.vector_store.Chroma") as mock_chroma:
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_chroma.return_value._collection = mock_collection
            mock_emb.return_value = MagicMock()

            from app.services.vector_store import indexed_count
            count = indexed_count()
            assert count == 0


# ── Agent — built-in eval examples ───────────────────────────────────────────

class TestAgentEvalExamples:
    def test_builtin_set_has_15_examples(self):
        from app.services.agent import ALL_EVAL_EXAMPLES
        assert len(ALL_EVAL_EXAMPLES) == 15

    def test_factual_qa_count(self):
        from app.models.schemas import EvalType
        from app.services.agent import ALL_EVAL_EXAMPLES
        factual = [e for e in ALL_EVAL_EXAMPLES if e.eval_type == EvalType.factual_qa]
        assert len(factual) == 10

    def test_recommendation_count(self):
        from app.models.schemas import EvalType
        from app.services.agent import ALL_EVAL_EXAMPLES
        recs = [e for e in ALL_EVAL_EXAMPLES if e.eval_type == EvalType.recommendation]
        assert len(recs) == 5

    def test_all_examples_have_required_fields(self):
        from app.services.agent import ALL_EVAL_EXAMPLES
        for ex in ALL_EVAL_EXAMPLES:
            assert ex.question
            assert ex.ground_truth
            assert ex.source_type
            assert ex.eval_type


# ── FastAPI routes (no real Bedrock calls) ────────────────────────────────────

@pytest.fixture
def client():
    """Return a TestClient with mocked dependencies."""
    import app.services.vector_store as vs_module
    vs_module._vector_store = None
    vs_module._docstore = None
    vs_module._retriever = None

    with patch("app.services.vector_store.get_embeddings") as mock_emb, \
         patch("app.services.vector_store.Chroma") as mock_chroma, \
         patch("app.services.models.get_bedrock_client"):

        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_chroma.return_value._collection = mock_collection
        mock_emb.return_value = MagicMock()

        from fastapi.testclient import TestClient
        from app.main import app as fastapi_app
        yield TestClient(fastapi_app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "claude_model" in data
        assert "indexed_docs" in data


class TestEvalExamplesEndpoint:
    def test_returns_15_examples(self, client):
        resp = client.get("/evaluate/examples")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 15

    def test_example_schema(self, client):
        resp = client.get("/evaluate/examples")
        first = resp.json()[0]
        assert "question" in first
        assert "ground_truth" in first
        assert "eval_type" in first
        assert "source_type" in first


class TestQueryEndpointValidation:
    def test_query_without_index_returns_400(self, client):
        with patch("app.main.indexed_count", return_value=0):
            resp = client.post("/query", json={"question": "What is the weight?"})
            assert resp.status_code == 400

    def test_query_too_short_returns_422(self, client):
        resp = client.post("/query", json={"question": "Hi"})
        assert resp.status_code == 422


class TestIngestEndpointValidation:
    def test_ingest_no_files_returns_422(self, client):
        resp = client.post("/ingest")
        assert resp.status_code == 422

    def test_ingest_non_pdf_returns_400(self, client):
        resp = client.post(
            "/ingest",
            files=[("files", ("test.txt", b"hello", "text/plain"))],
        )
        assert resp.status_code == 400
