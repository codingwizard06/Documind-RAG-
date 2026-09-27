"""
Unit tests for DocuMind.

Design note: these tests mock the embedding model and LLM API calls so the
suite runs fast, offline, and without API keys/GPU (this is what you want in
CI — hitting HuggingFace/Anthropic/OpenAI from a test suite is slow, flaky,
and costs money). The Chroma smoke test at the bottom exercises the real
storage/query engine end-to-end with a fake embedding function so it still
catches real integration bugs.
"""
import hashlib
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from chromadb import EmbeddingFunction

from src.config import settings, SAMPLE_DOCS_DIR
from src.ingest import load_documents, chunk_documents
from src.rag_pipeline import RAGPipeline, RetrievedChunk


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
class DeterministicFakeEmbedding(EmbeddingFunction):
    """Deterministic fake embeddings: same text always maps to the same
    vector, so tests are reproducible without downloading a real model."""

    def __call__(self, input):
        vectors = []
        for text in input:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=384)
            vectors.append((vec / np.linalg.norm(vec)).tolist())
        return vectors

    def name(self):
        return "deterministic_fake"


# --------------------------------------------------------------------------- #
# Ingestion tests
# --------------------------------------------------------------------------- #
def test_load_documents_finds_all_sample_files():
    docs = load_documents(SAMPLE_DOCS_DIR)
    sources = {d["source"] for d in docs}
    assert sources == {
        "data_privacy_policy.txt",
        "employee_handbook.txt",
        "incident_response_runbook.txt",
    }


def test_load_documents_skips_empty_and_unsupported(tmp_path):
    (tmp_path / "empty.txt").write_text("")
    (tmp_path / "good.txt").write_text("This has real content in it.")
    (tmp_path / "ignored.xyz").write_text("Should not be picked up.")

    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0]["source"] == "good.txt"


def test_chunk_documents_respects_metadata_and_produces_chunks():
    docs = [{"text": "A " * 2000, "source": "test.txt", "doc_id": "abc123"}]
    chunks = chunk_documents(docs)

    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk["metadata"]["source"] == "test.txt"
        assert chunk["metadata"]["chunk_index"] == i
        assert chunk["metadata"]["doc_id"] == "abc123"
        assert len(chunk["text"]) <= settings.chunk_size + settings.chunk_overlap


def test_chunk_documents_handles_empty_list():
    assert chunk_documents([]) == []


# --------------------------------------------------------------------------- #
# Retrieval tests (real Chroma engine, fake embeddings)
# --------------------------------------------------------------------------- #
@pytest.fixture
def pipeline_with_fake_embeddings(monkeypatch, tmp_path):
    """A real RAGPipeline instance, but wired to an ephemeral Chroma store
    with deterministic fake embeddings instead of downloading a real model."""
    monkeypatch.setattr(settings, "persist_directory", str(tmp_path / "vs"))
    monkeypatch.setattr(settings, "use_reranker", False)

    with patch(
        "src.rag_pipeline.embedding_functions.SentenceTransformerEmbeddingFunction",
        return_value=DeterministicFakeEmbedding(),
    ):
        pipeline = RAGPipeline()

    docs = load_documents(SAMPLE_DOCS_DIR)
    chunks = chunk_documents(docs)
    pipeline._collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return pipeline


def test_retrieve_returns_requested_number_of_chunks(pipeline_with_fake_embeddings):
    results = pipeline_with_fake_embeddings.retrieve("sick leave policy", top_k=3)
    assert len(results) == 3
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_retrieve_returns_valid_sources(pipeline_with_fake_embeddings):
    results = pipeline_with_fake_embeddings.retrieve("incident response", top_k=5)
    valid_sources = {
        "data_privacy_policy.txt", "employee_handbook.txt", "incident_response_runbook.txt"
    }
    for r in results:
        assert r.source in valid_sources


# --------------------------------------------------------------------------- #
# Prompt construction tests (pure function, no I/O)
# --------------------------------------------------------------------------- #
def test_build_prompt_includes_source_citations_and_question():
    chunks = [
        RetrievedChunk(text="Sick leave is 10 days per year.", source="handbook.txt",
                        chunk_index=1, score=0.9),
    ]
    prompt = RAGPipeline.build_prompt("How many sick days?", chunks)
    assert "How many sick days?" in prompt
    assert "[source: handbook.txt, chunk 1]" in prompt
    assert "Sick leave is 10 days per year." in prompt


def test_generate_with_no_chunks_returns_graceful_message():
    pipeline = RAGPipeline.__new__(RAGPipeline)  # skip __init__, no I/O needed
    result = pipeline.generate("irrelevant question", [])
    assert "couldn't find" in result.lower()


# --------------------------------------------------------------------------- #
# Generation tests (mocked LLM API — never call a real API in unit tests)
# --------------------------------------------------------------------------- #
def test_generate_retrieval_only_mode_returns_top_chunk(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "none")
    pipeline = RAGPipeline.__new__(RAGPipeline)
    chunks = [RetrievedChunk(text="Top answer text.", source="doc.txt", chunk_index=0, score=0.95)]
    result = pipeline.generate("a question", chunks)
    assert "Top answer text." in result
    assert "retrieval-only" in result.lower()


def test_generate_anthropic_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    pipeline = RAGPipeline.__new__(RAGPipeline)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        pipeline._generate_anthropic("some prompt")


def test_generate_anthropic_calls_api_with_expected_params(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "fake-key")
    monkeypatch.setattr(settings, "anthropic_model", "claude-sonnet-4-6")

    mock_text_block = MagicMock()
    mock_text_block.text = "Mocked grounded answer [source: doc.txt]"
    mock_response = MagicMock()
    mock_response.content = [mock_text_block]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("anthropic.Anthropic", return_value=mock_client):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        result = pipeline._generate_anthropic("some prompt")

    assert result == "Mocked grounded answer [source: doc.txt]"
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["messages"][0]["content"] == "some prompt"
