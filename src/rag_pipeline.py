"""
Core RAG pipeline: query -> retrieve -> (optional) rerank -> build grounded
prompt -> call LLM -> return answer with citations.

This module is provider-agnostic: it can run against Anthropic's API, OpenAI's
API, or in "retrieval-only" mode (no LLM call) so the retrieval half of the
system can be demoed/evaluated without any API key.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions

from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_reranker = None  # lazy-loaded singleton, cross-encoder models are heavy to init


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading reranker model: %s", settings.reranker_model)
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    score: float


@dataclass
class RAGResponse:
    answer: str
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    provider_used: str = "none"


class RAGPipeline:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=settings.persist_directory)
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=self._embed_fn,
        )

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        top_k = top_k or settings.top_k
        fetch_n = settings.rerank_top_n if settings.use_reranker else top_k
        fetch_n = max(fetch_n, top_k)

        results = self._collection.query(query_texts=[query], n_results=fetch_n)

        candidates = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            # Chroma cosine distance -> similarity score (higher is better)
            similarity = 1 - dist
            candidates.append(RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                chunk_index=meta.get("chunk_index", -1),
                score=similarity,
            ))

        if settings.use_reranker and candidates:
            candidates = self._rerank(query, candidates)

        return candidates[:top_k]

    def _rerank(self, query: str, candidates: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Cross-encoder reranking: much more accurate than bi-encoder similarity
        alone, at the cost of being slower (used only on the top-N shortlist)."""
        model = _get_reranker()
        pairs = [(query, c.text) for c in candidates]
        scores = model.predict(pairs)
        for c, s in zip(candidates, scores):
            c.score = float(s)
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_prompt(query: str, chunks: List[RetrievedChunk]) -> str:
        context_blocks = []
        for c in chunks:
            context_blocks.append(f"[source: {c.source}, chunk {c.chunk_index}]\n{c.text}")
        context = "\n\n---\n\n".join(context_blocks)
        return (
            f"Context excerpts:\n\n{context}\n\n"
            f"---\n\nQuestion: {query}\n\n"
            f"Answer the question using only the context above, and cite sources "
            f"using [source: filename] notation."
        )

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(self, query: str, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "I couldn't find anything relevant to that question in the ingested documents."

        prompt = self.build_prompt(query, chunks)
        provider = settings.llm_provider.lower()

        if provider == "anthropic":
            return self._generate_anthropic(prompt)
        elif provider == "openai":
            return self._generate_openai(prompt)
        else:
            # Retrieval-only fallback: no LLM call, just return the top chunk
            # so the retrieval half of the system is still demoable/testable
            # without an API key.
            top = chunks[0]
            return (
                f"[Retrieval-only mode — no LLM configured]\n\n"
                f"Most relevant excerpt (source: {top.source}):\n\n{top.text}"
            )

    def _generate_anthropic(self, prompt: str) -> str:
        import anthropic
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            system=settings.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if hasattr(block, "text"))

    def _generate_openai(self, prompt: str) -> str:
        from openai import OpenAI
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.openai_model,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            messages=[
                {"role": "system", "content": settings.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------ #
    # Public entrypoint
    # ------------------------------------------------------------------ #
    def answer(self, query: str, top_k: Optional[int] = None) -> RAGResponse:
        chunks = self.retrieve(query, top_k=top_k)
        answer_text = self.generate(query, chunks)
        return RAGResponse(
            answer=answer_text,
            retrieved_chunks=chunks,
            provider_used=settings.llm_provider,
        )


if __name__ == "__main__":
    pipeline = RAGPipeline()
    result = pipeline.answer("How many days of parental leave do primary caregivers get?")
    print("\nANSWER:\n", result.answer)
    print("\nSOURCES:")
    for c in result.retrieved_chunks:
        print(f"  - {c.source} (chunk {c.chunk_index}, score={c.score:.3f})")
