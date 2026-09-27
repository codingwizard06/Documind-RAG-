"""
Centralized configuration for DocuMind.
All tunables live here so the rest of the codebase never hardcodes paths,
model names, or magic numbers.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
EVAL_DIR = PROJECT_ROOT / "eval"


@dataclass
class Settings:
    # --- Embeddings (local, free, no API key required) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")

    # --- Chunking ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 800))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 120))

    # --- Retrieval ---
    top_k: int = int(os.getenv("TOP_K", 4))
    use_reranker: bool = os.getenv("USE_RERANKER", "true").lower() == "true"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", 10))

    # --- Vector store ---
    collection_name: str = "documind_collection"
    persist_directory: str = str(VECTOR_STORE_DIR)

    # --- LLM generation ---
    # Provider can be "anthropic", "openai", or "none" (retrieval-only mode).
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    max_tokens: int = int(os.getenv("MAX_TOKENS", 800))
    temperature: float = float(os.getenv("TEMPERATURE", 0.1))

    # --- Prompting ---
    system_prompt: str = (
        "You are a careful assistant answering questions strictly using the "
        "provided context excerpts from internal documents. Rules:\n"
        "1. Only use information present in the context. Do not use outside knowledge.\n"
        "2. If the context does not contain the answer, say so explicitly instead of guessing.\n"
        "3. Always cite the source document(s) you used, using the [source: filename] format "
        "given with each excerpt.\n"
        "4. Be concise and directly answer the question first, then add supporting detail."
    )


settings = Settings()

for d in (VECTOR_STORE_DIR, EVAL_DIR):
    d.mkdir(parents=True, exist_ok=True)
