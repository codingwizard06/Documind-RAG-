"""
Ingestion pipeline: load raw documents -> clean -> chunk -> embed -> persist to Chroma.

Run directly to (re)build the vector store from data/sample_docs (or any folder
passed via --source):

    python -m src.ingest --source data/sample_docs
"""
import argparse
import hashlib
import logging
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import settings, SAMPLE_DOCS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_text_from_file(path: Path) -> str:
    """Extract raw text from a single supported file."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(source_dir: Path) -> List[dict]:
    """Walk a directory and return a list of {text, source, doc_id} records."""
    documents = []
    files = sorted(
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        logger.warning("No supported files (.txt, .md, .pdf) found in %s", source_dir)

    for path in files:
        text = load_text_from_file(path).strip()
        if not text:
            logger.warning("Skipping empty document: %s", path.name)
            continue
        documents.append({
            "text": text,
            "source": path.name,
            "doc_id": hashlib.md5(str(path).encode()).hexdigest()[:12],
        })
        logger.info("Loaded %s (%d chars)", path.name, len(text))
    return documents


def chunk_documents(documents: List[dict]) -> List[dict]:
    """Split each document into overlapping chunks, preserving source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in documents:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append({
                "id": f"{doc['doc_id']}_{i}",
                "text": piece,
                "metadata": {
                    "source": doc["source"],
                    "chunk_index": i,
                    "doc_id": doc["doc_id"],
                },
            })
    logger.info("Created %d chunks from %d documents", len(chunks), len(documents))
    return chunks


def build_vector_store(chunks: List[dict], reset: bool = True) -> chromadb.Collection:
    """Embed chunks with a local sentence-transformer model and persist to Chroma."""
    client = chromadb.PersistentClient(path=settings.persist_directory)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )

    if reset:
        try:
            client.delete_collection(settings.collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    if chunks:
        collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )
    logger.info("Vector store built at %s with %d chunks", settings.persist_directory, len(chunks))
    return collection


def run_ingestion(source_dir: Path = SAMPLE_DOCS_DIR, reset: bool = True) -> chromadb.Collection:
    documents = load_documents(source_dir)
    chunks = chunk_documents(documents)
    return build_vector_store(chunks, reset=reset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the DocuMind vector store")
    parser.add_argument("--source", type=str, default=str(SAMPLE_DOCS_DIR),
                         help="Directory of .txt/.md/.pdf files to ingest")
    parser.add_argument("--no-reset", action="store_true",
                         help="Append to the existing collection instead of rebuilding it")
    args = parser.parse_args()

    run_ingestion(source_dir=Path(args.source), reset=not args.no_reset)
