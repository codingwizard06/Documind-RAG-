"""
DocuMind — FastAPI serving layer.

Exposes the RAG pipeline as a REST API, which is how this would actually be
integrated into a product (a Streamlit app is for demos; this is for
services calling in). Run with:

    uvicorn api:app --reload --port 8000

Then:
    curl -X POST http://localhost:8000/query \
         -H "Content-Type: application/json" \
         -d '{"question": "How many sick days do employees get?"}'
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings
from src.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Loading RAG pipeline (embedding model + vector store)...")
    pipeline = RAGPipeline()
    logger.info("Pipeline ready.")
    yield
    pipeline = None


app = FastAPI(
    title="DocuMind RAG API",
    description="Retrieval-Augmented Generation over internal documents.",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceChunk(BaseModel):
    source: str
    chunk_index: int
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    provider_used: str


@app.get("/health")
def health():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return {"status": "ok", "indexed_chunks": pipeline._collection.count()}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        result = pipeline.answer(request.question, top_k=request.top_k)
    except Exception as e:
        logger.exception("Error answering query")
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceChunk(source=c.source, chunk_index=c.chunk_index, score=c.score, text=c.text)
            for c in result.retrieved_chunks
        ],
        provider_used=result.provider_used,
    )


@app.get("/")
def root():
    return {
        "message": "DocuMind RAG API. POST /query with {'question': '...'} to get a grounded answer.",
        "llm_provider": settings.llm_provider,
    }
