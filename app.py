"""
DocuMind — Streamlit chat interface for the RAG pipeline.

Run with:
    streamlit run app.py
"""
import streamlit as st

from src.config import settings, SAMPLE_DOCS_DIR
from src.ingest import run_ingestion
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="DocuMind — Document QA", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def get_pipeline():
    return RAGPipeline()


def ensure_index_built():
    """Build the vector store on first run if it's empty."""
    pipeline = get_pipeline()
    count = pipeline._collection.count()
    if count == 0:
        with st.spinner("Building vector index from sample documents (first run only)..."):
            run_ingestion(source_dir=SAMPLE_DOCS_DIR, reset=True)
        st.cache_resource.clear()
    return get_pipeline()


st.title("📄 DocuMind")
st.caption(
    "Retrieval-Augmented Generation over internal documents — every answer is "
    "grounded in retrieved source chunks and cited."
)

with st.sidebar:
    st.header("⚙️ Configuration")
    st.write(f"**LLM Provider:** `{settings.llm_provider}`")
    st.write(f"**Embedding model:** `{settings.embedding_model.split('/')[-1]}`")
    st.write(f"**Reranker:** {'enabled' if settings.use_reranker else 'disabled'}")
    top_k = st.slider("Chunks to retrieve (top_k)", min_value=1, max_value=10, value=settings.top_k)

    st.divider()
    st.subheader("📚 Knowledge base")
    st.write("Currently indexed: data privacy policy, employee handbook, incident response runbook.")
    if st.button("🔄 Rebuild index from data/sample_docs"):
        with st.spinner("Rebuilding..."):
            run_ingestion(source_dir=SAMPLE_DOCS_DIR, reset=True)
        st.cache_resource.clear()
        st.success("Index rebuilt.")

    st.divider()
    if not settings.anthropic_api_key and not settings.openai_api_key:
        st.warning(
            "No LLM API key found. Running in **retrieval-only mode** — you'll "
            "see raw retrieved chunks instead of a generated answer. Add "
            "ANTHROPIC_API_KEY or OPENAI_API_KEY to your .env file to enable "
            "full generation."
        )

pipeline = ensure_index_built()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}** (chunk {s['chunk_index']}, score {s['score']:.3f})")
                    st.text(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))

query = st.chat_input("Ask a question about the indexed documents...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            result = pipeline.answer(query, top_k=top_k)
        st.markdown(result.answer)
        sources = [
            {"source": c.source, "chunk_index": c.chunk_index, "score": c.score, "text": c.text}
            for c in result.retrieved_chunks
        ]
        if sources:
            with st.expander("📎 Sources"):
                for s in sources:
                    st.markdown(f"**{s['source']}** (chunk {s['chunk_index']}, score {s['score']:.3f})")
                    st.text(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))

    st.session_state.messages.append({
        "role": "assistant", "content": result.answer, "sources": sources
    })
