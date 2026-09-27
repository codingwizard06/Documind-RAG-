FROM python:3.12-slim

WORKDIR /app

# System deps for building some Python packages (e.g. tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-build the vector index at image build time so the container starts
# ready to serve (comment out if you'd rather build it at runtime/startup).
RUN python -m src.ingest --source data/sample_docs

EXPOSE 8000 8501

# Default: run the FastAPI service. Override CMD to run Streamlit instead:
#   docker run -p 8501:8501 documind streamlit run app.py --server.address=0.0.0.0
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
