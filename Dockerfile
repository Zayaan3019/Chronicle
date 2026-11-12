# Chronicle on an always-on EC2 host (see deploy/ and infra/projects/chronicle).
#
# CPU-only PyTorch keeps the image ~1.5 GB smaller than the default CUDA wheel,
# and the embedding model and reranker are baked in at build time so a
# container start never depends on the Hugging Face Hub being reachable.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    USE_TF=0 \
    TRANSFORMERS_NO_TF=1 \
    HF_HOME=/opt/hf \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY . .

# Indexes live on the /state volume so they survive restarts and redeploys.
ENV HF_HUB_OFFLINE=1 \
    CHROMA_PERSIST_DIR=/state/live_chroma_db \
    BM25_INDEX_PATH=/state/live_bm25_index.pkl

RUN useradd --create-home --uid 1000 chronicle \
    && mkdir -p /state && chown -R chronicle /state /app
USER chronicle

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

# CORS/XSRF checks are off because CloudFront fronts the app on a different
# origin than the container sees; the app has no forms that act on a session
# cookie, and nothing it serves is private.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
