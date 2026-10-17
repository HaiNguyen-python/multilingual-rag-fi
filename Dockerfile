FROM python:3.12-slim-bookworm

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY corpus ./corpus

ENV MLRAG_CORPUS_DIR=/app/corpus \
    MLRAG_INDEX_DIR=/app/index \
    MLRAG_PROVIDER=offline
RUN mlrag index
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["mlrag", "serve", "--host", "0.0.0.0", "--port", "8000"]
