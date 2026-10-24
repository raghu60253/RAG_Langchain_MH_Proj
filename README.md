# PDF RAG API

FastAPI wrapper and Docker setup for asking questions about `AST-2.pdf`. The
API calls Ollama Cloud for chat completions and uses a local Hugging Face
sentence-transformer model for PDF embeddings.

## Configure

1. Copy `.env.example` to `.env`.
2. Set `OLLAMA_API_KEY` to a valid Ollama Cloud API key. Do not commit `.env`.
3. Optionally set `OLLAMA_CHAT_MODEL`, `OLLAMA_BASE_URL`,
   `HF_EMBEDDING_MODEL`, or `PDF_PATH`.

`OLLAMA_BASE_URL` defaults to `https://ollama.com/v1`. The PDF defaults to
`AST-2.pdf`. The first launch downloads the embedding model and indexes the PDF;
later launches reuse the persisted Chroma collection.

## Run locally

With the project virtual environment activated:

```powershell
pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## Run with Docker

Install Docker Desktop, then from the project directory:

```powershell
Copy-Item .env.example .env
# Edit .env and add your Ollama Cloud API key.
docker compose up --build
```

The first container start may take a while while it downloads the embedding
model and creates the PDF index. The Chroma index persists in the
`chroma-data` Docker volume.

Ask a question:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"Explain manifest.in"}'
```

Check `http://localhost:8000/health` for the health endpoint. Stop the service
with `Ctrl+C`, or run `docker compose down`. To remove the saved index as well,
run `docker compose down --volumes`; the next startup will rebuild it.

The API currently has no authentication. Keep it private or put it behind an
authenticated reverse proxy before exposing it publicly.
