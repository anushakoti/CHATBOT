# langchain-doc-chat — AI Assistant

A simplified, high-performance **Multimodal Retrieval-Augmented Generation (RAG)** system for product documentation, powered by AWS Bedrock (Claude + Titan Embeddings), ChromaDB, and LangChain.

---

## Key Features

- 📄 **Multimodal Extraction**: Processes PDFs to extract text, markdown-formatted tables, and high-resolution images.
- 🖼️ **Visual RAG**: The assistant retrieves and displays relevant images and tables from the documentation alongside text answers.
- 🚀 **Advanced Reranking**: Uses Cohere Rerank (with LLM-based fallback) to improve retrieval precision and answer quality.
- 📊 **RAGAS Evaluation (Optional)**: Built-in evaluation pipeline to measure Faithfulness, Answer Relevancy, Context Precision, and Context Recall. Note: Requires `ragas` and `datasets` which may need C++ Build Tools on Windows.
- 💻 **Product Recommendations**: Capable of recommending  products based on user needs and documentation context.

---

## Architecture

```
langchain-doc-chat/
├── backend/
│   ├── app/
│   │   ├── core.py                  ← Consolidated Settings, Schemas, and Model initialization
│   │   ├── services/
│   │   │   ├── document_loader.py   ← PDF extraction (text/tables/images)
│   │   │   ├── vector_store.py      ← ChromaDB multi-vector retriever
│   │   │   ├── tools.py             ← RAG chain (LCEL) + Cohere Reranker
│   │   │   └── agent.py             ← Orchestrator: ingest / query / RAGAS evaluate
│   │   ├── main.py                  ← FastAPI app & routes
│   │   └── __init__.py
│   └── requirements.txt
├── frontend/
│   ├── app.py                       ← Streamlit UI (Chat + Evaluation)
│   └── requirements.txt
├── workspace/                       ← Local storage for vector DB and extracted assets
└── README.md
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- AWS account with **Bedrock** access enabled for:
  - `anthropic.claude-3-5-sonnet`
  - `amazon.titan-embed-text-v2:0`

### 2. Configure Environment

Create a `.env` file in the `backend/` directory with your AWS credentials:

```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=ap-south-1

# Optional: Cohere Reranking (Recommended)
COHERE_API_KEY=your_cohere_api_key
COHERE_RERANK_MODEL=rerank-english-v3.0
```

### 3. Install Dependencies

```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd ../frontend
pip install -r requirements.txt
```

### 4. Run the Backend

```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### 5. Run the Streamlit Frontend

```bash
cd frontend
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## How it Works

1. **Ingestion**: Upload PDF manuals. The system extracts text chunks, parses tables into markdown, and saves images.
2. **Indexing**: Summaries and metadata are stored in ChromaDB using Titan Embeddings.
3. **Retrieval & Reranking**: When a user asks a question, the system retrieves the top $N$ candidates and then uses Cohere Rerank (or an LLM fallback) to rerank them for the best possible match.
4. **Generation**: Claude generates a comprehensive answer using the reranked context, including image references which the frontend renders as actual images.

cp backend/.env.example backend/.env

# Build and start both services
docker compose up --build -d

# Tail logs
docker compose logs -f
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:8501`

---

## API Reference

### `GET /health`
Liveness check. Returns service status, model IDs, and indexed document count.

### `POST /ingest`
Upload one or more brochure PDFs.

**Form data:**
| Field | Type | Description |
|---|---|---|
| `files` | `UploadFile[]` | PDF file(s) |
| `reset` | `bool` | Wipe existing vector store first (default: `false`) |

**Response:**
```json
{
  "status": "done",
  "pdfs_processed": 2,
  "texts": 2,
  "tables": 0,
  "images": 41,
  "summaries": 43,
  "indexed": 43,
  "message": "Successfully indexed 43 documents from 2 PDF(s)."
}
```

### `POST /query`
Ask a question. Returns a multimodal-grounded answer.

**Request body:**
```json
{
  "question": "Which Pro laptop supports 5G and is the lightest?",
  "k": 6,
  "include_sources": false
}
```

**Response:**
```json
{
  "question": "...",
  "answer": "...",
  "sources": [],
  "num_text_contexts": 4,
  "num_image_contexts": 2
}
```

### `POST /evaluate`
Run RAGAS evaluation. Omit the body to use the built-in 15-example set.

**Optional request body:** `list[EvalExample]`

**Response:**
```json
{
  "num_examples": 15,
  "scores": {
    "faithfulness": 0.933,
    "answer_relevancy": 0.868,
    "context_precision": 0.800,
    "context_recall": 0.956
  },
  "rows": [...],
  "errors": []
}
```

### `GET /evaluate/examples`
Returns the built-in 15-question evaluation dataset (10 factual Q&A + 5 recommendation scenarios).

---

## Running Tests

```bash
cd backend
pytest tests/ -v
```

Tests mock all AWS/Bedrock calls — no real API credentials required.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Pydantic Settings v2** | Type-safe env config, `.env` file support, IDE autocompletion |
| **`@lru_cache` on AWS clients** | One Bedrock client/LLM per process; avoids re-authentication overhead |
| **`asyncio.to_thread`** | CPU-heavy PDF extraction and Bedrock calls run in a thread pool, keeping FastAPI's event loop free |
| **Image resize before encoding** | Prevents `ValidationException` from Bedrock's 5 MB base64 image limit |
| **Summary-based retrieval** | ChromaDB stores *summaries* for semantic search; raw content (including full base64 images) is kept in `InMemoryStore` and swapped in at retrieval time |
| **LCEL RAG chain** | Composable, inspectable, supports streaming — identical logic to the notebook |
| **RAGAS `collections` import** | Uses the non-deprecated `ragas.metrics.collections` import path |
| **Docker multi-stage build** | Separate builder/runtime stages for a lean production image |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | *(required)* | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | *(required)* | AWS credentials |
| `AWS_DEFAULT_REGION` | `ap-south-1` | Bedrock region |
| `COHERE_API_KEY` | *(optional)* | Cohere API key for reranking |
| `COHERE_RERANK_MODEL` | `rerank-english-v3.0` | Cohere reranker model |
| `CLAUDE_MODEL_ID` | `global.anthropic.claude-sonnet-4-5-20250929-v1:0` | Bedrock model |
| `TITAN_EMBED_ID` | `amazon.titan-embed-text-v2:0` | Embedding model |
| `LLM_MAX_TOKENS` | `1024` | Max tokens for RAG answers |
| `RAGAS_MAX_TOKENS` | `2048` | Max tokens for RAGAS judge |
| `RETRIEVER_K` | `6` | Default retrieval top-k |
| `IMAGE_MAX_BYTES` | `5000000` | Max image size before resize (Bedrock limit) |
| `CHROMA_DIR` | `./workspace/chroma_db` | ChromaDB persistence path |
| `LOG_LEVEL` | `INFO` | Logging level |
| `CORS_ORIGINS` | `["http://localhost:8501"]` | Allowed CORS origins |

---

## RAGAS Scores (from notebook)

| Metric | Score |
|---|---|
| Faithfulness | **0.933** |
| Answer Relevancy | **0.868** |
| Context Precision | **0.800** |
| Context Recall | **0.956** |
