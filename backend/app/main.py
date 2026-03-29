from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import List
from contextlib import asynccontextmanager

from app.core import (
    HealthResponse, 
    EvaluationRequest, EvaluationResponse,
    IngestResponse, IngestStatus, QueryResponse, QueryRequest,
    model_manager
)
from app.services.agent import orchestrator
from app.services.vector_store import indexed_count

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("🚀 Starting Dell Chatbot API")
    try:
        test = model_manager.llm.invoke("Reply with OK only.")
        logger.info(f"LLM check: {test.content.strip()}")
        dim = len(model_manager.embeddings.embed_query("hello"))
        logger.info(f"Embedding dimension: {dim}")
        logger.info("✅ All models ready")
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
    yield
    logger.info("🛑 Shutting down Dell Chatbot API")


app = FastAPI(
    title="Dell Chatbot API",
    description="Multimodal RAG Chatbot for Dell Documentation",
    version="1.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        models_ready = model_manager.llm is not None
        vector_ready = orchestrator.document_loader is not None
        indexed = indexed_count()
        return HealthResponse(
            status="healthy" if models_ready and vector_ready else "degraded",
            models_ready=models_ready,
            vector_store_ready=vector_ready,
            indexed_docs=indexed
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            models_ready=False,
            vector_store_ready=False,
            indexed_docs=0
        )


@app.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest_pdfs(files: List[UploadFile] = File(...)):
    """Upload and process multiple PDF files"""
    total_texts, total_tables, total_images, total_summaries = 0, 0, 0, 0
    
    for file in files:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
        
        try:
            content = await file.read()
            result = await orchestrator.ingest_pdf(content, file.filename)
            total_texts += result.get('texts', 0)
            total_tables += result.get('tables', 0)
            total_images += result.get('images', 0)
            total_summaries += result.get('total_chunks', 0)
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process {file.filename}: {str(e)}")
    
    return IngestResponse(
        status=IngestStatus.done,
        pdfs_processed=len(files),
        texts=total_texts,
        tables=total_tables,
        images=total_images,
        summaries=total_summaries,
        indexed=indexed_count(),
        message="PDFs ingested successfully"
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Multimodal RAG query"""
    if indexed_count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed.")
    try:
        result = await orchestrator.query(message=req.question)
        num_text_contexts = sum(1 for s in result["sources"] if s.get("type") == "text")
        return QueryResponse(
            question=req.question,
            answer=result["answer"],
            sources=result["sources"] if req.include_sources else [],
            num_text_contexts=num_text_contexts,
            num_image_contexts=len(result["images"]),
            images=result["images"]
        )
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    """Evaluate RAG performance using RAGAS"""
    if len(request.questions) != len(request.ground_truth):
        raise HTTPException(status_code=400, detail="Mismatch between questions and ground truth")
    try:
        result = await orchestrator.evaluate(request.questions, request.ground_truth)
        return EvaluationResponse(**result)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
async def clear_vector_store():
    """Clear all indexed documents"""
    try:
        orchestrator.clear_vector_store()
        return {"message": "Vector store cleared successfully"}
    except Exception as e:
        logger.error(f"Clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)