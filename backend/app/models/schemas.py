from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class EvalType(str, Enum):
    factual_qa = "factual_qa"
    recommendation = "recommendation"


class SourceType(str, Enum):
    text = "text"
    table = "table"
    image = "image"


class RagasScores(BaseModel):
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None


class EvalExample(BaseModel):
    question: str
    ground_truth: str
    source_type: SourceType
    eval_type: EvalType


class EvalResponse(BaseModel):
    num_examples: int
    scores: RagasScores
    rows: List[Dict[str, Any]]
    errors: List[str]


class IngestStatus(str, Enum):
    done = "done"
    failed = "failed"


class IngestResponse(BaseModel):
    status: IngestStatus
    pdfs_processed: int
    texts: int
    tables: int
    images: int
    summaries: int
    indexed: int
    message: str


class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 6
    include_sources: Optional[bool] = False


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    num_text_contexts: int
    num_image_contexts: int
    images: List[Dict[str, Any]]


class PDFUploadResponse(BaseModel):
    filename: str
    texts: int
    tables: int
    images: int
    total_chunks: int


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    sources: Optional[List[Dict[str, Any]]] = None
    images: Optional[List[Dict[str, Any]]] = None
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    session_id: str


class HealthResponse(BaseModel):
    status: str
    models_ready: bool
    vector_store_ready: bool
    indexed_docs: int


class EvaluationRequest(BaseModel):
    questions: List[str]
    ground_truth: List[str]


class EvaluationResponse(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    individual_scores: List[Dict[str, Any]]