from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum

import boto3
from botocore.config import Config
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_aws import BedrockEmbeddings, ChatBedrock

# --- SETTINGS ---
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # AWS / Bedrock
    aws_access_key_id: str = Field(..., alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(..., alias="AWS_SECRET_ACCESS_KEY")
    aws_default_region: str = Field("ap-south-1", alias="AWS_DEFAULT_REGION")

    # Cohere
    cohere_api_key: Optional[str] = Field(None, alias="COHERE_API_KEY")
    cohere_rerank_model: str = Field("rerank-english-v3.0", alias="COHERE_RERANK_MODEL")

    # Model IDs
    claude_model_id: str = Field("anthropic.claude-3-5-sonnet-20240620-v1:0", alias="CLAUDE_MODEL_ID")
    titan_embed_id: str = Field("amazon.titan-embed-text-v2:0", alias="TITAN_EMBED_ID")

    # LLM knobs
    llm_temperature: float = Field(0.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(1000, alias="LLM_MAX_TOKENS")
    llm_max_concurrency: int = Field(2, alias="LLM_MAX_CONCURRENCY")

    # Retrieval
    retriever_k: int = Field(6, alias="RETRIEVER_K")
    rerank_k: int = Field(3, alias="RERANK_K")

    # Paths
    work_dir: Path = Field(Path("./workspace"), alias="WORK_DIR")
    chroma_dir: Path = Field(Path("./workspace/chroma_db"), alias="CHROMA_DIR")
    img_dir: Path = Field(Path("./workspace/extracted_images"), alias="IMG_DIR")
    chroma_collection: str = Field("dell_multimodal_rag", alias="CHROMA_COLLECTION")

    @field_validator("work_dir", "chroma_dir", "img_dir", mode="after")
    @classmethod
    def _ensure_dirs(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

# --- SCHEMAS ---
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
    method: Optional[str] = None

# --- MODELS ---
@lru_cache(maxsize=1)
def get_bedrock_client():
    settings = get_settings()
    config = Config(
        region_name=settings.aws_default_region,
        retries={'max_attempts': 3, 'mode': 'adaptive'},
        max_pool_connections=50 
    )
    return boto3.client(
        service_name="bedrock-runtime",
        config=config,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )

@lru_cache(maxsize=1)
def get_llm() -> ChatBedrock:
    settings = get_settings()
    client = get_bedrock_client()
    return ChatBedrock(
        client=client,
        model_id=settings.claude_model_id,
        model_kwargs={
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        },
    )

@lru_cache(maxsize=1)
def get_embeddings() -> BedrockEmbeddings:
    settings = get_settings()
    client = get_bedrock_client()
    return BedrockEmbeddings(
        client=client,
        model_id=settings.titan_embed_id,
    )

@lru_cache(maxsize=1)
def get_cohere_reranker():
    from langchain_cohere import CohereRerank
    settings = get_settings()
    if not settings.cohere_api_key:
        return None
    return CohereRerank(
        cohere_api_key=settings.cohere_api_key,
        model=settings.cohere_rerank_model,
        top_n=settings.rerank_k
    )

class ModelManager:
    def __init__(self):
        self._llm = None
        self._embeddings = None
        self._cohere_reranker = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    @property
    def cohere_reranker(self):
        if self._cohere_reranker is None:
            self._cohere_reranker = get_cohere_reranker()
        return self._cohere_reranker

model_manager = ModelManager()
