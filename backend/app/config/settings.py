"""
Application settings — all configuration via environment variables.
Compatible with Python 3.12+ (uses pydantic-settings v2).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── AWS / Bedrock ────────────────────────────────────────────────────────
    aws_access_key_id: str = Field(..., alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(..., alias="AWS_SECRET_ACCESS_KEY")
    aws_default_region: str = Field("ap-south-1", alias="AWS_DEFAULT_REGION")

    # ── OpenAI ───────────────────────────────────────────────────────────────
    #openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")

    # ── Model IDs ────────────────────────────────────────────────────────────
    claude_model_id: str = Field(
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        alias="CLAUDE_MODEL_ID",
    )
    titan_embed_id: str = Field(
        "amazon.titan-embed-text-v2:0",
        alias="TITAN_EMBED_ID",
    )

    # ── LLM knobs ────────────────────────────────────────────────────────────
    llm_temperature: float = Field(0.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(1000, alias="LLM_MAX_TOKENS")
    ragas_max_tokens: int = Field(4096, alias="RAGAS_MAX_TOKENS")
    llm_max_concurrency: int = Field(2, alias="LLM_MAX_CONCURRENCY")

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(10000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(0, alias="CHUNK_OVERLAP")
    unstructured_max_chars: int = Field(4000, alias="UNSTRUCTURED_MAX_CHARS")
    unstructured_new_after_n_chars: int = Field(3800, alias="UNSTRUCTURED_NEW_AFTER_N_CHARS")
    unstructured_combine_under_n_chars: int = Field(2000, alias="UNSTRUCTURED_COMBINE_UNDER_N_CHARS")

    # ── Retrieval ────────────────────────────────────────────────────────────
    retriever_k: int = Field(6, alias="RETRIEVER_K")
    image_min_dimension: int = Field(100, alias="IMAGE_MIN_DIMENSION")
    image_max_bytes: int = Field(5_000_000, alias="IMAGE_MAX_BYTES")  # Bedrock 5 MB limit

    # ── Paths ────────────────────────────────────────────────────────────────
    work_dir: Path = Field(Path("./workspace"), alias="WORK_DIR")
    upload_dir: Path = Field(Path("./workspace/uploads"), alias="UPLOAD_DIR")
    chroma_dir: Path = Field(Path("./workspace/chroma_db"), alias="CHROMA_DIR")
    img_dir: Path = Field(Path("./workspace/extracted_images"), alias="IMG_DIR")
    docstore_dir: Path = Field(Path("./workspace/docstore"), alias="DOCSTORE_DIR")
    chroma_collection: str = Field("dell_multimodal_rag", alias="CHROMA_COLLECTION")

    # ── API ──────────────────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    api_reload: bool = Field(False, alias="API_RELOAD")
    cors_origins: list[str] = Field(
        ["http://localhost:8501", "http://127.0.0.1:8501"],
        alias="CORS_ORIGINS",
    )
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # ── RAGAS eval examples ──────────────────────────────────────────────────
    eval_rate_limit_sleep: float = Field(2.0, alias="EVAL_RATE_LIMIT_SLEEP")

    @field_validator("work_dir", "upload_dir", "chroma_dir", "img_dir", mode="after")
    @classmethod
    def _ensure_dirs(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
