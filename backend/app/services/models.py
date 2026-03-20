
from __future__ import annotations

import logging
from functools import lru_cache

import boto3
from botocore.config import Config
from langchain_aws import BedrockEmbeddings, ChatBedrock

from app.config.settings import get_settings
from app.services.ragas_compat import AsyncCompatibleChatBedrock  # Only import what you need

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_bedrock_client():
    settings = get_settings()
    logger.info(
        "Creating Bedrock runtime client — region=%s", settings.aws_default_region
    )
    

    config = Config(
        region_name=settings.aws_default_region,
        retries={
            'max_attempts': 3,
            'mode': 'adaptive'
        },
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
    logger.info("Building LLM — model=%s", settings.claude_model_id)
    return ChatBedrock(
        client=client,
        model_id=settings.claude_model_id,
        model_kwargs={
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        },
    )


@lru_cache(maxsize=1)
def get_ragas_llm():
    """Get an async-compatible LLM for RAGAS."""
    settings = get_settings()
    client = get_bedrock_client()
    
    ragas_max_tokens = getattr(settings, 'ragas_max_tokens', 8192)
    
    logger.info("Building RAGAS LLM — model=%s, max_tokens=%d", 
                settings.claude_model_id, ragas_max_tokens)
    

    base_llm = ChatBedrock(
    client=client,
    model_id=settings.claude_model_id,
    model_kwargs={
        "temperature": 0,
        "max_tokens": 8192,
        "system": (
            "You are an evaluation assistant. Always respond with valid JSON only. "
            "Never omit required fields. For faithfulness verdicts, every statement "
            "object MUST include both 'verdict' (0 or 1) and 'reason' (string). "
            "Never wrap JSON in markdown code fences."
        ),
    },
)
    
  
    return AsyncCompatibleChatBedrock(base_llm)


@lru_cache(maxsize=1)
def get_embeddings() -> BedrockEmbeddings:
    settings = get_settings()
    client = get_bedrock_client()
    logger.info("Building embeddings — model=%s", settings.titan_embed_id)
    return BedrockEmbeddings(
        client=client,
        model_id=settings.titan_embed_id,
    )



class ModelManager:
    """Manager for Bedrock models"""
    
    @property
    def llm(self):
        return get_llm()
    
    @property
    def ragas_llm(self):
        return get_ragas_llm()
    
    @property
    def embeddings(self):
        return get_embeddings()



model_manager = ModelManager()