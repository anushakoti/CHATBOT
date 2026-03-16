"""Compatibility layer for RAGAS with AWS Bedrock"""

import asyncio
from typing import Any, List, Optional
import logging
import json
import re

from langchain_aws import ChatBedrock
from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from pydantic import Field, ConfigDict

logger = logging.getLogger(__name__)


class AsyncCompatibleChatBedrock(BaseLLM):
    """
    A wrapper that makes ChatBedrock compatible with async operations.
    """
    
    # Use Any type to avoid Pydantic validation issues with ChatBedrock
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    chat_bedrock: Any = Field(description="The underlying ChatBedrock instance")
    
    def __init__(self, chat_bedrock: ChatBedrock):
        # Call parent init with the field value
        super().__init__(chat_bedrock=chat_bedrock)
    
    def _clean_ragas_response(self, content: str) -> str:
        """Clean and repair LLM output so RAGAS Pydantic models validate cleanly."""
        
        # Strip markdown code fences
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Attempt to parse and patch missing verdict fields
        try:
            data = json.loads(content)
            
            # Handle {"statements": [...]} structure RAGAS uses for faithfulness
            if isinstance(data, dict) and "statements" in data:
                for item in data["statements"]:
                    if isinstance(item, dict) and "verdict" not in item:
                        item["verdict"] = 0  # Default: not supported
                    # Also ensure reason exists
                    if isinstance(item, dict) and "reason" not in item:
                        item["reason"] = "No reason provided"
            
            # Handle flat list of statements
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "verdict" not in item:
                        item["verdict"] = 0
                    if isinstance(item, dict) and "reason" not in item:
                        item["reason"] = "No reason provided"
            
            return json.dumps(data)
        
        except (json.JSONDecodeError, TypeError):
            # If JSON is broken, return a safe fallback structure
            logger.warning("Could not parse LLM JSON response, returning safe fallback")
            return json.dumps({"statements": []})


    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            response = self.chat_bedrock.invoke(prompt, stop=stop, **kwargs)
            content = response.content if hasattr(response, 'content') else str(response)
            content = self._clean_ragas_response(content)  # <-- clean here
            generations.append([Generation(text=content)])
        return LLMResult(generations=generations)


    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                None,
                lambda p=prompt: self.chat_bedrock.invoke(p, stop=stop, **kwargs)
            )
            for prompt in prompts
        ]
        responses = await asyncio.gather(*tasks)
        generations = []
        for resp in responses:
            content = resp.content if hasattr(resp, 'content') else str(resp)
            content = self._clean_ragas_response(content)  # <-- clean here
            generations.append([Generation(text=content)])
        return LLMResult(generations=generations)
    
    @property
    def _llm_type(self) -> str:
        return "async_compatible_chat_bedrock"