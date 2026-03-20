import asyncio
from typing import List, Dict, Any
import logging
from datetime import datetime
import uuid

from app.services.document_loader import DocumentLoader
from app.services.vector_store import vector_store_manager
from app.services.tools import rag_tool
from app.services.models import model_manager

logger = logging.getLogger(__name__)


class CleanJsonLLM:
    """Wrapper to clean JSON responses from Claude that are wrapped in ```json``` blocks.
    Designed to work properly with RAGAS."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def invoke(self, *args, **kwargs):
        response = self.llm.invoke(*args, **kwargs)
        if response is None:
            raise ValueError("LLM returned None response")
        

        if isinstance(response, str):
            content = response
        elif hasattr(response, 'content'):
            content = response.content
            if content is None:
                raise ValueError("LLM response content is None")
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")
        
       
        if content.startswith('```json') and content.endswith('```'):
            content = content[7:-3].strip()
        elif content.startswith('```') and content.endswith('```'):
            content = content[3:-3].strip()
        
       
        return content
    
    async def ainvoke(self, *args, **kwargs):
        """Async invoke - required for RAGAS"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, *args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Make the class callable"""
        return self.invoke(*args, **kwargs)
    
    
    def __getattr__(self, name):
        return getattr(self.llm, name)

class Orchestrator:
    """Orchestrates the entire RAG pipeline"""
    
    def __init__(self):
        self.document_loader = DocumentLoader()
        self.sessions = {} 
    
    async def ingest_pdf(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Ingest a PDF file: extract, summarize, and index
        """
        try:
        
            extracted = await self.document_loader.extract_pdf(file_content, filename)
            
            texts = extracted["texts"]
            tables = extracted["tables"]
            images = extracted["images"]
            
        
            await vector_store_manager.add_documents(texts, tables, images)
            
            return {
                "filename": filename,
                "texts": len(texts),
                "tables": len(tables),
                "images": len(images),
                "total_chunks": len(texts) + len(tables) + len(images)
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest PDF {filename}: {e}")
            raise
    
    async def query(self, message: str, session_id: str = None) -> Dict[str, Any]:
        """
        Process a user query
        """
        if not session_id:
            session_id = str(uuid.uuid4())
        
   
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "id": session_id,
                "created_at": datetime.now(),
                "history": []
            }
        
        try:

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, rag_tool.query, message)
            
           
            self.sessions[session_id]["history"].append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now()
            })
            self.sessions[session_id]["history"].append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "images": result["images"],
                "timestamp": datetime.now()
            })
            
            return {
                "answer": result["answer"],
                "sources": result["sources"],
                "images": result["images"],
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "answer": "I encountered an error while processing your question. Please try again.",
                "sources": [],
                "images": [],
                "session_id": session_id
            }
    
    async def evaluate(self, questions: List[str], ground_truth: List[str]) -> Dict[str, Any]:
        """
        Evaluate RAG performance using RAGAS with AWS Bedrock
        """
        try:
           
            import os
            os.environ["OPENAI_API_KEY"] = "dummy"
            
     
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            )
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from datasets import Dataset
            import pandas as pd
            
           
            bedrock_llm = model_manager.ragas_llm  
            bedrock_embeddings = model_manager.embeddings
            
     
            ragas_llm = LangchainLLMWrapper(bedrock_llm)
            ragas_embeddings = LangchainEmbeddingsWrapper(bedrock_embeddings)
            
     
            faithfulness_metric = faithfulness
            answer_relevancy_metric = answer_relevancy
            context_precision_metric = context_precision
            context_recall_metric = context_recall
            
          
            faithfulness_metric.llm = ragas_llm
            answer_relevancy_metric.llm = ragas_llm
            answer_relevancy_metric.embeddings = ragas_embeddings
            context_precision_metric.llm = ragas_llm
            context_recall_metric.llm = ragas_llm
            
            logger.info("✅RAGAS metrics configured with AWS Bedrock")
            
       
            data = {
                "question": questions,
                "ground_truth": ground_truth,
                "answer": [],
                "contexts": []
            }
            
           
            loop = asyncio.get_event_loop()
            for i, q in enumerate(questions):
                logger.info(f"Processing question {i+1}/{len(questions)}: {q[:50]}...")
                
          
                result = await loop.run_in_executor(None, rag_tool.query, q)
                data["answer"].append(result["answer"])
                
               
                docs = vector_store_manager.retriever.invoke(q)
                contexts = [doc.page_content for doc in docs]
                data["contexts"].append(contexts)
            
      
            logger.info("Creating evaluation dataset...")
            dataset = Dataset.from_dict(data)
            
          
            logger.info("Running RAGAS evaluation...")
            result = evaluate(
                dataset,
                metrics=[
                    faithfulness_metric,
                    answer_relevancy_metric,
                    context_precision_metric,
                    context_recall_metric
                ]
            )
            
         
            df = result.to_pandas()
            
         
            faithfulness_score = df["faithfulness"].mean() if "faithfulness" in df.columns else 0
            answer_relevancy_score = df["answer_relevancy"].mean() if "answer_relevancy" in df.columns else 0
            context_precision_score = df["context_precision"].mean() if "context_precision" in df.columns else 0
            context_recall_score = df["context_recall"].mean() if "context_recall" in df.columns else 0
            
        
            faithfulness_score = 0 if pd.isna(faithfulness_score) else float(faithfulness_score)
            answer_relevancy_score = 0 if pd.isna(answer_relevancy_score) else float(answer_relevancy_score)
            context_precision_score = 0 if pd.isna(context_precision_score) else float(context_precision_score)
            context_recall_score = 0 if pd.isna(context_recall_score) else float(context_recall_score)
            
            scores = df.to_dict('records')
            
            logger.info(f"Evaluation complete - Faithfulness: {faithfulness_score:.3f}")
            
            return {
                "faithfulness": faithfulness_score,
                "answer_relevancy": answer_relevancy_score,
                "context_precision": context_precision_score,
                "context_recall": context_recall_score,
                "individual_scores": scores
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            return {
            "error": str(e),
            "faithfulness": 0,
            "answer_relevancy": 0,
            "context_precision": 0,
            "context_recall": 0,
            "individual_scores": []
        }
    def clear_vector_store(self):
        """Clear all indexed documents"""
        vector_store_manager.clear()
        logger.info("Vector store cleared")



orchestrator = Orchestrator()