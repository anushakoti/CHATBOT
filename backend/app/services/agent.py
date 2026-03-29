import asyncio
from typing import List, Dict, Any
import logging

from app.services.document_loader import DocumentLoader
from app.services.vector_store import vector_store_manager
from app.services.tools import rag_tool
from app.core import model_manager

logger = logging.getLogger(__name__)


class CleanJsonLLM:
    """Wrapper to clean JSON responses from LLM for RAGAS evaluation."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def invoke(self, *args, **kwargs):
        response = self.llm.invoke(*args, **kwargs)
        if response is None:
            raise ValueError("LLM returned None response")
        
        content = response if isinstance(response, str) else getattr(response, 'content', str(response))
        if content is None:
            raise ValueError("LLM response content is None")
        
        if content.startswith('```json') and content.endswith('```'):
            content = content[7:-3].strip()
        elif content.startswith('```') and content.endswith('```'):
            content = content[3:-3].strip()
        
        return content
    
    async def ainvoke(self, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, *args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.llm, name)


class LLMEvaluator:
    """Lite evaluation using LLM as a judge when RAGAS is unavailable."""
    
    def __init__(self, llm):
        self.llm = llm
        from langchain_core.prompts import ChatPromptTemplate
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a RAG evaluation judge. Evaluate the following response based on:
- Faithfulness (0.0 to 1.0): Factuality relative to context.
- Answer Relevancy (0.0 to 1.0): How well it addresses the query.
- Context Precision (0.0 to 1.0): Quality of retrieved chunks.
- Context Recall (0.0 to 1.0): Coverage of the Ground Truth.

Return a JSON object only: {{"faithfulness": 0.9, "answer_relevancy": 0.8, "context_precision": 0.7, "context_recall": 0.9, "reasoning": "brief explanation"}}"""),
            ("human", "Question: {question}\n\nAnswer: {answer}\n\nContext: {context}\n\nGround Truth: {ground_truth}")
        ])

    async def evaluate(self, questions, ground_truths, answers, contexts):
        import json
        scores = []
        for q, gt, ans, ctx in zip(questions, ground_truths, answers, contexts):
            try:
                formatted_ctx = "\n".join(ctx) if isinstance(ctx, list) else str(ctx)
                res = await self.llm.ainvoke(self.prompt.format(
                    question=q, ground_truth=gt, answer=ans, context=formatted_ctx
                ))
                content = res.content if hasattr(res, 'content') else str(res)
                # Parse JSON
                start = content.find('{')
                end = content.rfind('}') + 1
                data = json.loads(content[start:end])
                data.update({"question": q, "ground_truth": gt, "answer": ans})
                scores.append(data)
            except Exception as e:
                logger.error(f"LLM Eval row failed: {e}")
                scores.append({"faithfulness": 0, "answer_relevancy": 0, "context_precision": 0, "context_recall": 0, "question": q, "error": str(e)})
        
        return scores

class Orchestrator:
    """Orchestrates the entire RAG pipeline"""
    
    def __init__(self):
        self.document_loader = DocumentLoader()
    
    async def ingest_pdf(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Ingest a PDF file: extract and index
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
    
    async def query(self, message: str) -> Dict[str, Any]:
        """
        Process a user query
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, rag_tool.query, message)
            
            return {
                "answer": result["answer"],
                "sources": result["sources"],
                "images": result["images"]
            }
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "answer": "I encountered an error while processing your question. Please try again.",
                "sources": [],
                "images": []
            }
    
    async def evaluate(self, questions: List[str], ground_truth: List[str]) -> Dict[str, Any]:
        """
        Evaluate RAG performance using RAGAS or LLM-based fallback.
        """
        try:
            import pandas as pd
            # 1. Prepare data
            data = {"question": questions, "ground_truth": ground_truth, "answer": [], "contexts": []}
            loop = asyncio.get_event_loop()
            for q in questions:
                result = await loop.run_in_executor(None, rag_tool.query, q)
                data["answer"].append(result["answer"])
                
                docs = vector_store_manager.retriever.invoke(q)
                data["contexts"].append([doc.page_content for doc in docs])

            # 2. Try RAGAS
            try:
                from ragas import evaluate
                from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
                from ragas.llms import LangchainLLMWrapper
                from ragas.embeddings import LangchainEmbeddingsWrapper
                from datasets import Dataset
                
                logger.info("Running RAGAS evaluation...")
                clean_llm = CleanJsonLLM(model_manager.llm)
                ragas_llm = LangchainLLMWrapper(clean_llm)
                ragas_embeddings = LangchainEmbeddingsWrapper(model_manager.embeddings)
                
                dataset = Dataset.from_dict(data)
                
                for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
                    metric.llm = ragas_llm
                    if hasattr(metric, 'embeddings'):
                        metric.embeddings = ragas_embeddings
                
                result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
                df = result.to_pandas()
                
                return {
                    "faithfulness": float(df["faithfulness"].mean() or 0),
                    "answer_relevancy": float(df["answer_relevancy"].mean() or 0),
                    "context_precision": float(df["context_precision"].mean() or 0),
                    "context_recall": float(df["context_recall"].mean() or 0),
                    "individual_scores": df.to_dict('records'),
                    "method": "RAGAS"
                }

            except ImportError:
                # 3. Use Lite LLM Evaluator (Silent Fallback)
                logger.info("RAGAS not found, using Lite LLM-based evaluation.")
                evaluator = LLMEvaluator(model_manager.llm)
                scores = await evaluator.evaluate(
                    questions=data["question"],
                    ground_truths=data["ground_truth"],
                    answers=data["answer"],
                    contexts=data["contexts"]
                )
                
                df = pd.DataFrame(scores)
                
                return {
                    "faithfulness": float(df["faithfulness"].mean() or 0),
                    "answer_relevancy": float(df["answer_relevancy"].mean() or 0),
                    "context_precision": float(df["context_precision"].mean() or 0),
                    "context_recall": float(df["context_recall"].mean() or 0),
                    "individual_scores": df.to_dict('records'),
                    "method": "LLM-Judge"
                }
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            return {"error": str(e), "faithfulness": 0, "answer_relevancy": 0, "context_precision": 0, "context_recall": 0, "individual_scores": []}

    def clear_vector_store(self):
        """Clear all indexed documents"""
        vector_store_manager.clear()
        logger.info("Vector store cleared")


orchestrator = Orchestrator()