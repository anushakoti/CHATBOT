from typing import List, Dict, Any
import logging

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.core import model_manager, get_settings
from app.services.vector_store import vector_store_manager

logger = logging.getLogger(__name__)


class MultimodalRAG:
    """Handles RAG chain construction and query processing with image and reranking support"""
    
    def __init__(self):
        self.llm = model_manager.llm
        self.retriever = vector_store_manager.retriever
        self.settings = get_settings()
        

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant for Dell product documentation and recommendations.
    
        - Answer questions using information explicitly present in the context.
        - If the context does not contain enough information to answer a factual question, say you don't have enough information.
        - If the user asks for product recommendations, use the information in the context first. If no specific context is available for a recommendation, you may provide general helpful advice based on Dell's typical product lines (like XPS for high performance, Inspiron for general use, G-Series/Alienware for gaming) while mentioning that they should check the official Dell website for the latest models.
        - When providing recommendations, be polite and helpful.
        - If tables or images are relevant, mention them in your response.

        Context:
        {context}
        """),
            ("human", "{question}")
        ])
        
   
        self.chain = (
            RunnablePassthrough.assign(context=self._get_context)
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _rerank_documents(self, question: str, docs: List[Document], k: int) -> List[Document]:
        """Cohere Rerank or LLM-based reranking to improve precision"""
        if not docs or len(docs) <= k:
            return docs[:k]
        
        logger.info(f"Reranking {len(docs)} documents down to {k}...")
        
        # 1. Try Cohere Rerank (Recommended)
        cohere_reranker = model_manager.cohere_reranker
        if cohere_reranker:
            try:
                # Use langchain-cohere's compressor
                # It accepts List[Document] and returns List[Document]
                reranked = cohere_reranker.compress_documents(docs, question)
                logger.info("Successfully reranked using Cohere.")
                return reranked[:k]
            except Exception as e:
                logger.error(f"Cohere Rerank failed: {e}. Falling back to LLM-based rerank.")

        # 2. Fallback to LLM-based reranking
        # This is a simplified reranker that uses the LLM to select the best documents
        rerank_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a reranking expert. Given a question and a list of documents, return the indices of the {k} most relevant documents as a comma-separated list of numbers (e.g., '0, 2, 5'). If fewer than {k} are relevant, return all relevant indices."),
            ("human", "Question: {question}\n\nDocuments:\n{docs}")
        ])
        
        docs_text = "\n".join([f"[{i}] {doc.page_content[:500]}" for i, doc in enumerate(docs)])
        
        try:
            res = self.llm.invoke(rerank_prompt.format(question=question, docs=docs_text, k=k))
            indices = [int(idx.strip()) for idx in res.content.split(",") if idx.strip().isdigit()]
            reranked = [docs[i] for i in indices if i < len(docs)]
            return reranked[:k] if reranked else docs[:k]
        except Exception as e:
            logger.error(f"LLM Reranking failed: {e}")
            return docs[:k]

    def _get_context(self, input_dict: dict) -> str:
        """Get formatted context for the chain with reranking"""
        question = input_dict.get("question", "")
        try:
            # Step 1: Initial Retrieval (Fetch documents)
            docs = self.retriever.invoke(question)
            
            # Step 2: Reranking
            reranked_docs = self._rerank_documents(question, docs, self.settings.rerank_k)
            
            if not reranked_docs:
                return "No relevant documents found."
            
            formatted = []
            for doc in reranked_docs:
                # If it's an image, the page_content already contains the summary description
                content = doc.page_content
                formatted.append(f"{content}")
            
            formatted = "\n\n".join(formatted)
            max_context_length = 20000
            if len(formatted) > max_context_length:
                formatted = formatted[:max_context_length] + "..."
            
            return formatted
        except Exception as e:
            logger.error(f"Error formatting docs: {e}")
            return "Error retrieving context. Please try again."
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a query and return answer with sources and images"""
        try:
            docs = self.retriever.invoke(question)
            reranked_docs = self._rerank_documents(question, docs, self.settings.rerank_k)
            
            sources = []
            images = []
            
            for doc in reranked_docs:
                is_image = doc.metadata.get("type") == "image"
                source_info = {
                    "source_pdf": doc.metadata.get("source_pdf", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "type": doc.metadata.get("type", "text"),
                    "has_image": is_image
                }
                
                if source_info not in sources:
                    sources.append(source_info)
                
                if is_image:
                    images.append({
                        "content": doc.metadata.get("image_data"), # Use metadata field instead
                        "source_pdf": doc.metadata.get("source_pdf", "Unknown"),
                        "page": doc.metadata.get("page", 0),
                        "width": doc.metadata.get("width"),
                        "height": doc.metadata.get("height"),
                    })
            
            try:
                # To provide truly multimodal context, we can send images to the chain if needed
                # For now, we'll just send the text/table context as formatted string
                answer = self.chain.invoke({"question": question})
            except Exception as e:
                logger.error(f"Chain invocation error: {e}")
                answer = "I encountered an error while processing your question. Please try again."
            
            return {
                "answer": answer,
                "sources": sources,
                "images": images
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "answer": "I encountered an error while processing your question. Please try again.",
                "sources": [],
                "images": []
            }
    
    def format_for_streamlit(self, question: str) -> tuple:
        """Format answer for Streamlit display with source citations"""
        result = self.query(question)
        formatted = result["answer"]
        
        if result["sources"]:
            formatted += "\n\n---\n**Sources:**\n"
            for source in result["sources"]:
                source_text = f"- {source['source_pdf']} (Page {source['page']}) - {source['type']}"
                if source.get('has_image'):
                    source_text += " 📷"
                formatted += source_text + "\n"
        
        return formatted, result["images"]



rag_tool = MultimodalRAG()