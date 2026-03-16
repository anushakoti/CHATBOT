from typing import List, Dict, Any
import logging
import base64
from io import BytesIO

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from app.services.models import model_manager
from app.services.vector_store import vector_store_manager

logger = logging.getLogger(__name__)


class MultimodalRAG:
    """Handles RAG chain construction and query processing with image support"""
    
    def __init__(self):
        self.llm = model_manager.llm
        self.retriever = vector_store_manager.retriever
        self._last_image_references = []
        
        # Create prompt template with image awareness
        self.prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant for Dell product documentation.
    
        IMPORTANT: Answer ONLY using information explicitly present in the context below.
        If the context does not contain enough information to answer, say:
        "I don't have enough information in the provided documents to answer this."
        Do NOT use any prior knowledge beyond what is in the context.

        Context:
        {context}
        ...
        """),
            ("human", "{question}")
        ])
        
        # Build RAG chain - NOT async
        self.chain = (
            RunnablePassthrough.assign(context=self._get_context)
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _get_context(self, input_dict: dict) -> str:
        """Get formatted context for the chain"""
        question = input_dict.get("question", "")
        try:
            docs = self.retriever.invoke(question)
            logger.info(f"Retrieved {len(docs)} documents for query: {question}")
            
            if not docs:
                return "No relevant documents found."
            
            # Limit to top 3 documents to avoid token limits
            docs = docs[:10]
            
            formatted = []
            self._last_image_references = []  # Reset image references
            
            for i, doc in enumerate(docs):
                logger.info(f"Context: Doc {i}: metadata keys: {list(doc.metadata.keys())}, content preview: {doc.page_content[:100]}")
                source = f"[Source: {doc.metadata.get('source_pdf', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}]"
                
                # Handle different document types
                if doc.metadata.get("type") == "image":
                    # Get full image data if available
                    try:
                        if doc.metadata.get("full_content"):
                            # Store image reference for later display
                            image_ref = {
                                "index": i,
                                "content": doc.metadata["full_content"],
                                "source_pdf": doc.metadata.get("source_pdf", "Unknown"),
                                "page": doc.metadata.get("page", "Unknown"),
                                "width": doc.metadata.get("width", 0),
                                "height": doc.metadata.get("height", 0)
                            }
                            self._last_image_references.append(image_ref)
                            continue  
                            #formatted.append(f"[IMAGE: {doc.metadata.get('source_pdf', 'Unknown')}, Page {doc.metadata.get('page', 'Unknown')}]\n{source}")
                        else:
                            formatted.append(f"[Image reference: {doc.metadata.get('source_pdf', 'Unknown')} page {doc.metadata.get('page', 'Unknown')}]\n{source}")
                    except Exception as e:
                        logger.error(f"Error retrieving image: {e}")
                        formatted.append(f"[Image reference: {doc.metadata.get('source_pdf', 'Unknown')} page {doc.metadata.get('page', 'Unknown')}]\n{source}")
                else:
                    content = doc.page_content
                    formatted.append(f"{content}\n{source}")
            
            formatted = "\n\n".join(formatted)
            
            # Limit context length to avoid token limits
            max_context_length = 20000
            if len(formatted) > max_context_length:
                formatted = formatted[:max_context_length] + "..."
            
            return formatted
        except Exception as e:
            logger.error(f"Error formatting docs: {e}")
            return "Error retrieving context. Please try again."
            return "Error retrieving context."
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a query and return answer with sources and images"""
        try:
            # Reset image references
            self._last_image_references = []
            
            # Get relevant docs
            docs = self.retriever.invoke(question)
            logger.info(f"Query: Retrieved {len(docs)} documents for query: {question}")
            
            # Prepare sources and collect images
            sources = []
            images = []
            
            for i, doc in enumerate(docs):
                logger.info(f"Query: Processing doc {i}, metadata: {doc.metadata}")
                
                source_info = {
                    "source_pdf": doc.metadata.get("source_pdf", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "type": doc.metadata.get("type", "text"),
                }
                
                # Add full content reference if it's an image
                if doc.metadata.get("type") == "image" and doc.metadata.get("full_content"):
                    logger.info(f"Query: Found image document: {source_info}")
                    try:
                        image_data = {
                            "content": doc.metadata["full_content"],
                            "source_pdf": doc.metadata.get("source_pdf", "Unknown"),
                            "page": doc.metadata.get("page", "Unknown"),
                            "width": doc.metadata.get("width", 0),
                            "height": doc.metadata.get("height", 0)
                        }
                        images.append(image_data)
                        source_info["has_image"] = True
                        logger.info(f"Query: Added image to results, total images: {len(images)}")
                    except Exception as e:
                        logger.error(f"Error retrieving full image: {e}")
                
                # Avoid duplicates
                if source_info not in sources:
                    sources.append(source_info)
            
            # Generate answer
            try:
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


# Global instance
rag_tool = MultimodalRAG()