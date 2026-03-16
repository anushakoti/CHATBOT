"""
ChromaDB vector store + InMemoryStore docstore, combined into a
custom MultiVectorRetriever.

The retriever embeds *summaries* (text/table/image) and returns the
corresponding *raw* content (text chunk, table string, or base64 image).
"""
from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.stores import InMemoryStore
from pydantic import ConfigDict

from app.config.settings import get_settings
from app.services.models import get_embeddings

logger = logging.getLogger(__name__)


# ── Multi-vector retriever ────────────────────────────────────────────────────

class MultiVectorRetriever(BaseRetriever):
    """Retrieve by summary similarity → return raw documents from docstore."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: Chroma
    docstore: InMemoryStore
    id_key: str = "doc_id"
    search_kwargs: dict = {}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> list[Document]:
        hits = self.vectorstore.similarity_search(query, **self.search_kwargs)
        doc_ids = [h.metadata[self.id_key] for h in hits if self.id_key in h.metadata]
        raw_docs = self.docstore.mget(doc_ids)
        docs = [d for d in raw_docs if d is not None]
        if not docs:
            # Fallback to summary docs if raw docs not found
            docs = hits
        return docs


# ── Singleton state (in-process) ─────────────────────────────────────────────
# We keep a module-level reference so the retriever survives across requests.

_vector_store: Chroma | None = None
_docstore: InMemoryStore | None = None
_retriever: MultiVectorRetriever | None = None


def _get_or_create_vector_store() -> Chroma:
    """Return the global Chroma vector store, initializing or repairing it as needed."""

    global _vector_store
    if _vector_store is None:
        settings = get_settings()
        _vector_store = Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=get_embeddings(),
            persist_directory=str(settings.chroma_dir),
        )

    # Chroma can sometimes end up with an uninitialized collection (e.g. if the
    # underlying directory changed or the previous process aborted). In that case,
    # accessing `_collection` raises ValueError and suggests calling `reset_collection()`.
    try:
        _ = _vector_store._collection
    except ValueError:
        logger.warning(
            "Chroma collection not initialized; resetting collection for '%s'.",
            _vector_store._collection_name,
        )
        _vector_store.reset_collection()

    return _vector_store


def _get_or_create_docstore() -> InMemoryStore:
    global _docstore
    if _docstore is None:
        _docstore = InMemoryStore()
    return _docstore


def get_retriever() -> MultiVectorRetriever:
    global _retriever
    if _retriever is None:
        settings = get_settings()
        _retriever = MultiVectorRetriever(
            vectorstore=_get_or_create_vector_store(),
            docstore=_get_or_create_docstore(),
            id_key="doc_id",
            search_kwargs={"k": settings.retriever_k},
        )
    return _retriever


def get_retriever_with_k(k: int) -> MultiVectorRetriever:
    """Get a retriever configured with a specific k value."""
    return MultiVectorRetriever(
        vectorstore=_get_or_create_vector_store(),
        docstore=_get_or_create_docstore(),
        id_key="doc_id",
        search_kwargs={"k": k},
    )


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_documents(
    raw_docs: list[Document],
    all_summaries: list[str],
) -> int:
    """
    Index raw documents into the retriever.
    raw_docs and all_summaries must have the same length.
    Returns the number of documents indexed.
    """
    assert len(raw_docs) == len(all_summaries), "lengths must match"

    try:
        vs = _get_or_create_vector_store()
        ds = _get_or_create_docstore()
    except ValueError as e:
        if "Chroma collection not initialized" in str(e):
            logger.warning("Collection not initialized during indexing, resetting stores...")
            reset_stores()
            vs = _get_or_create_vector_store()
            ds = _get_or_create_docstore()
        else:
            raise

    doc_ids = [str(uuid.uuid4()) for _ in raw_docs]

    summary_docs = []
    for i, s in enumerate(all_summaries):
        metadata = {"doc_id": doc_ids[i]}
        # Copy all metadata from raw_doc for persistence
        metadata.update(raw_docs[i].metadata)
        summary_docs.append(Document(page_content=s, metadata=metadata))

    ds.mset(list(zip(doc_ids, raw_docs)))
    vs.add_documents(summary_docs)

    logger.info("Indexed %d documents into vector store.", len(summary_docs))
    return len(summary_docs)


def indexed_count() -> int:
    """Return number of summaries currently in the vector store."""
    try:
        vs = _get_or_create_vector_store()
        return vs._collection.count()
    except ValueError as e:
        if "Chroma collection not initialized" in str(e):
            logger.warning("Collection not initialized, resetting...")
            vs.reset_collection()
            return vs._collection.count()
        else:
            raise


def reset_stores() -> None:
    """Clear all in-memory and persistent stores (useful for re-ingestion)."""
    global _vector_store, _docstore, _retriever
    try:
        if _vector_store is not None:
            _vector_store.delete_collection()
    except Exception as exc:
        logger.warning("Could not delete Chroma collection: %s", exc)
    _vector_store = None
    _docstore = None
    _retriever = None
    logger.info("Stores reset.")


# ── VectorStoreManager class ─────────────────────────────────────────────────

class VectorStoreManager:
    """Manages ChromaDB vector store and multi-vector retriever"""
    
    def __init__(self):
        self._retriever = None
        self._docstore = None
    
    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = get_retriever()
        return self._retriever
    
    @property
    def docstore(self):
        if self._docstore is None:
            self._docstore = _get_or_create_docstore()
        return self._docstore
    
    async def add_documents(self, texts: List[Dict], tables: List[Dict], images: List[Dict]):
        """Add documents to vector store"""
        # Prepare raw docs and summaries
        raw_docs = []
        summaries = []
        
        # Add texts
        for text in texts:
            raw_docs.append(Document(
                page_content=text["content"],
                metadata={
                    "source_pdf": text.get("source_pdf", "Unknown"),
                    "page": text.get("page", 0),
                    "type": "text"
                }
            ))
            summaries.append(text["content"][:1000])  # Truncate for summary
        
        # Add tables
        for table in tables:
            raw_docs.append(Document(
                page_content=table["content"],
                metadata={
                    "source_pdf": table.get("source_pdf", "Unknown"),
                    "page": table.get("page", 0),
                    "type": "table"
                }
            ))
            summaries.append(table["content"][:1000])
        
        # Add images
        for image in images:
            raw_docs.append(Document(
                page_content=f"Image data: {image['content'][:200]}...",
                metadata={
                    "source_pdf": image.get("source_pdf", "Unknown"),
                    "page": image.get("page", 0),
                    "type": "image",
                    "full_content": image.get("content", ""),
                    "width": image.get("width", 0),
                    "height": image.get("height", 0)
                }
            ))
            summaries.append(f"[Image: {image.get('source_pdf', 'Unknown')} page {image.get('page', 0)}]")
        
        # Index documents
        count = index_documents(raw_docs, summaries)
        logger.info(f"Indexed {count} documents")
    
    def get_document_by_id(self, doc_id: str):
        """Retrieve full document by ID"""
        if self.docstore:
            return self.docstore.mget([doc_id])[0]
        return None
    
    def get_retriever_with_k(self, k: int):
        """Get a retriever configured with a specific k value."""
        return get_retriever_with_k(k)
    
    def get_eval_retriever(self):          # <-- add this
        """Higher k retriever for evaluation — improves context recall."""
        return get_retriever_with_k(15)    
    
    def clear(self):
        """Clear all documents"""
        reset_stores()


# Global instance
vector_store_manager = VectorStoreManager()
