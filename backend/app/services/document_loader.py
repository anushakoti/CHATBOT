"""
PDF extraction (text, tables, images) and LLM-based summarisation.

Key design decisions vs. the notebook:
- Images larger than settings.image_max_bytes are resized before encoding so
  they never hit the Bedrock 5 MB limit.
- All operations are async-friendly: CPU-heavy work is run in a thread pool
  via asyncio.to_thread so FastAPI's event loop stays unblocked.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image as PILImage
import pdfplumber
from pypdf import PdfReader

from app.config.settings import get_settings
from app.services.models import get_llm

# Heavy PDF libs are imported lazily inside extract_pdf_sync so tests
# can import this module without requiring unstructured/pypdf system deps.

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

TEXT_SUMMARY_PROMPT = PromptTemplate.from_template(
    "You are an assistant creating concise retrieval-optimised summaries of "
    "Dell laptop product documentation.\n"
    "Summarise the following content in 2-4 sentences capturing all key "
    "product facts, specs, and features.\n\n"
    "Content: {element}"
)

IMAGE_SUMMARY_PROMPT = (
    "You are an assistant summarising product images from Dell laptop brochures "
    "for semantic search retrieval.\n"
    "Describe what you see: product photos, spec tables, diagrams, charts, or "
    "marketing graphics.\n"
    "Include any visible text, model names, specs, features, or data shown in the image.\n"
    "Keep the summary under 100 words, factual, and retrieval-optimised."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resize_image_bytes(data: bytes, max_bytes: int) -> bytes:
    """Down-scale an image until it fits under max_bytes (quality 85→50)."""
    if len(data) <= max_bytes:
        return data
    img = PILImage.open(io.BytesIO(data))
    buf = io.BytesIO()
    quality = 85
    while quality >= 50:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        if buf.tell() <= max_bytes:
            return buf.getvalue()
        # scale down by 10 %
        w, h = int(img.width * 0.9), int(img.height * 0.9)
        img = img.resize((w, h), PILImage.LANCZOS)
        quality -= 5
    return buf.getvalue()  # best effort


def _encode_bytes_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_pdf_sync(pdf_path: Path, img_dir: Path, settings) -> dict:
    """Blocking PDF extraction — call via asyncio.to_thread."""
    # Lazy imports so this module can be imported without system PDF libs
    from langchain_text_splitters import CharacterTextSplitter
    from pypdf import PdfReader
    from unstructured.partition.pdf import partition_pdf

    pdf_name = pdf_path.stem
    all_texts: list[str] = []
    all_tables: list[str] = []
    img_paths: list[Path] = []

    # ── Text & tables ─────────────────────────────────────────────────────────
    elements = partition_pdf(
        filename=str(pdf_path),
        extract_images_in_pdf=False,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=settings.unstructured_max_chars,
        new_after_n_chars=settings.unstructured_new_after_n_chars,
        combine_text_under_n_chars=settings.unstructured_combine_under_n_chars,
        strategy="fast",
    )

    raw_texts, raw_tables = [], []
    for el in elements:
        t = str(type(el))
        if "Table" in t:
            raw_tables.append(str(el))
        elif "CompositeElement" in t:
            raw_texts.append(str(el))

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    all_texts.extend(splitter.split_text(" ".join(raw_texts)))
    all_tables.extend(raw_tables)

    # ── Images ────────────────────────────────────────────────────────────────
    reader = PdfReader(str(pdf_path))
    for page_num, page in enumerate(reader.pages):
        for img_key, img_obj in enumerate(page.images):
            try:
                img = PILImage.open(io.BytesIO(img_obj.data))
                if img.width <= settings.image_min_dimension or img.height <= settings.image_min_dimension:
                    continue
                safe_name = pdf_name.replace(" ", "_")
                dest = img_dir / f"{safe_name}_p{page_num}_i{img_key}.png"
                img.save(dest, "PNG")
                img_paths.append(dest)
            except Exception as exc:
                logger.debug("Skipping unreadable image %s: %s", img_key, exc)

    logger.info(
        "%s → texts=%d tables=%d images=%d",
        pdf_path.name,
        len(all_texts),
        len(all_tables),
        len(img_paths),
    )
    return {"texts": all_texts, "tables": all_tables, "img_paths": img_paths}


async def extract_pdf(pdf_path: Path, img_dir: Path) -> dict:
    settings = get_settings()
    return await asyncio.to_thread(extract_pdf_sync, pdf_path, img_dir, settings)


# ── Summarisation ─────────────────────────────────────────────────────────────

def build_summarise_chain():
    llm = get_llm()
    return (
        {"element": lambda x: x}
        | TEXT_SUMMARY_PROMPT
        | llm.with_fallbacks([RunnableLambda(lambda _: AIMessage(content="Summary unavailable."))])
        | StrOutputParser()
    )


async def summarise_texts(texts: list[str]) -> list[str]:
    if not texts:
        return []
    chain = build_summarise_chain()
    settings = get_settings()
    return await asyncio.to_thread(
        chain.batch, texts, {"max_concurrency": settings.llm_max_concurrency}
    )


async def summarise_image_file(img_path: Path) -> tuple[str, str] | None:
    """Returns (b64_string, summary) or None on failure."""
    settings = get_settings()
    llm = get_llm()

    try:
        raw_bytes = img_path.read_bytes()
        raw_bytes = await asyncio.to_thread(_resize_image_bytes, raw_bytes, settings.image_max_bytes)
        b64 = _encode_bytes_b64(raw_bytes)

        msg = await asyncio.to_thread(
            llm.invoke,
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": IMAGE_SUMMARY_PROMPT},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        },
                    ]
                )
            ],
        )
        summary = msg.content if isinstance(msg.content, str) else str(msg.content)
        logger.debug("✓ %s", img_path.name)
        return b64, summary
    except Exception as exc:
        logger.warning("✗ %s: %s", img_path.name, exc)
        return None


async def summarise_images(img_paths: list[Path]) -> tuple[list[str], list[str]]:
    """Returns (img_base64_list, image_summaries)."""
    tasks = [summarise_image_file(p) for p in img_paths]
    results = await asyncio.gather(*tasks)
    b64_list, summaries = [], []
    for r in results:
        if r is not None:
            b64_list.append(r[0])
            summaries.append(r[1])
    return b64_list, summaries


# ── DocumentLoader class ──────────────────────────────────────────────────────

class DocumentLoader:
    """Handles PDF extraction and processing"""
    
    def __init__(self):
        settings = get_settings()
        self.image_base_dir = Path("./workspace/extracted_images")
        self.image_base_dir.mkdir(parents=True, exist_ok=True)
        self.llm = get_llm()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Default chunk size
            chunk_overlap=300,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    async def extract_pdf(self, file_content: bytes, filename: str) -> Dict[str, List[Dict]]:
        """
        Extract text chunks, tables, and images from a PDF file.
        Returns dict with keys: texts, tables, images.
        """
        pdf_name = Path(filename).stem
        image_dir = self.image_base_dir / pdf_name
        image_dir.mkdir(exist_ok=True)
        
        texts_out: List[Dict] = []
        tables_out: List[Dict] = []
        images_out: List[Dict] = []
        
        # Save temp file for processing
        temp_path = self.image_base_dir / filename
        with open(temp_path, "wb") as f:
            f.write(file_content)
        
        try:
            # Text & Tables via pdfplumber
            with pdfplumber.open(temp_path) as pdf:
                full_text_pages = []
                
                for pn, page in enumerate(pdf.pages, start=1):
                    # Extract tables
                    for tbl in page.extract_tables() or []:
                        if tbl and len(tbl) > 1:
                            header = tbl[0]
                            rows = tbl[1:]
                            md_rows = [
                                "| " + " | ".join(str(c or "") for c in header) + " |",
                                "| " + " | ".join("---" for _ in header) + " |"
                            ]
                            md_rows += [
                                "| " + " | ".join(str(c or "") for c in row) + " |"
                                for row in rows
                            ]
                            tables_out.append({
                                "content": "\n".join(md_rows),
                                "source_pdf": pdf_name,
                                "page": pn,
                                "type": "table",
                            })
                    
                    # Page text
                    txt = page.extract_text() or ""
                    if txt.strip():
                        full_text_pages.append((pn, txt))
                
                # Chunk text
                for chunk in self._chunk_text(full_text_pages):
                    chunk["source_pdf"] = pdf_name
                    texts_out.append(chunk)
            
            # Images via pypdf
            img_count = self._extract_images(temp_path, pdf_name, image_dir, images_out)
            
            logger.info(f"📄 {filename:45s} texts={len(texts_out):3d} tables={len(tables_out):3d} images={img_count:3d}")
            
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
        
        return {
            "texts": texts_out,
            "tables": tables_out,
            "images": images_out
        }
    
    def _chunk_text(self, full_text_pages: List[tuple]) -> List[Dict]:
        """Split text into chunks with page tracking"""
        chunks = []
        chunk_buf, chunk_pages = "", []
        
        for pn, txt in full_text_pages:
            if len(chunk_buf) + len(txt) > 1000 and chunk_buf:  # Use default chunk size
                chunks.append({
                    "content": chunk_buf.strip(),
                    "page": chunk_pages[0],
                    "type": "text",
                })
                chunk_buf, chunk_pages = "", []
            chunk_buf += " " + txt
            chunk_pages.append(pn)
        
        if chunk_buf.strip():
            chunks.append({
                "content": chunk_buf.strip(),
                "page": chunk_pages[0],
                "type": "text",
            })
        
        return chunks
    
    def _extract_images(self, pdf_path: Path, pdf_name: str, 
                       image_dir: Path, images_out: List[Dict]) -> int:
        """Extract images from PDF"""
        reader = PdfReader(pdf_path)
        img_count = 0
        
        for pn, page in enumerate(reader.pages, start=1):
            for ik, img_obj in enumerate(page.images):
                try:
                    pil = PILImage.open(io.BytesIO(img_obj.data))
                    if pil.width < 120 or pil.height < 120:
                        continue
                    
                    img_path = image_dir / f"p{pn:03d}_i{ik}.png"
                    pil.save(img_path, "PNG")
                    
                    with open(img_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    
                    images_out.append({
                        "content": b64,
                        "source_pdf": image_dir.name,  # pdf_name
                        "page": pn,
                        "type": "image",
                        "path": str(img_path),
                        "width": pil.width,
                        "height": pil.height,
                    })
                    img_count += 1
                except Exception as e:
                    logger.debug(f"Failed to extract image: {e}")
                    continue
        
        return img_count
