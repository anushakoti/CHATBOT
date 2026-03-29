import logging
import io
import base64
from pathlib import Path
from typing import Dict, List, Any
import pdfplumber
from pypdf import PdfReader
from PIL import Image as PILImage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core import get_settings

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Handles PDF extraction and processing"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        self.settings = get_settings()
    
    async def extract_pdf(self, file_content: bytes, filename: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract text chunks, tables, and images from a PDF file.
        """
        pdf_name = Path(filename).stem
        
        texts_out = []
        tables_out = []
        images_out = []
        
        temp_dir = Path("./workspace/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / filename
        with open(temp_path, "wb") as f:
            f.write(file_content)
        
        try:
            # Create a dedicated directory for images from this PDF
            pdf_img_dir = self.settings.img_dir / pdf_name
            pdf_img_dir.mkdir(parents=True, exist_ok=True)

            with pdfplumber.open(temp_path) as pdf:
                for pn, page in enumerate(pdf.pages, start=1):
                    # Extract Tables
                    for tbl in page.extract_tables() or []:
                        if tbl and len(tbl) > 1:
                            md_table = self._format_table_to_markdown(tbl)
                            tables_out.append({
                                "content": md_table,
                                "source_pdf": pdf_name,
                                "page": pn,
                                "type": "table",
                            })
                    
                    # Extract Text
                    text = page.extract_text() or ""
                    if text.strip():
                        chunks = self.text_splitter.split_text(text)
                        for chunk in chunks:
                            texts_out.append({
                                "content": chunk.strip(),
                                "source_pdf": pdf_name,
                                "page": pn,
                                "type": "text",
                            })
            
            # Extract Images
            img_count = self._extract_images(temp_path, pdf_name, pdf_img_dir, images_out)
            
            logger.info(f"Processed {filename}: {len(texts_out)} texts, {len(tables_out)} tables, {img_count} images")
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
        
        return {
            "texts": texts_out,
            "tables": tables_out,
            "images": images_out
        }
    
    def _format_table_to_markdown(self, table: List[List[Any]]) -> str:
        """Convert list of lists to a markdown table string"""
        if not table: return ""
        header = table[0]
        rows = table[1:]
        md = "| " + " | ".join(str(c or "") for c in header) + " |\n"
        md += "| " + " | ".join("---" for _ in header) + " |\n"
        for row in rows:
            md += "| " + " | ".join(str(c or "") for c in row) + " |\n"
        return md

    def _extract_images(self, pdf_path: Path, pdf_name: str, 
                       image_dir: Path, images_out: List[Dict[str, Any]]) -> int:
        """Extract images from PDF using PyMuPDF (fitz) - more robust than pypdf"""
        try:
            import fitz # PyMuPDF
            doc = fitz.open(str(pdf_path))
            img_count = 0
            
            logger.info(f"Extracting images from: {pdf_name} using PyMuPDF")
            
            for pn in range(len(doc)):
                page = doc[pn]
                image_list = page.get_images(full=True)
                
                for ik, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    try:
                        pil = PILImage.open(io.BytesIO(image_bytes))
                        
                        # Skip very small images (icons, etc.)
                        if pil.width < 50 or pil.height < 50:
                            continue
                        
                        # Standardize to RGB
                        if pil.mode != "RGB":
                            pil = pil.convert("RGB")
                        
                        img_filename = f"p{pn+1:03d}_i{ik}.png"
                        img_path = image_dir / img_filename
                        pil.save(img_path, "PNG")
                        
                        with open(img_path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode()
                        
                        images_out.append({
                            "content": b64,
                            "source_pdf": pdf_name,
                            "page": pn + 1,
                            "type": "image",
                            "path": str(img_path),
                            "width": pil.width,
                            "height": pil.height,
                        })
                        img_count += 1
                    except Exception as e:
                        logger.debug(f"Failed image {ik} on page {pn+1}: {str(e)}")
                        continue
            
            doc.close()
            logger.info(f"Final image count for {pdf_name}: {img_count}")
            return img_count
            
        except ImportError:
            logger.warning("PyMuPDF not found. Falling back to pypdf for image extraction.")
            # Fallback to the previous pypdf logic
            try:
                reader = PdfReader(str(pdf_path))
                img_count = 0
                for pn, page in enumerate(reader.pages, start=1):
                    if not page.images: continue
                    for ik, img_obj in enumerate(page.images):
                        try:
                            img_data = img_obj.data
                            if not img_data: continue
                            pil = PILImage.open(io.BytesIO(img_data))
                            if pil.width < 50 or pil.height < 50: continue
                            if pil.mode != "RGB": pil = pil.convert("RGB")
                            img_filename = f"p{pn:03d}_i{ik}.png"
                            img_path = image_dir / img_filename
                            pil.save(img_path, "PNG")
                            with open(img_path, "rb") as f:
                                b64 = base64.b64encode(f.read()).decode()
                            images_out.append({
                                "content": b64,
                                "source_pdf": pdf_name,
                                "page": pn,
                                "type": "image",
                                "path": str(img_path),
                                "width": pil.width,
                                "height": pil.height,
                            })
                            img_count += 1
                        except Exception: continue
                return img_count
            except Exception as e:
                logger.error(f"Fallback image extraction failure: {e}")
                return 0
        except Exception as e:
            logger.error(f"Global image extraction failure for {pdf_name}: {str(e)}")
            return 0
