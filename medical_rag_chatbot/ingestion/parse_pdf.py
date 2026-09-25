import os
import re
from pathlib import Path
from typing import List, Dict, Any
import pdfplumber
import pymupdf

from config import DATA_PROCESSED_IMAGES

def extract_claim_id(filename_or_text: str) -> str:
    """Extract claim ID formatted as CLM-YYYY-NNN from filename or text."""
    match = re.search(r'CLM-\d{4}-\d{3}', filename_or_text, re.IGNORECASE)
    if match:
        return match.group(0).upper()
    # Secondary pattern for flexible claim ids
    match2 = re.search(r'CLM[-_]?\d{3,6}', filename_or_text, re.IGNORECASE)
    if match2:
        return match2.group(0).upper()
    return "UNKNOWN"

def infer_doc_type(filename: str, text: str) -> str:
    """Infer document type: policy, claim_form, eob, billing, or prescription."""
    fn_lower = filename.lower()
    text_lower = text[:500].lower() if text else ""
    
    if "policy" in fn_lower or "coverage" in fn_lower or "benefit" in text_lower:
        return "policy"
    elif "eob" in fn_lower or "explanation of benefit" in text_lower:
        return "eob"
    elif "claim" in fn_lower or "claim form" in text_lower:
        return "claim_form"
    elif "bill" in fn_lower or "invoice" in fn_lower:
        return "billing"
    elif "prescription" in fn_lower or "rx" in fn_lower:
        return "prescription"
    return "pdf"

def parse_pdf(filepath: str | Path) -> List[Dict[str, Any]]:
    """
    Parse a PDF file, extracting clean text per page and saving any embedded images.
    Returns a list of dicts: {"text": page_text, "metadata": metadata_dict}
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PDF file not found: {filepath}")

    filename = filepath.name
    claim_id = extract_claim_id(filename)
    os.makedirs(DATA_PROCESSED_IMAGES, exist_ok=True)
    
    pages_data = []

    # Open with PyMuPDF for embedded images & fallback text
    try:
        doc = pymupdf.open(str(filepath))
    except Exception as e:
        doc = None

    # Open with pdfplumber for structured layout text extraction
    with pdfplumber.open(str(filepath)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            
            # If pdfplumber returned empty/scanned text, fall back to pymupdf
            if not text.strip() and doc and i - 1 < len(doc):
                fitz_page = doc[i - 1]
                text = fitz_page.get_text() or ""

            # Extract embedded images if any
            if doc and i - 1 < len(doc):
                fitz_page = doc[i - 1]
                image_list = fitz_page.get_images(full=True)
                for img_idx, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image.get("ext", "png")
                    img_filename = f"{filepath.stem}_p{i}_img{img_idx}.{image_ext}"
                    img_save_path = DATA_PROCESSED_IMAGES / img_filename
                    try:
                        with open(img_save_path, "wb") as f_img:
                            f_img.write(image_bytes)
                    except Exception:
                        pass

            if text.strip():
                # If claim_id was unknown in filename, check page text
                effective_claim_id = claim_id
                if effective_claim_id == "UNKNOWN":
                    text_claim_id = extract_claim_id(text)
                    if text_claim_id != "UNKNOWN":
                        effective_claim_id = text_claim_id

                doc_type = infer_doc_type(filename, text)
                
                metadata = {
                    "source_file": filename,
                    "page_number": i,
                    "claim_id": effective_claim_id,
                    "doc_type": doc_type,
                    "modality": "pdf",
                    "file_path": str(filepath)
                }
                pages_data.append({
                    "text": text.strip(),
                    "metadata": metadata
                })

    if doc:
        doc.close()

    return pages_data
