import os
import re
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image, ImageFilter, ImageEnhance
import pytesseract

from config import CAPTION_MODEL
from ingestion.parse_pdf import extract_claim_id

# Lazy-loaded BLIP processor and model
_blip_processor = None
_blip_model = None

def get_blip_model():
    """Lazy load BLIP image captioning model once at module level."""
    global _blip_processor, _blip_model
    if _blip_processor is None or _blip_model is None:
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            print(f"Loading image captioning model: {CAPTION_MODEL}...")
            _blip_processor = BlipProcessor.from_pretrained(CAPTION_MODEL)
            _blip_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL)
            print("BLIP model loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load BLIP model ({e}). Fallback caption generator will be used.")
            _blip_processor = False
            _blip_model = False
    return _blip_processor, _blip_model

def preprocess_image(image_path: str | Path) -> Image.Image:
    """Preprocess image: sharpen, enhance contrast, and convert to clean RGB."""
    img = Image.open(image_path).convert("RGB")
    # Apply sharpening for blurry scans
    img = img.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    return img

def generate_caption(image: Image.Image, fallback_text: str = "") -> str:
    """Generate image caption using BLIP vision-language model with fallback."""
    processor, model = get_blip_model()
    if processor and model:
        try:
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=100)
            caption = processor.decode(out[0], skip_special_tokens=True).strip()
            if caption:
                return caption
        except Exception as e:
            print(f"BLIP inference warning: {e}")

    # Fallback caption based on OCR or context
    if fallback_text and len(fallback_text.strip()) > 10:
        cleaned = " ".join(fallback_text.split()[:25])
        return f"Medical document scan containing text: {cleaned}"
    return "Medical document or prescription scan image"

def extract_ocr_text(image: Image.Image) -> str:
    """Run pytesseract OCR with fallback if tesseract binary is unavailable."""
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        # Pytesseract binary might not be installed on OS PATH
        return ""

def infer_image_doc_type(filename: str, ocr_text: str) -> str:
    """Infer image doc_type: prescription, scan, claim_form, or medical_record."""
    fn_lower = filename.lower()
    ocr_lower = ocr_text.lower()
    
    if "rx" in fn_lower or "prescrip" in fn_lower or "mg" in ocr_lower or "rx" in ocr_lower:
        return "prescription"
    elif "scan" in fn_lower or "xray" in fn_lower or "mri" in fn_lower or "radiology" in ocr_lower:
        return "scan"
    elif "claim" in fn_lower:
        return "claim_form"
    return "medical_record"

def parse_image(image_path: str | Path) -> Dict[str, Any]:
    """
    Parse an image: preprocess, extract OCR, generate caption, and assemble chunk text.
    Caption serves as chunk text; raw OCR stored as backup field in metadata.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    filename = image_path.name
    claim_id = extract_claim_id(filename)

    processed_img = preprocess_image(image_path)
    ocr_text = extract_ocr_text(processed_img)

    # If claim_id was unknown in filename, check OCR text
    if claim_id == "UNKNOWN" and ocr_text:
        ocr_claim_id = extract_claim_id(ocr_text)
        if ocr_claim_id != "UNKNOWN":
            claim_id = ocr_claim_id

    caption = generate_caption(processed_img, fallback_text=ocr_text)
    doc_type = infer_image_doc_type(filename, ocr_text)

    # Compose chunk text: Caption is primary; if OCR has rich prescription details, enrich chunk
    if ocr_text:
        chunk_text = f"Image Caption: {caption}. Extracted Text: {ocr_text}"
    else:
        chunk_text = f"Image Caption: {caption}"

    metadata = {
        "source_file": filename,
        "image_path": str(image_path),
        "page_number": 1,
        "claim_id": claim_id,
        "doc_type": doc_type,
        "modality": "image",
        "ocr_text": ocr_text[:500] if ocr_text else ""
    }

    return {
        "text": chunk_text,
        "metadata": metadata
    }
