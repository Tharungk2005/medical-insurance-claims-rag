import numpy as np
from typing import List, Union
from PIL import Image
from langchain_core.documents import Document

from config import EMBED_MODEL

# Lazy-loaded text embedder
_embed_model = None
_clip_model = None
_clip_processor = None

def get_embed_model():
    """Lazy load SentenceTransformer embedding model."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {EMBED_MODEL}...")
        _embed_model = SentenceTransformer(EMBED_MODEL)
        print("Embedding model loaded.")
    return _embed_model

def embed_chunks(chunks: List[Document], batch_size: int = 32) -> np.ndarray:
    """
    Batch encode Document chunks into normalized 384-dimensional dense vectors.
    """
    if not chunks:
        return np.empty((0, 384), dtype=np.float32)

    model = get_embed_model()
    texts = [c.page_content for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return np.array(embeddings, dtype=np.float32)

def embed_query(query_text: str) -> np.ndarray:
    """
    Encode a single query string into a normalized 384-dimensional dense vector.
    """
    model = get_embed_model()
    emb = model.encode([query_text], normalize_embeddings=True)[0]
    return np.array(emb, dtype=np.float32)

def get_clip_model():
    """Lazy load optional CLIP model for image-to-image search."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        try:
            from transformers import CLIPProcessor, CLIPModel
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        except Exception as e:
            print(f"Warning: CLIP model unavailable ({e})")
            _clip_model = False
            _clip_processor = False
    return _clip_processor, _clip_model

def embed_image(pil_image: Image.Image) -> Union[np.ndarray, None]:
    """Encode a PIL Image into a 512-dimensional CLIP embedding."""
    processor, model = get_clip_model()
    if processor and model:
        import torch
        inputs = processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0]
    return None
