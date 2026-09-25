import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import (
    DATA_RAW_DIR,
    DATA_RAW_PDFS,
    DATA_RAW_IMAGES,
    DATA_RAW_TABLES
)
from ingestion.parse_pdf import parse_pdf
from ingestion.parse_tables import parse_csv, parse_tables_from_pdf
from ingestion.parse_images import parse_image
from ingestion.mask_pii import mask_pii
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_chunks
from ingestion.vector_store import insert_chunks, get_collection
from retrieval.sparse_retriever import build_bm25_index
from retrieval.cache import invalidate_cache

def ingest_directory(data_dir: str | Path = DATA_RAW_DIR) -> List[Dict[str, Any]]:
    """
    Traverse raw data directories across PDFs, images, and tables.
    Parses documents, applies PII masking, and removes duplicate content.
    Returns unified list of document objects: {"text": str, "metadata": dict}
    """
    data_dir = Path(data_dir)
    pdf_dir = data_dir / "pdfs" if (data_dir / "pdfs").exists() else DATA_RAW_PDFS
    img_dir = data_dir / "images" if (data_dir / "images").exists() else DATA_RAW_IMAGES
    tbl_dir = data_dir / "tables" if (data_dir / "tables").exists() else DATA_RAW_TABLES

    all_raw_docs: List[Dict[str, Any]] = []

    # 1. Parse PDFs
    if pdf_dir.exists():
        for p in pdf_dir.glob("*.pdf"):
            print(f"Parsing PDF: {p.name}")
            try:
                # Text pages
                pdf_docs = parse_pdf(p)
                all_raw_docs.extend(pdf_docs)
                # Any embedded tables in PDF
                table_docs = parse_tables_from_pdf(p)
                all_raw_docs.extend(table_docs)
            except Exception as e:
                print(f"Error parsing PDF {p.name}: {e}")

    # 2. Parse Tables (CSV and Excel)
    if tbl_dir.exists():
        for p in list(tbl_dir.glob("*.csv")) + list(tbl_dir.glob("*.xlsx")) + list(tbl_dir.glob("*.xls")):
            print(f"Parsing Table: {p.name}")
            try:
                csv_docs = parse_csv(p)
                all_raw_docs.extend(csv_docs)
            except Exception as e:
                print(f"Error parsing table {p.name}: {e}")

    # 3. Parse Images (JPG, PNG, JPEG)
    if img_dir.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.tiff"]:
            for p in img_dir.glob(ext):
                print(f"Parsing Image: {p.name}")
                try:
                    img_doc = parse_image(p)
                    all_raw_docs.append(img_doc)
                except Exception as e:
                    print(f"Error parsing image {p.name}: {e}")

    # 4. Apply PII Masking and Deduplication
    processed_docs: List[Dict[str, Any]] = []
    seen_hashes = set()

    for doc in all_raw_docs:
        raw_text = doc.get("text", "").strip()
        if not raw_text:
            continue

        # Mask PII
        masked_text = mask_pii(raw_text)

        # Content hash deduplication
        content_hash = hashlib.md5(masked_text.encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        doc_copy = dict(doc)
        doc_copy["text"] = masked_text
        doc_copy["metadata"]["content_hash"] = content_hash
        processed_docs.append(doc_copy)

    print(f"Ingested {len(processed_docs)} total document elements after PII masking & deduplication.")
    return processed_docs

def ingest_and_store(data_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Complete end-to-end ingestion pipeline:
    1. Parse and mask all files from data/raw/
    2. Split/normalize into multimodal chunks
    3. Generate dense embeddings via BGE-small
    4. Upsert into ChromaDB vector store
    5. Build & serialize BM25 sparse index
    6. Invalidate query cache
    """
    target_dir = Path(data_dir) if data_dir else DATA_RAW_DIR
    print(f"\n--- Starting Unified Ingestion Pipeline on {target_dir} ---")
    
    docs = ingest_directory(target_dir)
    if not docs:
        print("No documents found to ingest.")
        return {"status": "empty", "num_chunks": 0, "num_docs": 0}

    # Chunking
    chunks = chunk_documents(docs)
    print(f"Generated {len(chunks)} chunks across modalities.")

    # Embedding
    embeddings = embed_chunks(chunks)
    print(f"Generated embeddings of shape {embeddings.shape}.")

    # Vector store insertion
    num_stored = insert_chunks(chunks, embeddings)
    print(f"Stored {num_stored} chunks in ChromaDB collection.")

    # BM25 sparse indexing
    build_bm25_index(chunks)
    print("BM25 sparse index created and saved.")

    # Invalidate query cache so new data is reflected
    invalidate_cache()
    print("Query cache invalidated.")

    unique_files = len(set(c.metadata.get("source_file", "") for c in chunks))
    print(f"=== Successfully Ingested {len(chunks)} Chunks across {unique_files} Documents ===\n")

    return {
        "status": "success",
        "num_chunks": len(chunks),
        "num_documents": unique_files,
        "collection_count": get_collection().count()
    }

if __name__ == "__main__":
    ingest_and_store()
