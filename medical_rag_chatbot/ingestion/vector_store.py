import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import chromadb
from langchain_core.documents import Document

from config import CHROMA_PATH, COLLECTION_NAME

_client = None
_collection = None

def get_client() -> chromadb.PersistentClient:
    """Initialize or return persistent ChromaDB client."""
    global _client
    if _client is None:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client

def get_collection():
    """Retrieve or create the claims_rag collection with cosine distance metric."""
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection

def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure ChromaDB metadata contains only supported primitive types."""
    clean = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif v is None:
            clean[k] = ""
        else:
            clean[k] = str(v)
    return clean

def insert_chunks(chunks: List[Document], embeddings: np.ndarray) -> int:
    """
    Insert or upsert document chunks and vectors into ChromaDB.
    Returns the count of inserted chunks.
    """
    if not chunks:
        return 0

    collection = get_collection()
    
    ids = []
    documents = []
    metadatas = []
    
    for i, c in enumerate(chunks):
        cid = c.metadata.get("chunk_id", f"{c.metadata.get('source_file', 'doc')}_chunk_{c.metadata.get('chunk_index', i)}")
        ids.append(cid)
        documents.append(c.page_content)
        metadatas.append(sanitize_metadata(c.metadata))

    embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

    # Use upsert to gracefully handle re-ingestion and updates
    collection.upsert(
        ids=ids,
        embeddings=embeddings_list,
        documents=documents,
        metadatas=metadatas
    )

    return len(ids)

def query_collection(
    query_embedding: np.ndarray,
    n_results: int = 10,
    where_filter: Optional[Dict[str, Any]] = None
) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
    """
    Query ChromaDB using cosine distance.
    Returns: (documents, metadatas, distances)
    """
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return [], [], []

    actual_n = min(n_results, count)
    
    emb_input = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
    if isinstance(emb_input[0], (float, int)):
        emb_input = [emb_input]

    # Clean where_filter if empty
    filter_arg = where_filter if where_filter and len(where_filter) > 0 else None

    results = collection.query(
        query_embeddings=emb_input,
        n_results=actual_n,
        where=filter_arg,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0] if results.get("documents") else []
    metas = results["metadatas"][0] if results.get("metadatas") else []
    distances = results["distances"][0] if results.get("distances") else []

    return docs, metas, distances

def delete_document(source_file: str) -> None:
    """Remove chunks associated with a specific source file."""
    collection = get_collection()
    collection.delete(where={"source_file": source_file})
