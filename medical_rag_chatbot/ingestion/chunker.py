from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_documents(docs: List[Dict[str, Any]]) -> List[Document]:
    """
    Chunk documents according to modality rules:
    - PDF: RecursiveCharacterTextSplitter with chunk_size and chunk_overlap
    - Table: row-level chunk preserved as-is
    - Image: caption/OCR chunk preserved as-is
    Appends chunk_index and chunk_id to every chunk's metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks: List[Document] = []
    
    # Track chunk index per source file
    file_chunk_counter: Dict[str, int] = {}

    for doc in docs:
        text = doc.get("text", "").strip()
        if not text:
            continue

        base_meta = dict(doc.get("metadata", {}))
        modality = base_meta.get("modality", "pdf")
        source_file = base_meta.get("source_file", "unknown_doc")

        if source_file not in file_chunk_counter:
            file_chunk_counter[source_file] = 0

        if modality == "pdf":
            # Split long PDF texts
            split_texts = splitter.split_text(text)
            for split_text in split_texts:
                if not split_text.strip():
                    continue
                c_idx = file_chunk_counter[source_file]
                file_chunk_counter[source_file] += 1

                meta = dict(base_meta)
                meta["chunk_index"] = c_idx
                meta["chunk_id"] = f"{source_file}_chunk_{c_idx}"
                all_chunks.append(Document(page_content=split_text.strip(), metadata=meta))

        elif modality in ["table", "image"]:
            # Tables and images are already atomic chunks
            c_idx = file_chunk_counter[source_file]
            file_chunk_counter[source_file] += 1

            meta = dict(base_meta)
            meta["chunk_index"] = c_idx
            meta["chunk_id"] = f"{source_file}_chunk_{c_idx}"
            all_chunks.append(Document(page_content=text, metadata=meta))

        else:
            # Fallback for general text
            split_texts = splitter.split_text(text)
            for split_text in split_texts:
                if not split_text.strip():
                    continue
                c_idx = file_chunk_counter[source_file]
                file_chunk_counter[source_file] += 1

                meta = dict(base_meta)
                meta["chunk_index"] = c_idx
                meta["chunk_id"] = f"{source_file}_chunk_{c_idx}"
                all_chunks.append(Document(page_content=split_text.strip(), metadata=meta))

    return all_chunks
