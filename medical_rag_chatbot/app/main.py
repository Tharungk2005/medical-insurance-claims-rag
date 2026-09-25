import os
import sys
import shutil
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.query_engine import answer_query
from ingestion.pipeline import ingest_and_store
from retrieval.cache import invalidate_cache, load_cache
from ingestion.vector_store import get_collection
from config import DATA_RAW_PDFS, DATA_RAW_IMAGES, DATA_RAW_TABLES

st.set_page_config(
    page_title="Claims Assistant — Multimodal RAG",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Aesthetics
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Container */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #0284c7 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
    }
    .main-header p {
        margin: 6px 0 0 0;
        font-size: 0.95rem;
        opacity: 0.9;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-pdf { background-color: #fee2e2; color: #991b1b; }
    .badge-image { background-color: #fef3c7; color: #92400e; }
    .badge-table { background-color: #e0e7ff; color: #3730a3; }
    
    .badge-high { background-color: #dcfce7; color: #166534; }
    .badge-med { background-color: #fef9c3; color: #854d0e; }
    .badge-low { background-color: #fee2e2; color: #991b1b; }
    .badge-cache { background-color: #f3e8ff; color: #6b21a8; }

    /* Source Box */
    .source-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metadata_filter" not in st.session_state:
    st.session_state.metadata_filter = None

# Sidebar Controls
with st.sidebar:
    st.markdown("### 🔍 Retrieval Filters")
    doc_type = st.selectbox(
        "Document Type",
        ["All", "policy", "claim_form", "eob", "billing", "prescription"],
        index=0
    )
    
    modalities = st.multiselect(
        "Modality Scope",
        ["pdf", "image", "table"],
        default=["pdf", "image", "table"]
    )
    
    claim_id_input = st.text_input("Claim ID Filter", placeholder="e.g. CLM-2024-001")
    use_cot_checkbox = st.checkbox("Enable Chain-of-Thought (CoT)", value=False)

    st.markdown("---")
    st.markdown("### 📤 Ingest Documents")
    uploaded_file = st.file_uploader(
        "Upload Claim Document",
        type=["pdf", "jpg", "jpeg", "png", "csv", "xlsx"]
    )

    if uploaded_file is not None:
        if st.button("Process & Index Document", use_container_width=True, type="primary"):
            fname = uploaded_file.name
            ext = Path(fname).suffix.lower()
            
            # Route to respective raw directory
            if ext == ".pdf":
                dest_dir = DATA_RAW_PDFS
            elif ext in [".csv", ".xlsx"]:
                dest_dir = DATA_RAW_TABLES
            else:
                dest_dir = DATA_RAW_IMAGES

            os.makedirs(dest_dir, exist_ok=True)
            save_path = dest_dir / fname
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Ingesting {fname}..."):
                res = ingest_and_store()
                st.success(f"Indexed successfully! Total chunks: {res.get('num_chunks', 0)}")

    st.markdown("---")
    st.markdown("### ⚙️ Database & Cache")
    col1, col2 = st.columns(2)
    with col1:
        try:
            coll_count = get_collection().count()
        except Exception:
            coll_count = 0
        st.metric("Vector Chunks", coll_count)
    with col2:
        cache_data = load_cache()
        st.metric("Cached Queries", len(cache_data))

    if st.button("Flush Cache", use_container_width=True):
        invalidate_cache()
        st.toast("Cache flushed successfully!")

    if st.button("Rebuild Vector DB", use_container_width=True):
        with st.spinner("Re-indexing full corpus..."):
            ingest_and_store()
            st.rerun()

# Build where_filter dict from sidebar selections
where_filter = {}
if claim_id_input.strip():
    where_filter["claim_id"] = claim_id_input.strip().upper()
if doc_type != "All":
    where_filter["doc_type"] = doc_type

# Main Panel
st.markdown("""
<div class="main-header">
    <h1>🏥 Medical Insurance Claims Processing Assistant</h1>
    <p>Multimodal RAG Chatbot powered by LLaMA 3.1 8B, BGE-Small embeddings, Hybrid BM25+RRF, and CrossEncoder Reranking</p>
</div>
""", unsafe_allow_html=True)

# Render Chat History
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])
        
        # Display sources if assistant message has metadata
        if role == "assistant" and msg.get("sources"):
            sources = msg["sources"]
            max_score = msg.get("max_score", 0.0)
            conf = msg.get("confidence", "Medium")
            cache_hit = msg.get("cache_hit", False)

            # Metadata header
            cols = st.columns([1, 1, 3])
            with cols[0]:
                badge_class = "badge-high" if conf == "High" else ("badge-med" if conf == "Medium" else "badge-low")
                st.markdown(f'<span class="badge {badge_class}">Confidence: {conf} ({max_score:.2f})</span>', unsafe_allow_html=True)
            with cols[1]:
                if cache_hit:
                    st.markdown('<span class="badge badge-cache">⚡ Cached Response</span>', unsafe_allow_html=True)

            with st.expander(f"View Retrieved Evidence ({len(sources)} sources)"):
                for idx, src in enumerate(sources, start=1):
                    meta = src.get("metadata", {})
                    s_file = meta.get("source_file", "Document")
                    mod = meta.get("modality", "pdf").lower()
                    page = meta.get("page_number", meta.get("row_index", "N/A"))
                    score = src.get("rerank_score", src.get("score", 0.0))

                    mod_badge = f'<span class="badge badge-{mod}">{mod.upper()}</span>'
                    st.markdown(f"**Source {idx}:** {mod_badge} `{s_file}` | Ref: {page} | Score: `{score:.2f}`", unsafe_allow_html=True)
                    st.text_area(f"Chunk text #{idx}", value=src.get("text", ""), height=100, disabled=True, key=f"src_{msg.get('msg_id', 0)}_{idx}")

# Chat Input Box
user_query = st.chat_input("Ask about claim denial reasons, policy coverage, copays, billing codes, or prescriptions...")

if user_query:
    # 1. Append user query to chat history
    msg_id = len(st.session_state.messages) + 1
    st.session_state.messages.append({"role": "user", "content": user_query, "msg_id": msg_id})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Process query with spinner
    with st.chat_message("assistant"):
        with st.spinner("Analyzing claim documents, billing tables, and medical scans..."):
            history_turns = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-10:-1]
            ]
            
            response = answer_query(
                query=user_query,
                metadata_filter=where_filter if where_filter else None,
                chat_history=history_turns,
                use_cot=use_cot_checkbox
            )

            answer_text = response["answer"]
            sources = response.get("sources", [])
            conf = response.get("confidence", "Medium")
            max_score = response.get("max_score", 0.0)
            cache_hit = response.get("cache_hit", False)

            st.markdown(answer_text)

            # Sources and confidence badges
            if sources:
                cols = st.columns([1, 1, 3])
                with cols[0]:
                    badge_class = "badge-high" if conf == "High" else ("badge-med" if conf == "Medium" else "badge-low")
                    st.markdown(f'<span class="badge {badge_class}">Confidence: {conf} ({max_score:.2f})</span>', unsafe_allow_html=True)
                with cols[1]:
                    if cache_hit:
                        st.markdown('<span class="badge badge-cache">⚡ Cached Response</span>', unsafe_allow_html=True)

                with st.expander(f"View Retrieved Evidence ({len(sources)} sources)"):
                    for idx, src in enumerate(sources, start=1):
                        meta = src.get("metadata", {})
                        s_file = meta.get("source_file", "Document")
                        mod = meta.get("modality", "pdf").lower()
                        page = meta.get("page_number", meta.get("row_index", "N/A"))
                        score = src.get("rerank_score", src.get("score", 0.0))

                        mod_badge = f'<span class="badge badge-{mod}">{mod.upper()}</span>'
                        st.markdown(f"**Source {idx}:** {mod_badge} `{s_file}` | Ref: {page} | Score: `{score:.2f}`", unsafe_allow_html=True)
                        st.text_area(f"Chunk text #{idx}", value=src.get("text", ""), height=100, disabled=True, key=f"src_live_{idx}")

            # Append assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "sources": sources,
                "confidence": conf,
                "max_score": max_score,
                "cache_hit": cache_hit,
                "msg_id": msg_id + 1
            })
