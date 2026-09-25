# 🏥 Medical Insurance Claims Processing — Multimodal RAG Chatbot

An enterprise-grade, fully free Multimodal Retrieval-Augmented Generation (RAG) chatbot designed to automate medical insurance claim inquiries. The system seamlessly ingests **PDFs** (policy documents, EOBs, claim forms), **Images** (prescriptions, MRI/X-ray scan reports), and **Tables** (billing codes, ICD-10 charges), performs **PII anonymization**, indexes data into **ChromaDB** with **BAAI/bge-small-en-v1.5** embeddings and **BM25**, uses **Reciprocal Rank Fusion (RRF)** with a **Cross-Encoder Reranker**, and delivers grounded answers via **LLaMA 3.1 8B** (Groq) with an interactive **Streamlit UI**.

---

## 🌟 Key Features

1. **Multimodal Data Ingestion**:
   - **PDFs**: Text and layout parsing via `pdfplumber` + embedded image extraction via `PyMuPDF (fitz)`.
   - **Tables**: Row-level structured chunking via `pandas` with column header context prepending.
   - **Images**: OCR text extraction via `pytesseract` + vision captioning via `Salesforce/blip-image-captioning-base`.
2. **HIPAA-Compliant PII Masking**:
   - Powered by Microsoft `presidio-analyzer` and `presidio-anonymizer` to mask patient names, SSNs, phone numbers, and dates before vector storage.
3. **Advanced Hybrid Retrieval Pipeline**:
   - **Dense Semantic Retrieval**: `BAAI/bge-small-en-v1.5` normalized embeddings (384-dim) stored locally in persistent `ChromaDB`.
   - **Sparse Keyword Retrieval**: `rank_bm25` (BM25Okapi) scoring over all document chunks.
   - **Hybrid Fusion**: Reciprocal Rank Fusion (`RRF`, $k=60$) combining semantic and keyword ranks.
   - **Cross-Encoder Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` with confidence calibration and insufficient context fallback.
4. **Dual-Tier Caching Engine**:
   - **Exact Cache**: MD5 query hashing with 24-hour TTL.
   - **Semantic Cache**: Cosine similarity caching ($\ge 0.95$ threshold) to eliminate redundant LLM calls.
5. **Prompt Engineering & Token Budgeting**:
   - Modality-aware provenance tags (`[From medical scan]`, `[From billing table]`, `[From policy PDF]`).
   - Dynamic few-shot medical claims demonstrations and Chain-of-Thought (`CoT`) reasoning toggle.
   - Context budget limiter using `tiktoken` (`cl100k_base`) with FIFO history eviction.
6. **Free-Tier LLM Architecture**:
   - Primary: **LLaMA 3.1 8B Instant** via Groq API (~500 tokens/sec).
   - Local Fallback: **Mistral 7B** and **LLaVA** vision via Ollama.
   - Offline Fallback: Deterministic citation extractor for demo and isolated environments.
7. **Comprehensive Evaluation Suite (Phases 11–13)**:
   - **30-Item Ground-Truth Golden Dataset** spanning all modalities.
   - Metrics: **Hit Rate@3**, **MRR**, **Precision@3**, and **RAGAS** context precision & recall.
   - **LLM-as-a-Judge** scoring for prompt variants (Zero-shot vs. Few-shot vs. CoT).
   - **Automated Regression Testing** checking against baseline scores.
   - Tracing hooks for **LangSmith**, **Arize Phoenix**, and **TruLens**.

---

## 🏗️ Technology Stack

| Component | Library / Model | Cost |
| :--- | :--- | :--- |
| **PDF Extraction** | `pdfplumber`, `PyMuPDF (pymupdf)` | Free |
| **Table Processing** | `pandas`, `openpyxl` | Free |
| **OCR & Vision** | `pytesseract`, `Pillow`, `Salesforce/blip-image-captioning-base` | Free (Local HuggingFace) |
| **PII Anonymization** | Microsoft `presidio-analyzer`, `presidio-anonymizer` | Free |
| **Text Chunking** | `langchain-text-splitters` (RecursiveCharacterTextSplitter) | Free |
| **Embedding Model** | `BAAI/bge-small-en-v1.5` (via `sentence-transformers`) | Free (Local) |
| **Vector Database** | `ChromaDB` (Persistent SQLite/HNSW) | Free (Local) |
| **Sparse Retrieval** | `rank-bm25` (BM25Okapi) | Free |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Free (Local) |
| **Caching Engine** | MD5 Exact Cache + Cosine Semantic Cache | Free |
| **Primary LLM** | Groq Cloud API (`llama-3.1-8b-instant`) | Free Tier |
| **Fallback LLMs** | Ollama (`mistral:7b`, `llava`) | Free (Local) |
| **Chatbot UI** | Streamlit | Free |
| **Evaluation** | `ragas`, custom Hit Rate/MRR engine, LLM-as-a-Judge | Free |
| **Observability** | `langsmith`, `arize-phoenix`, `trulens-eval` | Free Tier / Local |

---

## 📁 Repository Structure

```text
medical_rag_chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py                   # Streamlit chatbot application
│   └── query_engine.py           # Unified RAG execution engine
├── config.py                     # Global configurations, models, paths, and thresholds
├── data/
│   ├── raw/
│   │   ├── pdfs/                 # Policy documents, EOBs, claim forms
│   │   ├── images/               # Prescription scans, radiology reports
│   │   └── tables/               # Billing code CSVs and fee schedules
│   └── processed/
│       ├── bm25_index.pkl        # Serialized BM25 keyword index
│       └── images/               # Extracted embedded PDF images
├── evaluation/
│   ├── __init__.py
│   ├── baseline_scores.json      # Benchmark baseline for regression tracking
│   ├── eval_e2e.py               # End-to-end tracing and telemetry (TruLens / Phoenix)
│   ├── eval_prompts.py           # Prompt A/B testing & LLM-as-Judge scoring
│   ├── eval_retrieval.py         # Hit Rate, MRR, Precision, RAGAS context eval
│   ├── golden_dataset.json       # 30 ground-truth multimodal QA pairs
│   ├── prompt_eval_results.json  # Answers generated across prompt variants
│   ├── prompt_scores.csv         # Summary comparison table for prompts
│   ├── regression_check.py       # Automated regression test (< 5% delta gate)
│   └── retrieval_scores.csv      # Dense vs Hybrid vs Reranker retrieval scores
├── generate_sample_data.py       # Script creating realistic multimodal sample data
├── ingestion/
│   ├── __init__.py
│   ├── chunker.py                # Modality-aware text splitter
│   ├── embedder.py               # BGE-small embedding encoder & CLIP embedder
│   ├── mask_pii.py               # Presidio PII masking engine
│   ├── parse_images.py           # OCR + BLIP captioning pipeline
│   ├── parse_pdf.py              # PyMuPDF + pdfplumber layout parser
│   ├── parse_tables.py           # Structured CSV & PDF table normalizer
│   ├── pipeline.py               # Unified ingestion & store pipeline
│   └── vector_store.py           # ChromaDB persistent collection manager
├── llm/
│   ├── __init__.py
│   ├── llm_caller.py             # Groq, Ollama, and offline fallback caller
│   └── prompts.py                # System prompt, few-shots, CoT, token budgeting
├── retrieval/
│   ├── __init__.py
│   ├── cache.py                  # Exact MD5 and semantic cosine caching
│   ├── dense_retriever.py        # Dense vector search with similarity scores
│   ├── hybrid_retriever.py       # Reciprocal Rank Fusion (RRF) search
│   ├── reranker.py               # Cross-Encoder MiniLM reranking
│   └── sparse_retriever.py       # BM25Okapi keyword index & search
├── test_setup.py                 # Environment and dependency verification script
├── requirements.txt              # Pinned Python package dependencies
├── .env.example                  # Environment variable configuration template
└── .gitignore                    # Git ignore file (safeguarding API keys)
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup

Ensure you have Python 3.10+ installed.

```bash
# Clone the repository and navigate into the project root
cd medical_rag_chatbot

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
HUGGINGFACE_TOKEN=hf_your_huggingface_token_here
LANGCHAIN_API_KEY=lsv2_your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=medical-rag-chatbot
OLLAMA_BASE_URL=http://localhost:11434
```

> **Note**: Even without a Groq API key or running Ollama instance, the system includes a built-in grounded offline fallback that extracts and presents citations directly from retrieved records without failing.

### 3. Verify Setup

Run the verification test:
```bash
python test_setup.py
```

### 4. Ingest and Index Documents

Generate sample multimodal records and run the ingestion pipeline:

```bash
# Generate sample documents
python generate_sample_data.py

# Ingest, mask PII, embed, and store into ChromaDB & BM25
python ingestion/pipeline.py
```

---

## 💻 Running the Streamlit Chatbot

From the workspace root or project directory, start the Streamlit application:

```bash
# Method 1: Using the root launcher
python run_app.py

# Method 2: Direct Streamlit command
streamlit run medical_rag_chatbot/app/main.py
```

Open your browser at **`http://localhost:8501`**.

### UI Features:
- **Interactive Chat**: Conversational bubbles with full chat memory.
- **Source Inspection**: Expanders displaying the exact document name, page number, modality badge (`PDF`, `IMAGE`, `TABLE`), and raw text chunk.
- **Confidence Badges**: Real-time cross-encoder score badges (`High` $\ge 0.70$, `Medium` $0.40–0.70$, `Low` $< 0.40$).
- **Live Document Ingestion**: Upload new PDFs, prescription images, or billing CSVs directly through the sidebar with instant indexing.
- **Filter Controls**: Scope searches by claim ID (e.g. `CLM-2024-001`), document type, or modality.
- **Cache Management**: Button to flush exact and semantic caches.

---

## 📊 Evaluation & Observability Suite

### Phase 11: Retrieval Evaluation
Evaluates dense, hybrid RRF, and cross-encoder reranked retrieval against the 30 golden Q&A dataset:
```bash
python evaluation/eval_retrieval.py
```
Outputs:
- Generates `evaluation/retrieval_scores.csv`
- Evaluates **Hit Rate@3**, **MRR**, and **Precision@3**
- Evaluates **RAGAS context_precision** and **context_recall**
- Stores best configuration as `evaluation/baseline_scores.json`

### Phase 12: Prompting Strategy Evaluation
Performs A/B testing on prompt variants (Zero-shot vs. Few-shot vs. CoT) with LLM-as-a-Judge:
```bash
python evaluation/eval_prompts.py
```
Outputs:
- Generates `evaluation/prompt_eval_results.json` and `evaluation/prompt_scores.csv`
- Evaluates **Faithfulness**, **Answer Relevancy**, and **Judge Score (1–5)**
- Automatically selects and locks in the winning production prompt

### Phase 13: End-to-End Tracing & Telemetry
Runs full query tracing with TruLens, Arize Phoenix, and LangSmith:
```bash
python evaluation/eval_e2e.py
```

### Phase 13.4: Regression Testing Gate
Compares current pipeline retrieval metrics against baseline to prevent performance degradation:
```bash
python evaluation/regression_check.py
```
Fails with exit code 1 if any core retrieval metric regresses by more than **5%**.

---

## 🔒 Security & Privacy

- **PII Masking**: All text passes through Microsoft Presidio before entering any vector database or cache.
- **Secret Protection**: All sensitive tokens reside strictly in `.env`, which is permanently excluded via `.gitignore`.
- **Hallucination Prevention**: Prompts enforce strict grounding to retrieved evidence with low model temperature ($0.2$) and fallback flags when reranker confidence is below $0.30$.
