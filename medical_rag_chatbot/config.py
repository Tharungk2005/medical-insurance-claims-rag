import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# Document Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Retrieval Configuration
TOP_K_DENSE = 10
TOP_K_RERANK = 3
RERANK_THRESHOLD = 0.3
RRF_K = 60

# Caching Configuration
CACHE_SIMILARITY_THRESHOLD = 0.95
CACHE_FILE = BASE_DIR / ".cache" / "query_cache.json"
CACHE_TTL_SECONDS = 86400  # 24 hours

# AI Models
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
GROQ_MODEL = "llama-3.1-8b-instant"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_FALLBACK_MODEL = "mistral"
OLLAMA_VISION_MODEL = "llava"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Storage Paths
CHROMA_PATH = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "claims_rag"
BM25_INDEX_PATH = BASE_DIR / "data" / "processed" / "bm25_index.pkl"

# Raw & Processed Data Paths
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_RAW_PDFS = DATA_RAW_DIR / "pdfs"
DATA_RAW_IMAGES = DATA_RAW_DIR / "images"
DATA_RAW_TABLES = DATA_RAW_DIR / "tables"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_PROCESSED_IMAGES = DATA_PROCESSED_DIR / "images"

# LLM Context Limits
MAX_PROMPT_TOKENS = 3800
DEFAULT_TEMPERATURE = 0.2
MAX_RESPONSE_TOKENS = 1024

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "medical-rag-chatbot")
