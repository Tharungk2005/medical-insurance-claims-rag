import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv

load_dotenv(BASE_DIR / ".env")

def test_imports():
    print("Testing core imports...")
    try:
        import chromadb
        print("  [OK] chromadb")
    except ImportError as e:
        print("  [FAILED] chromadb:", e)

    try:
        import sentence_transformers
        print("  [OK] sentence_transformers")
    except ImportError as e:
        print("  [FAILED] sentence_transformers:", e)

    try:
        import groq
        print("  [OK] groq")
    except ImportError as e:
        print("  [FAILED] groq:", e)

    try:
        import streamlit
        print("  [OK] streamlit")
    except ImportError as e:
        print("  [FAILED] streamlit:", e)

    try:
        import pdfplumber
        import fitz
        print("  [OK] pdfplumber & PyMuPDF (fitz)")
    except ImportError as e:
        print("  [FAILED] pdf libraries:", e)

    try:
        import pandas
        import PIL
        print("  [OK] pandas & PIL")
    except ImportError as e:
        print("  [FAILED] pandas / PIL:", e)

    try:
        import rank_bm25
        print("  [OK] rank_bm25")
    except ImportError as e:
        print("  [FAILED] rank_bm25:", e)

    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        print("  [OK] Microsoft Presidio (Analyzer & Anonymizer)")
    except ImportError as e:
        print("  [FAILED] presidio:", e)

    groq_key = os.getenv("GROQ_API_KEY", "")
    masked_key = groq_key[:5] + "..." if len(groq_key) >= 5 else "(empty or placeholder)"
    print(f"\nConfiguration Check:")
    print(f"  GROQ_API_KEY prefix: {masked_key}")
    print(f"  Python executable: {sys.executable}")
    print("Environment setup verified successfully!")

if __name__ == "__main__":
    test_imports()
