import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent / "medical_rag_chatbot"
VENV_PYTHON = BASE_DIR / "venv" / "Scripts" / "python.exe"
MAIN_APP = BASE_DIR / "app" / "main.py"

def main():
    print("=" * 60)
    print("🏥 Starting Medical Insurance Claims Multimodal RAG Assistant")
    print("=" * 60)
    if not VENV_PYTHON.exists():
        print(f"Error: Virtual environment python not found at {VENV_PYTHON}")
        sys.exit(1)

    cmd = [
        str(VENV_PYTHON),
        "-m",
        "streamlit",
        "run",
        str(MAIN_APP),
        "--server.port=8501",
        "--server.headless=false",
        "--browser.gatherUsageStats=false"
    ]
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(BASE_DIR))

if __name__ == "__main__":
    main()
