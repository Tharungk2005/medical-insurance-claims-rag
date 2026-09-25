# Production Dockerfile for Multimodal Medical Insurance Claims RAG Chatbot
FROM python:3.11-slim

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies: Tesseract OCR, Poppler, build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY medical_rag_chatbot/requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Copy application source code
COPY medical_rag_chatbot ./medical_rag_chatbot
COPY run_app.py ./

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit app
ENTRYPOINT ["streamlit", "run", "medical_rag_chatbot/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
