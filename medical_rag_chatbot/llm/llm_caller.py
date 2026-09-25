import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    DEFAULT_TEMPERATURE,
    MAX_RESPONSE_TOKENS,
    OLLAMA_FALLBACK_MODEL,
    OLLAMA_VISION_MODEL,
    OLLAMA_BASE_URL
)

# Lazy-loaded Groq client
_groq_client = None

def get_groq_client():
    global _groq_client
    if _groq_client is None:
        key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
        if key and not key.startswith("your_") and not key.startswith("gsk_your_"):
            try:
                from groq import Groq
                _groq_client = Groq(api_key=key)
            except Exception as e:
                print(f"Warning: Failed to initialize Groq client: {e}")
                _groq_client = False
        else:
            _groq_client = False
    return _groq_client

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def call_groq(messages: List[Dict[str, str]]) -> str:
    """Call Groq API using LLaMA 3.1 8B Instant model."""
    client = get_groq_client()
    if not client:
        raise ValueError("Groq client is not configured or GROQ_API_KEY is missing/placeholder.")

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=MAX_RESPONSE_TOKENS,
        temperature=DEFAULT_TEMPERATURE
    )
    return response.choices[0].message.content

def call_ollama(messages: List[Dict[str, str]], model: str = OLLAMA_FALLBACK_MODEL) -> Optional[str]:
    """Call local Ollama server running Mistral 7B or similar."""
    try:
        url = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": DEFAULT_TEMPERATURE
            }
        }
        res = requests.post(url, json=payload, timeout=30)
        if res.status_code == 200:
            return res.json().get("message", {}).get("content", "")
    except Exception as e:
        # Ollama server might not be running locally
        pass
    return None

def call_llava(image_path: str | Path, question: str) -> Optional[str]:
    """Pass raw image bytes and question to local LLaVA vision model via Ollama."""
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            return None

        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode("utf-8")

        url = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": OLLAMA_VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": question,
                    "images": [encoded_image]
                }
            ],
            "stream": False
        }
        res = requests.post(url, json=payload, timeout=60)
        if res.status_code == 200:
            return res.json().get("message", {}).get("content", "")
    except Exception as e:
        print(f"LLaVA vision inference warning: {e}")
    return None

def offline_grounded_answer(messages: List[Dict[str, str]]) -> str:
    """
    Intelligent offline fallback synthesizer used when neither Groq API nor Ollama is reachable.
    Directly extracts and summarizes factual statements from the retrieved context.
    Ensures zero crashes in offline/demo environments while maintaining citation fidelity.
    """
    user_msg = messages[-1].get("content", "")
    context_part = ""
    question_part = ""

    if "Context from medical records, policies, and billing tables:" in user_msg:
        parts = user_msg.split("Question:")
        context_part = parts[0].replace("Context from medical records, policies, and billing tables:", "").strip()
        if len(parts) > 1:
            question_part = parts[1].strip()
    else:
        context_part = user_msg

    if not context_part or "No relevant context found" in context_part:
        return "I do not have enough information in the available documents to answer this."

    # Parse individual source chunks
    chunks = context_part.split("---")
    first_chunk = chunks[0].strip() if chunks else context_part

    # Generate a grounded citation-based response
    lines = [line.strip() for line in first_chunk.split("\n") if line.strip()]
    header = lines[0] if lines else "Retrieved Source"
    body = "\n".join(lines[1:6]) if len(lines) > 1 else first_chunk

    return (
        f"Based on the retrieved medical records ({header}):\n\n"
        f"{body}\n\n"
        f"*(Note: Generated via grounded local extraction. Set a valid GROQ_API_KEY in .env for full LLaMA 3.1 generation)*"
    )

def call_llm(messages: List[Dict[str, str]]) -> str:
    """
    Hierarchical LLM dispatcher:
    1. Primary: Groq Cloud API (LLaMA 3.1 8B Instant)
    2. Fallback: Local Ollama (Mistral 7B)
    3. Safe Fallback: Offline Grounded Context Extractor
    """
    # 1. Try Groq
    try:
        if get_groq_client():
            return call_groq(messages)
    except Exception as e:
        print(f"Groq API call failed ({e}). Attempting Ollama fallback...")

    # 2. Try Ollama local
    ollama_resp = call_ollama(messages)
    if ollama_resp and ollama_resp.strip():
        return ollama_resp

    # 3. Safe grounded offline fallback
    return offline_grounded_answer(messages)
