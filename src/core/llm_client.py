"""
Centralized LLM client interface for the Dynamic Causal Character Graphs system.

Handles:
- API key configuration (GEMINI_API_KEY - optional)
- Free Hugging Face Inference API (no key required)
- Model selection
- Structured generation (JSON output)
- Text generation
"""

import os
import json
import logging
import requests
from typing import Any, Dict, Optional, Type, TypeVar, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL_NAME = "gemini-2.5-flash"

# Track which backend is active
_active_backend = None  # "gemini", "huggingface", or None

# Embedding configuration


def _get_gemini_embedding_model() -> str:
    return os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2-preview")




def get_embedding_provider() -> Optional[str]:
    """Return the active embedding provider name, if configured."""
    return "gemini"


def is_embedding_available(provider: Optional[str] = None) -> bool:
    """Check if the configured embedding backend is ready to use."""
    provider = provider or get_embedding_provider()
    if provider == "gemini":
        return get_api_key() is not None
    return False


def get_embedding(text: str, provider: Optional[str] = None) -> Optional[List[float]]:
    """Fetch an embedding vector for the given text using the configured provider."""
    provider = provider or get_embedding_provider()
    if provider == "gemini":
        api_key = get_api_key()
        if not api_key:
            return None
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)  # type: ignore[attr-defined, reportPrivateImportUsage]
            response = genai.embed_content(  # type: ignore[attr-defined, reportPrivateImportUsage]
                model=_get_gemini_embedding_model(),
                content=text,
            )
            if isinstance(response, dict):
                return response.get("embedding")
            embedding = getattr(response, "embedding", None)
            if embedding is not None:
                return list(embedding)
        except Exception as exc:
            logger.warning(f"Gemini embedding failed: {exc}")
        return None

    return None

def get_api_key() -> Optional[str]:
    """Retrieve the API key from environment variables."""
    return os.getenv("GEMINI_API_KEY")

def configure_client() -> bool:
    """
    Configure the LLM client. required for Gemini Embeddings 2.
    """
    global _active_backend
    
    api_key = get_api_key()
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)  # type: ignore[attr-defined, reportPrivateImportUsage]
            _active_backend = "gemini"
            return True
        except Exception as e:
            logger.warning(f"Failed to configure Gemini client: {e}")
            raise e
    
    return False

def is_llm_available() -> bool:
    """Check if any LLM backend is configured and available."""
    return get_api_key() is not None

def generate_text(prompt: str) -> Optional[str]:
    """Generate text using the active LLM backend."""
    if _active_backend == "gemini":
        try:
            import google.generativeai as genai
            model = genai.GenerativeModel(DEFAULT_MODEL_NAME)
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return None
    return None
