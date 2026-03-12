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
from typing import Any, Dict, Optional, Type, TypeVar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL_NAME = "gemini-2.0-flash-exp"

# Free Hugging Face models (no API key required)
HF_FREE_MODEL = "microsoft/DialoGPT-medium"
HF_FREE_API_URL = "https://api-inference.huggingface.co/models/"

# Track which backend is active
_active_backend = None  # "gemini", "huggingface", or None

def get_api_key() -> Optional[str]:
    """Retrieve the API key from environment variables."""
    return os.getenv("GEMINI_API_KEY")

def configure_client() -> bool:
    """
    Configure the LLM client.
    Returns True if any backend is available, False otherwise.
    Tries Gemini first (if key present), then falls back to rule-based.
    """
    global _active_backend
    
    api_key = get_api_key()
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            _active_backend = "gemini"
            return True
        except Exception as e:
            logger.warning(f"Failed to configure Gemini client: {e}")
    
    # No external API - use rule-based fallback
    _active_backend = None
    return False

def generate_text(
    prompt: str,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> Optional[str]:
    """
    Generate plain text response from the LLM.
    
    Returns None if the client is not configured or generation fails.
    Uses rule-based fallback when no API is available.
    """
    global _active_backend
    
    api_key = get_api_key()
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            return response.text
        except Exception as e:
            logger.warning(f"Gemini generation failed: {e}")
    
    # Return None to trigger rule-based fallback
    return None

def generate_structured(
    prompt: str,
    response_schema: Dict[str, Any],
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = 0.2
) -> Optional[Dict[str, Any]]:
    """
    Generate a JSON structured response using the model's JSON mode capabilities.
    
    The prompt should explicitly ask for JSON output matching the schema.
    """
    api_key = get_api_key()
    if not api_key:
        return None

    full_prompt = (
        f"{prompt}\n\n"
        f"You must respond with valid JSON matching this schema:\n"
        f"{json.dumps(response_schema, indent=2)}\n"
        f"Response:"
    )

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                response_mime_type="application/json"
            )
        )
        
        # Parse the JSON response
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response from LLM.")
            return None
            
    except Exception as e:
        logger.error(f"Structured generation failed: {e}")
        return None

def is_llm_available() -> bool:
    """Check if any LLM backend is configured and available."""
    return get_api_key() is not None
