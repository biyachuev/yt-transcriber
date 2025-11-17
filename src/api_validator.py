"""
API key validation module.

This module provides functions to validate API keys early in the pipeline
by making test API calls, preventing wasted processing time.
"""
from typing import Literal
from .logger import logger
from .config import settings


def validate_openai_api_key() -> bool:
    """
    Validate OpenAI API key by making a test API call.

    Returns:
        True if the API key is valid, False otherwise.
    """
    if not settings.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not set in environment")
        logger.error("Set it in .env file: OPENAI_API_KEY=your-key-here")
        return False

    try:
        import openai
        from openai import OpenAI

        # Create client with the API key
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        # Make a minimal test API call to validate the key
        # Using the models list endpoint as it's lightweight
        logger.info("Validating OpenAI API key...")
        client.models.list()

        logger.info("✓ OpenAI API key is valid")
        return True

    except openai.AuthenticationError as e:
        logger.error("❌ Invalid OpenAI API key")
        logger.error("Error: %s", str(e))
        logger.error("Please check your OPENAI_API_KEY in .env file")
        return False

    except openai.APIConnectionError as e:
        logger.error("❌ Cannot connect to OpenAI API")
        logger.error("Error: %s", str(e))
        logger.error("Please check your internet connection")
        return False

    except ImportError:
        logger.error("OpenAI library not installed")
        logger.error("Install it with: pip install openai>=1.6.0")
        return False

    except Exception as e:
        logger.error("❌ Error validating OpenAI API key: %s", str(e))
        return False


def validate_api_keys_for_operation(
    transcribe_method: str | None = None,
    translate_methods: list[str] | None = None,
    refine_backend: str | None = None,
    summarize_backend: str | None = None,
) -> bool:
    """
    Validate all API keys required for the specified operations.

    Args:
        transcribe_method: Transcription method (e.g., 'whisper_openai_api').
        translate_methods: List of translation methods (e.g., ['openai_api']).
        refine_backend: Refinement backend (e.g., 'openai_api').
        summarize_backend: Summarization backend (e.g., 'openai_api').

    Returns:
        True if all required API keys are valid, False otherwise.
    """
    needs_openai = False

    # Check if OpenAI is needed for transcription
    if transcribe_method == "whisper_openai_api":
        needs_openai = True

    # Check if OpenAI is needed for translation
    if translate_methods:
        for method in translate_methods:
            method_normalized = method.lower().replace('-', '_')
            if method_normalized == "openai_api":
                needs_openai = True
                break

    # Check if OpenAI is needed for refinement
    if refine_backend == "openai_api":
        needs_openai = True

    # Check if OpenAI is needed for summarization
    if summarize_backend == "openai_api":
        needs_openai = True

    # Validate OpenAI key if needed
    if needs_openai:
        if not validate_openai_api_key():
            return False

    return True
