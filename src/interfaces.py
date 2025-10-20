"""
Protocol interfaces for external services.

This module defines abstract interfaces for external dependencies,
enabling dependency injection and easier testing.
"""
from typing import Protocol, List, Dict, Any, Optional
from pathlib import Path


class TranscriptionServiceProtocol(Protocol):
    """Protocol for transcription services (Whisper, OpenAI API, etc.)"""

    def transcribe(
        self,
        audio_path: Path | str,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None
    ) -> List[Any]:
        """
        Transcribe audio file to text segments.

        Args:
            audio_path: Path to audio file
            language: Language code (optional)
            initial_prompt: Prompt hint for transcription (optional)

        Returns:
            List of TranscriptionSegment objects
        """
        ...


class TranslationServiceProtocol(Protocol):
    """Protocol for translation services (NLLB, OpenAI, etc.)"""

    def translate_segments(
        self,
        segments: List[Any],
        source_lang: str,
        target_lang: str
    ) -> List[Any]:
        """
        Translate transcription segments.

        Args:
            segments: List of TranscriptionSegment objects
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            List of translated TranscriptionSegment objects
        """
        ...

    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """
        Translate plain text.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Translated text
        """
        ...


class LLMServiceProtocol(Protocol):
    """Protocol for LLM services (Ollama, OpenAI API)"""

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text completion from prompt.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        ...


class CacheServiceProtocol(Protocol):
    """Protocol for caching services"""

    def get(self, key: str) -> Optional[Any]:
        """Get cached value by key"""
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cache value with optional TTL"""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        ...


class CostTrackerProtocol(Protocol):
    """Protocol for API cost tracking"""

    def add_transcription(self, audio_duration_seconds: float) -> None:
        """Track transcription cost"""
        ...

    def add_translation(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Track translation cost"""
        ...

    def add_refinement(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Track refinement cost"""
        ...

    def add_summarization(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Track summarization cost"""
        ...

    def print_summary(self) -> None:
        """Print cost summary"""
        ...

    @property
    def total_cost(self) -> float:
        """Get total cost"""
        ...

    @property
    def total_tokens(self) -> int:
        """Get total tokens"""
        ...
