"""
Application configuration.
"""
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import Field
import os


class Settings(BaseSettings):
    """Application settings."""

    # Paths.
    BASE_DIR: Path = Path(__file__).parent.parent
    OUTPUT_DIR: Path = BASE_DIR / "output"
    TEMP_DIR: Path = BASE_DIR / "temp"
    LOGS_DIR: Path = BASE_DIR / "logs"
    CACHE_DIR: Path = BASE_DIR / "cache"
    
    # API keys.
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    
    # Whisper configuration.
    WHISPER_MODEL_DIR: Path = BASE_DIR / "models" / "whisper"
    WHISPER_DEVICE: str = "cpu"  # "cpu", "cuda", or "mps"
    # Note: on Apple Silicon we default to CPU (MPS has issues with Whisper).
    # Performance on M1 CPU: whisper_base 0.06x, whisper_small 0.19x.
    
    # NLLB configuration.
    NLLB_MODEL_NAME: str = "facebook/nllb-200-distilled-1.3B"
    NLLB_MODEL_DIR: Path = BASE_DIR / "models" / "nllb"
    # Performance on M1 CPU: ~0.25x (better quality, ~2× slower).
    
    # Supported languages.
    SUPPORTED_LANGUAGES: list[str] = ["ru", "en"]
    
    # Logging configuration.
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    # Document formatting.
    DEFAULT_FONT: str = "Arial"  # Typeface with Cyrillic support.
    DEFAULT_FONT_SIZE: int = 11
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure all required directories exist.
        self.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        self.TEMP_DIR.mkdir(exist_ok=True, parents=True)
        self.LOGS_DIR.mkdir(exist_ok=True, parents=True)
        self.CACHE_DIR.mkdir(exist_ok=True, parents=True)
        self.WHISPER_MODEL_DIR.mkdir(exist_ok=True, parents=True)
        self.NLLB_MODEL_DIR.mkdir(exist_ok=True, parents=True)
        
        # Force CPU on Apple Silicon because Whisper struggles with MPS kernels.
        if os.uname().machine == 'arm64':
            self.WHISPER_DEVICE = "cpu"
            import logging
            logging.getLogger("youtube_transcriber").info(
            "Apple Silicon detected: Using CPU (MPS has compatibility issues with Whisper)"
        )


# Global settings instance.
settings = Settings()


class TranscribeOptions:
    """Available transcription backends."""
    WHISPER_BASE = "whisper_base"
    WHISPER_SMALL = "whisper_small"
    WHISPER_MEDIUM = "whisper_medium"
    WHISPER_OPENAI_API = "whisper_openai_api"

    ALL = [WHISPER_BASE, WHISPER_SMALL, WHISPER_MEDIUM, WHISPER_OPENAI_API]


class TranslateOptions:
    """Available translation backends."""
    NLLB = "NLLB"
    OPENAI_API = "openai_api"

    ALL = [NLLB, OPENAI_API]


class RefineOptions:
    """Available text refinement backends."""
    OLLAMA = "ollama"
    OPENAI_API = "openai_api"

    ALL = [OLLAMA, OPENAI_API]


class SummarizeOptions:
    """Available summarization backends."""
    OLLAMA = "ollama"
    OPENAI_API = "openai_api"

    ALL = [OLLAMA, OPENAI_API]
