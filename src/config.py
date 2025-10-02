"""
Конфигурация приложения
"""
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import Field
import os


class Settings(BaseSettings):
    """Настройки приложения"""
    
    # Пути
    BASE_DIR: Path = Path(__file__).parent.parent
    OUTPUT_DIR: Path = BASE_DIR / "output"
    TEMP_DIR: Path = BASE_DIR / "temp"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # API ключи
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    
    # Whisper настройки
    WHISPER_MODEL_DIR: Path = BASE_DIR / "models" / "whisper"
    WHISPER_DEVICE: str = "cpu"  # "cpu" или "cuda" или "mps" (для M1)
    
    # NLLB настройки
    NLLB_MODEL_NAME: str = "facebook/nllb-200-distilled-600M"
    NLLB_MODEL_DIR: Path = BASE_DIR / "models" / "nllb"
    
    # Языки
    SUPPORTED_LANGUAGES: list[str] = ["ru", "en"]
    
    # Логирование
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    # Форматирование документов
    DEFAULT_FONT: str = "Arial"  # Шрифт с кириллицей
    DEFAULT_FONT_SIZE: int = 11
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Создаем директории если их нет
        self.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        self.TEMP_DIR.mkdir(exist_ok=True, parents=True)
        self.LOGS_DIR.mkdir(exist_ok=True, parents=True)
        self.WHISPER_MODEL_DIR.mkdir(exist_ok=True, parents=True)
        self.NLLB_MODEL_DIR.mkdir(exist_ok=True, parents=True)
        
        #  Для M1/M2 используем CPU из-за проблем с MPS в Whisper
        if os.uname().machine == 'arm64':
            self.WHISPER_DEVICE = "cpu"
            import logging
            logging.getLogger("youtube_transcriber").info(
            "Apple Silicon detected: Using CPU (MPS has compatibility issues with Whisper)"
        )


# Глобальный экземпляр настроек
settings = Settings()


class TranscribeOptions:
    """Доступные методы транскрибирования"""
    WHISPER_BASE = "whisper_base"
    WHISPER_SMALL = "whisper_small"
    WHISPER_OPENAI_API = "whisper_openai_api"
    
    ALL = [WHISPER_BASE, WHISPER_SMALL, WHISPER_OPENAI_API]


class TranslateOptions:
    """Доступные методы перевода"""
    NLLB = "NLLB"
    OPENAI_API = "openai_api"
    
    ALL = [NLLB, OPENAI_API]