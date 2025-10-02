"""
Модуль для транскрибирования аудио
"""
from pathlib import Path
from typing import List, Dict, Optional
import whisper
from tqdm import tqdm
import torch

from .config import settings, TranscribeOptions
from .logger import logger
from .utils import format_timestamp, estimate_processing_time


class TranscriptionSegment:
    """Класс для представления сегмента транскрипции"""
    
    def __init__(self, start: float, end: float, text: str, speaker: Optional[str] = None):
        self.start = start
        self.end = end
        self.text = text.strip()
        self.speaker = speaker
    
    def __repr__(self):
        speaker_prefix = f"[{self.speaker}] " if self.speaker else ""
        return f"[{format_timestamp(self.start)}] {speaker_prefix}{self.text}"
    
    def to_dict(self) -> dict:
        """Преобразование в словарь"""
        return {
            'start': self.start,
            'end': self.end,
            'text': self.text,
            'timestamp': format_timestamp(self.start),
            'speaker': self.speaker
        }


class Transcriber:
    """Класс для транскрибирования аудио"""
    
    def __init__(self, method: str = TranscribeOptions.WHISPER_BASE):
        self.method = method
        self.model = None
        self.device = self._get_device()
        logger.info(f"Используется устройство: {self.device}")
    
    def _get_device(self) -> str:
        """Определение доступного устройства для вычислений"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        """Загрузка модели Whisper"""
        if self.model is not None:
            return
        
        logger.info(f"Загрузка модели Whisper ({self.method})...")
        
        if self.method == TranscribeOptions.WHISPER_BASE:
            model_name = "base"
        elif self.method == TranscribeOptions.WHISPER_SMALL:
            model_name = "small"
        else:
            raise ValueError(f"Неподдерживаемый метод: {self.method}")
        
        self.model = whisper.load_model(
            model_name,
            device=self.device,
            download_root=str(settings.WHISPER_MODEL_DIR)
        )
        logger.info("Модель загружена успешно")
    
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        with_speakers: bool = False
    ) -> List[TranscriptionSegment]:
        """
        Транскрибирование аудиофайла
        
        Args:
            audio_path: Путь к аудиофайлу
            language: Код языка ('ru' или 'en'), None для автоопределения
            with_speakers: Выполнить ли speaker diarization
            
        Returns:
            Список сегментов транскрипции
        """
        logger.info(f"Начало транскрибирования: {audio_path.name}")
        
        if with_speakers:
            logger.warning("Speaker diarization будет доступно в расширенной версии")
        
        # Загружаем модель
        self._load_model()
        
        # Получаем информацию о длительности для оценки времени
        import subprocess
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 
                 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                 str(audio_path)],
                capture_output=True,
                text=True
            )
            duration = float(result.stdout.strip())
            logger.info(f"Оценочное время обработки: {estimate_processing_time(duration, 'transcribe')}")
        except Exception as e:
            logger.debug(f"Не удалось определить длительность: {e}")
        
        # Параметры транскрибирования
        transcribe_options = {
            'language': language,
            'task': 'transcribe',
            'verbose': False,
            'fp16': False if self.device == "cpu" else True,
        }
        
        logger.info("Транскрибирование в процессе...")
        
        # Выполняем транскрибирование
        result = self.model.transcribe(
            str(audio_path),
            **transcribe_options
        )
        
        # Определяем язык
        detected_language = result.get('language', 'unknown')
        logger.info(f"Обнаруженный язык: {detected_language}")
        
        # Преобразуем результаты в наш формат
        segments = []
        for segment in tqdm(result['segments'], desc="Обработка сегментов"):
            segments.append(
                TranscriptionSegment(
                    start=segment['start'],
                    end=segment['end'],
                    text=segment['text']
                )
            )
        
        logger.info(f"Транскрибирование завершено. Создано {len(segments)} сегментов")
        
        return segments
    
    def segments_to_text(self, segments: List[TranscriptionSegment]) -> str:
        """
        Преобразование сегментов в текст
        
        Args:
            segments: Список сегментов
            
        Returns:
            Текст транскрипции
        """
        return '\n\n'.join([seg.text for seg in segments])
    
    def segments_to_text_with_timestamps(
        self,
        segments: List[TranscriptionSegment],
        with_speakers: bool = False
    ) -> str:
        """
        Преобразование сегментов в текст с таймкодами
        
        Args:
            segments: Список сегментов
            with_speakers: Включить ли информацию о спикерах
            
        Returns:
            Текст с таймкодами
        """
        lines = []
        for seg in segments:
            timestamp = format_timestamp(seg.start)
            speaker = f"[{seg.speaker}] " if with_speakers and seg.speaker else ""
            lines.append(f"[{timestamp}] {speaker}{seg.text}")
        
        return '\n\n'.join(lines)