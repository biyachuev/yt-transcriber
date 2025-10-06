"""
Модуль для транскрибирования аудио
"""
from pathlib import Path
from typing import List, Dict, Optional
import whisper
from tqdm import tqdm
import torch

from src.config import settings, TranscribeOptions
from src.logger import logger
from src.utils import format_timestamp, estimate_processing_time


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
        elif self.method == TranscribeOptions.WHISPER_MEDIUM:
            model_name = "medium"
        else:
            raise ValueError(f"Неподдерживаемый метод: {self.method}")
        
        # Пытаемся загрузить на выбранное устройство, при ошибке SparseMPS переключаемся на CPU
        try:
            self.model = whisper.load_model(
                model_name,
                device=self.device,
                download_root=str(settings.WHISPER_MODEL_DIR)
            )
        except NotImplementedError as e:
            message = str(e)
            if "SparseMPS" in message or "_sparse_coo_tensor" in message:
                logger.warning("На MPS отсутствует поддержка требуемой операции. Переключаемся на CPU...")
                self.device = "cpu"
                self.model = whisper.load_model(
                    model_name,
                    device=self.device,
                    download_root=str(settings.WHISPER_MODEL_DIR)
                )
            else:
                raise
        except RuntimeError as e:
            # Подстраховка на случай других ошибок MPS
            message = str(e)
            if "MPS" in message and ("sparse" in message.lower() or "_sparse_coo_tensor" in message):
                logger.warning("Возникла ошибка MPS при загрузке модели. Переключаемся на CPU...")
                self.device = "cpu"
                self.model = whisper.load_model(
                    model_name,
                    device=self.device,
                    download_root=str(settings.WHISPER_MODEL_DIR)
                )
            else:
                raise
        
        logger.info("Модель загружена успешно")
    
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        with_speakers: bool = False,
        initial_prompt: Optional[str] = None
    ) -> List[TranscriptionSegment]:
        """
        Транскрибирование аудиофайла

        Args:
            audio_path: Путь к аудиофайлу
            language: Код языка ('ru' или 'en'), None для автоопределения
            with_speakers: Выполнить ли speaker diarization
            initial_prompt: Промпт для Whisper (помогает распознавать имена, термины)

        Returns:
            Список сегментов транскрипции
        """
        logger.info(f"Начало транскрибирования: {audio_path.name}")

        if initial_prompt:
            logger.info(f"Используется промпт для улучшения качества: {initial_prompt}")
        
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
            logger.info(f"Оценочное время обработки: {estimate_processing_time(duration, 'transcribe', self.method)}")
        except Exception as e:
            logger.debug(f"Не удалось определить длительность: {e}")
        
        # Параметры транскрибирования
        transcribe_options = {
            'language': language,
            'task': 'transcribe',
            'verbose': False,
            # Включаем fp16 только на CUDA. На CPU/MPS оставляем fp32 для стабильности
            'fp16': True if self.device == "cuda" else False,
        }

        # Добавляем промпт если он передан
        if initial_prompt:
            transcribe_options['initial_prompt'] = initial_prompt
        
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

    def update_segments_from_text(self, segments: List[TranscriptionSegment], refined_text: str) -> List[TranscriptionSegment]:
        """
        Обновление сегментов с улучшенным текстом

        Разбивает улучшенный текст обратно на сегменты, сохраняя временные метки

        Args:
            segments: Исходные сегменты с временными метками
            refined_text: Улучшенный текст

        Returns:
            Обновленные сегменты
        """
        # Разбиваем улучшенный текст на абзацы
        paragraphs = [p.strip() for p in refined_text.split('\n\n') if p.strip()]

        # Если количество абзацев совпадает с сегментами - просто заменяем текст
        if len(paragraphs) == len(segments):
            updated_segments = []
            for seg, new_text in zip(segments, paragraphs):
                updated_seg = TranscriptionSegment(
                    start=seg.start,
                    end=seg.end,
                    text=new_text,
                    speaker=seg.speaker
                )
                updated_segments.append(updated_seg)
            return updated_segments

        # Иначе пропорционально распределяем временные метки
        logger.warning(f"Количество абзацев ({len(paragraphs)}) не совпадает с сегментами ({len(segments)})")
        logger.info("Создание новых сегментов на основе улучшенного текста...")

        if not segments:
            return []

        total_duration = segments[-1].end - segments[0].start
        segment_duration = total_duration / len(paragraphs) if paragraphs else 0

        updated_segments = []
        start_time = segments[0].start

        for i, paragraph in enumerate(paragraphs):
            end_time = start_time + segment_duration if i < len(paragraphs) - 1 else segments[-1].end

            updated_seg = TranscriptionSegment(
                start=start_time,
                end=end_time,
                text=paragraph,
                speaker=None
            )
            updated_segments.append(updated_seg)
            start_time = end_time

        return updated_segments
    
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