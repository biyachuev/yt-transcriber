"""
Модуль для перевода текста
"""
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm
import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

from .config import settings, TranslateOptions
from .logger import logger
from .utils import chunk_text, detect_language


class Translator:
    """Класс для перевода текста"""
    
    def __init__(self, method: str = TranslateOptions.NLLB):
        self.method = method
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = self._get_device()
        logger.info(f"Переводчик использует устройство: {self.device}")
    
    def _get_device(self) -> str:
        """Определение доступного устройства"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_nllb_model(self):
        """Загрузка модели NLLB"""
        if self.model is not None:
            return
        
        logger.info("Загрузка модели NLLB... Это может занять несколько минут при первом запуске")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.NLLB_MODEL_NAME,
            cache_dir=str(settings.NLLB_MODEL_DIR)
        )
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            settings.NLLB_MODEL_NAME,
            cache_dir=str(settings.NLLB_MODEL_DIR)
        )
        
        # Перемещаем модель на нужное устройство
        if self.device != "cpu":
            self.model = self.model.to(self.device)
        
        logger.info("Модель NLLB загружена успешно")
    
    def _get_nllb_language_code(self, lang: str) -> str:
        """
        Преобразование кода языка в формат NLLB
        
        Args:
            lang: Код языка ('ru' или 'en')
            
        Returns:
            Код языка для NLLB
        """
        mapping = {
            'ru': 'rus_Cyrl',
            'en': 'eng_Latn'
        }
        return mapping.get(lang, 'eng_Latn')
    
    def translate_text(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: str = "ru"
    ) -> str:
        """
        Перевод текста
        
        Args:
            text: Текст для перевода
            source_lang: Исходный язык (None для автоопределения)
            target_lang: Целевой язык
            
        Returns:
            Переведенный текст
        """
        if not text.strip():
            return text
        
        # Определяем исходный язык если не задан
        if source_lang is None:
            source_lang = detect_language(text)
            logger.info(f"Определен исходный язык: {source_lang}")
        
        # Если исходный язык = целевому, возвращаем оригинал
        if source_lang == target_lang:
            logger.info("Исходный язык совпадает с целевым, перевод не требуется")
            return text
        
        logger.info(f"Начало перевода с {source_lang} на {target_lang}")
        
        if self.method == TranslateOptions.NLLB:
            return self._translate_with_nllb(text, source_lang, target_lang)
        elif self.method == TranslateOptions.OPENAI_API:
            return self._translate_with_openai(text, source_lang, target_lang)
        else:
            raise ValueError(f"Неподдерживаемый метод перевода: {self.method}")
    
    def _translate_with_nllb(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """
        Перевод с использованием NLLB
        
        Args:
            text: Текст для перевода
            source_lang: Исходный язык
            target_lang: Целевой язык
            
        Returns:
            Переведенный текст
        """
        self._load_nllb_model()
        
        # Разбиваем текст на чанки для обработки
        chunks = chunk_text(text, max_tokens=500)
        logger.info(f"Текст разбит на {len(chunks)} частей для перевода")
        
        translated_chunks = []
        
        src_lang_code = self._get_nllb_language_code(source_lang)
        tgt_lang_code = self._get_nllb_language_code(target_lang)
        
        for chunk in tqdm(chunks, desc="Перевод"):
            # Токенизация
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Перемещаем на устройство
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Устанавливаем язык источника
            self.tokenizer.src_lang = src_lang_code
            
            # Генерация перевода
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang_code],
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
            
            # Декодирование
            translated_text = self.tokenizer.batch_decode(
                translated_tokens,
                skip_special_tokens=True
            )[0]
            
            translated_chunks.append(translated_text)
        
        result = '\n\n'.join(translated_chunks)
        logger.info("Перевод завершен успешно")
        
        return result
    
    def _translate_with_openai(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """
        Перевод с использованием OpenAI API (для расширенной версии)
        
        Args:
            text: Текст для перевода
            source_lang: Исходный язык
            target_lang: Целевой язык
            
        Returns:
            Переведенный текст
        """
        logger.warning("OpenAI API перевод будет доступен в расширенной версии")
        
        # Заглушка для MVP
        return text
    
    def translate_segments(
        self,
        segments: List,
        source_lang: Optional[str] = None,
        target_lang: str = "ru"
    ) -> List:
        """
        Перевод списка сегментов транскрипции
        
        Args:
            segments: Список объектов TranscriptionSegment
            source_lang: Исходный язык
            target_lang: Целевой язык
            
        Returns:
            Список переведенных сегментов
        """
        from .transcriber import TranscriptionSegment
        
        # Собираем весь текст для перевода
        texts = [seg.text for seg in segments]
        full_text = '\n\n'.join(texts)
        
        # Переводим
        translated_text = self.translate_text(full_text, source_lang, target_lang)
        
        # Разбиваем обратно на сегменты
        translated_parts = translated_text.split('\n\n')
        
        # Создаем новые сегменты с переведенным текстом
        translated_segments = []
        for i, seg in enumerate(segments):
            translated_seg = TranscriptionSegment(
                start=seg.start,
                end=seg.end,
                text=translated_parts[i] if i < len(translated_parts) else seg.text,
                speaker=seg.speaker
            )
            translated_segments.append(translated_seg)
        
        return translated_segments