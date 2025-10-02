"""
Тесты для модуля utils
"""
import pytest
from src.utils import (
    sanitize_filename,
    format_timestamp,
    detect_language,
    chunk_text,
    estimate_processing_time
)


class TestSanitizeFilename:
    """Тесты для функции sanitize_filename"""
    
    def test_removes_invalid_characters(self):
        """Проверка удаления недопустимых символов"""
        filename = 'test<>:"/\\|?*file.txt'
        result = sanitize_filename(filename)
        assert '<' not in result
        assert '>' not in result
        assert ':' not in result
        assert '"' not in result
    
    def test_replaces_spaces_with_underscores(self):
        """Проверка замены пробелов"""
        filename = 'my test file name'
        result = sanitize_filename(filename)
        assert ' ' not in result
        assert '_' in result
    
    def test_truncates_long_names(self):
        """Проверка обрезки длинных имен"""
        filename = 'a' * 300
        result = sanitize_filename(filename, max_length=200)
        assert len(result) <= 200
    
    def test_handles_empty_string(self):
        """Проверка обработки пустой строки"""
        result = sanitize_filename('')
        assert result == 'untitled'
    
    def test_cyrillic_characters_preserved(self):
        """Проверка сохранения кириллицы"""
        filename = 'Тестовый файл'
        result = sanitize_filename(filename)
        assert 'Тестовый' in result


class TestFormatTimestamp:
    """Тесты для функции format_timestamp"""
    
    def test_formats_seconds_only(self):
        """Проверка форматирования секунд"""
        assert format_timestamp(45) == "00:45"
    
    def test_formats_minutes_and_seconds(self):
        """Проверка форматирования минут и секунд"""
        assert format_timestamp(125) == "02:05"
    
    def test_formats_hours_minutes_seconds(self):
        """Проверка форматирования часов, минут и секунд"""
        assert format_timestamp(3665) == "01:01:05"
    
    def test_handles_zero(self):
        """Проверка обработки нуля"""
        assert format_timestamp(0) == "00:00"


class TestDetectLanguage:
    """Тесты для функции detect_language"""
    
    def test_detects_russian(self):
        """Проверка определения русского языка"""
        text = "Это тестовый текст на русском языке"
        assert detect_language(text) == "ru"
    
    def test_detects_english(self):
        """Проверка определения английского языка"""
        text = "This is a test text in English language"
        assert detect_language(text) == "en"
    
    def test_mixed_text_dominant_language(self):
        """Проверка смешанного текста"""
        # Больше русских символов
        text = "Это текст with some English words"
        assert detect_language(text) == "ru"
        
        # Больше английских символов
        text = "This is text с несколькими русскими словами"
        assert detect_language(text) == "en"


class TestChunkText:
    """Тесты для функции chunk_text"""
    
    def test_splits_long_text(self):
        """Проверка разбиения длинного текста"""
        text = ' '.join(['word'] * 3000)
        chunks = chunk_text(text, max_tokens=1000)
        assert len(chunks) > 1
        
        # Проверяем, что каждый чанк не превышает лимит
        for chunk in chunks:
            word_count = len(chunk.split())
            assert word_count <= 1000
    
    def test_preserves_paragraphs(self):
        """Проверка сохранения абзацев"""
        text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        chunks = chunk_text(text, max_tokens=100)
        
        # Все параграфы должны быть сохранены
        combined = '\n\n'.join(chunks)
        assert "Paragraph 1" in combined
        assert "Paragraph 2" in combined
        assert "Paragraph 3" in combined
    
    def test_short_text_returns_single_chunk(self):
        """Проверка короткого текста"""
        text = "Short text"
        chunks = chunk_text(text, max_tokens=1000)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestEstimateProcessingTime:
    """Тесты для функции estimate_processing_time"""
    
    def test_transcribe_estimation(self):
        """Проверка оценки времени транскрибирования"""
        result = estimate_processing_time(3600, operation="transcribe")
        assert "минут" in result
    
    def test_translate_estimation(self):
        """Проверка оценки времени перевода"""
        result = estimate_processing_time(3600, operation="translate")
        assert "минут" in result
    
    def test_short_duration(self):
        """Проверка короткой длительности"""
        result = estimate_processing_time(30, operation="transcribe")
        assert "менее" in result or "около" in result