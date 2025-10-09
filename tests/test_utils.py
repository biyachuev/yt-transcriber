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
    
    def test_removes_exclamation_marks(self):
        """Проверка удаления восклицательных знаков (проблема с Cursor terminal)"""
        filename = 'NEW_FIDE_HIKARULE_DRAMA!!.mp3'
        result = sanitize_filename(filename)
        assert '!' not in result
        assert 'NEW_FIDE_HIKARULE_DRAMA__' in result or 'NEW_FIDE_HIKARULE_DRAMA_' in result
    
    def test_removes_multiple_special_chars(self):
        """Проверка удаления множественных специальных символов"""
        filename = 'Test!@#$%^&*()_+{}|:<>?[]\\;\'",./`~file.mp3'
        result = sanitize_filename(filename)
        # Проверяем что основные проблемные символы удалены
        problematic_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', 
                           '{', '}', '|', ':', '<', '>', '?', '[', ']', '\\', 
                           ';', "'", '"', ',', '/', '`', '~']
        for char in problematic_chars:
            assert char not in result


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
        text = "Это текст на русском языке который содержит with some English words"
        assert detect_language(text) == "ru"

        # Больше английских символов
        text = "This is a longer English text с парой русских слов"
        assert detect_language(text) == "en"


class TestChunkText:
    """Тесты для функции chunk_text"""
    
    def test_splits_long_text(self):
        """Проверка разбиения длинного текста"""
        # Создаём текст с параграфами (chunk_text разбивает по параграфам)
        paragraphs = [' '.join(['word'] * 500) for _ in range(10)]
        text = '\n\n'.join(paragraphs)
        chunks = chunk_text(text, max_tokens=1000)
        assert len(chunks) > 1

        # Проверяем, что каждый чанк не превышает лимит
        for chunk in chunks:
            word_count = len(chunk.split())
            assert word_count <= 1100  # Небольшой запас
    
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
        assert "менее" in result or "около" in result or "секунд" in result

    def test_different_models(self):
        """Проверка разных моделей"""
        result_base = estimate_processing_time(1000, "transcribe", "whisper_base")
        result_small = estimate_processing_time(1000, "transcribe", "whisper_small")
        result_medium = estimate_processing_time(1000, "transcribe", "whisper_medium")

        # Все должны вернуть строку
        assert isinstance(result_base, str)
        assert isinstance(result_small, str)
        assert isinstance(result_medium, str)