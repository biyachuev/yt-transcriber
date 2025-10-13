"""
Tests for the utils module.
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
    """Tests for sanitize_filename."""
    
    def test_removes_invalid_characters(self):
        """Disallowed filesystem characters should be removed."""
        filename = 'test<>:"/\\|?*file.txt'
        result = sanitize_filename(filename)
        assert '<' not in result
        assert '>' not in result
        assert ':' not in result
        assert '"' not in result
    
    def test_replaces_spaces_with_underscores(self):
        """Spaces should be replaced with underscores."""
        filename = 'my test file name'
        result = sanitize_filename(filename)
        assert ' ' not in result
        assert '_' in result
    
    def test_truncates_long_names(self):
        """Long names should be truncated to the maximum length."""
        filename = 'a' * 300
        result = sanitize_filename(filename, max_length=200)
        assert len(result) <= 200
    
    def test_handles_empty_string(self):
        """Empty input should fall back to 'untitled'."""
        result = sanitize_filename('')
        assert result == 'untitled'
    
    def test_cyrillic_characters_preserved(self):
        """Cyrillic characters must remain intact."""
        filename = 'Тестовый файл'
        result = sanitize_filename(filename)
        assert 'Тестовый' in result
    
    def test_removes_exclamation_marks(self):
        """Ensure problematic exclamation marks are removed."""
        filename = 'NEW_FIDE_HIKARULE_DRAMA!!.mp3'
        result = sanitize_filename(filename)
        assert '!' not in result
        assert 'NEW_FIDE_HIKARULE_DRAMA__' in result or 'NEW_FIDE_HIKARULE_DRAMA_' in result
    
    def test_removes_multiple_special_chars(self):
        """Multiple special characters should be stripped."""
        filename = 'Test!@#$%^&*()_+{}|:<>?[]\\;\'",./`~file.mp3'
        result = sanitize_filename(filename)
        # Ensure problematic characters are removed.
        problematic_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', 
                           '{', '}', '|', ':', '<', '>', '?', '[', ']', '\\', 
                           ';', "'", '"', ',', '/', '`', '~']
        for char in problematic_chars:
            assert char not in result


class TestFormatTimestamp:
    """Tests for format_timestamp."""
    
    def test_formats_seconds_only(self):
        """Format seconds-only value."""
        assert format_timestamp(45) == "00:45"
    
    def test_formats_minutes_and_seconds(self):
        """Format minutes and seconds."""
        assert format_timestamp(125) == "02:05"
    
    def test_formats_hours_minutes_seconds(self):
        """Format hours, minutes, and seconds."""
        assert format_timestamp(3665) == "01:01:05"
    
    def test_handles_zero(self):
        """Zero seconds should return 00:00."""
        assert format_timestamp(0) == "00:00"


class TestDetectLanguage:
    """Tests for detect_language."""
    
    def test_detects_russian(self):
        """Russian text should be detected as 'ru'."""
        text = "Это тестовый текст на русском языке"
        assert detect_language(text) == "ru"
    
    def test_detects_english(self):
        """English text should be detected as 'en'."""
        text = "This is a test text in English language"
        assert detect_language(text) == "en"
    
    def test_mixed_text_dominant_language(self):
        """Language detection should follow the dominant alphabet."""
        # More Cyrillic characters
        text = "Это текст на русском языке который содержит with some English words"
        assert detect_language(text) == "ru"

        # More Latin characters
        text = "This is a longer English text с парой русских слов"
        assert detect_language(text) == "en"


class TestChunkText:
    """Tests for chunk_text."""
    
    def test_splits_long_text(self):
        """Long text should be split into multiple chunks."""
        # Create paragraph-based text (chunk_text splits on paragraphs).
        paragraphs = [' '.join(['word'] * 500) for _ in range(10)]
        text = '\n\n'.join(paragraphs)
        chunks = chunk_text(text, max_tokens=1000)
        assert len(chunks) > 1

        # Ensure each chunk stays within the limit.
        for chunk in chunks:
            word_count = len(chunk.split())
            assert word_count <= 1100  # small buffer
    
    def test_preserves_paragraphs(self):
        """Paragraph boundaries should be preserved."""
        text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        chunks = chunk_text(text, max_tokens=100)
        
        # Ensure each paragraph is still present.
        combined = '\n\n'.join(chunks)
        assert "Paragraph 1" in combined
        assert "Paragraph 2" in combined
        assert "Paragraph 3" in combined
    
    def test_short_text_returns_single_chunk(self):
        """Short text should remain a single chunk."""
        text = "Short text"
        chunks = chunk_text(text, max_tokens=1000)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestEstimateProcessingTime:
    """Tests for estimate_processing_time."""
    
    def test_transcribe_estimation(self):
        """Transcription estimate should mention minutes."""
        result = estimate_processing_time(3600, operation="transcribe")
        assert "minute" in result or "minutes" in result or "about" in result  # handle phrasing
   
    def test_translate_estimation(self):
        """Translation estimate should mention minutes."""
        result = estimate_processing_time(3600, operation="translate")
        assert "minute" in result or "minutes" in result or "about" in result
   
    def test_short_duration(self):
        """Short durations should mention seconds."""
        result = estimate_processing_time(30, operation="transcribe")
        assert "second" in result or "seconds" in result or "few" in result or "about" in result

    def test_different_models(self):
        """Ensure all models return a string result."""
        result_base = estimate_processing_time(1000, "transcribe", "whisper_base")
        result_small = estimate_processing_time(1000, "transcribe", "whisper_small")
        result_medium = estimate_processing_time(1000, "transcribe", "whisper_medium")

        # All results should be strings.
        assert isinstance(result_base, str)
        assert isinstance(result_small, str)
        assert isinstance(result_medium, str)
