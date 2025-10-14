"""
Unit tests for Whisper hallucination cleanup functionality.
"""
import pytest
from src.transcriber import Transcriber, TranscriptionSegment


class TestHallucinationCleanup:
    """Test suite for hallucination detection and cleanup."""

    def setup_method(self):
        """Set up test fixtures."""
        self.transcriber = Transcriber()

    def test_clean_korean_characters(self):
        """Test removal of Korean characters from Russian text."""
        segments = [
            TranscriptionSegment(
                start=0.0,
                end=5.0,
                text="Нормальный текст",
                speaker=None
            ),
            TranscriptionSegment(
                start=5.0,
                end=10.0,
                text="Текст с 사람 корейскими символами",
                speaker=None
            ),
            TranscriptionSegment(
                start=10.0,
                end=15.0,
                text="Ещё нормальный текст",
                speaker=None
            ),
        ]

        cleaned = self.transcriber._clean_hallucinations(segments, expected_language="ru")

        # The segment with Korean characters should be removed (>5% non-standard chars)
        assert len(cleaned) == 2
        assert cleaned[0].text == "Нормальный текст"
        assert cleaned[1].text == "Ещё нормальный текст"

    def test_clean_random_names(self):
        """Test removal of hallucinated random names."""
        segments = [
            TranscriptionSegment(
                start=0.0,
                end=5.0,
                text="Нормальная речь о технологиях",
                speaker=None
            ),
            TranscriptionSegment(
                start=5.0,
                end=10.0,
                text="Shepherd Bettsies",
                speaker=None
            ),
            TranscriptionSegment(
                start=10.0,
                end=15.0,
                text="campo",
                speaker=None
            ),
        ]

        cleaned = self.transcriber._clean_hallucinations(segments, expected_language="ru")

        # Hallucinated names should be removed
        assert len(cleaned) == 1
        assert cleaned[0].text == "Нормальная речь о технологиях"

    def test_clean_mixed_language_hallucinations(self):
        """Test removal of segments with suspicious language mixing."""
        segments = [
            TranscriptionSegment(
                start=0.0,
                end=5.0,
                text="Работаю в компании GenAI",  # Normal - technical term
                speaker=None
            ),
            TranscriptionSegment(
                start=5.0,
                end=10.0,
                text="got interesting questionable",  # Hallucination - too much English
                speaker=None
            ),
            TranscriptionSegment(
                start=10.0,
                end=15.0,
                text="Продолжаем обсуждение AI",  # Normal - AI is accepted term
                speaker=None
            ),
        ]

        cleaned = self.transcriber._clean_hallucinations(segments, expected_language="ru")

        # Should keep segments with Russian + technical terms, remove pure English hallucination
        assert len(cleaned) == 2
        assert "GenAI" in cleaned[0].text or "AI" in cleaned[0].text

    def test_clean_subscription_phrases(self):
        """Test removal of typical YouTube subscription hallucinations."""
        segments = [
            TranscriptionSegment(
                start=0.0,
                end=5.0,
                text="Нормальное содержание",
                speaker=None
            ),
            TranscriptionSegment(
                start=5.0,
                end=10.0,
                text="Thanks for watching",
                speaker=None
            ),
            TranscriptionSegment(
                start=10.0,
                end=15.0,
                text="Please subscribe and hit the bell",
                speaker=None
            ),
        ]

        cleaned = self.transcriber._clean_hallucinations(segments, expected_language="ru")

        # Subscription phrases should be removed
        assert len(cleaned) == 1
        assert cleaned[0].text == "Нормальное содержание"

    def test_clean_very_short_segments(self):
        """Test removal of very short nonsensical segments."""
        segments = [
            TranscriptionSegment(
                start=0.0,
                end=5.0,
                text="Нормальный длинный сегмент текста",
                speaker=None
            ),
            TranscriptionSegment(
                start=5.0,
                end=10.0,
                text="ab",  # Too short
                speaker=None
            ),
            TranscriptionSegment(
                start=10.0,
                end=15.0,
                text="Ещё один нормальный сегмент",
                speaker=None
            ),
        ]

        cleaned = self.transcriber._clean_hallucinations(segments, expected_language="ru")

        # Very short segments should be removed
        assert len(cleaned) == 2

    def test_preserve_speaker_labels(self):
        """Test that speaker labels are preserved during cleanup."""
        segments = [
            TranscriptionSegment(
                start=0.0,
                end=5.0,
                text="Текст первого спикера",
                speaker="SPEAKER_01"
            ),
            TranscriptionSegment(
                start=5.0,
                end=10.0,
                text="Shepherd Bettsies",  # Hallucination
                speaker="SPEAKER_02"
            ),
            TranscriptionSegment(
                start=10.0,
                end=15.0,
                text="Текст второго спикера",
                speaker="SPEAKER_02"
            ),
        ]

        cleaned = self.transcriber._clean_hallucinations(segments, expected_language="ru")

        # Speaker labels should be preserved
        assert len(cleaned) == 2
        assert cleaned[0].speaker == "SPEAKER_01"
        assert cleaned[1].speaker == "SPEAKER_02"

    def test_clean_special_characters(self):
        """Test removal of segments with too many special characters."""
        segments = [
            TranscriptionSegment(
                start=0.0,
                end=5.0,
                text="Нормальный текст с пунктуацией.",
                speaker=None
            ),
            TranscriptionSegment(
                start=5.0,
                end=10.0,
                text="###$$%%%^^^&&&***",  # Too many special chars
                speaker=None
            ),
            TranscriptionSegment(
                start=10.0,
                end=15.0,
                text="Ещё текст",
                speaker=None
            ),
        ]

        cleaned = self.transcriber._clean_hallucinations(segments, expected_language="ru")

        # Segments with too many special characters should be removed
        assert len(cleaned) == 2

    def test_preserve_technical_terms(self):
        """Test that technical terms and brand names are preserved."""
        segments = [
            TranscriptionSegment(
                start=0.0,
                end=5.0,
                text="Я работаю в Тбанке или Тинькофф",
                speaker=None
            ),
            TranscriptionSegment(
                start=5.0,
                end=10.0,
                text="Мы используем технологии ChatGPT и OpenAI",
                speaker=None
            ),
            TranscriptionSegment(
                start=10.0,
                end=15.0,
                text="Применяем подходы GenAI и RAG в продуктах",
                speaker=None
            ),
        ]

        cleaned = self.transcriber._clean_hallucinations(segments, expected_language="ru")

        # All segments with technical terms mixed with Russian should be preserved
        assert len(cleaned) == 3
        assert "Тбанке" in cleaned[0].text or "Тинькофф" in cleaned[0].text

    def test_empty_segments_removed(self):
        """Test that empty segments are removed."""
        segments = [
            TranscriptionSegment(
                start=0.0,
                end=5.0,
                text="Текст",
                speaker=None
            ),
            TranscriptionSegment(
                start=5.0,
                end=10.0,
                text="   ",  # Only whitespace
                speaker=None
            ),
            TranscriptionSegment(
                start=10.0,
                end=15.0,
                text="",  # Empty
                speaker=None
            ),
        ]

        cleaned = self.transcriber._clean_hallucinations(segments, expected_language="ru")

        # Only non-empty segment should remain
        assert len(cleaned) == 1

    def test_partial_cleanup_within_segment(self):
        """Test that Korean chars are removed from mixed segments."""
        segments = [
            TranscriptionSegment(
                start=0.0,
                end=5.0,
                text="Текст с 사람한국어 символами продолжается нормально",
                speaker=None
            ),
        ]

        cleaned = self.transcriber._clean_hallucinations(segments, expected_language="ru")

        # The segment should be removed due to suspicious mixing (>5% Korean chars)
        # (Korean characters in Russian text is a strong hallucination signal)
        assert len(cleaned) == 0

    def test_english_transcription(self):
        """Test that English transcriptions work correctly."""
        segments = [
            TranscriptionSegment(
                start=0.0,
                end=5.0,
                text="This is normal English text",
                speaker=None
            ),
            TranscriptionSegment(
                start=5.0,
                end=10.0,
                text="사람한국어말 Korean in English",
                speaker=None
            ),
            TranscriptionSegment(
                start=10.0,
                end=15.0,
                text="More normal text",
                speaker=None
            ),
        ]

        cleaned = self.transcriber._clean_hallucinations(segments, expected_language="en")

        # Korean characters should trigger removal even in English (>5% threshold)
        assert len(cleaned) == 2
        assert cleaned[0].text == "This is normal English text"
        assert cleaned[1].text == "More normal text"
