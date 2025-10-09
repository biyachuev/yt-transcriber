"""
Тесты для модуля transcriber
"""
import pytest
from unittest.mock import patch, MagicMock
from src.transcriber import TranscriptionSegment, Transcriber
from src.config import TranscribeOptions


class TestTranscriptionSegment:
    """Тесты для класса TranscriptionSegment"""
    
    def test_segment_creation(self):
        """Проверка создания сегмента"""
        segment = TranscriptionSegment(
            start=10.5,
            end=15.3,
            text="Test text"
        )
        
        assert segment.start == 10.5
        assert segment.end == 15.3
        assert segment.text == "Test text"
        assert segment.speaker is None
    
    def test_segment_with_speaker(self):
        """Проверка сегмента со спикером"""
        segment = TranscriptionSegment(
            start=0,
            end=5,
            text="Hello",
            speaker="Speaker 1"
        )
        
        assert segment.speaker == "Speaker 1"
    
    def test_segment_strips_text(self):
        """Проверка удаления пробелов из текста"""
        segment = TranscriptionSegment(
            start=0,
            end=5,
            text="  Text with spaces  "
        )
        
        assert segment.text == "Text with spaces"
    
    def test_segment_to_dict(self):
        """Проверка преобразования в словарь"""
        segment = TranscriptionSegment(
            start=65,
            end=70,
            text="Test"
        )
        
        result = segment.to_dict()
        assert 'start' in result
        assert 'end' in result
        assert 'text' in result
        assert 'timestamp' in result
        assert result['timestamp'] == "01:05"


class TestTranscriber:
    """Тесты для класса Transcriber"""

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_transcriber_initialization(self, mock_whisper, mock_torch):
        """Проверка инициализации транскрайбера"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber(method=TranscribeOptions.WHISPER_BASE)

        assert transcriber.method == TranscribeOptions.WHISPER_BASE
        assert transcriber.model is None  # Модель загружается по требованию
        assert transcriber.device in ['cpu', 'cuda', 'mps']

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_segments_to_text(self, mock_whisper, mock_torch):
        """Проверка преобразования сегментов в текст"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        segments = [
            TranscriptionSegment(0, 5, "First segment"),
            TranscriptionSegment(5, 10, "Second segment"),
            TranscriptionSegment(10, 15, "Third segment")
        ]

        result = transcriber.segments_to_text(segments)

        assert "First segment" in result
        assert "Second segment" in result
        assert "Third segment" in result
        assert result.count("\n\n") == 2  # Два разделителя между тремя сегментами

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_segments_to_text_with_timestamps(self, mock_whisper, mock_torch):
        """Проверка преобразования сегментов в текст с таймкодами"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        segments = [
            TranscriptionSegment(0, 5, "First"),
            TranscriptionSegment(65, 70, "Second")
        ]

        result = transcriber.segments_to_text_with_timestamps(segments)

        assert "[00:00]" in result
        assert "[01:05]" in result
        assert "First" in result
        assert "Second" in result

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_segments_to_text_with_timestamps_and_speakers(self, mock_whisper, mock_torch):
        """Проверка преобразования с таймкодами и спикерами"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        segments = [
            TranscriptionSegment(0, 5, "Hello", speaker="Speaker 1"),
            TranscriptionSegment(5, 10, "Hi", speaker="Speaker 2")
        ]

        result = transcriber.segments_to_text_with_timestamps(
            segments,
            with_speakers=True
        )

        assert "[Speaker 1]" in result
        assert "[Speaker 2]" in result
        assert "[00:00]" in result
        assert "[00:05]" in result


# Интеграционные тесты (требуют аудиофайла)
@pytest.mark.integration
class TestTranscriberIntegration:
    """Интеграционные тесты (пропускаются по умолчанию)"""
    
    @pytest.mark.skip(reason="Требуется тестовый аудиофайл")
    def test_transcribe_audio_file(self):
        """Проверка транскрибирования реального файла"""
        # Этот тест требует наличия тестового аудиофайла
        pass