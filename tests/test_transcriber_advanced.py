"""Advanced tests for transcriber module to increase coverage"""
import pytest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from src.transcriber import TranscriptionSegment, Transcriber
from src.config import TranscribeOptions


@pytest.fixture
def temp_dir():
    """Create temporary directory"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


class TestTranscriberAdvanced:
    """Advanced tests for Transcriber class"""

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_transcriber_device_detection_cuda(self, mock_whisper, mock_torch):
        """Test device detection with CUDA available"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        assert transcriber.device == 'cuda'

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_transcriber_device_detection_mps(self, mock_whisper, mock_torch):
        """Test device detection with MPS available"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        transcriber = Transcriber()

        assert transcriber.device == 'mps'

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_transcriber_device_detection_cpu(self, mock_whisper, mock_torch):
        """Test device detection falls back to CPU"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        assert transcriber.device == 'cpu'

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_transcriber_different_methods(self, mock_whisper, mock_torch):
        """Test transcriber with different methods"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        for method in [TranscribeOptions.WHISPER_BASE, TranscribeOptions.WHISPER_SMALL, TranscribeOptions.WHISPER_MEDIUM]:
            transcriber = Transcriber(method=method)
            assert transcriber.method == method

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_segments_to_text_preserves_order(self, mock_whisper, mock_torch):
        """Test that segments_to_text preserves order"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        segments = [
            TranscriptionSegment(0, 5, "First"),
            TranscriptionSegment(5, 10, "Second"),
            TranscriptionSegment(10, 15, "Third"),
            TranscriptionSegment(15, 20, "Fourth"),
        ]

        result = transcriber.segments_to_text(segments)

        # Check order is preserved
        assert result.index("First") < result.index("Second")
        assert result.index("Second") < result.index("Third")
        assert result.index("Third") < result.index("Fourth")

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_segments_to_text_with_timestamps_formats_correctly(self, mock_whisper, mock_torch):
        """Test timestamp formatting"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        segments = [
            TranscriptionSegment(0, 5, "Start"),
            TranscriptionSegment(65, 70, "After minute"),
            TranscriptionSegment(3665, 3670, "After hour"),
        ]

        result = transcriber.segments_to_text_with_timestamps(segments)

        # Check timestamp formats
        assert "[00:00]" in result
        assert "[01:05]" in result
        assert "[01:01:05]" in result

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_segments_to_text_with_speakers_and_timestamps(self, mock_whisper, mock_torch):
        """Test combined speakers and timestamps"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        segments = [
            TranscriptionSegment(0, 5, "Hello", speaker="Alice"),
            TranscriptionSegment(5, 10, "Hi", speaker="Bob"),
        ]

        result = transcriber.segments_to_text_with_timestamps(segments, with_speakers=True)

        # Check both speakers and timestamps are present
        assert "[Alice]" in result
        assert "[Bob]" in result
        assert "[00:00]" in result
        assert "[00:05]" in result

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_load_model_lazy_loading(self, mock_whisper, mock_torch):
        """Test that model is loaded lazily"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        # Model should not be loaded initially
        assert transcriber.model is None

        # Mock whisper.load_model
        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model

        # Load model
        transcriber._load_model()

        # Model should now be loaded
        assert transcriber.model is not None
        mock_whisper.load_model.assert_called_once()


class TestTranscriptionSegmentAdvanced:
    """Advanced tests for TranscriptionSegment"""

    def test_segment_to_dict_with_speaker(self):
        """Test to_dict with speaker"""
        segment = TranscriptionSegment(65, 70, "Test", speaker="Speaker 1")

        result = segment.to_dict()

        assert result['speaker'] == "Speaker 1"
        assert result['text'] == "Test"
        assert result['start'] == 65
        assert result['end'] == 70

    def test_segment_to_dict_without_speaker(self):
        """Test to_dict without speaker"""
        segment = TranscriptionSegment(0, 5, "Test")

        result = segment.to_dict()

        assert result['speaker'] is None

    def test_segment_timestamp_formatting(self):
        """Test different timestamp formats"""
        test_cases = [
            (0, "00:00"),
            (30, "00:30"),
            (60, "01:00"),
            (90, "01:30"),
            (3600, "01:00:00"),
            (3665, "01:01:05"),
        ]

        for seconds, expected in test_cases:
            segment = TranscriptionSegment(seconds, seconds + 5, "Test")
            result = segment.to_dict()
            assert result['timestamp'] == expected

    def test_segment_equality(self):
        """Test segment comparison"""
        seg1 = TranscriptionSegment(0, 5, "Test")
        seg2 = TranscriptionSegment(0, 5, "Test")
        seg3 = TranscriptionSegment(0, 5, "Different")

        # Same text and timing
        assert seg1.text == seg2.text
        assert seg1.start == seg2.start

        # Different text
        assert seg1.text != seg3.text

    def test_segment_text_cleaning(self):
        """Test text is properly cleaned"""
        test_cases = [
            ("  Text  ", "Text"),
            ("\nText\n", "Text"),
            ("\t Text \t", "Text"),
            ("   Multiple   Spaces   ", "Multiple   Spaces"),  # Internal spaces preserved
        ]

        for input_text, expected in test_cases:
            segment = TranscriptionSegment(0, 5, input_text)
            assert segment.text == expected
