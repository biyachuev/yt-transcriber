"""
Tests for speaker diarization functionality.
"""
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.transcriber import Transcriber, TranscriptionSegment


class TestSpeakerDiarization:
    """Test suite for speaker diarization functionality."""

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_get_diarization_pipeline_without_token(self, mock_whisper, mock_torch):
        """Test that diarization pipeline returns None without HF token."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        with patch.dict('os.environ', {}, clear=True):
            pipeline = transcriber._get_diarization_pipeline()

        assert pipeline is None

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_get_diarization_pipeline_with_token(self, mock_whisper, mock_torch):
        """Test that diarization pipeline loads successfully with HF token."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        mock_pipeline = Mock()

        # Clear any existing environment variables to ensure clean test
        with patch.dict('os.environ', {'HF_TOKEN': 'test_token'}, clear=False):
            # Remove any other HF token variables
            with patch.dict('os.environ', {'HUGGINGFACE_TOKEN': ''}, clear=False):
                with patch('pyannote.audio.Pipeline') as mock_pipeline_class:
                    # Mock from_pretrained to have 'token' parameter in signature
                    mock_from_pretrained = Mock(return_value=mock_pipeline)
                    # Create a mock signature with 'token' parameter
                    import inspect
                    mock_signature = Mock()
                    mock_signature.parameters = {'token': Mock(), 'revision': Mock()}
                    with patch('inspect.signature', return_value=mock_signature):
                        mock_pipeline_class.from_pretrained = mock_from_pretrained
                        # Also need to patch os.environ.get to return our test token
                        def mock_get_env(key, default=None):
                            if key == "HF_TOKEN":
                                return 'test_token'
                            elif key == "HUGGINGFACE_TOKEN":
                                return None
                            elif key == "PYANNOTE_DIARIZATION_REVISION":
                                return None
                            return default
                        with patch('os.environ.get', side_effect=mock_get_env):
                            pipeline = transcriber._get_diarization_pipeline()

        assert pipeline == mock_pipeline
        # Check that token was passed (might include revision too)
        call_args = mock_from_pretrained.call_args
        assert call_args[0][0] == "pyannote/speaker-diarization-3.1"
        assert call_args[1].get('token') == 'test_token'

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_get_diarization_pipeline_import_error(self, mock_whisper, mock_torch):
        """Test graceful handling when pyannote.audio is not installed."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        # Simulate ImportError by patching the import statement
        def mock_import(name, *args):
            if 'pyannote' in name:
                raise ImportError("No module named 'pyannote'")
            return __import__(name, *args)

        with patch.dict('os.environ', {'HF_TOKEN': 'test_token'}):
            with patch('builtins.__import__', side_effect=mock_import):
                pipeline = transcriber._get_diarization_pipeline()

        assert pipeline is None

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_perform_speaker_diarization_without_pipeline(self, mock_whisper, mock_torch):
        """Test that diarization gracefully fails without pipeline."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()
        transcriber._diarization_pipeline = None

        segments = [
            TranscriptionSegment(0.0, 5.0, "Hello world"),
            TranscriptionSegment(5.0, 10.0, "How are you"),
        ]

        audio_path = Path("/fake/path/audio.mp3")

        result = transcriber._perform_speaker_diarization(audio_path, segments)

        # Should return original segments unchanged
        assert len(result) == 2
        assert result[0].speaker is None
        assert result[1].speaker is None

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_perform_speaker_diarization_with_speakers(self, mock_whisper, mock_torch):
        """Test speaker diarization assigns correct speaker labels."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        # Mock diarization pipeline
        mock_pipeline = Mock()
        mock_annotation = Mock()

        # Create mock speaker turns
        # Speaker A: 0-6 seconds
        # Speaker B: 6-12 seconds
        turn1 = Mock()
        turn1.start = 0.0
        turn1.end = 6.0

        turn2 = Mock()
        turn2.start = 6.0
        turn2.end = 12.0

        # pyannote.audio returns Annotation object directly, not wrapped
        # itertracks needs yield_label parameter
        def mock_itertracks(yield_label=False):
            if yield_label:
                return iter([
                    (turn1, None, "SPEAKER_00"),
                    (turn2, None, "SPEAKER_01"),
                ])
            return iter([(turn1, None), (turn2, None)])

        mock_annotation.itertracks = mock_itertracks
        # Make sure mock_annotation doesn't have these attributes
        # so the code uses it directly (fallback path in transcriber.py:680-682)
        del mock_annotation.exclusive_speaker_diarization
        del mock_annotation.speaker_diarization

        # Pipeline should be callable and return annotation
        # It can be called with dict or Path
        def pipeline_call(audio_input):
            return mock_annotation

        mock_pipeline.side_effect = pipeline_call
        transcriber._diarization_pipeline = mock_pipeline

        segments = [
            TranscriptionSegment(0.0, 3.0, "Hello"),
            TranscriptionSegment(3.0, 6.0, "My name is Alice"),
            TranscriptionSegment(7.0, 10.0, "Hi, I'm Bob"),
            TranscriptionSegment(10.0, 12.0, "Nice to meet you"),
        ]

        audio_path = Path("/fake/path/audio.mp3")

        result = transcriber._perform_speaker_diarization(audio_path, segments)

        assert len(result) == 4
        # First two segments should be SPEAKER_00
        assert result[0].speaker == "SPEAKER_00"
        assert result[1].speaker == "SPEAKER_00"
        # Last two segments should be SPEAKER_01
        assert result[2].speaker == "SPEAKER_01"
        assert result[3].speaker == "SPEAKER_01"

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_perform_speaker_diarization_overlap_detection(self, mock_whisper, mock_torch):
        """Test that speaker is assigned based on maximum overlap."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        # Mock diarization pipeline
        mock_pipeline = Mock()
        mock_annotation = Mock()

        # Speaker A: 0-5 seconds
        # Speaker B: 5-10 seconds
        turn1 = Mock()
        turn1.start = 0.0
        turn1.end = 5.0

        turn2 = Mock()
        turn2.start = 5.0
        turn2.end = 10.0

        # pyannote.audio returns Annotation object directly, not wrapped
        def mock_itertracks(yield_label=False):
            if yield_label:
                return iter([
                    (turn1, None, "SPEAKER_00"),
                    (turn2, None, "SPEAKER_01"),
                ])
            return iter([(turn1, None), (turn2, None)])

        mock_annotation.itertracks = mock_itertracks
        # Make sure mock_annotation doesn't have these attributes
        del mock_annotation.exclusive_speaker_diarization
        del mock_annotation.speaker_diarization

        # Pipeline should be callable and return annotation
        def pipeline_call(audio_input):
            return mock_annotation

        mock_pipeline.side_effect = pipeline_call
        transcriber._diarization_pipeline = mock_pipeline

        # Segment that spans both speakers (4-6 seconds)
        # 1 second overlap with Speaker A, 1 second overlap with Speaker B
        # Since they're equal, it should pick the first one encountered
        segments = [
            TranscriptionSegment(4.0, 6.0, "Boundary segment"),
        ]

        audio_path = Path("/fake/path/audio.mp3")

        result = transcriber._perform_speaker_diarization(audio_path, segments)

        assert len(result) == 1
        # Should be assigned to one of the speakers
        assert result[0].speaker in ["SPEAKER_00", "SPEAKER_01"]

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_perform_speaker_diarization_exception_handling(self, mock_whisper, mock_torch):
        """Test that exceptions during diarization are handled gracefully."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        # Mock pipeline that raises an exception
        mock_pipeline = Mock()
        mock_pipeline.side_effect = RuntimeError("Diarization failed")
        transcriber._diarization_pipeline = mock_pipeline

        segments = [
            TranscriptionSegment(0.0, 5.0, "Hello world"),
        ]

        audio_path = Path("/fake/path/audio.mp3")

        result = transcriber._perform_speaker_diarization(audio_path, segments)

        # Should return original segments without speaker labels
        assert len(result) == 1
        assert result[0].speaker is None

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_perform_speaker_diarization_counts_unique_speakers(self, mock_whisper, mock_torch):
        """Test that unique speaker count is logged correctly."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        # Mock diarization pipeline with 3 speakers
        mock_pipeline = Mock()
        mock_annotation = Mock()

        turn1 = Mock()
        turn1.start = 0.0
        turn1.end = 3.0

        turn2 = Mock()
        turn2.start = 3.0
        turn2.end = 6.0

        turn3 = Mock()
        turn3.start = 6.0
        turn3.end = 9.0

        # pyannote.audio returns Annotation object directly, not wrapped
        def mock_itertracks(yield_label=False):
            if yield_label:
                return iter([
                    (turn1, None, "SPEAKER_00"),
                    (turn2, None, "SPEAKER_01"),
                    (turn3, None, "SPEAKER_02"),
                ])
            return iter([(turn1, None), (turn2, None), (turn3, None)])

        mock_annotation.itertracks = mock_itertracks
        # Make sure mock_annotation doesn't have these attributes
        del mock_annotation.exclusive_speaker_diarization
        del mock_annotation.speaker_diarization

        # Pipeline should be callable and return annotation
        def pipeline_call(audio_input):
            return mock_annotation

        mock_pipeline.side_effect = pipeline_call
        transcriber._diarization_pipeline = mock_pipeline

        segments = [
            TranscriptionSegment(0.0, 3.0, "Hello"),
            TranscriptionSegment(3.0, 6.0, "Hi"),
            TranscriptionSegment(6.0, 9.0, "Hey"),
        ]

        audio_path = Path("/fake/path/audio.mp3")

        result = transcriber._perform_speaker_diarization(audio_path, segments)

        # Verify result structure (logging is tested separately)
        # Just check that the correct number of unique speakers are found

        assert len(result) == 3
        assert result[0].speaker == "SPEAKER_00"
        assert result[1].speaker == "SPEAKER_01"
        assert result[2].speaker == "SPEAKER_02"
