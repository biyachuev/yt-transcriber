"""
Tests for the transcriber module.
"""
import pytest
from unittest.mock import patch, MagicMock
from src.transcriber import TranscriptionSegment, Transcriber
from src.config import TranscribeOptions


class TestTranscriptionSegment:
    """Unit tests covering TranscriptionSegment behaviour."""
    
    def test_segment_creation(self):
        """Ensure the segment stores basic attributes."""
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
        """Ensure speaker information is preserved."""
        segment = TranscriptionSegment(
            start=0,
            end=5,
            text="Hello",
            speaker="Speaker 1"
        )
        
        assert segment.speaker == "Speaker 1"
    
    def test_segment_strips_text(self):
        """Leading and trailing whitespace should be removed."""
        segment = TranscriptionSegment(
            start=0,
            end=5,
            text="  Text with spaces  "
        )
        
        assert segment.text == "Text with spaces"
    
    def test_segment_to_dict(self):
        """Dictionary representation contains expected keys."""
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
    """Unit tests covering the Transcriber helper."""

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_transcriber_initialization(self, mock_whisper, mock_torch):
        """Ensure device selection and lazy model loading."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber(method=TranscribeOptions.WHISPER_BASE)

        assert transcriber.method == TranscribeOptions.WHISPER_BASE
        assert transcriber.model is None  # model loads lazily
        assert transcriber.device in ['cpu', 'cuda', 'mps']

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_segments_to_text(self, mock_whisper, mock_torch):
        """Join segments into a single text block."""
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
        assert result.count("\n\n") == 2  # two separators between three segments

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_segments_to_text_with_timestamps(self, mock_whisper, mock_torch):
        """Ensure timestamps are added when requested."""
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
        """Ensure timestamps and speaker labels are included."""
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


class TestChunking:
    """Tests for audio chunking logic."""

    def test_chunk_metadata_preservation(self):
        """Ensure _split_audio_file returns chunk metadata (path, start, end)."""
        from pathlib import Path

        transcriber = Transcriber(method=TranscribeOptions.WHISPER_OPENAI_API)

        # Mock the actual file operations
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(stdout="100.0", returncode=0)

            # Mock _find_speech_boundaries to return empty list (simple splitting)
            with patch.object(transcriber, '_find_speech_boundaries', return_value=[]):
                # Create a fake audio path
                fake_path = Path("/fake/audio.mp3")

                # Mock file size to trigger chunking
                with patch.object(Path, 'stat') as mock_stat:
                    mock_stat.return_value.st_size = 50 * 1024 * 1024  # 50 MB

                    with patch.object(Path, 'suffix', '.mp3'):
                        with patch.object(Path, 'stem', 'audio'):
                            with patch.object(Path, 'parent', Path('/fake')):
                                chunks = transcriber._split_audio_file(fake_path)

                                # Should return list of tuples
                                assert isinstance(chunks, list)
                                assert len(chunks) > 0

                                # Each chunk should be (path, start_time, end_time)
                                for chunk in chunks:
                                    assert isinstance(chunk, tuple)
                                    assert len(chunk) == 3
                                    path, start, end = chunk
                                    assert isinstance(start, float)
                                    assert isinstance(end, float)
                                    assert start < end

    def test_silent_chunk_timestamp_handling(self):
        """
        Regression test: timestamps must be correct even when a chunk contains no speech.
        Bug was: cumulative_time only advanced when segments were found.
        Fix: use chunk_start instead of cumulative tracking.
        """
        from pathlib import Path

        transcriber = Transcriber(method=TranscribeOptions.WHISPER_OPENAI_API)

        # Simulate 3 chunks: chunk 0 has speech, chunk 1 is silent, chunk 2 has speech
        mock_chunks = [
            (Path("/fake/chunk_0.mp3"), 0.0, 100.0),    # 0-100s: speech
            (Path("/fake/chunk_1.mp3"), 100.0, 200.0),  # 100-200s: SILENT
            (Path("/fake/chunk_2.mp3"), 200.0, 300.0),  # 200-300s: speech
        ]

        # Mock OpenAI responses
        def mock_transcribe_single(chunk_path, *args, **kwargs):
            if "chunk_0" in str(chunk_path):
                # Chunk 0: speech from 0-10s (relative to chunk)
                return [TranscriptionSegment(0.0, 10.0, "First segment")]
            elif "chunk_1" in str(chunk_path):
                # Chunk 1: SILENT - no segments
                return []
            elif "chunk_2" in str(chunk_path):
                # Chunk 2: speech from 0-10s (relative to chunk)
                return [TranscriptionSegment(0.0, 10.0, "Third segment")]
            return []

        with patch.object(transcriber, '_split_audio_file', return_value=mock_chunks):
            with patch.object(transcriber, '_transcribe_single_file_with_openai', side_effect=mock_transcribe_single):
                # Mock file size check
                fake_path = Path("/fake/large_audio.mp3")
                with patch.object(Path, 'stat') as mock_stat:
                    mock_stat.return_value.st_size = 50 * 1024 * 1024  # 50 MB

                    # Call the main transcription method
                    segments = transcriber._transcribe_with_openai_api(
                        fake_path,
                        language='en',
                        initial_prompt=None
                    )

                    # Should have 2 segments (chunk 1 was silent)
                    assert len(segments) == 2

                    # First segment: from chunk 0, starts at 0 + chunk_start = 0.0
                    assert segments[0].start == 0.0
                    assert segments[0].end == 10.0
                    assert segments[0].text == "First segment"

                    # Second segment: from chunk 2, starts at 0 + chunk_start = 200.0
                    # This is the key test - it should NOT be shifted by chunk 1's missing segments
                    assert segments[1].start == 200.0
                    assert segments[1].end == 210.0
                    assert segments[1].text == "Third segment"


# Integration tests (require audio files)
@pytest.mark.integration
class TestTranscriberIntegration:
    """Integration tests (skipped unless audio samples are present)."""

    @pytest.mark.skip(reason="Requires test audio file")
    def test_transcribe_audio_file(self):
        """Placeholder for real-file transcription test."""
        # Requires a sample audio file to run
        pass
