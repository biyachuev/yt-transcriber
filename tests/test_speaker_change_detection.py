"""
Tests for speaker change detection in text formatting.
"""
from unittest.mock import patch

from src.transcriber import Transcriber, TranscriptionSegment


class TestSpeakerChangeDetection:
    """Test suite for speaker change detection in output formatting."""

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_segments_to_text_with_speakers_shows_speaker_only_on_change(self, mock_whisper, mock_torch):
        """Test that speaker labels appear only when speaker changes."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        segments = [
            TranscriptionSegment(0.0, 3.0, "Hello, how are you?", speaker="SPEAKER_00"),
            TranscriptionSegment(3.0, 6.0, "I'm doing great.", speaker="SPEAKER_00"),
            TranscriptionSegment(6.0, 9.0, "That's good to hear.", speaker="SPEAKER_01"),
            TranscriptionSegment(9.0, 12.0, "Thanks for asking.", speaker="SPEAKER_01"),
            TranscriptionSegment(12.0, 15.0, "You're welcome.", speaker="SPEAKER_00"),
        ]

        result = transcriber.segments_to_text_with_speakers(segments)

        # First speaker should have label
        assert "[SPEAKER_00] Hello, how are you?" in result
        # Second segment from same speaker should NOT have label
        assert "[SPEAKER_00] I'm doing great." not in result
        assert "I'm doing great." in result

        # Speaker change should show new label
        assert "[SPEAKER_01] That's good to hear." in result
        # Same speaker continuing should NOT have label
        assert "[SPEAKER_01] Thanks for asking." not in result
        assert "Thanks for asking." in result

        # Another speaker change
        assert "[SPEAKER_00] You're welcome." in result

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_segments_to_text_with_speakers_no_speakers(self, mock_whisper, mock_torch):
        """Test formatting when no speaker information is available."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        segments = [
            TranscriptionSegment(0.0, 3.0, "Hello, how are you?"),
            TranscriptionSegment(3.0, 6.0, "I'm doing great."),
        ]

        result = transcriber.segments_to_text_with_speakers(segments)

        # No speaker labels should appear
        assert "SPEAKER" not in result
        assert "Hello, how are you?" in result
        assert "I'm doing great." in result

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_segments_to_text_with_timestamps_shows_speaker_only_on_change(self, mock_whisper, mock_torch):
        """Test that speaker labels in timestamped output appear only when speaker changes."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        segments = [
            TranscriptionSegment(0.0, 3.0, "Hello there", speaker="SPEAKER_00"),
            TranscriptionSegment(3.0, 6.0, "Nice to meet you", speaker="SPEAKER_00"),
            TranscriptionSegment(6.0, 9.0, "Hi back", speaker="SPEAKER_01"),
        ]

        result = transcriber.segments_to_text_with_timestamps(segments, with_speakers=True)

        # First speaker should have label with timestamp
        assert "[00:00] [SPEAKER_00] Hello there" in result
        # Second segment from same speaker should NOT have speaker label
        assert "[00:03] Nice to meet you" in result
        assert "[00:03] [SPEAKER_00]" not in result
        # Speaker change should show new label
        assert "[00:06] [SPEAKER_01] Hi back" in result

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_segments_to_text_with_timestamps_no_speaker_flag(self, mock_whisper, mock_torch):
        """Test that speaker labels don't appear when with_speakers=False."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        segments = [
            TranscriptionSegment(0.0, 3.0, "Hello there", speaker="SPEAKER_00"),
            TranscriptionSegment(3.0, 6.0, "Nice to meet you", speaker="SPEAKER_01"),
        ]

        result = transcriber.segments_to_text_with_timestamps(segments, with_speakers=False)

        # No speaker labels should appear
        assert "SPEAKER" not in result
        assert "[00:00] Hello there" in result
        assert "[00:03] Nice to meet you" in result

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_segments_to_text_with_speakers_single_speaker(self, mock_whisper, mock_torch):
        """Test formatting when only one speaker is present."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        segments = [
            TranscriptionSegment(0.0, 3.0, "First sentence", speaker="SPEAKER_00"),
            TranscriptionSegment(3.0, 6.0, "Second sentence", speaker="SPEAKER_00"),
            TranscriptionSegment(6.0, 9.0, "Third sentence", speaker="SPEAKER_00"),
        ]

        result = transcriber.segments_to_text_with_speakers(segments)

        # Only first segment should have speaker label
        assert "[SPEAKER_00] First sentence" in result
        # Remaining segments should not have speaker labels
        assert result.count("SPEAKER_00") == 1

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_segments_to_text_with_speakers_alternating_speakers(self, mock_whisper, mock_torch):
        """Test formatting when speakers alternate frequently."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()

        segments = [
            TranscriptionSegment(0.0, 2.0, "A says this", speaker="SPEAKER_00"),
            TranscriptionSegment(2.0, 4.0, "B responds", speaker="SPEAKER_01"),
            TranscriptionSegment(4.0, 6.0, "A replies", speaker="SPEAKER_00"),
            TranscriptionSegment(6.0, 8.0, "B answers", speaker="SPEAKER_01"),
        ]

        result = transcriber.segments_to_text_with_speakers(segments)

        # Every segment should have a speaker label since they alternate
        assert result.count("SPEAKER_00") == 2
        assert result.count("SPEAKER_01") == 2
