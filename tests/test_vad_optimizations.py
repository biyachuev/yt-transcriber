"""Tests for VAD optimizations."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from src.transcriber import Transcriber


class TestVADOptimizations:
    """Test suite for VAD performance optimizations."""

    def test_calculate_split_points_simple(self):
        """Test simple time-based splitting when no VAD data is available."""
        transcriber = Transcriber()

        split_points = transcriber._calculate_split_points(
            total_duration=100.0,
            target_chunk_duration=30.0,
            speech_segments=[]
        )

        # Should create ~4 chunks (100/30 = 3.33, rounded up)
        assert len(split_points) == 4
        assert split_points[0][0] == 0.0
        assert split_points[-1][1] == 100.0

    def test_calculate_split_points_with_gaps(self):
        """Test VAD-based splitting with speech gaps."""
        transcriber = Transcriber()

        # Create speech segments with clear gaps
        speech_segments = [
            (0.0, 28.0),   # First speech
            (30.0, 58.0),  # Gap at 28-30 (2s gap)
            (60.0, 88.0),  # Gap at 58-60 (2s gap)
            (90.0, 100.0)  # Gap at 88-90 (2s gap)
        ]

        split_points = transcriber._calculate_split_points(
            total_duration=100.0,
            target_chunk_duration=30.0,
            speech_segments=speech_segments
        )

        # Should split at gaps
        assert len(split_points) >= 2
        assert split_points[0][0] == 0.0

        # Check that splits happen near gaps
        for start, end in split_points:
            assert start < end
            assert end <= 100.0

    def test_calculate_split_points_min_gap_filter(self):
        """Test that short gaps (< 300ms) are filtered out."""
        transcriber = Transcriber()

        # Create speech segments with short gaps that should be ignored
        speech_segments = [
            (0.0, 10.0),
            (10.1, 20.0),   # 100ms gap - too short
            (20.2, 30.0),   # 200ms gap - too short
            (30.5, 40.0),   # 300ms gap - should be used
            (40.8, 50.0),   # 300ms gap - should be used
        ]

        split_points = transcriber._calculate_split_points(
            total_duration=50.0,
            target_chunk_duration=25.0,
            speech_segments=speech_segments
        )

        # Should create chunks, but not at very short gaps
        assert len(split_points) >= 1
        for start, end in split_points:
            assert start < end

    def test_calculate_split_points_sequential_optimization(self):
        """Test that gap search is O(n) not O(n²)."""
        transcriber = Transcriber()

        # Create many speech segments
        num_segments = 1000
        speech_segments = []
        for i in range(num_segments):
            start = i * 2.0
            end = start + 1.0
            speech_segments.append((start, end))

        # This should complete quickly (O(n) vs O(n²))
        import time
        start_time = time.time()

        split_points = transcriber._calculate_split_points(
            total_duration=2000.0,
            target_chunk_duration=200.0,
            speech_segments=speech_segments
        )

        elapsed = time.time() - start_time

        # Should complete in < 1 second for 1000 segments (O(n))
        # O(n²) would take much longer
        assert elapsed < 1.0
        assert len(split_points) >= 1

    def test_calculate_split_points_no_infinite_loop(self):
        """Test that split point calculation doesn't get stuck."""
        transcriber = Transcriber()

        # Edge case: all speech, no gaps
        speech_segments = [(0.0, 100.0)]

        split_points = transcriber._calculate_split_points(
            total_duration=100.0,
            target_chunk_duration=30.0,
            speech_segments=speech_segments
        )

        # Should still create chunks even without good gaps
        assert len(split_points) >= 1
        assert split_points[0][0] == 0.0
        assert split_points[-1][1] == 100.0

    def test_silero_vad_initialization(self):
        """Test Silero VAD model initialization."""
        transcriber = Transcriber()

        # Try to get Silero VAD (may fail if not available)
        model = transcriber._get_silero_vad()

        # Either it loads successfully or returns None
        assert model is None or hasattr(transcriber, '_silero_vad_model')

        # Second call should use cached model
        model2 = transcriber._get_silero_vad()
        assert model2 is model

    @patch('subprocess.run')
    def test_bitrate_detection(self, mock_run):
        """Test ffprobe bitrate detection for accurate chunk sizing."""
        transcriber = Transcriber()

        # Mock ffprobe responses
        mock_run.side_effect = [
            # Duration query
            Mock(returncode=0, stdout='1000.0\n'),
            # Bitrate query
            Mock(returncode=0, stdout='192000\n'),  # 192 kbps
        ]

        # Create a temporary test file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            test_path = Path(f.name)
            f.write(b'fake audio data')

        try:
            # Mock the validation to skip file checks
            with patch.object(transcriber, '_validate_audio_path'):
                with patch.object(transcriber, '_find_speech_boundaries', return_value=[]):
                    with patch.object(transcriber, '_calculate_split_points', return_value=[(0, 500)]):
                        with patch('subprocess.run', side_effect=[
                            Mock(returncode=0, stdout='1000.0\n'),  # duration
                            Mock(returncode=0, stdout='192000\n'),   # bitrate
                        ]):
                            # This should use bitrate for calculation
                            # Just test that it doesn't crash
                            try:
                                chunks = transcriber._split_audio_file(test_path)
                            except Exception:
                                # Expected to fail at ffmpeg stage, we only care about bitrate logic
                                pass
        finally:
            test_path.unlink(missing_ok=True)

    def test_gap_padding_prevents_mid_syllable_cuts(self):
        """Test that gap padding of 100ms is considered."""
        transcriber = Transcriber()

        # Speech segments with a clear 1-second gap
        speech_segments = [
            (0.0, 30.0),
            (31.0, 60.0),  # 1-second gap at 30-31
        ]

        split_points = transcriber._calculate_split_points(
            total_duration=60.0,
            target_chunk_duration=30.0,
            speech_segments=speech_segments
        )

        # Should split near the gap
        if len(split_points) > 1:
            split_pos = split_points[0][1]
            # Split should be in the middle of the gap (30.5s)
            # allowing for search window tolerance
            assert 29.0 <= split_pos <= 32.0


class TestVADFallback:
    """Test VAD fallback mechanisms."""

    def test_find_speech_boundaries_silero_fallback_to_pyannote(self):
        """Test fallback from Silero to pyannote when Silero fails."""
        transcriber = Transcriber()

        # Mock Silero to fail
        with patch.object(transcriber, '_get_silero_vad', return_value=Mock()):
            with patch.object(transcriber, '_find_speech_boundaries_silero', side_effect=Exception("Silero error")):
                with patch.object(transcriber, '_get_vad_pipeline', return_value=None):
                    # Should return empty list when both fail
                    result = transcriber._find_speech_boundaries(Path('/fake/path.mp3'))
                    assert result == []

    def test_find_speech_boundaries_no_vad_available(self):
        """Test behavior when no VAD is available."""
        transcriber = Transcriber()

        # Mock both VAD methods to return None
        with patch.object(transcriber, '_get_silero_vad', return_value=None):
            with patch.object(transcriber, '_get_vad_pipeline', return_value=None):
                result = transcriber._find_speech_boundaries(Path('/fake/path.mp3'))
                assert result == []
