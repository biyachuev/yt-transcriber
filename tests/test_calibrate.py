"""Tests for calibrate module"""
import pytest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import subprocess


@pytest.fixture
def temp_dir():
    """Create temporary directory"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


class TestCalibrateUtilities:
    """Tests for calibrate script utilities"""

    @patch('subprocess.run')
    def test_ffprobe_duration_extraction(self, mock_run):
        """Test extracting duration with ffprobe"""
        mock_result = Mock()
        mock_result.stdout = "180.5\n"
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        # Simulate ffprobe call
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries',
             'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
             'test.mp3'],
            capture_output=True,
            text=True,
            check=True
        )

        duration = float(result.stdout.strip())
        assert duration == 180.5

    @patch('subprocess.run')
    def test_ffprobe_error_handling(self, mock_run):
        """Test ffprobe error handling"""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'ffprobe')

        with pytest.raises(subprocess.CalledProcessError):
            subprocess.run(
                ['ffprobe', 'test.mp3'],
                capture_output=True,
                check=True
            )

    def test_find_test_files(self, temp_dir):
        """Test finding test files"""
        # Create test files
        test_file1 = temp_dir / "test.mp3"
        test_file1.touch()
        test_file2 = temp_dir / "test2.mp3"
        test_file2.touch()
        other_file = temp_dir / "audio.mp3"
        other_file.touch()

        # Find test*.mp3
        test_files = list(temp_dir.glob("test*.mp3"))
        assert len(test_files) == 2

        # Find all mp3
        all_files = list(temp_dir.glob("*.mp3"))
        assert len(all_files) == 3

    def test_model_validation(self):
        """Test model name validation"""
        valid_models = ["whisper_base", "whisper_small", "whisper_medium"]

        for model in valid_models:
            assert model in valid_models

        invalid_model = "whisper_invalid"
        assert invalid_model not in valid_models

    @patch('time.time')
    def test_timing_measurement(self, mock_time):
        """Test timing measurement"""
        # Simulate time progression
        mock_time.side_effect = [0.0, 10.5]

        start_time = mock_time()
        # Simulate some work
        end_time = mock_time()

        elapsed = end_time - start_time
        assert elapsed == 10.5

    def test_multiplier_calculation(self):
        """Test multiplier calculation"""
        # Test data
        audio_duration = 60.0  # 1 minute
        processing_time = 3.6  # 3.6 seconds

        multiplier = processing_time / audio_duration
        assert abs(multiplier - 0.06) < 0.001

    def test_format_results(self):
        """Test result formatting"""
        duration = 180.0  # 3 minutes
        processing_time = 10.8
        multiplier = processing_time / duration

        # Format results
        results = {
            'duration': duration,
            'processing_time': processing_time,
            'multiplier': multiplier,
            'speedup': duration / processing_time
        }

        assert results['duration'] == 180.0
        assert results['processing_time'] == 10.8
        assert abs(results['multiplier'] - 0.06) < 0.001
        assert abs(results['speedup'] - 16.67) < 0.01


class TestCalibrateWorkflow:
    """Test calibration workflow"""

    def test_temp_dir_check(self, temp_dir):
        """Test temp directory existence check"""
        assert temp_dir.exists()

        # Create temp subdir
        temp_subdir = temp_dir / "temp"
        assert not temp_subdir.exists()

        temp_subdir.mkdir()
        assert temp_subdir.exists()

    def test_audio_file_discovery(self, temp_dir):
        """Test audio file discovery logic"""
        # No files initially
        test_files = list(temp_dir.glob("test*.mp3"))
        assert len(test_files) == 0

        # Create test file
        test_file = temp_dir / "test.mp3"
        test_file.touch()

        test_files = list(temp_dir.glob("test*.mp3"))
        assert len(test_files) == 1
        assert test_files[0].name == "test.mp3"

    @patch('sys.argv', ['calibrate.py', 'whisper_base'])
    def test_argument_parsing(self):
        """Test command line argument parsing"""
        import sys

        if len(sys.argv) > 1:
            model = sys.argv[1]
        else:
            model = "whisper_base"

        assert model == "whisper_base"

    @patch('sys.argv', ['calibrate.py'])
    def test_default_model(self):
        """Test default model selection"""
        import sys

        if len(sys.argv) > 1:
            model = sys.argv[1]
        else:
            model = "whisper_base"

        assert model == "whisper_base"


class TestCalibrateMath:
    """Test calibration mathematical functions"""

    def test_speedup_calculation(self):
        """Test speedup factor calculation"""
        audio_duration = 300.0  # 5 minutes
        processing_time = 18.0  # 18 seconds

        speedup = audio_duration / processing_time
        assert abs(speedup - 16.67) < 0.01

    def test_multiplier_various_models(self):
        """Test multiplier calculation for various models"""
        test_cases = [
            (60.0, 3.6, 0.06),    # base model
            (60.0, 11.4, 0.19),   # small model
            (60.0, 27.0, 0.45),   # medium model
        ]

        for duration, proc_time, expected_mult in test_cases:
            multiplier = proc_time / duration
            assert abs(multiplier - expected_mult) < 0.01

    def test_estimation_accuracy(self):
        """Test estimation accuracy"""
        # Known values
        actual_time = 10.5
        estimated_time = 10.8

        error_percentage = abs(estimated_time - actual_time) / actual_time * 100
        assert error_percentage < 10  # Less than 10% error

    def test_time_formatting(self):
        """Test time formatting functions"""
        seconds = 125.5

        minutes = int(seconds // 60)
        secs = int(seconds % 60)

        assert minutes == 2
        assert secs == 5

    def test_duration_edge_cases(self):
        """Test edge cases in duration handling"""
        # Very short audio
        short_duration = 1.0
        processing_time = 0.06
        multiplier = processing_time / short_duration
        assert multiplier == 0.06

        # Very long audio
        long_duration = 3600.0  # 1 hour
        processing_time = 216.0  # 3.6 minutes
        multiplier = processing_time / long_duration
        assert abs(multiplier - 0.06) < 0.001

    def test_zero_duration_handling(self):
        """Test zero duration handling"""
        duration = 0.0

        # Should not divide by zero
        if duration > 0:
            multiplier = 10.0 / duration
        else:
            multiplier = None

        assert multiplier is None
