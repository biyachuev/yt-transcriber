"""
Tests for video_processor module.
"""
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.video_processor import VideoProcessor


@pytest.fixture
def video_processor():
    """Create a VideoProcessor instance."""
    return VideoProcessor()


@pytest.fixture
def mock_video_file(tmp_path):
    """Create a mock video file."""
    video_file = tmp_path / "test_video.mp4"
    video_file.write_text("fake video content")
    return video_file


def test_video_processor_init(video_processor):
    """Test VideoProcessor initialization."""
    assert video_processor.temp_dir is not None


def test_extract_audio_file_not_found(video_processor, tmp_path):
    """Test extract_audio with non-existent file."""
    non_existent = tmp_path / "non_existent.mp4"

    with pytest.raises(FileNotFoundError) as exc_info:
        video_processor.extract_audio(non_existent)

    assert "Video file not found" in str(exc_info.value)


@patch("subprocess.run")
def test_extract_audio_success(mock_run, video_processor, mock_video_file, tmp_path):
    """Test successful audio extraction."""
    # Setup temp_dir to tmp_path
    video_processor.temp_dir = tmp_path

    expected_output = tmp_path / "test_video.mp3"

    # Mock successful ffmpeg execution and create output file
    def side_effect(*args, **kwargs):
        # Create the output file when ffmpeg is "run"
        expected_output.write_text("fake audio content")
        return Mock(returncode=0, stderr="FFmpeg output")

    mock_run.side_effect = side_effect

    result = video_processor.extract_audio(mock_video_file)

    assert result == expected_output
    assert mock_run.called

    # Verify ffmpeg was called with correct arguments
    call_args = mock_run.call_args[0][0]
    assert "ffmpeg" in call_args
    assert "-i" in call_args
    assert str(mock_video_file) in call_args
    assert "-vn" in call_args


@patch("subprocess.run")
def test_extract_audio_already_exists(mock_run, video_processor, mock_video_file, tmp_path):
    """Test extract_audio when output file already exists."""
    video_processor.temp_dir = tmp_path

    # Create the output file beforehand
    expected_output = tmp_path / "test_video.mp3"
    expected_output.write_text("existing audio")

    result = video_processor.extract_audio(mock_video_file)

    # Should return existing file without calling ffmpeg
    assert result == expected_output
    assert not mock_run.called


@patch("subprocess.run")
def test_extract_audio_ffmpeg_fails(mock_run, video_processor, mock_video_file, tmp_path):
    """Test extract_audio when ffmpeg fails."""
    video_processor.temp_dir = tmp_path

    # Mock ffmpeg failure
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1,
        cmd="ffmpeg",
        stderr="FFmpeg error"
    )

    with pytest.raises(RuntimeError) as exc_info:
        video_processor.extract_audio(mock_video_file)

    assert "Failed to extract audio" in str(exc_info.value)


@patch("subprocess.run")
def test_extract_audio_ffmpeg_not_found(mock_run, video_processor, mock_video_file, tmp_path):
    """Test extract_audio when ffmpeg is not installed."""
    video_processor.temp_dir = tmp_path

    # Mock ffmpeg not found
    mock_run.side_effect = FileNotFoundError()

    with pytest.raises(RuntimeError) as exc_info:
        video_processor.extract_audio(mock_video_file)

    assert "FFmpeg not found" in str(exc_info.value)


@patch("subprocess.run")
def test_extract_audio_output_not_created(mock_run, video_processor, mock_video_file, tmp_path):
    """Test extract_audio when output file is not created."""
    video_processor.temp_dir = tmp_path

    # Mock successful ffmpeg execution but don't create output file
    mock_run.return_value = Mock(returncode=0, stderr="FFmpeg output")

    with pytest.raises(RuntimeError) as exc_info:
        video_processor.extract_audio(mock_video_file)

    assert "output file not created" in str(exc_info.value)


@patch("subprocess.run")
def test_get_video_duration_success(mock_run, video_processor, mock_video_file):
    """Test successful video duration retrieval."""
    # Mock ffprobe output
    mock_run.return_value = Mock(returncode=0, stdout="123.456\n", stderr="")

    duration = video_processor.get_video_duration(mock_video_file)

    assert duration == 123.456
    assert mock_run.called

    # Verify ffprobe was called
    call_args = mock_run.call_args[0][0]
    assert "ffprobe" in call_args
    assert str(mock_video_file) in call_args


@patch("subprocess.run")
def test_get_video_duration_failure(mock_run, video_processor, mock_video_file):
    """Test get_video_duration when ffprobe fails."""
    # Mock ffprobe failure
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1,
        cmd="ffprobe",
        stderr="Error"
    )

    duration = video_processor.get_video_duration(mock_video_file)

    assert duration is None


@patch("subprocess.run")
def test_get_video_duration_invalid_output(mock_run, video_processor, mock_video_file):
    """Test get_video_duration with invalid output."""
    # Mock invalid ffprobe output
    mock_run.return_value = Mock(returncode=0, stdout="invalid\n", stderr="")

    duration = video_processor.get_video_duration(mock_video_file)

    assert duration is None


@patch("subprocess.run")
def test_get_video_duration_empty_output(mock_run, video_processor, mock_video_file):
    """Test get_video_duration with empty output."""
    # Mock empty ffprobe output
    mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

    duration = video_processor.get_video_duration(mock_video_file)

    assert duration is None
