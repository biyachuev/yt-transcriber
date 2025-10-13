"""
Video processor utility for extracting audio from video files.
"""
import subprocess
from pathlib import Path
from typing import Optional

from .config import settings
from .logger import logger
from .utils import sanitize_filename


class VideoProcessor:
    """Extract audio from video files using FFmpeg."""

    def __init__(self):
        self.temp_dir = settings.TEMP_DIR

    def extract_audio(self, video_path: Path) -> Path:
        """
        Extract audio from a video file and convert it to MP3.

        Args:
            video_path: Path to the video file (mp4, mkv, avi, etc.).

        Returns:
            Path to the extracted audio file (MP3).

        Raises:
            FileNotFoundError: If the video file does not exist.
            RuntimeError: If FFmpeg extraction fails.
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info("Extracting audio from video: %s", video_path.name)

        # Generate output path.
        video_title = sanitize_filename(video_path.stem)
        audio_output = self.temp_dir / f"{video_title}.mp3"

        # Check if audio file already exists (avoid re-extraction).
        if audio_output.exists():
            logger.info("Audio file already exists, skipping extraction: %s", audio_output)
            return audio_output

        # Build FFmpeg command.
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "libmp3lame",
            "-q:a", "2",  # High quality
            "-y",  # Overwrite output file if exists
            str(audio_output)
        ]

        logger.info("Running FFmpeg to extract audio...")
        logger.debug("FFmpeg command: %s", " ".join(ffmpeg_cmd))

        try:
            result = subprocess.run(
                ffmpeg_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            logger.debug("FFmpeg output: %s", result.stderr)

        except subprocess.CalledProcessError as e:
            logger.error("FFmpeg failed with return code %d", e.returncode)
            logger.error("FFmpeg stderr: %s", e.stderr)
            raise RuntimeError(f"Failed to extract audio from video: {e.stderr}")

        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg:\n"
                "  macOS: brew install ffmpeg\n"
                "  Linux: sudo apt install ffmpeg\n"
                "  Windows: Download from https://ffmpeg.org/download.html"
            )

        if not audio_output.exists():
            raise RuntimeError(f"Audio extraction failed: output file not created at {audio_output}")

        logger.info("Audio extracted successfully: %s", audio_output)

        return audio_output

    def get_video_duration(self, video_path: Path) -> Optional[float]:
        """
        Get video duration in seconds using FFprobe.

        Args:
            video_path: Path to the video file.

        Returns:
            Duration in seconds, or None if unable to determine.
        """
        ffprobe_cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]

        try:
            result = subprocess.run(
                ffprobe_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            duration_str = result.stdout.strip()
            if duration_str:
                return float(duration_str)

        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            logger.warning("Unable to determine video duration")

        return None
