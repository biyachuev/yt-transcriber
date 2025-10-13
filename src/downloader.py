"""
YouTube downloader utility used to fetch audio and metadata.
"""
from pathlib import Path
from typing import Tuple, Optional

import yt_dlp
from tqdm import tqdm

from .config import settings
from .logger import logger
from .utils import sanitize_filename


class YouTubeDownloader:
    """Download YouTube audio tracks together with auxiliary metadata."""

    def __init__(self):
        self.temp_dir = settings.TEMP_DIR
        self.progress_bar: Optional[tqdm] = None

    def _progress_hook(self, data: dict):
        """Hook passed to yt_dlp to display download progress."""
        status = data.get("status")
        if status == "downloading":
            if self.progress_bar is None:
                total = data.get("total_bytes") or data.get("total_bytes_estimate", 0)
                self.progress_bar = tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc="Downloading",
                )

            downloaded = data.get("downloaded_bytes", 0)
            if self.progress_bar.n < downloaded:
                self.progress_bar.update(downloaded - self.progress_bar.n)

        elif status == "finished":
            if self.progress_bar:
                self.progress_bar.close()
                self.progress_bar = None
            logger.info("Download complete, starting post-processing...")

    def download_audio(self, url: str) -> Tuple[Path, str, float, dict]:
        """
        Download the best audio stream from YouTube and convert it to MP3.

        Args:
            url: YouTube video URL.

        Returns:
            Tuple containing audio file path, title, duration (seconds), and metadata.
        """
        logger.info("Starting YouTube download: %s", url)

        metadata = self.extract_metadata(url)

        video_title = metadata["title"]
        duration = metadata["duration"]

        logger.info("Video title: %s", video_title)
        logger.info("Duration: %d min %d sec", duration // 60, duration % 60)

        clean_title = sanitize_filename(video_title)
        output_path = self.temp_dir / f"{clean_title}.%(ext)s"

        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": str(output_path),
            "progress_hooks": [self._progress_hook],
            "quiet": False,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        audio_file = self.temp_dir / f"{clean_title}.mp3"

        if not audio_file.exists():
            raise FileNotFoundError(f"Expected audio file not found: {audio_file}")

        logger.info("Audio saved to: %s", audio_file)

        return audio_file, video_title, duration, metadata

    def extract_metadata(self, url: str) -> dict:
        """
        Fetch metadata (including subtitles) without downloading the media.

        Args:
            url: YouTube video URL.

        Returns:
            Metadata dictionary (title, description, tags, subtitles, etc.).
        """
        logger.info("Extracting video metadata...")

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "ru"],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        metadata = {
            "title": info.get("title", "untitled"),
            "duration": info.get("duration", 0),
            "description": info.get("description", ""),
            "uploader": info.get("uploader", ""),
            "upload_date": info.get("upload_date", ""),
            "tags": info.get("tags", []),
            "categories": info.get("categories", []),
            "channel": info.get("channel", ""),
        }

        logger.info("Video metadata:")
        logger.info("  Title: %s", metadata["title"])
        logger.info("  Uploader: %s", metadata["uploader"])
        logger.info("  Channel: %s", metadata["channel"])
        if metadata["tags"]:
            logger.info("  Tags: %s", ", ".join(metadata["tags"][:10]))
        if metadata["categories"]:
            logger.info("  Categories: %s", ", ".join(metadata["categories"]))

        subtitles_info = info.get("subtitles", {})
        automatic_captions = info.get("automatic_captions", {})

        subtitle_sample = ""
        if subtitles_info or automatic_captions:
            logger.info("Subtitles detected:")

            subtitle_url = None

            if "en" in subtitles_info:
                logger.info("  - English subtitles (manual)")
                if subtitles_info["en"]:
                    subtitle_url = subtitles_info["en"][0].get("url")
            elif "en" in automatic_captions:
                logger.info("  - English subtitles (auto)")
                if automatic_captions["en"]:
                    subtitle_url = automatic_captions["en"][0].get("url")
            elif "ru" in subtitles_info:
                logger.info("  - Russian subtitles (manual)")
                if subtitles_info["ru"]:
                    subtitle_url = subtitles_info["ru"][0].get("url")
            elif "ru" in automatic_captions:
                logger.info("  - Russian subtitles (auto)")
                if automatic_captions["ru"]:
                    subtitle_url = automatic_captions["ru"][0].get("url")

            if subtitle_url:
                try:
                    import requests

                    response = requests.get(subtitle_url, timeout=10)
                    if response.status_code == 200:
                        import re

                        subtitle_text = response.text
                        subtitle_text = re.sub(r"<[^>]+>", "", subtitle_text)
                        subtitle_text = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3}", "", subtitle_text)
                        subtitle_sample = subtitle_text[:2000].strip()
                        logger.info(
                            "  Loaded subtitle sample (%d chars)", len(subtitle_sample)
                        )
                except Exception as e:  # pragma: no cover
                    logger.warning("  Failed to download subtitles: %s", e)

            metadata["has_subtitles"] = True
            metadata["subtitles_sample"] = subtitle_sample
        else:
            logger.info("No subtitles available")
            metadata["has_subtitles"] = False
            metadata["subtitles_sample"] = ""

        return metadata

    def get_video_info(self, url: str) -> dict:
        """
        Convenience wrapper returning metadata without downloading media.

        Args:
            url: YouTube video URL.

        Returns:
            Metadata dictionary.
        """
        return self.extract_metadata(url)
