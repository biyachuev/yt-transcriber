"""
Tests for downloader module
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from src.downloader import YouTubeDownloader


@pytest.fixture
def temp_dir():
    """Create temporary directory"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def downloader(temp_dir, monkeypatch):
    """Create downloader with temp directory"""
    from src import config
    monkeypatch.setattr(config.settings, 'TEMP_DIR', temp_dir)
    return YouTubeDownloader()


class TestYouTubeDownloader:
    """Tests for YouTubeDownloader class"""

    def test_initialization(self, downloader, temp_dir):
        """Test initialization"""
        assert downloader.temp_dir == temp_dir
        assert downloader.progress_bar is None

    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires internet connection")
    def test_real_video_metadata(self):
        """Real test for metadata extraction (skipped by default)"""
        downloader = YouTubeDownloader()
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        metadata = downloader.extract_metadata(url)

        assert 'title' in metadata
        assert 'duration' in metadata
        assert metadata['duration'] > 0
