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

    @patch('src.downloader.yt_dlp.YoutubeDL')
    def test_extract_metadata(self, mock_yt_dlp, downloader):
        """Test metadata extraction"""
        mock_ydl_instance = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance

        mock_info = {
            'title': 'Test Video',
            'duration': 180,
            'uploader': 'Test Channel',
            'view_count': 1000
        }
        mock_ydl_instance.extract_info.return_value = mock_info

        url = "https://www.youtube.com/watch?v=test"
        metadata = downloader.extract_metadata(url)

        assert metadata['title'] == 'Test Video'
        assert metadata['duration'] == 180
        mock_ydl_instance.extract_info.assert_called_once_with(url, download=False)

    @patch('src.downloader.yt_dlp.YoutubeDL')
    def test_download_audio(self, mock_yt_dlp, downloader, temp_dir):
        """Test audio download"""
        mock_ydl_instance = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance

        mock_info = {
            'title': 'Test Video',
            'duration': 180,
        }
        mock_ydl_instance.extract_info.return_value = mock_info

        # Create fake audio file
        audio_file = temp_dir / "Test_Video.mp3"
        audio_file.touch()

        url = "https://www.youtube.com/watch?v=test"
        result_path, title, duration, metadata = downloader.download_audio(url)

        assert result_path.exists()
        assert title == 'Test Video'
        assert duration == 180
        mock_ydl_instance.download.assert_called_once()

    def test_progress_hook_downloading(self, downloader):
        """Test progress hook during download"""
        d = {
            'status': 'downloading',
            'total_bytes': 1000000,
            'downloaded_bytes': 500000
        }

        downloader._progress_hook(d)

        assert downloader.progress_bar is not None
        assert downloader.progress_bar.total == 1000000

    def test_progress_hook_finished(self, downloader):
        """Test progress hook when finished"""
        # First trigger downloading to create progress bar
        d_downloading = {
            'status': 'downloading',
            'total_bytes': 1000000,
            'downloaded_bytes': 500000
        }
        downloader._progress_hook(d_downloading)

        # Then trigger finished
        d_finished = {'status': 'finished'}
        downloader._progress_hook(d_finished)

        assert downloader.progress_bar is None

    @patch('src.downloader.yt_dlp.YoutubeDL')
    def test_download_audio_file_not_found(self, mock_yt_dlp, downloader):
        """Test error when downloaded file is missing"""
        mock_ydl_instance = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance

        mock_info = {
            'title': 'Missing Video',
            'duration': 180,
        }
        mock_ydl_instance.extract_info.return_value = mock_info

        url = "https://www.youtube.com/watch?v=test"

        with pytest.raises(FileNotFoundError):
            downloader.download_audio(url)

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
