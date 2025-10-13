"""Edge case tests for various modules"""
import pytest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from src.transcriber import TranscriptionSegment, Transcriber
from src.translator import Translator
from src.text_reader import TextReader
from src.text_refiner import TextRefiner
from src.downloader import YouTubeDownloader
from src.utils import sanitize_filename, chunk_text, detect_language, format_timestamp


@pytest.fixture
def temp_dir():
    """Create temporary directory"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


class TestTranscriberEdgeCases:
    """Edge cases for Transcriber"""

    def test_segment_with_negative_time(self):
        """Test segment with negative start time"""
        # Should create segment even with negative time (no validation)
        segment = TranscriptionSegment(start=-1, end=5, text="Test")
        assert segment.start == -1

    def test_segment_with_end_before_start(self):
        """Test segment where end < start"""
        # Should create segment but may have invalid timestamp
        segment = TranscriptionSegment(start=10, end=5, text="Test")
        assert segment.start == 10
        assert segment.end == 5

    def test_segment_with_very_long_text(self):
        """Test segment with extremely long text"""
        long_text = "A" * 10000
        segment = TranscriptionSegment(start=0, end=100, text=long_text)
        assert len(segment.text) == 10000

    def test_segment_with_empty_text(self):
        """Test segment with empty text"""
        segment = TranscriptionSegment(start=0, end=5, text="")
        assert segment.text == ""

    def test_segment_with_special_characters(self):
        """Test segment with special characters"""
        text = "Test 🎉 \n\t\r Special: <>&\"'"
        segment = TranscriptionSegment(start=0, end=5, text=text)
        assert "🎉" in segment.text

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_segments_to_text_empty_list(self, mock_whisper, mock_torch):
        """Test converting empty segment list"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()
        result = transcriber.segments_to_text([])
        assert result == ""

    @patch('src.transcriber.torch')
    @patch('src.transcriber.whisper')
    def test_segments_to_text_single_segment(self, mock_whisper, mock_torch):
        """Test converting single segment"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        transcriber = Transcriber()
        segments = [TranscriptionSegment(0, 5, "Single")]
        result = transcriber.segments_to_text(segments)
        assert result == "Single"


class TestTranslatorEdgeCases:
    """Edge cases for Translator"""

    @patch('src.translator.torch')
    def test_translate_empty_text(self, mock_torch):
        """Test translating empty text"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        translator = Translator()
        result = translator.translate_text("")
        assert result == ""

    @patch('src.translator.torch')
    def test_translate_only_whitespace(self, mock_torch):
        """Test translating only whitespace"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        translator = Translator()
        result = translator.translate_text("   \n\t\r   ")
        # Should handle gracefully
        assert isinstance(result, str)


class TestTextReaderEdgeCases:
    """Edge cases for TextReader"""

    def test_read_text_with_bom(self, temp_dir):
        """Test reading file with BOM"""
        test_file = temp_dir / "bom.txt"
        # Write with BOM
        with open(test_file, 'w', encoding='utf-8-sig') as f:
            f.write("Test content")

        reader = TextReader()
        result = reader.read_text(str(test_file))
        assert "Test content" in result

    def test_read_very_large_file(self, temp_dir):
        """Test reading very large file"""
        test_file = temp_dir / "large.txt"
        # Create 1MB file
        content = "A" * 1000000
        test_file.write_text(content)

        reader = TextReader()
        result = reader.read_text(str(test_file))
        assert len(result) == 1000000

    def test_strip_markdown_complex(self):
        """Test stripping complex markdown"""
        reader = TextReader()
        markdown = """
        # Heading 1
        ## Heading 2
        **Bold** and *italic* and ***both***
        [Link](http://example.com)
        ![Image](image.png)
        `code` and ```code block```
        - List item
        1. Numbered
        > Quote
        """
        result = reader._strip_markdown(markdown)

        # Should remove markdown but keep text
        assert "Heading 1" in result
        assert "Bold" in result
        assert "italic" in result
        assert "**" not in result

    def test_detect_language_very_short(self):
        """Test language detection on very short text"""
        reader = TextReader()
        result = reader.detect_language("Hi")
        assert result in ['en', 'ru']

    def test_detect_language_numbers_only(self):
        """Test language detection on numbers"""
        reader = TextReader()
        result = reader.detect_language("123456")
        assert result in ['en', 'ru']

    def test_detect_language_mixed(self):
        """Test language detection on mixed content"""
        reader = TextReader()
        text = "English текст 123 !@#$%"
        result = reader.detect_language(text)
        assert result in ['en', 'ru']


class TestTextRefinerEdgeCases:
    """Edge cases for TextRefiner"""

    @patch('src.text_refiner.requests.get')
    def test_split_very_long_sentence(self, mock_get):
        """Test splitting very long sentence"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'models': [{'name': 'qwen2.5:3b'}]}
        mock_get.return_value = mock_response

        refiner = TextRefiner(model_name="qwen2.5:3b")

        # Create text with long sentences
        long_sentence = "This is a sentence. " * 200
        chunks = refiner._split_text_into_chunks(long_sentence, max_chunk_size=500)

        # Should split into multiple chunks
        assert len(chunks) >= 1

    @patch('src.text_refiner.requests.get')
    def test_split_only_punctuation(self, mock_get):
        """Test splitting text with only punctuation"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'models': [{'name': 'qwen2.5:3b'}]}
        mock_get.return_value = mock_response

        refiner = TextRefiner(model_name="qwen2.5:3b")
        text = "!!! ??? ..."
        chunks = refiner._split_text_into_chunks(text)
        assert len(chunks) >= 1

    @patch('src.text_refiner.requests.get')
    def test_detect_topic_empty_response(self, mock_get):
        """Test topic detection with empty response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'models': [{'name': 'qwen2.5:3b'}]}
        mock_get.return_value = mock_response

        refiner = TextRefiner(model_name="qwen2.5:3b")

        with patch.object(refiner, '_call_ollama', return_value=""):
            topic = refiner._detect_topic("Some text")
            assert topic == "общая"  # Default fallback


class TestDownloaderEdgeCases:
    """Edge cases for Downloader"""

    def test_progress_hook_with_estimate(self, temp_dir, monkeypatch):
        """Test progress hook with estimated bytes"""
        from src import config
        monkeypatch.setattr(config.settings, 'TEMP_DIR', temp_dir)

        downloader = YouTubeDownloader()

        d = {
            'status': 'downloading',
            'total_bytes_estimate': 1000000,  # Only estimate available
            'downloaded_bytes': 500000
        }

        downloader._progress_hook(d)
        assert downloader.progress_bar is not None

    def test_progress_hook_no_total(self, temp_dir, monkeypatch):
        """Test progress hook without total bytes"""
        from src import config
        monkeypatch.setattr(config.settings, 'TEMP_DIR', temp_dir)

        downloader = YouTubeDownloader()

        d = {
            'status': 'downloading',
            'downloaded_bytes': 500000
        }

        downloader._progress_hook(d)
        # Should handle missing total gracefully


class TestUtilsEdgeCases:
    """Edge cases for utility functions"""

    def test_sanitize_filename_only_invalid(self):
        """Test sanitizing filename with only invalid characters"""
        result = sanitize_filename("???///|||")
        assert result == "untitled"

    def test_sanitize_filename_very_long(self):
        """Test sanitizing very long filename"""
        long_name = "A" * 500
        result = sanitize_filename(long_name)
        assert len(result) <= 200

    def test_sanitize_filename_unicode_emoji(self):
        """Test sanitizing filename with emoji"""
        result = sanitize_filename("Test 🎉 File")
        # Emoji may be preserved or removed depending on implementation
        assert isinstance(result, str)
        assert len(result) > 0

    def test_chunk_text_empty(self):
        """Test chunking empty text"""
        result = chunk_text("")
        assert result == [""]

    def test_chunk_text_single_word(self):
        """Test chunking single word"""
        result = chunk_text("word")
        assert result == ["word"]

    def test_chunk_text_no_paragraphs(self):
        """Test chunking text without paragraph breaks"""
        text = "word " * 1000
        result = chunk_text(text)
        # Should split into chunks
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_detect_language_empty(self):
        """Test language detection on empty string"""
        result = detect_language("")
        assert result in ['en', 'ru']

    def test_format_timestamp_very_large(self):
        """Test formatting very large timestamp"""
        # 100 hours
        result = format_timestamp(360000)
        assert ":" in result

    def test_format_timestamp_fractional(self):
        """Test formatting fractional seconds"""
        result = format_timestamp(65.7)
        assert result == "01:05"
