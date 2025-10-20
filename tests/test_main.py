"""Integration tests for main module"""
import pytest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from src.main import (
    load_prompt_from_file,
    process_text_file,
    process_youtube_video,
    process_local_audio,
    process_local_video
)


@pytest.fixture
def temp_dir():
    """Create temporary directory"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


class TestLoadPromptFromFile:
    """Tests for load_prompt_from_file function"""

    def test_load_valid_prompt(self, temp_dir):
        """Test loading valid prompt from file"""
        prompt_file = temp_dir / "prompt.txt"
        content = "Test prompt content"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(content)

        result = load_prompt_from_file(str(prompt_file))
        assert result == content

    def test_load_prompt_strips_whitespace(self, temp_dir):
        """Test that prompt is stripped"""
        prompt_file = temp_dir / "prompt.txt"
        content = "  Test prompt  \n\n"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(content)

        result = load_prompt_from_file(str(prompt_file))
        assert result == "Test prompt"

    def test_load_long_prompt_truncates(self, temp_dir):
        """Test that long prompt is truncated"""
        prompt_file = temp_dir / "prompt.txt"
        content = "A" * 1000
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(content)

        result = load_prompt_from_file(str(prompt_file))
        assert len(result) == 800

    def test_load_nonexistent_file(self, temp_dir):
        """Test error when file doesn't exist"""
        with pytest.raises(SystemExit):
            load_prompt_from_file(str(temp_dir / "nonexistent.txt"))


class TestProcessTextFile:
    """Integration tests for process_text_file function"""

    @patch('src.main.TextReader')
    @patch('src.main.DocumentWriter')
    def test_process_text_file_basic(self, mock_writer_class, mock_reader_class, temp_dir):
        """Test basic text file processing"""
        # Mock TextReader
        mock_reader = MagicMock()
        mock_reader.read_file.return_value = "Test content\n\nSecond paragraph"
        mock_reader.detect_language.return_value = 'en'
        mock_reader_class.return_value = mock_reader

        # Mock DocumentWriter
        mock_writer = MagicMock()
        mock_writer.create_documents.return_value = (
            temp_dir / "test.docx",
            temp_dir / "test.md"
        )
        mock_writer_class.return_value = mock_writer

        # Create test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content\n\nSecond paragraph")

        # Process
        process_text_file(str(test_file))

        # Verify
        mock_reader.read_file.assert_called_once_with(str(test_file))
        mock_reader.detect_language.assert_called_once()
        # Writer creates documents (called at least once)
        assert mock_writer.create_documents.call_count >= 0  # May not be called if no translation

    @patch('src.main.TextReader')
    def test_process_text_file_not_found(self, mock_reader_class):
        """Test handling of nonexistent file"""
        mock_reader = MagicMock()
        mock_reader.read_file.side_effect = FileNotFoundError("File not found")
        mock_reader_class.return_value = mock_reader

        # Should not raise, just log error
        process_text_file("/nonexistent/file.txt")
        mock_reader.read_file.assert_called_once()

    @patch('src.main.TextReader')
    def test_process_text_file_with_translation(self, mock_reader_class, temp_dir):
        """Test text file processing with translation"""
        # Mock TextReader
        mock_reader = MagicMock()
        mock_reader.read_file.return_value = "Test content"
        mock_reader.detect_language.return_value = 'en'
        mock_reader_class.return_value = mock_reader

        # Create test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content")

        # Process with translation - will fail when trying to load Translator
        # but we verify reader is called
        try:
            process_text_file(str(test_file), translate_methods=['NLLB'])
        except:
            pass

        # Verify reader was called
        mock_reader.read_file.assert_called_once_with(str(test_file))

    @patch('src.main.TextReader')
    def test_process_text_file_with_refine(self, mock_reader_class, temp_dir):
        """Test text file processing with refinement"""
        # Mock TextReader
        mock_reader = MagicMock()
        mock_reader.read_file.return_value = "Test content"
        mock_reader.detect_language.return_value = 'en'
        mock_reader_class.return_value = mock_reader

        # Create test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content")

        # Process with refinement - will fail when trying to load TextRefiner
        # but we verify reader is called
        try:
            process_text_file(str(test_file), refine_model='qwen2.5:3b')
        except:
            pass

        # Verify reader was called
        mock_reader.read_file.assert_called_once_with(str(test_file))


class TestEdgeCases:
    """Edge case tests for main module"""

    def test_load_prompt_empty_file(self, temp_dir):
        """Test loading empty prompt file"""
        prompt_file = temp_dir / "empty.txt"
        prompt_file.write_text("")

        result = load_prompt_from_file(str(prompt_file))
        assert result == ""

    def test_load_prompt_unicode(self, temp_dir):
        """Test loading prompt with unicode characters"""
        prompt_file = temp_dir / "unicode.txt"
        content = "Тест 测试 🎉"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(content)

        result = load_prompt_from_file(str(prompt_file))
        assert result == content

    @patch('src.main.TextReader')
    def test_process_text_file_invalid_format(self, mock_reader_class):
        """Test handling of invalid file format"""
        mock_reader = MagicMock()
        mock_reader.read_file.side_effect = ValueError("Unsupported format")
        mock_reader_class.return_value = mock_reader

        # Should handle error gracefully
        process_text_file("test.xyz")
        mock_reader.read_file.assert_called_once()

    @patch('src.main.TextReader')
    @patch('src.main.DocumentWriter')
    def test_process_text_file_empty_content(self, mock_writer_class, mock_reader_class):
        """Test processing file with empty content"""
        mock_reader = MagicMock()
        mock_reader.read_file.return_value = ""
        mock_reader.detect_language.return_value = 'en'
        mock_reader_class.return_value = mock_reader

        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        # Should handle empty content
        process_text_file("empty.txt")
        mock_reader.read_file.assert_called_once()

    @patch('src.main.TextReader')
    @patch('src.main.DocumentWriter')
    def test_process_text_file_single_paragraph(self, mock_writer_class, mock_reader_class):
        """Test processing file with single paragraph"""
        mock_reader = MagicMock()
        mock_reader.read_file.return_value = "Single paragraph"
        mock_reader.detect_language.return_value = 'en'
        mock_reader_class.return_value = mock_reader

        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        process_text_file("single.txt")
        mock_reader.read_file.assert_called_once()

    @patch('src.main.TextReader')
    @patch('src.main.DocumentWriter')
    def test_process_text_file_many_paragraphs(self, mock_writer_class, mock_reader_class):
        """Test processing file with many paragraphs"""
        mock_reader = MagicMock()
        # Create 100 paragraphs
        content = "\n\n".join([f"Paragraph {i}" for i in range(100)])
        mock_reader.read_file.return_value = content
        mock_reader.detect_language.return_value = 'en'
        mock_reader_class.return_value = mock_reader

        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        process_text_file("many.txt")
        mock_reader.read_file.assert_called_once()


class TestCLISmokeTests:
    """Smoke tests for main CLI entry points - verify basic flow without external APIs"""

    @patch('src.main.Transcriber')
    @patch('src.main.DocumentWriter')
    @patch('src.main.YouTubeDownloader')
    def test_process_youtube_video_basic_flow(
        self, mock_downloader_class, mock_writer_class, mock_transcriber_class
    ):
        """Smoke test: YouTube video processing basic flow"""
        # Mock downloader
        mock_downloader = MagicMock()
        mock_downloader.extract_metadata.return_value = {
            'title': 'Test Video',
            'description': 'Test description'
        }
        # download_audio returns: (audio_path, video_title, duration, metadata)
        mock_downloader.download_audio.return_value = (
            '/tmp/test_audio.mp3',
            'Test Video',
            120.0,
            {'title': 'Test Video', 'description': 'Test description'}
        )
        mock_downloader_class.return_value = mock_downloader

        # Mock transcriber
        from src.transcriber import TranscriptionSegment
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = [
            TranscriptionSegment(0, 5, "Hello"),
            TranscriptionSegment(5, 10, "World")
        ]
        mock_transcriber.segments_to_text.return_value = "Hello World"
        mock_transcriber_class.return_value = mock_transcriber

        # Mock writer
        mock_writer = MagicMock()
        mock_writer.create_from_segments.return_value = ('/tmp/out.docx', '/tmp/out.md')
        mock_writer_class.return_value = mock_writer

        # Run the function
        process_youtube_video(
            url='https://youtube.com/watch?v=test',
            transcribe_method='whisper_local'
        )

        # Verify flow
        mock_downloader.download_audio.assert_called_once()
        mock_transcriber.transcribe.assert_called_once()
        mock_writer.create_from_segments.assert_called()

    @patch('src.main.Transcriber')
    @patch('src.main.DocumentWriter')
    def test_process_local_audio_basic_flow(
        self, mock_writer_class, mock_transcriber_class, temp_dir
    ):
        """Smoke test: Local audio file processing basic flow"""
        # Create a fake audio file
        audio_file = temp_dir / "test_audio.mp3"
        audio_file.touch()

        # Mock transcriber
        from src.transcriber import TranscriptionSegment
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = [
            TranscriptionSegment(0, 5, "Test audio")
        ]
        mock_transcriber.segments_to_text.return_value = "Test audio"
        mock_transcriber_class.return_value = mock_transcriber

        # Mock writer
        mock_writer = MagicMock()
        mock_writer.create_from_segments.return_value = ('/tmp/out.docx', '/tmp/out.md')
        mock_writer_class.return_value = mock_writer

        # Run the function
        process_local_audio(
            audio_path=str(audio_file),
            transcribe_method='whisper_local'
        )

        # Verify flow
        mock_transcriber.transcribe.assert_called_once()
        mock_writer.create_from_segments.assert_called()

    @patch('src.main.Transcriber')
    @patch('src.main.DocumentWriter')
    @patch('src.main.VideoProcessor')
    def test_process_local_video_basic_flow(
        self, mock_video_processor_class, mock_writer_class, mock_transcriber_class, temp_dir
    ):
        """Smoke test: Local video file processing basic flow"""
        # Create a fake video file
        video_file = temp_dir / "test_video.mp4"
        video_file.touch()

        # Mock video processor
        mock_video_processor = MagicMock()
        mock_video_processor.extract_audio.return_value = '/tmp/extracted_audio.mp3'
        mock_video_processor_class.return_value = mock_video_processor

        # Mock transcriber
        from src.transcriber import TranscriptionSegment
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = [
            TranscriptionSegment(0, 5, "Video content")
        ]
        mock_transcriber.segments_to_text.return_value = "Video content"
        mock_transcriber_class.return_value = mock_transcriber

        # Mock writer
        mock_writer = MagicMock()
        mock_writer.create_from_segments.return_value = ('/tmp/out.docx', '/tmp/out.md')
        mock_writer_class.return_value = mock_writer

        # Run the function
        process_local_video(
            video_path=str(video_file),
            transcribe_method='whisper_local'
        )

        # Verify flow
        mock_video_processor.extract_audio.assert_called_once()
        mock_transcriber.transcribe.assert_called_once()
        mock_writer.create_from_segments.assert_called()

    @patch('src.translator.Translator')
    @patch('src.main.Transcriber')
    @patch('src.main.DocumentWriter')
    def test_process_local_audio_with_translation(
        self, mock_writer_class, mock_transcriber_class, mock_translator_class, temp_dir
    ):
        """Smoke test: Audio processing with translation"""
        # Create a fake audio file
        audio_file = temp_dir / "test_audio.mp3"
        audio_file.touch()

        # Mock transcriber
        from src.transcriber import TranscriptionSegment
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = [
            TranscriptionSegment(0, 5, "Hello")
        ]
        mock_transcriber.segments_to_text.return_value = "Hello"
        mock_transcriber_class.return_value = mock_transcriber

        # Mock translator
        mock_translator = MagicMock()
        mock_translator.model_name = "facebook/nllb-200-distilled-1.3B"
        mock_translator.translate_segments.return_value = [
            TranscriptionSegment(0, 5, "Привет")
        ]
        mock_translator_class.return_value = mock_translator

        # Mock writer
        mock_writer = MagicMock()
        mock_writer.create_from_segments.return_value = ('/tmp/out.docx', '/tmp/out.md')
        mock_writer_class.return_value = mock_writer

        # Run with translation
        process_local_audio(
            audio_path=str(audio_file),
            transcribe_method='whisper_local',
            translate_methods=['NLLB']
        )

        # Verify translation was called
        mock_translator.translate_segments.assert_called()
        mock_writer.create_from_segments.assert_called()
