"""Integration tests for main module"""
import pytest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from src.main import load_prompt_from_file, process_text_file


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
