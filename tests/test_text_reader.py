"""Tests for text_reader module"""
import pytest
from pathlib import Path
import tempfile
import shutil
from docx import Document
from src.text_reader import TextReader


@pytest.fixture
def temp_dir():
    """Create temporary directory"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def reader():
    """Create TextReader instance"""
    return TextReader()


class TestTextReader:
    """Tests for TextReader class"""

    def test_initialization(self, reader):
        """Test initialization"""
        assert reader is not None

    def test_read_text_file(self, reader, temp_dir):
        """Test reading text file"""
        test_file = temp_dir / "test.txt"
        content = "Test content\nSecond line"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(content)

        result = reader.read_text(str(test_file))
        assert result == content

    def test_detect_language_russian(self, reader):
        """Test detecting Russian language"""
        text = "Это тестовый текст на русском языке"
        result = reader.detect_language(text)
        assert result == 'ru'

    def test_detect_language_english(self, reader):
        """Test detecting English language"""
        text = "This is a test text in English"
        result = reader.detect_language(text)
        assert result == 'en'
