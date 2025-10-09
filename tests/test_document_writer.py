"""Tests for document_writer module"""
import pytest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from src.document_writer import DocumentWriter
from src.transcriber import TranscriptionSegment


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def writer(temp_output_dir, monkeypatch):
    """Create DocumentWriter with temp directory"""
    from src import config
    monkeypatch.setattr(config.settings, 'OUTPUT_DIR', temp_output_dir)
    return DocumentWriter()


class TestDocumentWriter:
    """Tests for DocumentWriter class"""

    def test_initialization(self, writer, temp_output_dir):
        """Test initialization"""
        assert writer.output_dir == temp_output_dir

    def test_create_documents_basic(self, writer, temp_output_dir):
        """Test creating basic documents"""
        title = "Test Document"
        sections = [{'title': 'Section 1', 'method': 'Test Method', 'content': 'Test content'}]

        docx_path, md_path = writer.create_documents(title, sections)

        assert docx_path.exists()
        assert md_path.exists()
        assert docx_path.suffix == '.docx'
        assert md_path.suffix == '.md'

    @patch('src.transcriber.Transcriber')
    def test_create_from_segments(self, mock_transcriber_class, writer, temp_output_dir):
        """Test creating documents from segments"""
        # Mock Transcriber to avoid heavy imports
        mock_transcriber = MagicMock()
        mock_transcriber.segments_to_text.return_value = "First segment Second segment"
        mock_transcriber_class.return_value = mock_transcriber

        title = "Segments Test"
        segments = [
            TranscriptionSegment(0, 5, "First segment"),
            TranscriptionSegment(5, 10, "Second segment")
        ]

        docx_path, md_path = writer.create_from_segments(
            title=title,
            transcription_segments=segments,
            transcribe_method="whisper_base",
            with_timestamps=False
        )

        assert docx_path.exists()
        assert md_path.exists()
        mock_transcriber.segments_to_text.assert_called_once()
