# Unit Tests for YouTube Transcriber

This directory contains unit tests for the YouTube Transcriber project.

## Test Structure

```
tests/
├── __init__.py
├── test_utils.py              # Tests for utility functions
├── test_transcriber.py        # Tests for transcription functionality
├── test_translator.py         # Tests for translation functionality
├── test_document_writer.py    # Tests for document generation
├── test_text_reader.py        # Tests for text reading
├── test_text_refiner.py       # Tests for LLM-based text refinement
└── test_downloader.py         # Tests for YouTube downloader
```

## Running Tests

### Run all tests:
```bash
pytest
```

### Run specific test file:
```bash
pytest tests/test_utils.py -v
```

### Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

### Run without coverage:
```bash
pytest --no-cov
```

### Run only unit tests (skip integration tests):
```bash
pytest -m "not integration"
```

### Run only integration tests:
```bash
pytest -m integration
```

## Test Markers

- `@pytest.mark.integration` - Integration tests requiring external services (Ollama, YouTube, etc.)
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.unit` - Fast unit tests

## Coverage Reports

After running tests with coverage, you can view the HTML report:

```bash
open htmlcov/index.html
```

## Current Test Coverage

- **utils.py**: Utility functions (sanitize_filename, format_timestamp, etc.)
- **transcriber.py**: TranscriptionSegment and Transcriber classes
- **translator.py**: Translator class
- **document_writer.py**: DocumentWriter class
- **text_reader.py**: TextReader class
- **text_refiner.py**: TextRefiner class
- **downloader.py**: YouTubeDownloader class

## Writing New Tests

### Test Structure Example:

```python
import pytest
from src.module_name import ClassName


class TestClassName:
    """Tests for ClassName"""

    def test_basic_functionality(self):
        """Test description"""
        obj = ClassName()
        result = obj.method()
        assert result == expected_value

    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires external service")
    def test_integration(self):
        """Integration test (skipped by default)"""
        pass
```

### Fixtures

Use fixtures for setup and teardown:

```python
@pytest.fixture
def temp_dir():
    """Create temporary directory"""
    import tempfile
    import shutil
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)
```

## Configuration

Test configuration is in `pytest.ini` at the root of the project.

Key settings:
- Minimum coverage: 20%
- Integration tests are skipped by default
- HTML and XML coverage reports are generated
