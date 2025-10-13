"""Advanced tests for utils module to increase coverage"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.utils import create_whisper_prompt, create_whisper_prompt_with_llm


class TestCreateWhisperPrompt:
    """Tests for create_whisper_prompt function"""

    def test_create_prompt_with_full_metadata(self):
        """Test prompt creation with full metadata"""
        metadata = {
            'title': 'Test Video Title',
            'description': 'Test description',
            'channel': 'Test Channel',
            'tags': ['tag1', 'tag2', 'tag3']
        }

        result = create_whisper_prompt(metadata)

        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain video title
        assert 'Test Video Title' in result or 'Test' in result

    def test_create_prompt_with_minimal_metadata(self):
        """Test prompt creation with minimal metadata"""
        metadata = {'title': 'Simple Title'}

        result = create_whisper_prompt(metadata)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_create_prompt_with_empty_metadata(self):
        """Test prompt creation with empty metadata"""
        metadata = {}

        result = create_whisper_prompt(metadata)

        # Should handle gracefully
        assert isinstance(result, str)

    def test_create_prompt_with_long_description(self):
        """Test prompt creation with very long description"""
        metadata = {
            'title': 'Test',
            'description': 'A' * 10000  # Very long description
        }

        result = create_whisper_prompt(metadata)

        # Should limit prompt length
        assert isinstance(result, str)
        assert len(result) < 1000  # Reasonable limit

    def test_create_prompt_with_many_tags(self):
        """Test prompt creation with many tags"""
        metadata = {
            'title': 'Test',
            'tags': [f'tag{i}' for i in range(100)]  # 100 tags
        }

        result = create_whisper_prompt(metadata)

        assert isinstance(result, str)
        assert len(result) > 0


class TestCreateWhisperPromptWithLLM:
    """Tests for create_whisper_prompt_with_llm function"""

    def test_without_ollama(self):
        """Test when ollama is disabled"""
        metadata = {'title': 'Test'}

        result = create_whisper_prompt_with_llm(metadata, use_ollama=False)

        assert isinstance(result, str)
        # Should fallback to basic prompt
        assert len(result) > 0

    @patch('src.utils.requests.post')
    def test_with_ollama_success(self, mock_post):
        """Test successful ollama call"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'Подкаст о программировании. Python, AI, машинное обучение.'
        }
        mock_post.return_value = mock_response

        metadata = {
            'title': 'Python AI Podcast',
            'description': 'Discussion about AI'
        }

        result = create_whisper_prompt_with_llm(metadata, use_ollama=True)

        assert isinstance(result, str)
        assert len(result) > 0
        mock_post.assert_called_once()

    @patch('src.utils.requests.post')
    def test_with_ollama_failure(self, mock_post):
        """Test ollama call failure"""
        mock_post.side_effect = Exception("Connection error")

        metadata = {'title': 'Test'}

        result = create_whisper_prompt_with_llm(metadata, use_ollama=True)

        # Should fallback to basic prompt
        assert isinstance(result, str)
        assert len(result) > 0

    @patch('src.utils.requests.post')
    def test_with_ollama_bad_status(self, mock_post):
        """Test ollama bad status code"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        metadata = {'title': 'Test'}

        result = create_whisper_prompt_with_llm(metadata, use_ollama=True)

        # Should fallback to basic prompt
        assert isinstance(result, str)

    @patch('src.utils.requests.post')
    def test_with_ollama_long_response(self, mock_post):
        """Test ollama with very long response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'A' * 10000  # Very long response
        }
        mock_post.return_value = mock_response

        metadata = {'title': 'Test'}

        result = create_whisper_prompt_with_llm(metadata, use_ollama=True)

        # Should truncate
        assert isinstance(result, str)
        assert len(result) <= 1000

    @patch('src.utils.requests.post')
    def test_with_subtitles_sample(self, mock_post):
        """Test with subtitles sample in metadata"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'Test prompt with context.'
        }
        mock_post.return_value = mock_response

        metadata = {
            'title': 'Test',
            'subtitles_sample': 'Hello, this is the first sentence.'
        }

        result = create_whisper_prompt_with_llm(metadata, use_ollama=True)

        assert isinstance(result, str)
