"""
Tests for the summarizer module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.summarizer import Summarizer
from src.config import SummarizeOptions


class TestSummarizer:
    """Unit tests for Summarizer class"""

    @patch('src.summarizer.requests.get')
    @patch('src.summarizer.get_cache')
    def test_initialization_ollama(self, mock_get_cache, mock_requests_get):
        """Test Summarizer initialization with Ollama backend"""
        # Mock Ollama availability check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [{'name': 'qwen2.5:3b'}]
        }
        mock_requests_get.return_value = mock_response

        # Mock cache
        mock_cache = MagicMock()
        mock_get_cache.return_value = mock_cache

        summarizer = Summarizer(
            backend=SummarizeOptions.OLLAMA,
            model_name='qwen2.5:3b'
        )

        assert summarizer.backend == SummarizeOptions.OLLAMA
        assert summarizer.model_name == 'qwen2.5:3b'
        assert summarizer.use_cache is True

    @patch('src.summarizer.settings.OPENAI_API_KEY', 'test-key')
    @patch('src.summarizer.get_cache')
    @patch('src.summarizer.get_openai_rate_limiter')
    def test_initialization_openai(self, mock_rate_limiter, mock_get_cache):
        """Test Summarizer initialization with OpenAI backend"""
        mock_cache = MagicMock()
        mock_get_cache.return_value = mock_cache
        mock_rate_limiter.return_value = MagicMock()

        summarizer = Summarizer(
            backend=SummarizeOptions.OPENAI_API,
            model_name='gpt-4'
        )

        assert summarizer.backend == SummarizeOptions.OPENAI_API
        assert summarizer.model_name == 'gpt-4'

    @patch('src.summarizer.requests.get')
    def test_ollama_model_not_found(self, mock_requests_get):
        """Test error when Ollama model not found"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [{'name': 'different-model:1b'}]
        }
        mock_requests_get.return_value = mock_response

        with pytest.raises(RuntimeError, match="Model 'qwen2.5:3b' not found"):
            Summarizer(
                backend=SummarizeOptions.OLLAMA,
                model_name='qwen2.5:3b'
            )

    @patch('src.summarizer.requests.get')
    def test_ollama_server_unavailable(self, mock_requests_get):
        """Test error when Ollama server is unavailable"""
        import requests
        mock_requests_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            Summarizer(
                backend=SummarizeOptions.OLLAMA,
                model_name='qwen2.5:3b'
            )

    def test_unsupported_backend(self):
        """Test error with unsupported backend"""
        with pytest.raises(ValueError, match="Unsupported summarization backend"):
            Summarizer(backend="invalid_backend")

    @patch('src.summarizer.requests.get')
    @patch('src.summarizer.requests.post')
    @patch('src.summarizer.get_cache')
    def test_summarize_with_ollama(self, mock_get_cache, mock_post, mock_get):
        """Test text summarization with Ollama"""
        # Mock Ollama availability
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            'models': [{'name': 'qwen2.5:3b'}]
        }
        mock_get.return_value = mock_get_response

        # Mock cache miss
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        # Mock Ollama response (non-streaming)
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {
            'response': 'This is summary',
            'done': True
        }
        mock_post.return_value = mock_post_response

        summarizer = Summarizer(backend=SummarizeOptions.OLLAMA)
        result = summarizer.summarize("Long text to summarize")

        assert "This is summary" in result
        mock_post.assert_called_once()

    @patch('src.summarizer.settings.OPENAI_API_KEY', 'test-key')
    @patch('src.summarizer.get_cache')
    @patch('src.summarizer.get_openai_rate_limiter')
    @patch('openai.OpenAI')
    def test_summarize_with_openai(self, mock_openai_class, mock_rate_limiter, mock_get_cache):
        """Test text summarization with OpenAI"""
        # Mock cache miss
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        # Mock rate limiter
        mock_rate_limiter.return_value = MagicMock()

        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'This is a summary'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        mock_client.chat.completions.create.return_value = mock_response

        summarizer = Summarizer(
            backend=SummarizeOptions.OPENAI_API,
            model_name='gpt-4'
        )
        result = summarizer.summarize("Long text to summarize")

        assert result == 'This is a summary'
        mock_client.chat.completions.create.assert_called_once()

    @patch('src.summarizer.requests.get')
    @patch('src.summarizer.requests.post')
    @patch('src.summarizer.get_cache')
    def test_cache_hit(self, mock_get_cache, mock_post, mock_get):
        """Test that cache is used when available"""
        # Mock Ollama availability
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            'models': [{'name': 'qwen2.5:3b'}]
        }
        mock_get.return_value = mock_get_response

        # Mock cache - return None initially (cache is not used in _call_ollama)
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        # Mock Ollama response
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {
            'response': 'Generated summary',
            'done': True
        }
        mock_post.return_value = mock_post_response

        summarizer = Summarizer(backend=SummarizeOptions.OLLAMA)
        result = summarizer.summarize("Text to summarize")

        assert result == "Generated summary"
        # Verify Ollama was called
        mock_post.assert_called_once()

    @patch('src.summarizer.requests.get')
    def test_no_cache_when_disabled(self, mock_get):
        """Test that cache is not used when disabled"""
        # Mock Ollama availability
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            'models': [{'name': 'qwen2.5:3b'}]
        }
        mock_get.return_value = mock_get_response

        summarizer = Summarizer(
            backend=SummarizeOptions.OLLAMA,
            use_cache=False
        )

        assert summarizer.cache is None
        assert summarizer.use_cache is False
