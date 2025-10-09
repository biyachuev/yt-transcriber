"""Tests for text_refiner module"""
import pytest
from unittest.mock import Mock, patch
import requests
from src.text_refiner import TextRefiner


class TestTextRefiner:
    """Tests for TextRefiner class"""

    @patch('src.text_refiner.requests.get')
    def test_initialization(self, mock_get):
        """Test initialization"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'models': [{'name': 'qwen2.5:3b'}]}
        mock_get.return_value = mock_response

        refiner = TextRefiner(model_name="qwen2.5:3b")
        assert refiner.model_name == "qwen2.5:3b"
        assert refiner.ollama_url == "http://localhost:11434"

    @patch('src.text_refiner.requests.get')
    def test_check_ollama_unavailable(self, mock_get):
        """Test handling unavailable Ollama server"""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

        with pytest.raises(RuntimeError):
            TextRefiner(model_name="qwen2.5:3b")

    @patch('src.text_refiner.requests.get')
    def test_check_model_not_found(self, mock_get):
        """Test handling missing model"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'models': [{'name': 'other-model'}]}
        mock_get.return_value = mock_response

        with pytest.raises(RuntimeError, match="не найдена в Ollama"):
            TextRefiner(model_name="qwen2.5:3b")

    @patch('src.text_refiner.requests.get')
    def test_split_text_into_chunks(self, mock_get):
        """Test text chunking"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'models': [{'name': 'qwen2.5:3b'}]}
        mock_get.return_value = mock_response

        refiner = TextRefiner(model_name="qwen2.5:3b")

        text = "First sentence. Second sentence. Third sentence."
        chunks = refiner._split_text_into_chunks(text, max_chunk_size=30)

        assert len(chunks) > 1
        assert all(len(chunk) <= 50 for chunk in chunks)  # Allow some overhead

    @patch('src.text_refiner.requests.get')
    @patch('src.text_refiner.requests.post')
    def test_refine_chunk(self, mock_post, mock_get):
        """Test chunk refinement"""
        # Mock Ollama availability check
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {'models': [{'name': 'qwen2.5:3b'}]}
        mock_get.return_value = mock_get_response

        # Mock refinement response (stream=False returns json)
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {'response': 'Refined text'}
        mock_post.return_value = mock_post_response

        refiner = TextRefiner(model_name="qwen2.5:3b")
        result = refiner.refine_chunk("Original text")

        assert "Refined text" in result
        mock_post.assert_called()

    @patch('src.text_refiner.requests.get')
    @patch('src.text_refiner.TextRefiner._call_ollama')
    def test_refine_text_multiple_chunks(self, mock_call_ollama, mock_get):
        """Test refining text with multiple chunks"""
        # Mock Ollama availability check
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {'models': [{'name': 'qwen2.5:3b'}]}
        mock_get.return_value = mock_get_response

        # Mock refinement
        mock_call_ollama.return_value = "Refined chunk"

        refiner = TextRefiner(model_name="qwen2.5:3b")

        # Create long text that will be split
        long_text = "Sentence. " * 500
        result = refiner.refine_text(long_text)

        assert len(result) > 0
        assert mock_call_ollama.call_count > 1  # Multiple chunks processed

    @patch('src.text_refiner.requests.get')
    @patch('src.text_refiner.requests.post')
    def test_refine_chunk_error_fallback(self, mock_post, mock_get):
        """Test fallback to original text on error"""
        # Mock Ollama availability check
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {'models': [{'name': 'qwen2.5:3b'}]}
        mock_get.return_value = mock_get_response

        # Mock error response
        mock_post.side_effect = requests.exceptions.RequestException("Error")

        refiner = TextRefiner(model_name="qwen2.5:3b")
        original_text = "Original text"
        result = refiner.refine_chunk(original_text)

        assert result == original_text  # Should return original on error

    @patch('src.text_refiner.requests.get')
    def test_detect_topic(self, mock_get):
        """Test topic detection"""
        # Mock Ollama availability check
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {'models': [{'name': 'qwen2.5:3b'}]}
        mock_get.return_value = mock_get_response

        refiner = TextRefiner(model_name="qwen2.5:3b")

        with patch.object(refiner, '_call_ollama', return_value="программирование дополнительный текст"):
            topic = refiner._detect_topic("Some text about programming")

            assert "программирование" == topic

    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires Ollama server")
    def test_real_refinement(self):
        """Real text refinement test (skipped by default)"""
        refiner = TextRefiner(model_name="qwen2.5:3b")
        text = "This is uh you know a test text"
        result = refiner.refine_chunk(text)
        assert len(result) > 0
