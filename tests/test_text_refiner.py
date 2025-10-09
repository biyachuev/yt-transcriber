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

    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires Ollama server")
    def test_real_refinement(self):
        """Real text refinement test (skipped by default)"""
        refiner = TextRefiner(model_name="qwen2.5:3b")
        text = "This is uh you know a test text"
        result = refiner.refine_chunk(text)
        assert len(result) > 0
