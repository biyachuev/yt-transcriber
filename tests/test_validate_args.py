"""
Tests for argument validation in main.py
"""
import argparse
from unittest.mock import patch, MagicMock
import pytest
from src.main import validate_args


class TestValidateArgs:
    """Tests for validate_args function"""

    def create_args(self, **kwargs):
        """Helper to create args namespace with defaults"""
        defaults = {
            'url': None,
            'input_audio': None,
            'input_video': None,
            'input_text': None,
            'transcribe': None,
            'translate': None,
            'refine_model': None,
            'refine_backend': 'ollama',
            'refine_translation': None,
            'summarize': False,
            'summarize_model': None,
            'summarize_backend': 'ollama',
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_validate_no_input_source(self):
        """Test validation fails when no input source is provided"""
        args = self.create_args()
        assert validate_args(args) is False

    def test_validate_multiple_input_sources(self):
        """Test validation fails when multiple input sources are provided"""
        args = self.create_args(
            url="https://youtube.com/watch?v=test",
            input_audio="test.mp3"
        )
        assert validate_args(args) is False

    def test_validate_audio_without_transcribe(self):
        """Test validation fails when audio input without transcribe method"""
        args = self.create_args(input_audio="test.mp3")
        assert validate_args(args) is False

    @patch('src.main.Path')
    def test_validate_nonexistent_audio_file(self, mock_path):
        """Test validation fails when audio file doesn't exist"""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        args = self.create_args(
            input_audio="nonexistent.mp3",
            transcribe="whisper_base"
        )
        assert validate_args(args) is False

    def test_validate_invalid_transcribe_method(self):
        """Test validation fails with invalid transcription method"""
        args = self.create_args(
            url="https://youtube.com/watch?v=test",
            transcribe="invalid_method"
        )
        assert validate_args(args) is False

    def test_validate_invalid_translation_method(self):
        """Test validation fails with invalid translation method"""
        args = self.create_args(
            url="https://youtube.com/watch?v=test",
            transcribe="whisper_base",
            translate="invalid_translation"
        )
        assert validate_args(args) is False

    @patch('src.main.settings')
    def test_validate_openai_transcribe_without_key(self, mock_settings):
        """Test validation fails when using OpenAI transcribe without API key"""
        mock_settings.OPENAI_API_KEY = None

        args = self.create_args(
            url="https://youtube.com/watch?v=test",
            transcribe="whisper_openai_api"
        )
        assert validate_args(args) is False

    @patch('src.main.settings')
    def test_validate_openai_translate_without_key(self, mock_settings):
        """Test validation fails when using OpenAI translate without API key"""
        mock_settings.OPENAI_API_KEY = None

        args = self.create_args(
            url="https://youtube.com/watch?v=test",
            transcribe="whisper_base",
            translate="openai_api"
        )
        assert validate_args(args) is False

    @patch('src.main.settings')
    @patch('requests.get')
    def test_validate_ollama_refine_model_not_found(self, mock_get, mock_settings):
        """Test validation fails when Ollama model doesn't exist"""
        mock_settings.OPENAI_API_KEY = "test_key"

        # Mock Ollama server response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [
                {'name': 'qwen2.5:7b'},
                {'name': 'llama3:8b'}
            ]
        }
        mock_get.return_value = mock_response

        args = self.create_args(
            url="https://youtube.com/watch?v=test",
            transcribe="whisper_base",
            refine_model="nonexistent_model",
            refine_backend="ollama"
        )
        assert validate_args(args) is False

    @patch('src.main.settings')
    @patch('requests.get')
    def test_validate_ollama_server_unavailable(self, mock_get, mock_settings):
        """Test validation fails when Ollama server is unavailable"""
        mock_settings.OPENAI_API_KEY = "test_key"

        # Mock connection error
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Connection refused")

        args = self.create_args(
            url="https://youtube.com/watch?v=test",
            transcribe="whisper_base",
            refine_model="qwen2.5:7b",
            refine_backend="ollama"
        )
        assert validate_args(args) is False

    @patch('src.main.settings')
    def test_validate_openai_refine_without_key(self, mock_settings):
        """Test validation fails when using OpenAI refine without API key"""
        mock_settings.OPENAI_API_KEY = None

        args = self.create_args(
            url="https://youtube.com/watch?v=test",
            transcribe="whisper_base",
            refine_model="gpt-4o-mini",
            refine_backend="openai_api"
        )
        assert validate_args(args) is False

    @patch('src.main.settings')
    @patch('requests.get')
    def test_validate_summarize_ollama_model_not_found(self, mock_get, mock_settings):
        """Test validation fails when summarize Ollama model doesn't exist"""
        mock_settings.OPENAI_API_KEY = "test_key"

        # Mock Ollama server response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [{'name': 'qwen2.5:7b'}]
        }
        mock_get.return_value = mock_response

        args = self.create_args(
            url="https://youtube.com/watch?v=test",
            transcribe="whisper_base",
            summarize=True,
            summarize_model="nonexistent_summary_model",
            summarize_backend="ollama"
        )
        assert validate_args(args) is False

    @patch('src.main.settings')
    @patch('requests.get')
    def test_validate_successful_with_all_params(self, mock_get, mock_settings):
        """Test validation succeeds with all valid parameters"""
        mock_settings.OPENAI_API_KEY = "test_key"

        # Mock Ollama server response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [
                {'name': 'qwen2.5:7b'},
                {'name': 'llama3:8b'}
            ]
        }
        mock_get.return_value = mock_response

        args = self.create_args(
            url="https://youtube.com/watch?v=test",
            transcribe="whisper_base",
            translate="NLLB",
            refine_model="qwen2.5:7b",
            refine_backend="ollama"
        )
        assert validate_args(args) is True

    def test_validate_multiple_translation_methods(self):
        """Test validation with multiple translation methods"""
        args = self.create_args(
            url="https://youtube.com/watch?v=test",
            transcribe="whisper_base",
            translate="NLLB,openai_api"
        )
        # Should fail because openai_api requires API key
        with patch('src.main.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = None
            assert validate_args(args) is False

    @patch('src.main.settings')
    @patch('requests.get')
    def test_validate_refine_translation_without_ollama(self, mock_get, mock_settings):
        """Test validation fails when refine-translation is used but Ollama is unavailable"""
        mock_settings.OPENAI_API_KEY = "test_key"

        # Mock connection error
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Connection refused")

        args = self.create_args(
            url="https://youtube.com/watch?v=test",
            transcribe="whisper_base",
            translate="NLLB",
            refine_translation="qwen2.5:7b"
        )
        assert validate_args(args) is False


class TestValidateArgsIntegration:
    """Integration tests for validate_args"""

    def create_args(self, **kwargs):
        """Helper to create args namespace with defaults"""
        defaults = {
            'url': None,
            'input_audio': None,
            'input_video': None,
            'input_text': None,
            'transcribe': None,
            'translate': None,
            'refine_model': None,
            'refine_backend': 'ollama',
            'refine_translation': None,
            'summarize': False,
            'summarize_model': None,
            'summarize_backend': 'ollama',
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_validate_prevents_expensive_operations(self):
        """
        Test that validation catches errors before expensive operations
        This is the key scenario the user asked about
        """
        # Scenario: User specifies nonexistent model for refinement
        # Validation should catch this BEFORE downloading YouTube video
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'models': [{'name': 'qwen2.5:7b'}]
            }
            mock_get.return_value = mock_response

            args = self.create_args(
                url="https://youtube.com/watch?v=Y4u9EOTwjqw",
                transcribe="whisper_base",
                refine_model="nonexistent_model_12345",
                refine_backend="ollama"
            )

            # Validation should fail immediately
            result = validate_args(args)
            assert result is False, "Validation should fail before downloading video"

    def test_validate_catches_missing_api_key_early(self):
        """Test that missing API key is caught before transcription"""
        with patch('src.main.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = None

            args = self.create_args(
                url="https://youtube.com/watch?v=test",
                transcribe="whisper_openai_api"
            )

            # Should fail immediately
            result = validate_args(args)
            assert result is False, "Should catch missing API key before transcription"
