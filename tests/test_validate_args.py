"""
Tests for argument validation in main.py
"""
import argparse
from unittest.mock import patch, MagicMock
import pytest
from src.main import validate_args


class TestValidateArgs:
    """Tests for validate_args function with subcommand structure"""

    def create_args(self, command='youtube', **kwargs):
        """Helper to create args namespace with defaults for a specific command"""
        defaults = {
            'refine_model': None,
            'refine_backend': 'ollama',
            'refine_translation': None,
            'summarize_model': None,
            'summarize_backend': 'ollama',
        }

        # Command-specific defaults
        if command == 'youtube':
            defaults.update({
                'url': None,
                'transcribe': None,
                'translate': None,
                'speakers': False,
            })
        elif command in ['audio', 'video']:
            defaults.update({
                'input': None,
                'transcribe': None,
                'translate': None,
                'speakers': False,
            })
        elif command == 'text':
            defaults.update({
                'input': None,
                'translate': None,
            })

        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_validate_youtube_without_transcribe(self):
        """Test validation fails when youtube command lacks transcribe method"""
        args = self.create_args(command='youtube', url="https://youtube.com/watch?v=test")
        assert validate_args('youtube', args) is False

    def test_validate_audio_without_transcribe(self):
        """Test validation fails when audio command lacks transcribe method"""
        args = self.create_args(command='audio', input="test.mp3")
        assert validate_args('audio', args) is False

    @patch('src.main.Path')
    def test_validate_nonexistent_audio_file(self, mock_path):
        """Test validation fails when audio file doesn't exist"""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        args = self.create_args(
            command='audio',
            input="nonexistent.mp3",
            transcribe="whisper-base"
        )
        assert validate_args('audio', args) is False

    def test_validate_invalid_transcribe_method(self):
        """Test validation fails with invalid transcription method"""
        args = self.create_args(
            command='youtube',
            url="https://youtube.com/watch?v=test",
            transcribe="invalid_method"
        )
        assert validate_args('youtube', args) is False

    def test_validate_invalid_translation_method(self):
        """Test validation fails with invalid translation method"""
        args = self.create_args(
            command='youtube',
            url="https://youtube.com/watch?v=test",
            transcribe="whisper_base",
            translate="invalid_translation"
        )
        assert validate_args('youtube', args) is False

    @patch('src.main.settings')
    def test_validate_openai_transcribe_without_key(self, mock_settings):
        """Test validation fails when using OpenAI transcribe without API key"""
        mock_settings.OPENAI_API_KEY = None

        args = self.create_args(
            command='youtube',
            url="https://youtube.com/watch?v=test",
            transcribe="whisper_openai_api"
        )
        assert validate_args('youtube', args) is False

    @patch('src.main.settings')
    def test_validate_openai_translate_without_key(self, mock_settings):
        """Test validation fails when using OpenAI translate without API key"""
        mock_settings.OPENAI_API_KEY = None

        args = self.create_args(
            command='youtube',
            url="https://youtube.com/watch?v=test",
            transcribe="whisper_base",
            translate="openai_api"
        )
        assert validate_args('youtube', args) is False

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
            command='youtube',
            url="https://youtube.com/watch?v=test",
            transcribe="whisper_base",
            refine_model="nonexistent_model",
            refine_backend="ollama"
        )
        assert validate_args('youtube', args) is False

    @patch('src.main.settings')
    @patch('requests.get')
    def test_validate_ollama_server_unavailable(self, mock_get, mock_settings):
        """Test validation fails when Ollama server is unavailable"""
        mock_settings.OPENAI_API_KEY = "test_key"

        # Mock connection error
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Connection refused")

        args = self.create_args(
            command='youtube',
            url="https://youtube.com/watch?v=test",
            transcribe="whisper_base",
            refine_model="qwen2.5:7b",
            refine_backend="ollama"
        )
        assert validate_args('youtube', args) is False

    @patch('src.main.settings')
    def test_validate_openai_refine_without_key(self, mock_settings):
        """Test validation fails when using OpenAI refine without API key"""
        mock_settings.OPENAI_API_KEY = None

        args = self.create_args(
            command='youtube',
            url="https://youtube.com/watch?v=test",
            transcribe="whisper_base",
            refine_model="gpt-4o-mini",
            refine_backend="openai_api"
        )
        assert validate_args('youtube', args) is False

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
            command='youtube',
            url="https://youtube.com/watch?v=test",
            transcribe="whisper_base",
            summarize_model="nonexistent_summary_model",
            summarize_backend="ollama"
        )
        assert validate_args('youtube', args) is False

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
            command='youtube',
            url="https://youtube.com/watch?v=test",
            transcribe="whisper_base",
            translate="NLLB",
            refine_model="qwen2.5:7b",
            refine_backend="ollama"
        )
        assert validate_args('youtube', args) is True

    def test_validate_multiple_translation_methods(self):
        """Test validation with multiple translation methods"""
        args = self.create_args(
            command='youtube',
            url="https://youtube.com/watch?v=test",
            transcribe="whisper_base",
            translate="NLLB,openai_api"
        )
        # Should fail because openai_api requires API key
        with patch('src.main.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = None
            assert validate_args('youtube', args) is False

    @patch('src.main.settings')
    @patch('requests.get')
    def test_validate_refine_translation_without_ollama(self, mock_get, mock_settings):
        """Test validation fails when refine-translation is used but Ollama is unavailable"""
        mock_settings.OPENAI_API_KEY = "test_key"

        # Mock connection error
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Connection refused")

        args = self.create_args(
            command='youtube',
            url="https://youtube.com/watch?v=test",
            transcribe="whisper_base",
            translate="NLLB",
            refine_translation="qwen2.5:7b"
        )
        assert validate_args('youtube', args) is False

    def test_validate_text_command_without_transcribe(self):
        """Test text command doesn't require transcribe"""
        args = self.create_args(
            command='text',
            input="document.docx",
            translate="NLLB"
        )
        # Text command should not require transcribe
        # This test validates that text processing works differently
        with patch('src.main.Path') as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            # Should succeed without transcribe for text command
            assert validate_args('text', args) is True


class TestValidateArgsIntegration:
    """Integration tests for validate_args with new command structure"""

    def create_args(self, command='youtube', **kwargs):
        """Helper to create args namespace with defaults"""
        defaults = {
            'refine_model': None,
            'refine_backend': 'ollama',
            'refine_translation': None,
            'summarize_model': None,
            'summarize_backend': 'ollama',
        }

        if command == 'youtube':
            defaults.update({
                'url': None,
                'transcribe': None,
                'translate': None,
                'speakers': False,
            })
        elif command in ['audio', 'video']:
            defaults.update({
                'input': None,
                'transcribe': None,
                'translate': None,
                'speakers': False,
            })
        elif command == 'text':
            defaults.update({
                'input': None,
                'translate': None,
            })

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
                command='youtube',
                url="https://youtube.com/watch?v=Y4u9EOTwjqw",
                transcribe="whisper_base",
                refine_model="nonexistent_model_12345",
                refine_backend="ollama"
            )

            # Validation should fail immediately
            result = validate_args('youtube', args)
            assert result is False, "Validation should fail before downloading video"

    def test_validate_catches_missing_api_key_early(self):
        """Test that missing API key is caught before transcription"""
        with patch('src.main.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = None

            args = self.create_args(
                command='youtube',
                url="https://youtube.com/watch?v=test",
                transcribe="whisper_openai_api"
            )

            # Should fail immediately
            result = validate_args('youtube', args)
            assert result is False, "Should catch missing API key before transcription"

    def test_validate_new_command_structure(self):
        """Test that new command-based structure works correctly"""
        # All commands should be validated independently
        commands_with_transcribe = ['youtube', 'audio', 'video']

        for command in commands_with_transcribe:
            args = self.create_args(command=command)
            # Should fail without transcribe
            result = validate_args(command, args)
            assert result is False, f"{command} should require transcribe"
