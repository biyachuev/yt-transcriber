"""
Tests for API key validation module.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.api_validator import validate_openai_api_key, validate_api_keys_for_operation


class TestValidateOpenAIAPIKey:
    """Tests for validate_openai_api_key function."""

    @patch("src.api_validator.settings")
    def test_no_api_key_set(self, mock_settings):
        """Test validation fails when API key is not set."""
        mock_settings.OPENAI_API_KEY = ""
        assert validate_openai_api_key() is False

    @patch("src.api_validator.settings")
    @patch("openai.OpenAI")
    def test_valid_api_key(self, mock_openai_class, mock_settings):
        """Test validation succeeds with valid API key."""
        mock_settings.OPENAI_API_KEY = "sk-valid-key"
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.models.list.return_value = []

        assert validate_openai_api_key() is True
        mock_client.models.list.assert_called_once()

    @patch("src.api_validator.settings")
    @patch("openai.OpenAI")
    def test_invalid_api_key(self, mock_openai_class, mock_settings):
        """Test validation fails with invalid API key."""
        from openai import AuthenticationError

        mock_settings.OPENAI_API_KEY = "sk-invalid-key"
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.models.list.side_effect = AuthenticationError(
            message="Invalid API key", response=MagicMock(status_code=401), body=None
        )

        assert validate_openai_api_key() is False

    @patch("src.api_validator.settings")
    @patch("openai.OpenAI")
    def test_connection_error(self, mock_openai_class, mock_settings):
        """Test validation fails on connection error."""
        from openai import APIConnectionError

        mock_settings.OPENAI_API_KEY = "sk-valid-key"
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.models.list.side_effect = APIConnectionError(
            message="Connection failed", request=MagicMock()
        )

        assert validate_openai_api_key() is False


class TestValidateAPIKeysForOperation:
    """Tests for validate_api_keys_for_operation function."""

    @patch("src.api_validator.validate_openai_api_key")
    def test_no_api_required(self, mock_validate):
        """Test when no API keys are required."""
        result = validate_api_keys_for_operation(
            transcribe_method="whisper_base",
            translate_methods=["nllb"],
            refine_backend="ollama",
            summarize_backend="ollama",
        )
        assert result is True
        mock_validate.assert_not_called()

    @patch("src.api_validator.validate_openai_api_key")
    def test_openai_required_for_transcription(self, mock_validate):
        """Test OpenAI key validation when required for transcription."""
        mock_validate.return_value = True

        result = validate_api_keys_for_operation(transcribe_method="whisper_openai_api")
        assert result is True
        mock_validate.assert_called_once()

    @patch("src.api_validator.validate_openai_api_key")
    def test_openai_required_for_translation(self, mock_validate):
        """Test OpenAI key validation when required for translation."""
        mock_validate.return_value = True

        result = validate_api_keys_for_operation(translate_methods=["openai_api"])
        assert result is True
        mock_validate.assert_called_once()

    @patch("src.api_validator.validate_openai_api_key")
    def test_openai_required_for_refinement(self, mock_validate):
        """Test OpenAI key validation when required for refinement."""
        mock_validate.return_value = True

        result = validate_api_keys_for_operation(refine_backend="openai_api")
        assert result is True
        mock_validate.assert_called_once()

    @patch("src.api_validator.validate_openai_api_key")
    def test_openai_required_for_summarization(self, mock_validate):
        """Test OpenAI key validation when required for summarization."""
        mock_validate.return_value = True

        result = validate_api_keys_for_operation(summarize_backend="openai_api")
        assert result is True
        mock_validate.assert_called_once()

    @patch("src.api_validator.validate_openai_api_key")
    def test_openai_validation_failure(self, mock_validate):
        """Test operation fails when OpenAI key validation fails."""
        mock_validate.return_value = False

        result = validate_api_keys_for_operation(transcribe_method="whisper_openai_api")
        assert result is False
        mock_validate.assert_called_once()

    @patch("src.api_validator.validate_openai_api_key")
    def test_multiple_operations_require_openai(self, mock_validate):
        """Test OpenAI key validated once when multiple operations need it."""
        mock_validate.return_value = True

        result = validate_api_keys_for_operation(
            transcribe_method="whisper_openai_api",
            translate_methods=["openai_api"],
            refine_backend="openai_api",
        )
        assert result is True
        # Should be called only once even though multiple operations need it
        mock_validate.assert_called_once()

    @patch("src.api_validator.validate_openai_api_key")
    def test_mixed_translation_methods(self, mock_validate):
        """Test with mixed translation methods (some need OpenAI, some don't)."""
        mock_validate.return_value = True

        result = validate_api_keys_for_operation(
            translate_methods=["nllb", "openai_api"]
        )
        assert result is True
        mock_validate.assert_called_once()
