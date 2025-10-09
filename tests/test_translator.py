"""Tests for translator module"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.translator import Translator
from src.transcriber import TranscriptionSegment
from src.config import TranslateOptions


class TestTranslator:
    """Tests for Translator class"""

    @patch('src.translator.torch')
    def test_initialization(self, mock_torch):
        """Test translator initialization"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        translator = Translator(method=TranslateOptions.NLLB)
        assert translator.method == TranslateOptions.NLLB
        assert translator.model is None
        assert translator.device in ['cpu', 'cuda', 'mps']

    @patch('src.translator.torch')
    def test_translate_empty_text(self, mock_torch):
        """Test translating empty text"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        translator = Translator()
        result = translator.translate_text("")
        assert result == ""

    @patch('src.translator.torch')
    def test_translate_same_language(self, mock_torch):
        """Test translation when source and target languages are the same"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        translator = Translator()
        text = "Hello world"
        result = translator.translate_text(text, source_lang='en', target_lang='en')
        assert result == text

    @patch('src.translator.torch')
    def test_translate_with_nllb(self, mock_torch):
        """Test translation with NLLB model"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        translator = Translator(method=TranslateOptions.NLLB)

        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        translator.model = mock_model
        translator.tokenizer = mock_tokenizer

        # Mock tokenizer behavior
        mock_tokenizer.return_value = {'input_ids': MagicMock()}
        mock_tokenizer.lang_code_to_id = {'eng_Latn': 123, 'rus_Cyrl': 456}
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_tokenizer.batch_decode.return_value = ["Translated text"]

        result = translator.translate_text("Hello", source_lang='en', target_lang='ru')

        assert result == "Translated text"

    @patch('src.translator.torch')
    def test_translate_segments(self, mock_torch):
        """Test translating segments"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        translator = Translator()

        # Mock translate_text to avoid model loading
        with patch.object(translator, 'translate_text', return_value="Translated"):
            segments = [
                TranscriptionSegment(0, 5, "First"),
                TranscriptionSegment(5, 10, "Second")
            ]

            result = translator.translate_segments(segments, source_lang='en', target_lang='ru')

            assert len(result) == 2
            assert all(seg.text == "Translated" for seg in result)

    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires NLLB model")
    def test_real_translation(self):
        """Real translation test (skipped by default)"""
        translator = Translator(method=TranslateOptions.NLLB)
        text = "Hello world"
        result = translator.translate_text(text, source_lang='en', target_lang='ru')
        assert len(result) > 0
