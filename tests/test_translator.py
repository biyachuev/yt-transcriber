"""Tests for translator module"""
import pytest
from unittest.mock import Mock, patch
from src.translator import Translator
from src.transcriber import TranscriptionSegment
from src.config import TranslateOptions


class TestTranslator:
    """Tests for Translator class"""

    def test_initialization(self):
        """Test translator initialization"""
        translator = Translator(method=TranslateOptions.NLLB)
        assert translator.method == TranslateOptions.NLLB
        assert translator.model is None
        assert translator.device in ['cpu', 'cuda', 'mps']

    def test_translate_empty_text(self):
        """Test translating empty text"""
        translator = Translator()
        result = translator.translate_text("")
        assert result == ""

    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires NLLB model")
    def test_real_translation(self):
        """Real translation test (skipped by default)"""
        translator = Translator(method=TranslateOptions.NLLB)
        text = "Hello world"
        result = translator.translate_text(text, source_lang='en', target_lang='ru')
        assert len(result) > 0
