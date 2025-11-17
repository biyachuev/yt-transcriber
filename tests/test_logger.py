"""
Tests for logger module.
"""
import pytest
from src.logger import format_orange


class TestFormatOrange:
    """Tests for format_orange function."""

    def test_format_orange_basic(self):
        """Test basic orange formatting."""
        text = "$0.1234"
        result = format_orange(text)

        # Check that result contains ANSI codes for orange and reset
        assert "\033[38;5;208m" in result
        assert "\033[0m" in result
        assert text in result

    def test_format_orange_empty_string(self):
        """Test orange formatting with empty string."""
        result = format_orange("")
        assert result == "\033[38;5;208m\033[0m"

    def test_format_orange_preserves_content(self):
        """Test that orange formatting preserves original content."""
        text = "Test $1.23 value"
        result = format_orange(text)

        # Strip ANSI codes to check content
        stripped = result.replace("\033[38;5;208m", "").replace("\033[0m", "")
        assert stripped == text

    def test_format_orange_structure(self):
        """Test the exact structure of formatted output."""
        text = "$0.50"
        result = format_orange(text)
        expected = f"\033[38;5;208m{text}\033[0m"
        assert result == expected
