"""Custom exceptions for yt-transcriber"""


class TranscriberError(Exception):
    """Base exception for transcriber errors"""
    pass


class DownloadError(TranscriberError):
    """Error during video/audio download"""
    pass


class TranscriptionError(TranscriberError):
    """Error during transcription"""
    pass


class TranslationError(TranscriberError):
    """Error during translation"""
    pass


class FileReadError(TranscriberError):
    """Error reading file"""
    pass


class ModelNotFoundError(TranscriberError):
    """Model not found or not loaded"""
    pass


class InvalidFormatError(TranscriberError):
    """Invalid file format"""
    pass


class OllamaError(TranscriberError):
    """Error with Ollama server"""
    pass


class ConfigurationError(TranscriberError):
    """Configuration error"""
    pass
