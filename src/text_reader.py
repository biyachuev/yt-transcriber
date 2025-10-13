"""
Utilities for reading text-based documents (DOCX, Markdown, plain text, PDF).
"""
from pathlib import Path
from typing import Optional
import re

from .logger import logger


class TextReader:
    """Helper class capable of reading several common document formats."""

    def __init__(self):
        """Initialise the reader (no state required)."""
        pass

    def read_file(self, file_path: str) -> str:
        """
        Auto-detect the file type and return its textual contents.

        Args:
            file_path: Path to the document.

        Returns:
            Extracted text.

        Raises:
            ValueError: Unsupported file type.
            FileNotFoundError: File does not exist.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = path.suffix.lower()

        if extension == ".docx":
            return self.read_docx(file_path)
        if extension == ".md":
            return self.read_markdown(file_path)
        if extension == ".txt":
            return self.read_text(file_path)
        if extension == ".pdf":
            return self.read_pdf(file_path)

        raise ValueError(f"Unsupported file format: {extension}")

    def read_docx(self, file_path: str) -> str:
        """
        Read text from a .docx file.

        Args:
            file_path: Path to the DOCX document.

        Returns:
            Extracted text.
        """
        try:
            from docx import Document
        except ImportError as exc:
            raise ImportError(
                "python-docx is required to handle .docx files.\n"
                "Install it with: pip install python-docx"
            ) from exc

        logger.info("Reading DOCX file: %s", file_path)

        try:
            doc = Document(file_path)

            paragraphs = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    paragraphs.append(text)

            full_text = "\n\n".join(paragraphs)

            logger.info("Extracted %d paragraphs, %d characters", len(paragraphs), len(full_text))

            return full_text

        except Exception as e:  # pragma: no cover
            logger.error("Failed to read DOCX file: %s", e)
            raise

    def read_markdown(self, file_path: str) -> str:
        """
        Read text from a Markdown file and strip formatting.

        Args:
            file_path: Path to the Markdown file.

        Returns:
            Plain text content.
        """
        logger.info("Reading Markdown file: %s", file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            text = self._strip_markdown(content)

            logger.info("Extracted %d characters", len(text))

            return text

        except Exception as e:  # pragma: no cover
            logger.error("Failed to read Markdown file: %s", e)
            raise

    def read_text(self, file_path: str) -> str:
        """
        Read a UTF-8 encoded plain text file.

        Args:
            file_path: Path to the file.

        Returns:
            Text content.
        """
        logger.info("Reading text file: %s", file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            logger.info("Extracted %d characters", len(content))

            return content

        except Exception as e:  # pragma: no cover
            logger.error("Failed to read text file: %s", e)
            raise

    def _strip_markdown(self, markdown_text: str) -> str:
        """
        Remove Markdown formatting from the provided text.

        Args:
            markdown_text: Markdown content.

        Returns:
            Cleaned plain text.
        """
        text = markdown_text

        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`[^`]+`", "", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", "", text)
        text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^[\s]*[-*_]{3,}[\s]*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def read_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to the PDF document.

        Returns:
            Extracted text content.
        """
        try:
            import PyPDF2
        except ImportError as exc:
            raise ImportError(
                "PyPDF2 is required to process PDF documents.\n"
                "Install it with: pip install PyPDF2"
            ) from exc

        logger.info("Reading PDF file: %s", file_path)

        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_parts = []

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text and text.strip():
                        text_parts.append(text.strip())

                full_text = "\n\n".join(text_parts)

                if not full_text.strip():
                    logger.warning("PDF file appears empty or text extraction failed")
                    return ""

                logger.info("Extracted %d pages from PDF", len(text_parts))
                return full_text

        except Exception as e:  # pragma: no cover
            logger.error("Failed to read PDF file: %s", e)
            raise

    def detect_language(self, text: str) -> str:
        """
        Simple heuristic-based language detection.

        Args:
            text: Text to analyse.

        Returns:
            'ru' for Russian, 'en' for English.
        """
        cyrillic_chars = sum(1 for c in text if "\u0400" <= c <= "\u04FF")
        total_chars = sum(1 for c in text if c.isalpha())

        if total_chars == 0:
            return "en"

        return "ru" if (cyrillic_chars / total_chars) > 0.3 else "en"
