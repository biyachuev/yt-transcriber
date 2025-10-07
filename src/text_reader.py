"""
Модуль для чтения текстовых файлов (docx, md)
"""
from pathlib import Path
from typing import Optional
import re

from .logger import logger


class TextReader:
    """Класс для чтения текстов из различных форматов"""

    def __init__(self):
        """Инициализация"""
        pass

    def read_file(self, file_path: str) -> str:
        """
        Чтение текста из файла (автоматическое определение формата)

        Args:
            file_path: Путь к файлу

        Returns:
            Текст из файла

        Raises:
            ValueError: Если формат не поддерживается
            FileNotFoundError: Если файл не найден
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        # Определяем формат по расширению
        extension = path.suffix.lower()

        if extension == '.docx':
            return self.read_docx(file_path)
        elif extension == '.md':
            return self.read_markdown(file_path)
        elif extension == '.txt':
            return self.read_text(file_path)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {extension}")

    def read_docx(self, file_path: str) -> str:
        """
        Чтение текста из .docx файла

        Args:
            file_path: Путь к .docx файлу

        Returns:
            Текст из документа
        """
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "Для работы с .docx файлами необходима библиотека python-docx.\n"
                "Установите её: pip install python-docx"
            )

        logger.info(f"Чтение .docx файла: {file_path}")

        try:
            doc = Document(file_path)

            # Извлекаем весь текст из параграфов
            paragraphs = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    paragraphs.append(text)

            full_text = '\n'.join(paragraphs)

            logger.info(f"Прочитано {len(paragraphs)} параграфов, {len(full_text)} символов")

            return full_text

        except Exception as e:
            logger.error(f"Ошибка при чтении .docx файла: {e}")
            raise

    def read_markdown(self, file_path: str) -> str:
        """
        Чтение текста из .md файла (убирает markdown разметку)

        Args:
            file_path: Путь к .md файлу

        Returns:
            Текст без markdown разметки
        """
        logger.info(f"Чтение .md файла: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Убираем markdown разметку
            text = self._strip_markdown(content)

            logger.info(f"Прочитано {len(text)} символов")

            return text

        except Exception as e:
            logger.error(f"Ошибка при чтении .md файла: {e}")
            raise

    def read_text(self, file_path: str) -> str:
        """
        Чтение текста из .txt файла

        Args:
            file_path: Путь к .txt файлу

        Returns:
            Текст из файла
        """
        logger.info(f"Чтение .txt файла: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            logger.info(f"Прочитано {len(content)} символов")

            return content

        except Exception as e:
            logger.error(f"Ошибка при чтении .txt файла: {e}")
            raise

    def _strip_markdown(self, markdown_text: str) -> str:
        """
        Убирает markdown разметку из текста

        Args:
            markdown_text: Текст с markdown разметкой

        Returns:
            Чистый текст
        """
        text = markdown_text

        # Убираем заголовки (# ## ###)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

        # Убираем код блоки
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)

        # Убираем жирный текст и курсив
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)

        # Убираем ссылки [text](url) → text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

        # Убираем изображения ![alt](url)
        text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)

        # Убираем списки (- * +)
        text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)

        # Убираем нумерованные списки
        text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)

        # Убираем горизонтальные линии
        text = re.sub(r'^[\s]*[-*_]{3,}[\s]*$', '', text, flags=re.MULTILINE)

        # Убираем blockquotes
        text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)

        # Убираем множественные пустые строки
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def detect_language(self, text: str) -> str:
        """
        Определение языка текста (простое эвристическое определение)

        Args:
            text: Текст для определения языка

        Returns:
            'ru' для русского, 'en' для английского
        """
        # Считаем кириллицу
        cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        total_chars = sum(1 for c in text if c.isalpha())

        if total_chars == 0:
            return 'en'

        # Если больше 30% кириллицы - считаем русским
        return 'ru' if (cyrillic_chars / total_chars) > 0.3 else 'en'