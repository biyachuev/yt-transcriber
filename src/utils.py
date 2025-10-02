"""
Вспомогательные функции
"""
import re
from pathlib import Path
from typing import Optional


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """
    Очистка имени файла от недопустимых символов
    
    Args:
        filename: Исходное имя файла
        max_length: Максимальная длина имени
        
    Returns:
        Очищенное имя файла
    """
    # Удаляем недопустимые символы для файловой системы
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    clean_name = re.sub(invalid_chars, '_', filename)
    
    # Удаляем множественные пробелы и подчеркивания
    clean_name = re.sub(r'[\s_]+', '_', clean_name)
    
    # Обрезаем до максимальной длины
    if len(clean_name) > max_length:
        clean_name = clean_name[:max_length]
    
    # Удаляем точки в начале и конце
    clean_name = clean_name.strip('._')
    
    return clean_name or "untitled"


def format_timestamp(seconds: float) -> str:
    """
    Форматирование временной метки в формат MM:SS или HH:MM:SS
    
    Args:
        seconds: Время в секундах
        
    Returns:
        Отформатированная временная метка
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def detect_language(text: str) -> str:
    """
    Простое определение языка текста (русский или английский)
    
    Args:
        text: Текст для анализа
        
    Returns:
        Код языка ('ru' или 'en')
    """
    # Подсчитываем кириллические и латинские символы
    cyrillic_count = len(re.findall(r'[а-яА-ЯёЁ]', text))
    latin_count = len(re.findall(r'[a-zA-Z]', text))
    
    if cyrillic_count > latin_count:
        return "ru"
    else:
        return "en"


def chunk_text(text: str, max_tokens: int = 2000) -> list[str]:
    """
    Разбивка текста на чанки для перевода
    
    Args:
        text: Текст для разбивки
        max_tokens: Максимальный размер чанка (приблизительно в словах)
        
    Returns:
        Список чанков текста
    """
    # Разбиваем по абзацам
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(para.split())
        
        if current_length + para_length > max_tokens and current_chunk:
            # Сохраняем текущий чанк и начинаем новый
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_length = para_length
        else:
            current_chunk.append(para)
            current_length += para_length
    
    # Добавляем последний чанк
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks


def estimate_processing_time(duration_seconds: float, operation: str = "transcribe") -> str:
    """
    Оценка времени обработки
    
    Args:
        duration_seconds: Длительность аудио в секундах
        operation: Тип операции ('transcribe' или 'translate')
        
    Returns:
        Строка с оценкой времени
    """
    if operation == "transcribe":
        # Whisper обычно обрабатывает в реальном времени или чуть медленнее
        estimated = duration_seconds * 1.5
    else:  # translate
        # Перевод обычно быстрее
        estimated = duration_seconds * 0.3
    
    minutes = int(estimated // 60)
    if minutes < 1:
        return "менее 1 минуты"
    elif minutes == 1:
        return "около 1 минуты"
    else:
        return f"около {minutes} минут"