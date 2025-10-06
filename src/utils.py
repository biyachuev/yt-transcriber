"""
Вспомогательные функции
"""
import re
from pathlib import Path
from typing import Optional
import requests


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """
    Очистка имени файла от недопустимых символов
    
    Args:
        filename: Исходное имя файла
        max_length: Максимальная длина имени
        
    Returns:
        Очищенное имя файла
    """
    # Удаляем недопустимые символы для файловой системы и терминала
    # Добавляем восклицательные знаки (!) и другие символы, которые вызывают проблемы в терминале
    invalid_chars = r'[<>:"/\\|?*!@#$%^&*()+={}[\]|:;,\'",./`~\x00-\x1f]'
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


def estimate_processing_time(
    duration_seconds: float, 
    operation: str = "transcribe",
    model: str = "whisper_base"
) -> str:
    """
    Оценка времени обработки (калиброванные значения для M1 MacBook Air)
    
    Args:
        duration_seconds: Длительность аудио в секундах
        operation: Тип операции ('transcribe' или 'translate')
        model: Модель для транскрибирования
        
    Returns:
        Строка с оценкой времени
    """
    if operation == "transcribe":
        # Множители для разных моделей (M1 CPU, откалиброванные)
        multipliers = {
            "whisper_base": 0.06,   # Очень быстро!
            "whisper_small": 0.19,  # Всё ещё быстро
            "whisper_medium": 0.45, # Медленнее, но лучше качество (оценка)
        }
        estimated = duration_seconds * multipliers.get(model, 0.10)
    else:  # translate
        # NLLB на M1 CPU
        estimated = duration_seconds * 0.47
    
    minutes = int(estimated // 60)
    seconds = int(estimated % 60)
    
    # Детальное форматирование
    if estimated < 5:
        return "несколько секунд"
    elif estimated < 10:
        return f"{int(estimated)} секунд"
    elif estimated < 30:
        return f"около {seconds} секунд"
    elif estimated < 45:
        return "около 30 секунд"
    elif estimated < 60:
        return "около минуты"
    elif estimated < 90:
        return "1-1.5 минуты"
    elif minutes < 3:
        return f"около {minutes} минут"
    elif minutes < 5:
        return f"{minutes}-{minutes+1} минут"
    elif minutes < 10:
        return f"около {minutes} минут"
    else:
        return f"около {minutes} минут (±10%)"


def create_whisper_prompt_with_llm(metadata: dict, use_ollama: bool = True, model: str = "qwen2.5:3b") -> str:
    """
    Создание промпта для Whisper из метаданных с помощью LLM

    Args:
        metadata: Словарь с метаданными видео
        use_ollama: Использовать Ollama для генерации промпта
        model: Модель Ollama для использования

    Returns:
        Строка-промпт для Whisper
    """
    from .logger import logger

    if not use_ollama:
        return create_whisper_prompt(metadata)

    # Собираем контекст из метаданных
    context_parts = []

    title = metadata.get('title', '')
    if title:
        context_parts.append(f"Title: {title}")

    description = metadata.get('description', '')
    if description:
        # Берём первые 500 символов описания
        short_desc = description[:500] if len(description) > 500 else description
        context_parts.append(f"Description: {short_desc}")

    tags = metadata.get('tags', [])
    if tags:
        context_parts.append(f"Tags: {', '.join(tags[:20])}")  # Первые 20 тегов

    channel = metadata.get('channel', '')
    if channel:
        context_parts.append(f"Channel: {channel}")

    # Добавляем субтитры если есть (первые 1000 символов для извлечения терминов)
    subtitles = metadata.get('subtitles_sample', '')
    if subtitles:
        context_parts.append(f"Subtitles sample: {subtitles[:1000]}")

    context = '\n'.join(context_parts)

    # Промпт для LLM (финальная версия с обязательной структурой на русском)
    llm_prompt = f"""Ты — ассистент, который генерирует идеальные промпты для модели транскрибации Whisper. Твоя единственная задача — создать один короткий, максимально информативный и **СТРОГО НА РУССКОМ ЯЗЫКЕ** текст.

**Входные данные:**
{context}

**Инструкции по генерации промпта:**
1.  **ЯЗЫКОВАЯ ДИСЦИПЛИНА:** **КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО** генерировать текст на английском или любом другом языке, кроме **РУССКОГО**.
2.  **Извлечение:** Извлеки все **собственные имена** (людей, компаний), **названия брендов/продуктов** и **специфические технические термины/аббревиатуры** из всех входных данных.
3.  **Обогащение:** Определи широкую тематику (AI, Финансы, и т.д.) и сгенерируй **3-5 дополнительных высокочастотных тематических терминов** для расширения лексикона Whisper.
4.  **ФОРМАТ ВЫВОДА (ОБЯЗАТЕЛЬНЫЙ ПОРЯДОК):**
    * **Часть 1 (Контекст):** Сформулируй одно-два связных предложения, описывающих тему ролика и ключевых лиц (например, "Подкаст о... Гость: [Имя]").
    * **Часть 2 (Лексика):** После контекста, через запятую, перечисли **ВСЕ** извлеченные и сгенерированные термины. Закончи точкой.
    * **Часть 3 (Привязка):** **ЗАКОНЧИ ПРОМПТ ТОЧНЫМ ТЕКСТОМ ИЗ ПОЛЯ Subtitles sample (первое предложение).** Это обязательный элемент, который должен стоять последним.

**Вывод:** Сгенерируй только сам промпт для Whisper, как единый, непрерывный текст.
**НЕ ПИШИ** "Часть 1", "Часть 2", "Часть 3", "Лексика", "Привязка" или другие заголовки.
Просто напиши текст промпта слитно.
**ВАЖНО:** Промпт должен быть не длиннее 500 символов (ограничение Whisper).

ПРОМПТ ДЛЯ WHISPER:"""

    try:
        # Вызов Ollama
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": llm_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 400,  # Больше токенов для генерации дополнительных терминов
                    "stop": ["\n\n\n", "**", "EXPLANATION:", "Note:", "Пояснение:"]
                }
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            prompt = result.get('response', '').strip()

            # Очистка результата
            # Убираем заголовки "Часть 1", "Часть 2", "Часть 3" и т.д.
            import re
            prompt = re.sub(r'Часть \d+[:\(][^\)]*[\):]?\s*', '', prompt)
            prompt = re.sub(r'\(Контекст\)|\(Лексика\)|\(Привязка\)', '', prompt)

            # Убираем возможные артефакты
            if ':' in prompt and prompt.index(':') < 20:
                prompt = prompt.split(':', 1)[1].strip()

            # Ограничиваем длину (Whisper лимит ~224 токена ≈ 600 символов для русского)
            MAX_PROMPT_LENGTH = 600  # Безопасный лимит для русского текста

            if len(prompt) > MAX_PROMPT_LENGTH:
                logger.warning(f"Промпт слишком длинный ({len(prompt)} символов), обрезается до {MAX_PROMPT_LENGTH}")

                # Пытаемся сохранить структуру: контекст + термины + первое предложение привязки
                # Находим последнюю точку перед лимитом (это может быть конец терминов)
                truncated = prompt[:MAX_PROMPT_LENGTH]

                # Если есть точка, обрезаем до неё
                last_period = truncated.rfind('.')
                if last_period > MAX_PROMPT_LENGTH * 0.5:  # Точка не слишком близко к началу
                    truncated = truncated[:last_period + 1]
                else:
                    # Иначе обрезаем по последней запятой
                    last_comma = truncated.rfind(',')
                    if last_comma > 0:
                        truncated = truncated[:last_comma]

                # Добавляем первое предложение из субтитров как привязку (если потеряли)
                if 'subtitles_sample' in str(metadata):
                    first_sentence = metadata.get('subtitles_sample', '').split('.')[0]
                    if first_sentence and first_sentence not in truncated:
                        # Добавляем если есть место
                        if len(truncated) + len(first_sentence) + 2 <= MAX_PROMPT_LENGTH:
                            truncated = truncated.rstrip() + ' ' + first_sentence + '.'

                prompt = truncated
                logger.info(f"Промпт обрезан до {len(prompt)} символов")

            logger.info(f"LLM создал промпт для Whisper ({len(prompt)} символов):")
            logger.info(f"  {prompt}")

            return prompt
        else:
            logger.warning(f"Ошибка при вызове Ollama: {response.status_code}")
            return create_whisper_prompt(metadata)

    except Exception as e:
        logger.warning(f"Не удалось создать промпт через LLM: {e}")
        logger.info("Используется стандартный метод создания промпта")
        return create_whisper_prompt(metadata)


def create_whisper_prompt(metadata: dict) -> str:
    """
    Создание промпта для Whisper из метаданных видео

    Whisper использует промпт для улучшения качества транскрипции:
    - Правильное распознавание имён, терминов и специфичных слов
    - Сохранение стиля и пунктуации

    Args:
        metadata: Словарь с метаданными видео

    Returns:
        Строка-промпт для Whisper
    """
    prompt_parts = []

    # Список общих стоп-слов, которые не нужны в промпте
    stop_words = {
        'the', 'and', 'for', 'with', 'from', 'this', 'that', 'these', 'those',
        'how', 'what', 'where', 'when', 'why', 'who', 'which', 'can', 'will',
        'should', 'would', 'could', 'may', 'might', 'must', 'video', 'tutorial',
        'guide', 'tips', 'tricks', 'best', 'top', 'new', 'learn', 'beginner',
        'advanced', 'full', 'complete', 'explained', 'easy', 'simple'
    }

    # Извлекаем ключевые слова из названия
    title = metadata.get('title', '')
    if title:
        # Слова с заглавными буквами (имена, бренды, аббревиатуры)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b|\b[A-Z]{2,}\b', title)
        for word in capitalized_words:
            if word.lower() not in stop_words:
                prompt_parts.append(word)

    # Добавляем ВСЕ теги (они уже отфильтрованы автором видео)
    tags = metadata.get('tags', [])
    if tags:
        # Берём все теги, фильтруем только по стоп-словам
        for tag in tags:
            # Пропускаем слишком общие фразы
            if tag.lower() not in stop_words and len(tag) > 2:
                prompt_parts.append(tag)

    # Добавляем имя канала (часто содержит имя автора)
    channel = metadata.get('channel', '')
    if channel:
        prompt_parts.append(channel)

    # Формируем промпт
    if prompt_parts:
        # Убираем дубликаты и объединяем
        unique_parts = []
        seen = set()
        for part in prompt_parts:
            if part.lower() not in seen:
                unique_parts.append(part)
                seen.add(part.lower())

        # Whisper имеет ограничение на длину промпта (~224 токена или ~1000 символов)
        # Собираем промпт постепенно, проверяя длину
        MAX_PROMPT_LENGTH = 800  # Оставляем запас
        prompt_list = []
        current_length = 0

        for part in unique_parts:
            # +2 для ", " разделителя
            part_length = len(part) + 2
            if current_length + part_length > MAX_PROMPT_LENGTH:
                break
            prompt_list.append(part)
            current_length += part_length

        prompt = ', '.join(prompt_list)

        # Импортируем logger локально чтобы избежать циклических импортов
        from .logger import logger
        logger.info(f"Создан промпт для Whisper ({len(prompt)} символов, {len(prompt_list)} терминов):")
        logger.info(f"  {prompt}")

        return prompt

    return ""