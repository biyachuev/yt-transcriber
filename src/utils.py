"""
General utility helpers used across the project.
"""
import re
from pathlib import Path
from typing import Optional

import requests


def format_log_preview(text: str, max_length: int = 80) -> str:
    """
    Create a short single-line preview suitable for logging.

    Args:
        text: Original text.
        max_length: Maximum preview length.

    Returns:
        Sanitised preview string.
    """
    if not text:
        return ""

    single_line = " ".join(text.split())
    if len(single_line) > max_length:
        return single_line[:max_length] + "..."
    return single_line


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """
    Remove characters that are unsafe for file systems and terminals.

    Args:
        filename: Original file name.
        max_length: Maximum allowed length.

    Returns:
        Sanitised file name suitable for saving.
    """
    invalid_chars = r'[<>:"/\\|?*!@#$%^&*()+={}[\]|:;,\'",./`~\x00-\x1f]'
    clean_name = re.sub(invalid_chars, "_", filename)

    clean_name = re.sub(r"[\s_]+", "_", clean_name)

    if len(clean_name) > max_length:
        clean_name = clean_name[:max_length]

    clean_name = clean_name.strip("._")

    return clean_name or "untitled"


def format_timestamp(seconds: float) -> str:
    """
    Convert a timestamp in seconds to MM:SS or HH:MM:SS format.

    Args:
        seconds: Timestamp value.

    Returns:
        Formatted timestamp string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def detect_language(text: str) -> str:
    """
    Perform a simple Russian/English language heuristic.

    Args:
        text: Text to analyse.

    Returns:
        'ru' for Russian, 'en' for English.
    """
    cyrillic_count = len(re.findall(r"[а-яА-ЯёЁ]", text))
    latin_count = len(re.findall(r"[a-zA-Z]", text))

    if cyrillic_count > latin_count:
        return "ru"
    return "en"


def chunk_text(text: str, max_tokens: int = 2000) -> list[str]:
    """
    Split text into chunks sized for translation pipelines.

    Args:
        text: Original text.
        max_tokens: Approximate maximum number of words per chunk.

    Returns:
        List of text chunks.
    """
    paragraphs = text.split("\n\n")

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for para in paragraphs:
        para_length = len(para.split())

        if current_length + para_length > max_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_length = para_length
        else:
            current_chunk.append(para)
            current_length += para_length

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def estimate_processing_time(
    duration_seconds: float,
    operation: str = "transcribe",
    model: str = "whisper_base",
) -> str:
    """
    Estimate processing time based on calibrated M1 MacBook Air numbers.

    Args:
        duration_seconds: Duration of source audio.
        operation: Either 'transcribe' or 'translate'.
        model: Whisper model identifier (for transcription).

    Returns:
        Human-friendly estimate string.
    """
    if operation == "transcribe":
        multipliers = {
            "whisper_base": 0.06,
            "whisper_small": 0.19,
            "whisper_medium": 0.45,
        }
        estimated = duration_seconds * multipliers.get(model, 0.10)
    else:  # translate
        estimated = duration_seconds * 0.47

    minutes = int(estimated // 60)
    seconds = int(estimated % 60)

    if estimated < 5:
        return "a few seconds"
    if estimated < 10:
        return f"{int(estimated)} seconds"
    if estimated < 30:
        return f"around {seconds} seconds"
    if estimated < 45:
        return "around 30 seconds"
    if estimated < 60:
        return "about a minute"
    if estimated < 90:
        return "1–1.5 minutes"
    if minutes < 3:
        return f"about {minutes} minutes"
    if minutes < 5:
        return f"{minutes}-{minutes + 1} minutes"
    if minutes < 10:
        return f"about {minutes} minutes"
    return f"around {minutes} minutes (±10%)"


def create_whisper_prompt_with_llm(
    metadata: dict,
    use_ollama: bool = True,
    model: str = "qwen2.5:3b",
) -> str:
    """
    Build a Whisper prompt from metadata using a local LLM.

    Args:
        metadata: Video metadata dictionary.
        use_ollama: Whether to call Ollama for prompt generation.
        model: Ollama model name.

    Returns:
        Generated Whisper prompt.
    """
    from .logger import logger

    if not use_ollama:
        return create_whisper_prompt(metadata)

    context_parts = []

    title = metadata.get("title", "")
    if title:
        context_parts.append(f"Title: {title}")

    description = metadata.get("description", "")
    if description:
        short_desc = description[:500] if len(description) > 500 else description
        context_parts.append(f"Description: {short_desc}")

    tags = metadata.get("tags", [])
    if tags:
        context_parts.append(f"Tags: {', '.join(tags[:20])}")

    channel = metadata.get("channel", "")
    if channel:
        context_parts.append(f"Channel: {channel}")

    subtitles = metadata.get("subtitles_sample", "")
    if subtitles:
        context_parts.append(f"Subtitles sample: {subtitles[:1000]}")

    context = "\n".join(context_parts)

    # Preserve the Russian prompt: it is intentionally crafted for Whisper.
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
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": llm_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 400,
                    "stop": ["\n\n\n", "**", "EXPLANATION:", "Note:", "Пояснение:"],
                },
            },
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            prompt = result.get("response", "").strip()

            prompt = re.sub(r'Часть \d+[:\(][^\)]*[\):]?\s*', '', prompt)
            prompt = re.sub(r'\(Контекст\)|\(Лексика\)|\(Привязка\)', '', prompt)

            if ":" in prompt and prompt.index(":") < 20:
                prompt = prompt.split(":", 1)[1].strip()

            MAX_PROMPT_LENGTH = 600

            if len(prompt) > MAX_PROMPT_LENGTH:
                logger.warning(
                    "Generated prompt is too long (%d chars); trimming to %d",
                    len(prompt),
                    MAX_PROMPT_LENGTH,
                )

                truncated = prompt[:MAX_PROMPT_LENGTH]
                last_period = truncated.rfind(".")
                if last_period > MAX_PROMPT_LENGTH * 0.5:
                    truncated = truncated[: last_period + 1]
                else:
                    last_comma = truncated.rfind(",")
                    if last_comma > 0:
                        truncated = truncated[:last_comma]

                if "subtitles_sample" in str(metadata):
                    first_sentence = metadata.get("subtitles_sample", "").split(".")[0]
                    if first_sentence and first_sentence not in truncated:
                        if len(truncated) + len(first_sentence) + 2 <= MAX_PROMPT_LENGTH:
                            truncated = truncated.rstrip() + " " + first_sentence + "."

                prompt = truncated
                logger.info("Prompt trimmed to %d characters", len(prompt))

            logger.info("LLM-generated Whisper prompt (%d chars)", len(prompt))
            logger.debug("Prompt preview: %s", format_log_preview(prompt))

            return prompt

        logger.warning("Ollama returned status %s; falling back to standard prompt", response.status_code)
        return create_whisper_prompt(metadata)

    except Exception as e:
        logger.warning("Failed to build prompt via LLM: %s", e)
        logger.info("Falling back to standard prompt generator")
        return create_whisper_prompt(metadata)


def create_whisper_prompt(metadata: dict) -> str:
    """
    Build a Whisper prompt directly from metadata without LLM assistance.

    Args:
        metadata: Video metadata dictionary.

    Returns:
        Prompt string joined by commas.
    """
    prompt_parts: list[str] = []

    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "these",
        "those",
        "how",
        "what",
        "where",
        "when",
        "why",
        "who",
        "which",
        "can",
        "will",
        "should",
        "would",
        "could",
        "may",
        "might",
        "must",
        "video",
        "tutorial",
        "guide",
        "tips",
        "tricks",
        "best",
        "top",
        "new",
        "learn",
        "beginner",
        "advanced",
        "full",
        "complete",
        "explained",
        "easy",
        "simple",
    }

    title = metadata.get("title", "")
    if title:
        capitalised_words = re.findall(
            r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b|\b[A-Z]{2,}\b", title
        )
        for word in capitalised_words:
            if word.lower() not in stop_words:
                prompt_parts.append(word)

    tags = metadata.get("tags", [])
    if tags:
        for tag in tags:
            if tag.lower() not in stop_words and len(tag) > 2:
                prompt_parts.append(tag)

    channel = metadata.get("channel", "")
    if channel:
        prompt_parts.append(channel)

    if prompt_parts:
        unique_parts: list[str] = []
        seen = set()
        for part in prompt_parts:
            if part.lower() not in seen:
                unique_parts.append(part)
                seen.add(part.lower())

        MAX_PROMPT_LENGTH = 800
        prompt_list: list[str] = []
        current_length = 0

        for part in unique_parts:
            part_length = len(part) + 2  # account for ", "
            if current_length + part_length > MAX_PROMPT_LENGTH:
                break
            prompt_list.append(part)
            current_length += part_length

        prompt = ", ".join(prompt_list)

        from .logger import logger

        logger.info(
            "Generated metadata-based Whisper prompt (%d chars, %d terms)",
            len(prompt),
            len(prompt_list),
        )
        logger.debug("Prompt preview: %s", format_log_preview(prompt))

        return prompt

    return ""
