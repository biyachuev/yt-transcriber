"""
Главный модуль приложения
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

from .config import settings, TranscribeOptions, TranslateOptions
from .logger import logger
from .downloader import YouTubeDownloader
from .transcriber import Transcriber
from .document_writer import DocumentWriter
from .utils import detect_language, sanitize_filename, create_whisper_prompt, create_whisper_prompt_with_llm
from .text_reader import TextReader


def load_prompt_from_file(prompt_file_path: str) -> str:
    """
    Загрузка промпта из текстового файла

    Args:
        prompt_file_path: Путь к файлу с промптом

    Returns:
        Промпт в виде строки
    """
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()

        # Ограничиваем ��лину промпта (Whisper имеет лимит)
        MAX_PROMPT_LENGTH = 800
        if len(prompt) > MAX_PROMPT_LENGTH:
            logger.warning(f"Промпт из файла слишком длинный ({len(prompt)} символов), обрезается до {MAX_PROMPT_LENGTH}")
            prompt = prompt[:MAX_PROMPT_LENGTH]

        logger.info(f"Загружен пользовательский промпт из файла ({len(prompt)} символов):")
        logger.info(f"  {prompt}")

        return prompt
    except FileNotFoundError:
        logger.error(f"Файл с промптом не найден: {prompt_file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ошибка при чтении файла промпта: {e}")
        sys.exit(1)


def print_help():
    """Вывод справки по использованию"""
    help_text = """
YouTube Transcriber & Translator
=================================

Использование:
    python -m src.main [OPTIONS]

Примеры:
    # Транскрибирование и перевод YouTube видео
    python -m src.main --url "https://youtube.com/watch?v=..." --transcribe whisper_base --translate NLLB

    # Только транскрибирование
    python -m src.main --url "https://youtube.com/watch?v=..." --transcribe whisper_base

    # Обработка аудиофайла
    python -m src.main --input_audio audio.mp3 --transcribe whisper_base --translate NLLB

    # Использование пользовательского промпта для улучшения качества
    python -m src.main --url "https://youtube.com/watch?v=..." --transcribe whisper_base --prompt prompt.txt

    # Улучшение транскрипции с помощью локальной LLM
    python -m src.main --input_audio audio.mp3 --transcribe whisper_medium --refine-model llama3.2:3b

    # С улучшением И переводом (создаст 3 документа: original, refined, translated)
    python -m src.main --input_audio audio.mp3 --transcribe whisper_medium --translate NLLB --refine-model llama3.2:3b

Опции:
    --url URL                   URL видео на YouTube
    --input_audio PATH          Путь к аудиофайлу (mp3, wav и др.)
    --input_text PATH           Путь к текстовому файлу (docx, md)

    --transcribe METHOD         Метод транскрибирования
    --translate METHOD          Метод перевода (можно указать несколько через запятую)

    --prompt PATH               Путь к текстовому файлу с промптом для Whisper
                                (помогает правильно распознавать имена, термины)
                                Для YouTube: если не указан, промпт создаётся из метаданных
                                Для аудиофайлов: рекомендуется указывать вручную

    --refine-model MODEL        Модель Ollama для улучшения транскрипции
                                (например: qwen2.5:3b, llama3:8b)
                                Требует запущенный Ollama сервер


    --refine-translation MODEL  Модель Ollama для улучшения перевода
                                (например: qwen2.5:3b, llama3:8b)
                                Делает перевод более естественным
                                Требует запущенный Ollama сервер

    --speakers                  Включить определение спикеров (в разработке)

    --translate-model MODEL     Модель NLLB для перевода
                                (по умолчанию: facebook/nllb-200-distilled-1.3B)
                                Доступные: facebook/nllb-200-distilled-600M (быстрее),
                                facebook/nllb-200-distilled-1.3B (лучше качество),
                                facebook/nllb-200-3.3B (самое лучшее, но медленно)

    --help, -h                  Показать эту справку

Доступные методы транскрибирования:
    - whisper_base              Whisper Base (локально, быстро)
    - whisper_small             Whisper Small (локально, средняя скорость)
    - whisper_medium            Whisper Medium (локально, медленно, высокое качество)
    - whisper_openai_api        Whisper через OpenAI API (в разработке)

Доступные методы перевода:
    - NLLB                      NLLB от Meta (локально)
    - openai_api                OpenAI API (в разработке)

Промпт для Whisper:
    Промпт помогает модели правильно распознавать специфичные слова:
    - Имена людей (например: "Hikaru Nakamura, Magnus Carlsen")
    - Технические термины (например: "FIDE, chess tournament")
    - Бренды и названия (например: "OpenAI, ChatGPT")

    Формат файла промпта: обычный текстовый файл с ключевыми словами через запятую
    Пример содержимого prompt.txt:
        FIDE, Hikaru Nakamura, Magnus Carlsen, chess tournament, bongcloud

Улучшение транскрипции с помощью LLM (--refine-model):
    После транскрипции текст можно улучшить с помощью локальной языковой модели:
    - Исправление терминологии и имён собственных
    - Добавление правильной пунктуации
    - Удаление слов-паразитов и оговорок
    - Структурирование текста на абзацы

    Структура выходных файлов:
    - Без --refine-model: название.docx, название.md
    - С --refine-model: название (original).docx/md, название (refined).docx/md
    - С --refine-model --translate: добавляется название (translated).docx/md

    Требования:
    1. Установленный Ollama: https://ollama.ai
    2. Загруженная модель: ollama pull qwen2.5:3b
    3. Запущенный сервер: ollama serve

    Рекомендуемые модели:
    - llama3.2:3b   - быстрая, хорошее качество
    - qwen2.5:3b    - быстрая, отлично для русского и английского
    - llama3:8b     - медленнее, но качественнее
    - mistral:7b    - хороший баланс

    ВНИМАНИЕ: qwen3:4b использует "chain of thought" и работает ОЧЕНЬ медленно (не рекомендуется)

Примечания:
    - Результаты сохраняются в папку 'output/' в форматах .docx и .md
    - Промежуточные файлы сохраняются в папку 'temp/'
    - Логи записываются в папку 'logs/'
    """
    print(help_text)


def process_text_file(
    text_path: str,
    translate_methods: Optional[list[str]] = None,
    refine_model: Optional[str] = None,
    refine_translation_model: Optional[str] = None,
    translate_model: Optional[str] = None
):
    """
    Обработка текстового файла (docx, md, txt)

    Args:
        text_path: Путь к текстовому файлу
        translate_methods: Список методов перевода
        refine_model: Модель Ollama для улучшения текста
        refine_translation_model: Модель Ollama для улучшения перевода
    """
    logger.info("=" * 60)
    logger.info("Начало обработки текстового файла")
    logger.info("=" * 60)

    text_path_obj = Path(text_path)
    text_title = sanitize_filename(text_path_obj.stem)

    # 1. Чтение текста из файла
    logger.info(f"\n[1/3] Чтение текста из файла: {text_path}")

    text_reader = TextReader()
    try:
        text_content = text_reader.read_file(text_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Ошибка чтения файла: {e}")
        return

    # Определяем язык
    detected_language = text_reader.detect_language(text_content)
    logger.info(f"Определён язык: {"русский" if detected_language == "ru" else "английский"}")

    # Создаём псевдо-сегменты для совместимости с существующей архитектурой
    paragraphs = [p.strip() for p in text_content.split("\n\n") if p.strip()]

    original_segments = []
    for i, para in enumerate(paragraphs):
        if para:
            original_segments.append({
                "text": para,
                "start": None,
                "end": None,
                "speaker": None
            })

    logger.info(f"Текст разбит на {len(original_segments)} параграфов")

    # 1.5. Улучшение текста с помощью LLM
    refined_segments = None
    if refine_model:
        logger.info(f"\n[1.5/3] Улучшение текста с помощью {refine_model}...")

        try:
            from .text_refiner import TextRefiner

            refiner = TextRefiner(model_name=refine_model)
            refined_text = refiner.refine_text(text_content)
            refined_paragraphs = [p.strip() for p in refined_text.split("\n\n") if p.strip()]

            refined_segments = []
            for para in refined_paragraphs:
                if para:
                    refined_segments.append({
                        "text": para,
                        "start": None,
                        "end": None,
                        "speaker": None
                    })

            logger.info(f"✅ Улучшение завершено ({len(refined_segments)} параграфов)")

        except ImportError:
            logger.warning("⚠️  text_refiner не доступен, пропускаем улучшение")
        except Exception as e:
            logger.error(f"Ошибка при улучшении текста: {e}")
            logger.warning("Продолжаем без улучшения")

    # 2. Перевод
    translated_segments_dict = {}
    if translate_methods:
        logger.info(f"\n[2/3] Перевод текста...")

        for method in translate_methods:
            logger.info(f"\n  Метод перевода: {method}")
            from .translator import Translator


            translator = Translator(method=method)  # TODO: Add model_name parameter
            segments_to_translate = refined_segments if refined_segments else original_segments

            try:
                translated_segments = translator.translate_segments(
                    segments_to_translate,
                    source_lang=detected_language,
                    target_lang="ru" if detected_language == "en" else "en"
                )

                if refine_translation_model and translated_segments:
                    logger.info(f"  Улучшение перевода с помощью {refine_translation_model}...")

                    try:
                        from .text_refiner import TextRefiner

                        translation_refiner = TextRefiner(model_name=refine_translation_model)
                        translated_text = "\n\n".join([seg["text"] for seg in translated_segments])
                        refined_translation = translation_refiner.refine_translation(translated_text)
                        refined_translation_paragraphs = [p.strip() for p in refined_translation.split("\n\n") if p.strip()]

                        for i, para in enumerate(refined_translation_paragraphs):
                            if i < len(translated_segments):
                                translated_segments[i]["text"] = para

                        logger.info(f"  ✅ Улучшение перевода завершено")

                    except Exception as e:
                        logger.error(f"Ошибка при улучшении перевода: {e}")
                        logger.warning("Используем неулучшенный перевод")

                translated_segments_dict[method] = translated_segments

            except Exception as e:
                logger.error(f"Ошибка при переводе методом {method}: {e}")

    # 3. Создание документов
    logger.info(f"\n[3/3] Создание документов...")

    writer = DocumentWriter()

    # Создаем документ с улучшенной версией (если есть)
    if refined_segments:
        logger.info(f"  Создание документа с улучшенным текстом...")
        docx_path_refined, md_path_refined = writer.create_from_segments(
            title=f"{text_title}_refined",
            transcription_segments=refined_segments,
            translation_segments=None,
            transcribe_method=f"Refined with {refine_model}",
            translate_method="",
            with_timestamps=False
        )
        logger.info(f"  ✅ Улучшенный: {md_path_refined}")

    # Создаем документы с переводами
    if translated_segments_dict:
        for method, translated_segs in translated_segments_dict.items():
            logger.info(f"  Создание документа с переводом ({method})...")

            docx_path_trans, md_path_trans = writer.create_from_segments(
                title=f"{text_title}_translated_{method}",
                transcription_segments=refined_segments if refined_segments else original_segments,
                translation_segments=translated_segs,
                transcribe_method=f"Loaded from {text_path_obj.suffix}" + (f" + {refine_model}" if refine_model else ""),
                translate_method=method,
                with_timestamps=False
            )
            logger.info(f"  ✅ Перевод: {md_path_trans}")

    # Если ничего не создали, выводим предупреждение
    if not refined_segments and not translated_segments_dict:
        logger.warning("⚠️  Не указаны параметры --refine-model или --translate")
        logger.info("Исходный файл не изменен. Укажите --refine-model для улучшения или --translate для перевода.")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Обработка текстового файла завершена!")
    logger.info("=" * 60)



def validate_args(args) -> bool:
    """
    Валидация аргументов командной строки
    
    Args:
        args: Аргументы из argparse
        
    Returns:
        True если валидация прошла успешно
    """
    # Проверяем что задан только один входной параметр
    input_count = sum([
        bool(args.url),
        bool(args.input_audio),
        bool(args.input_text)
    ])
    
    if input_count == 0:
        logger.error("Не указан входной параметр (url, input_audio или input_text)")
        return False
    
    if input_count > 1:
        logger.error("Можно указать только один входной параметр")
        return False
    
    # Для audio/video требуется метод транскрибирования
    if (args.url or args.input_audio) and not args.transcribe:
        logger.error("Для аудио/видео необходимо указать метод транскрибирования (--transcribe)")
        return False
    
    
    # Проверяем существование файлов
    if args.input_audio:
        if not Path(args.input_audio).exists():
            logger.error(f"Аудиофайл не найден: {args.input_audio}")
            return False
    
    if args.input_text:
        if not Path(args.input_text).exists():
            logger.error(f"Текстовый файл не найден: {args.input_text}")
            return False
    
    return True


def process_youtube_video(
    url: str,
    transcribe_method: str,
    translate_methods: Optional[list[str]] = None,
    with_speakers: bool = False,
    custom_prompt: Optional[str] = None,
    refine_model: Optional[str] = None,
    refine_translation_model: Optional[str] = None,
    translate_model: Optional[str] = None
):
    """
    Обработка YouTube видео

    Args:
        url: URL видео
        transcribe_method: Метод транскрибирования
        translate_methods: Список методов перевода
        with_speakers: Включить speaker diarization
        custom_prompt: Пользовательский промпт для Whisper (если None, генерируется из метаданных)
        refine_model: Модель Ollama для улучшения транскрипции
        refine_translation_model: Модель Ollama для улучшения перевода
    """
    logger.info("=" * 60)
    logger.info("Начало обработки YouTube видео")
    logger.info("=" * 60)
    
    # 1. Скачивание аудио и извлечение метаданных
    logger.info("\n[1/4] Скачивание аудио с YouTube...")
    downloader = YouTubeDownloader()
    audio_path, video_title, duration, metadata = downloader.download_audio(url)

    # 2. Транскрибирование
    logger.info("\n[2/4] Транскрибирование аудио...")

    # Определяем промпт: пользовательски�� или из метаданных
    if custom_prompt:
        whisper_prompt = custom_prompt
        logger.info("Используется пользовательский промпт")
    else:
        # Проверяем доступность Ollama для создания промпта
        try:
            import requests
            ollama_available = requests.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
        except:
            ollama_available = False

        if ollama_available and refine_model:
            # Используем ту же модель что и для улучшения транскрипции
            whisper_prompt = create_whisper_prompt_with_llm(metadata, use_ollama=True, model=refine_model)
            logger.info("Промпт создан с помощью LLM из метаданных видео")
        else:
            whisper_prompt = create_whisper_prompt(metadata)
            logger.info("Промпт создан из метаданных видео (стандартный метод)")

    transcriber = Transcriber(method=transcribe_method)
    transcription_segments = transcriber.transcribe(
        audio_path,
        language=None,  # Автоопределение
        with_speakers=with_speakers,
        initial_prompt=whisper_prompt
    )

    # Сохраняем оригинальные сегменты для создания двух версий документа
    original_transcription_segments = transcription_segments

    # 2.5. Улучшение транскрипции с помощью LLM (если указана модель)
    refined_transcription_segments = None
    if refine_model:
        logger.info(f"\n[2.5/4] Улучшение транскрипции с помощью {refine_model}...")
        try:
            from .text_refiner import TextRefiner

            refiner = TextRefiner(model_name=refine_model)

            # Получаем текст из сегментов
            original_text = transcriber.segments_to_text(transcription_segments)

            # Улучшаем текст (используем whisper_prompt как контекст)
            refined_text = refiner.refine_text(original_text, context=whisper_prompt)

            # Создаем улучшенные сегменты
            refined_transcription_segments = transcriber.update_segments_from_text(
                transcription_segments,
                refined_text
            )

            logger.info("Транскрипция улучшена")
        except Exception as e:
            logger.error(f"Ошибка при улучшении транскрипции: {e}")
            logger.warning("Продолжаем с оригинальной транскрипцией")
            refined_transcription_segments = None

    # 3. Перевод (если требуется)
    translation_segments = None
    translation_segments_refined = None
    translate_method_str = ""

    if translate_methods:
        logger.info("\n[3/4] Перевод текста...")

        # Определяем язык оригинала
        original_text = transcriber.segments_to_text(original_transcription_segments)
        source_lang = detect_language(original_text)

        # Используем первый метод перевода (в MVP)
        translate_method = translate_methods[0]

        from .translator import Translator

        if source_lang == "en":
            translator = Translator(method=translate_method)

            # Если есть улучшенная версия, переводим только её
            if refined_transcription_segments:
                logger.info("Перевод улучшенной транскрипции...")
                translation_segments_refined = translator.translate_segments(
                    refined_transcription_segments,
                    source_lang="en",
                    target_lang="ru"
                )
                logger.info("Перевод улучшенной транскрипции выполнен")
            else:
                # Если нет улучшенной версии, переводим оригинал
                translation_segments = translator.translate_segments(
                    original_transcription_segments,
                    source_lang="en",
                    target_lang="ru"
                )
                logger.info("Перевод оригинальной транскрипции выполнен")
        else:
            logger.info("Видео на русском языке, перевод не требуется")
    else:
        logger.info("\n[3/4] Перевод не требуется")

    # 3.5. Улучшение перевода с помощью LLM (если указана модель и есть перевод)
    translation_segments_refined_llm = None
    if refine_translation_model and (translation_segments_refined or translation_segments):
        logger.info(f"\n[3.5/4] Улучшение перевода с помощью {refine_translation_model}...")
        try:
            from .text_refiner import TextRefiner

            refiner = TextRefiner(model_name=refine_translation_model)

            # Определяем какой перевод улучшать
            segments_to_refine = translation_segments_refined if translation_segments_refined else translation_segments

            # Получаем текст перевода
            translated_text = transcriber.segments_to_text(segments_to_refine)

            # Улучшаем перевод
            refined_translation_text = refiner.refine_translation(translated_text, context=whisper_prompt)

            # Создаем улучшенные сегменты перевода
            translation_segments_refined_llm = transcriber.update_segments_from_text(
                segments_to_refine,
                refined_translation_text
            )

            logger.info("Перевод улучшен")
        except Exception as e:
            logger.error(f"Ошибка при улучшении перевода: {e}")
            logger.warning("Продолжаем с оригинальным переводом")
            translation_segments_refined_llm = None

    # 4. Создание документов
    logger.info("\n[4/4] Создание выходных документов...")
    writer = DocumentWriter()

    # Если есть улучшенная версия транскрипции
    if refined_transcription_segments:
        # Документ с оригинальной версией (без перевода)
        logger.info("Создание документа с оригинальной транскрипцией...")
        docx_path_orig, md_path_orig = writer.create_from_segments(
            title=f"{video_title} (original)",
            transcription_segments=original_transcription_segments,
            translation_segments=None,  # без перевода
            transcribe_method=transcribe_method,
            translate_method="",
            with_timestamps=False
        )

        # Документ с улучшенной версией (с переводом если есть, но не улучшенным)
        logger.info("Создание документа с улучшенной транскрипцией...")
        docx_path_refined, md_path_refined = writer.create_from_segments(
            title=f"{video_title} (refined)",
            transcription_segments=refined_transcription_segments,
            translation_segments=translation_segments_refined,  # перевод улучшенной версии
            transcribe_method=f"{transcribe_method} + {refine_model}",
            translate_method=translate_method_str,
            with_timestamps=False
        )

        # Если есть улучшенный перевод, создаем дополнительный документ
        if translation_segments_refined_llm:
            logger.info("Соз��ание документа с улучшенным переводом...")
            docx_path_trans_refined, md_path_trans_refined = writer.create_from_segments(
                title=f"{video_title} (translated refined)",
                transcription_segments=refined_transcription_segments if refined_transcription_segments else original_transcription_segments,
                translation_segments=translation_segments_refined_llm,
                transcribe_method=f"{transcribe_method} + {refine_model}" if refine_model else transcribe_method,
                translate_method=f"{translate_method_str} + {refine_translation_model}",
                with_timestamps=False
            )

            logger.info("\n" + "=" * 60)
            logger.info("Обработка завершена успешно!")
            logger.info(f"Результаты сохранены:")
            logger.info(f"\nОригинальная версия:")
            logger.info(f"  - {docx_path_orig}")
            logger.info(f"  - {md_path_orig}")
            logger.info(f"\nУлучшенная транскрипция:")
            logger.info(f"  - {docx_path_refined}")
            logger.info(f"  - {md_path_refined}")
            logger.info(f"\nУлучшенный перевод:")
            logger.info(f"  - {docx_path_trans_refined}")
            logger.info(f"  - {md_path_trans_refined}")
            logger.info("=" * 60)
        else:
            logger.info("\n" + "=" * 60)
            logger.info("Обработка завершена успешно!")
            logger.info(f"Результаты сохранены:")
            logger.info(f"\nОригинальная версия:")
            logger.info(f"  - {docx_path_orig}")
            logger.info(f"  - {md_path_orig}")
            logger.info(f"\nУлучшенная версия:")
            logger.info(f"  - {docx_path_refined}")
            logger.info(f"  - {md_path_refined}")
            logger.info("=" * 60)
    else:
        # Только оригинальная версия
        # Если есть улучшенный перевод (без улучшенной транскрипции)
        if translation_segments_refined_llm:
            # Документ с оригинальным переводом
            logger.info("Создание документа с оригинальным переводом...")
            docx_path_orig, md_path_orig = writer.create_from_segments(
                title=f"{video_title} (translated)",
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments,
                transcribe_method=transcribe_method,
                translate_method=translate_method_str,
                with_timestamps=False
            )

            # Документ с улучшенным переводом
            logger.info("Создание документа с улучшенным переводом...")
            docx_path_refined, md_path_refined = writer.create_from_segments(
                title=f"{video_title} (translated refined)",
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments_refined_llm,
                transcribe_method=transcribe_method,
                translate_method=f"{translate_method_str} + {refine_translation_model}",
                with_timestamps=False
            )

            logger.info("\n" + "=" * 60)
            logger.info("Обработка завершена успешно!")
            logger.info(f"Результаты сохранены:")
            logger.info(f"\nОригинальный перевод:")
            logger.info(f"  - {docx_path_orig}")
            logger.info(f"  - {md_path_orig}")
            logger.info(f"\nУлучшенный перевод:")
            logger.info(f"  - {docx_path_refined}")
            logger.info(f"  - {md_path_refined}")
            logger.info("=" * 60)
        else:
            # Только один документ (оригинал с переводом или без)
            docx_path, md_path = writer.create_from_segments(
                title=video_title,
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments,
                transcribe_method=transcribe_method,
                translate_method=translate_method_str,
                with_timestamps=False
            )

            logger.info("\n" + "=" * 60)
            logger.info("Обработка завершена успешно!")
            logger.info(f"Результаты сохранены:")
            logger.info(f"  - {docx_path}")
            logger.info(f"  - {md_path}")
            logger.info("=" * 60)


def process_local_audio(
    audio_path: str,
    transcribe_method: str,
    translate_methods: Optional[list[str]] = None,
    with_speakers: bool = False,
    custom_prompt: Optional[str] = None,
    refine_model: Optional[str] = None,
    refine_translation_model: Optional[str] = None,
    translate_model: Optional[str] = None
):
    """
    Обработка локального аудиофайла

    Args:
        audio_path: Путь к аудиофайлу
        transcribe_method: Метод транскрибирования
        translate_methods: Список методов перевода
        with_speakers: Включить speaker diarization
        custom_prompt: Пользовательский промпт для Whisper
        refine_model: Модель Ollama для улучшения транскрипции
        refine_translation_model: Модель Ollama для улучшения перевода
    """
    logger.info("=" * 60)
    logger.info("Начало обработки локального аудиофайла")
    logger.info("=" * 60)

    audio_path_obj = Path(audio_path)
    audio_title = sanitize_filename(audio_path_obj.stem)  # Очищенное имя файла без расширения

    # 1. Транскрибирование
    logger.info("\n[1/3] Транскрибирование аудио...")

    # Используем пользовательский промпт если передан
    if custom_prompt:
        logger.info("Используется пользовательский промпт")
    else:
        logger.info("Промпт не задан (будет использоваться автоматическое распознавание)")

    transcriber = Transcriber(method=transcribe_method)
    transcription_segments = transcriber.transcribe(
        audio_path_obj,
        language=None,  # Автоопределение
        with_speakers=with_speakers,
        initial_prompt=custom_prompt
    )

    # Сохраняем оригинальные сегменты для создания двух версий документа
    original_transcription_segments = transcription_segments

    # 1.5. Улучшение транскрипции с помощью LLM (если указана модель)
    refined_transcription_segments = None
    if refine_model:
        logger.info(f"\n[1.5/3] Улучшение транскрипции с помощью {refine_model}...")
        try:
            from .text_refiner import TextRefiner

            refiner = TextRefiner(model_name=refine_model)

            # Получаем текст из сегментов
            original_text = transcriber.segments_to_text(transcription_segments)

            # Улучшаем текст (используем custom_prompt как контекст)
            refined_text = refiner.refine_text(original_text, context=custom_prompt)

            # Создаем улучшенные сегменты
            refined_transcription_segments = transcriber.update_segments_from_text(
                transcription_segments,
                refined_text
            )

            logger.info("Транскрипция улучшена")
        except Exception as e:
            logger.error(f"Ошибка при улучшении транскрипции: {e}")
            logger.warning("Продолжаем с оригинальной транскрипцией")
            refined_transcription_segments = None

    # 2. Перевод (если требуется)
    translation_segments = None
    translation_segments_refined = None
    translate_method_str = ""

    if translate_methods:
        logger.info("\n[2/3] Перевод текста...")

        # Определяем язык оригинала
        original_text = transcriber.segments_to_text(transcription_segments)
        source_lang = detect_language(original_text)

        # Используем первый метод перевода (в MVP)

        translate_method = translate_methods[0]
        translate_method_str = translate_method

        from .translator import Translator


        if source_lang == "en":
            translator = Translator(method=translate_method)

            # Если есть улучшенная версия, переводим только её
            if refined_transcription_segments:
                logger.info("Перевод ул��чшенной транскрипции...")
                translation_segments_refined = translator.translate_segments(
                    refined_transcription_segments,
                    source_lang="en",
                    target_lang="ru"
                )
                logger.info("Перевод улучшенной транскрипции выполнен")
            else:
                # Если нет улучшенной версии, переводим оригинал
                translation_segments = translator.translate_segments(
                    transcription_segments,
                    source_lang="en",
                    target_lang="ru"
                )
                logger.info("Перевод оригинальной транскрипции выполнен")
        else:
            logger.info("Аудио на русском языке, перевод не требуется")
    else:
        logger.info("\n[2/3] Перевод не требуется")

    # 2.5. Улучшение перевода с помощью LLM (если указана модель и есть перевод)
    translation_segments_refined_llm = None
    if refine_translation_model and (translation_segments_refined or translation_segments):
        logger.info(f"\n[2.5/3] Улучшение перевода с помощью {refine_translation_model}...")
        try:
            from .text_refiner import TextRefiner

            refiner = TextRefiner(model_name=refine_translation_model)

            # Определяем какой перевод улучшать
            segments_to_refine = translation_segments_refined if translation_segments_refined else translation_segments

            # Получаем текст перевода
            translated_text = transcriber.segments_to_text(segments_to_refine)

            # Улучшаем перевод
            refined_translation_text = refiner.refine_translation(translated_text, context=custom_prompt)

            # Создаем улучшенные сегменты перевода
            translation_segments_refined_llm = transcriber.update_segments_from_text(
                segments_to_refine,
                refined_translation_text
            )

            logger.info("Перевод улучшен")
        except Exception as e:
            logger.error(f"Ошибка при улучшении перевода: {e}")
            logger.warning("Продолжаем с оригинальным переводом")
            translation_segments_refined_llm = None

    # 3. Создание документов
    logger.info("\n[3/3] Создание выходных документов...")
    writer = DocumentWriter()

    # Если есть улучшенная версия транскрипции
    if refined_transcription_segments:
        # Документ с оригинальной версией (без перевода)
        logger.info("Создание документа с оригинальной транскрипцией...")
        docx_path_orig, md_path_orig = writer.create_from_segments(
            title=f"{audio_title} (original)",
            transcription_segments=original_transcription_segments,
            translation_segments=None,  # без перевода
            transcribe_method=transcribe_method,
            translate_method="",
            with_timestamps=False
        )

        # Документ с улучшенной версией (с переводом если есть, но не улучшенным)
        logger.info("Создание документа с улучшенной транскрипцией...")
        docx_path_refined, md_path_refined = writer.create_from_segments(
            title=f"{audio_title} (refined)",
            transcription_segments=refined_transcription_segments,
            translation_segments=translation_segments_refined,  # перевод улучшенной версии
            transcribe_method=f"{transcribe_method} + {refine_model}",
            translate_method=translate_method_str,
            with_timestamps=False
        )

        # Если есть улучшенный перевод, создаем дополнительный документ
        if translation_segments_refined_llm:
            logger.info("Создание документа с улучшенным переводом...")
            docx_path_trans_refined, md_path_trans_refined = writer.create_from_segments(
                title=f"{audio_title} (translated refined)",
                transcription_segments=refined_transcription_segments if refined_transcription_segments else original_transcription_segments,
                translation_segments=translation_segments_refined_llm,
                transcribe_method=f"{transcribe_method} + {refine_model}" if refine_model else transcribe_method,
                translate_method=f"{translate_method_str} + {refine_translation_model}",
                with_timestamps=False
            )

            logger.info("\n" + "=" * 60)
            logger.info("Обработка завершена успешно!")
            logger.info(f"Результаты сохранены:")
            logger.info(f"\nОригинальная версия:")
            logger.info(f"  - {docx_path_orig}")
            logger.info(f"  - {md_path_orig}")
            logger.info(f"\nУлучшенная транскрипция:")
            logger.info(f"  - {docx_path_refined}")
            logger.info(f"  - {md_path_refined}")
            logger.info(f"\nУлучшенный перевод:")
            logger.info(f"  - {docx_path_trans_refined}")
            logger.info(f"  - {md_path_trans_refined}")
            logger.info("=" * 60)
        else:
            logger.info("\n" + "=" * 60)
            logger.info("Обработка завершена успешно!")
            logger.info(f"Результаты сохранены:")
            logger.info(f"\nОригинальная версия:")
            logger.info(f"  - {docx_path_orig}")
            logger.info(f"  - {md_path_orig}")
            logger.info(f"\nУлучшенная версия:")
            logger.info(f"  - {docx_path_refined}")
            logger.info(f"  - {md_path_refined}")
            logger.info("=" * 60)
    else:
        # Только оригинальная версия
        # Если есть улучшенный перевод (без улучшенной транскрипции)
        if translation_segments_refined_llm:
            # Документ с оригинальным переводом
            logger.info("Создание документа с оригинальным переводом...")
            docx_path_orig, md_path_orig = writer.create_from_segments(
                title=f"{audio_title} (translated)",
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments,
                transcribe_method=transcribe_method,
                translate_method=translate_method_str,
                with_timestamps=False
            )

            # Документ с улучшенным переводом
            logger.info("Создание документа с улучшенным переводом...")
            docx_path_refined, md_path_refined = writer.create_from_segments(
                title=f"{audio_title} (translated refined)",
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments_refined_llm,
                transcribe_method=transcribe_method,
                translate_method=f"{translate_method_str} + {refine_translation_model}",
                with_timestamps=False
            )

            logger.info("\n" + "=" * 60)
            logger.info("Обработка завершена успешно!")
            logger.info(f"Результаты сохранены:")
            logger.info(f"\nОригинальный перевод:")
            logger.info(f"  - {docx_path_orig}")
            logger.info(f"  - {md_path_orig}")
            logger.info(f"\nУлучшенный перевод:")
            logger.info(f"  - {docx_path_refined}")
            logger.info(f"  - {md_path_refined}")
            logger.info("=" * 60)
        else:
            # Только один документ (оригинал с переводом или без)
            docx_path, md_path = writer.create_from_segments(
                title=audio_title,
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments,
                transcribe_method=transcribe_method,
                translate_method=translate_method_str,
                with_timestamps=False
            )

            logger.info("\n" + "=" * 60)
            logger.info("Обработка завершена успешно!")
            logger.info(f"Результаты сохранены:")
            logger.info(f"  - {docx_path}")
            logger.info(f"  - {md_path}")
            logger.info("=" * 60)


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description="YouTube Transcriber & Translator",
        add_help=False
    )

    # Входные данные
    parser.add_argument('--url', type=str, help='URL видео на YouTube')
    parser.add_argument('--input_audio', type=str, help='Путь к аудиофайлу')
    parser.add_argument('--input_text', type=str, help='Путь к текстовому файлу')

    # Методы обработки
    parser.add_argument('--transcribe', type=str, help='Метод транскрибирования')
    parser.add_argument('--translate', type=str, help='Метод перевода (через запятую)')

    # Дополнительные опции
    parser.add_argument('--prompt', type=str, help='Путь к текстовому файлу с промптом для Whisper')
    parser.add_argument('--refine-model', type=str, help='Модель Ollama для улучшения транскрипции (например: qwen2.5:3b)')
    parser.add_argument('--refine-translation', type=str, help='Модель Ollama для улучшения перевода (например: qwen2.5:3b)')
    parser.add_argument('--speakers', action='store_true', help='Включить определение спикеров')
    parser.add_argument('--translate-model', type=str, help='Модель NLLB для перевода (по умолчанию: facebook/nllb-200-distilled-1.3B)')

    parser.add_argument('--help', '-h', action='store_true', help='Показать справку')

    args = parser.parse_args()

    # Показываем справку
    if args.help or len(sys.argv) == 1:
        print_help()
        sys.exit(0)

    # Валидация аргументов
    if not validate_args(args):
        sys.exit(1)

    try:
        # Парсим методы перевода
        translate_methods = None
        if args.translate:
            translate_methods = [m.strip() for m in args.translate.split(',')]

        # Загружаем пользовательский промпт если указан
        custom_prompt = None
        if args.prompt:
            custom_prompt = load_prompt_from_file(args.prompt)

        # Обработка в зависимости от типа входных данных
        if args.url:
            process_youtube_video(
                url=args.url,
                transcribe_method=args.transcribe,
                translate_methods=translate_methods,
                with_speakers=args.speakers,
                custom_prompt=custom_prompt,
                refine_model=args.refine_model,
                refine_translation_model=args.refine_translation,
                translate_model=args.translate_model
            )
        elif args.input_audio:
            process_local_audio(
                audio_path=args.input_audio,
                transcribe_method=args.transcribe,
                translate_methods=translate_methods,
                with_speakers=args.speakers,
                custom_prompt=custom_prompt,
                refine_model=args.refine_model,
                refine_translation_model=args.refine_translation,
                translate_model=args.translate_model
            )
        elif args.input_text:
            process_text_file(
                text_path=args.input_text,
                translate_methods=translate_methods,
                refine_model=args.refine_model,
                refine_translation_model=args.refine_translation,
                translate_model=args.translate_model
            )


    except KeyboardInterrupt:
        logger.info("\nОбработка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Произошла ошибка: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()