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
from .translator import Translator
from .document_writer import DocumentWriter
from .utils import detect_language


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

Опции:
    --url URL                   URL видео на YouTube
    --input_audio PATH          Путь к аудиофайлу (mp3, wav и др.)
    --input_text PATH           Путь к текстовому файлу (docx, md)
    
    --transcribe METHOD         Метод транскрибирования
    --translate METHOD          Метод перевода (можно указать несколько через запятую)
    
    --speakers                  Включить определение спикеров (в разработке)
    --help, -h                  Показать эту справку

Доступные методы транскрибирования:
    - whisper_base              Whisper Base (локально)
    - whisper_small             Whisper Small (локально)
    - whisper_openai_api        Whisper через OpenAI API (в разработке)

Доступные методы перевода:
    - NLLB                      NLLB от Meta (локально)
    - openai_api                OpenAI API (в разработке)

Примечания:
    - Результаты сохраняются в папку 'output/' в форматах .docx и .md
    - Промежуточные файлы сохраняются в папку 'temp/'
    - Логи записываются в папку 'logs/'
    """
    print(help_text)


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
    
    # Для текста требуется метод перевода
    if args.input_text and not args.translate:
        logger.error("Для текстового файла необходимо указать метод перевода (--translate)")
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
    with_speakers: bool = False
):
    """
    Обработка YouTube видео
    
    Args:
        url: URL видео
        transcribe_method: Метод транскрибирования
        translate_methods: Список методов перевода
        with_speakers: Включить speaker diarization
    """
    logger.info("=" * 60)
    logger.info("Начало обработки YouTube видео")
    logger.info("=" * 60)
    
    # 1. Скачивание аудио
    logger.info("\n[1/4] Скачивание аудио с YouTube...")
    downloader = YouTubeDownloader()
    audio_path, video_title, duration = downloader.download_audio(url)
    
    # 2. Транскрибирование
    logger.info("\n[2/4] Транскрибирование аудио...")
    transcriber = Transcriber(method=transcribe_method)
    transcription_segments = transcriber.transcribe(
        audio_path,
        language=None,  # Автоопределение
        with_speakers=with_speakers
    )
    
    # 3. Перевод (если требуется)
    translation_segments = None
    translate_method_str = ""
    
    if translate_methods:
        logger.info("\n[3/4] Перевод текста...")
        
        # Определяем язык оригинала
        original_text = transcriber.segments_to_text(transcription_segments)
        source_lang = detect_language(original_text)
        
        # Используем первый метод перевода (в MVP)
        translate_method = translate_methods[0]
        translate_method_str = translate_method
        
        if source_lang == "en":
            translator = Translator(method=translate_method)
            translation_segments = translator.translate_segments(
                transcription_segments,
                source_lang="en",
                target_lang="ru"
            )
            logger.info("Перевод выполнен")
        else:
            logger.info("Видео на русском языке, перевод не требуется")
    else:
        logger.info("\n[3/4] Перевод не требуется")
    
    # 4. Создание документов
    logger.info("\n[4/4] Создание выходных документов...")
    writer = DocumentWriter()
    docx_path, md_path = writer.create_from_segments(
        title=video_title,
        transcription_segments=transcription_segments,
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
    parser.add_argument('--speakers', action='store_true', help='Включить определение спикеров')
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
        
        # Обработка в зависимости от типа входных данных
        if args.url:
            process_youtube_video(
                url=args.url,
                transcribe_method=args.transcribe,
                translate_methods=translate_methods,
                with_speakers=args.speakers
            )
        elif args.input_audio:
            logger.info("Обработка аудиофайлов будет доступна в следующей версии")
            sys.exit(1)
        elif args.input_text:
            logger.info("Обработка текстовых файлов будет доступна в следующей версии")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("\nОбработка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Произошла ошибка: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()