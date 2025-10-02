"""
Полный тест транскрибирования и перевода без скачивания
"""
from pathlib import Path
from src.transcriber import Transcriber
from src.translator import Translator
from src.document_writer import DocumentWriter
from src.logger import logger
from src.utils import detect_language
import sys

print("=" * 70)
print("🎬 ТЕСТИРОВАНИЕ ТРАНСКРИБИРОВАНИЯ И ПЕРЕВОДА")
print("=" * 70)

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

# Путь к аудиофайлу
AUDIO_FILE = "temp/test.mp3"  # Или test_10s.mp3 для быстрого теста

# Методы
TRANSCRIBE_METHOD = "whisper_base"
TRANSLATE_METHOD = "NLLB"

# Создать документы?
CREATE_DOCUMENTS = True

# ============================================================================
# ПРОВЕРКИ
# ============================================================================

audio_path = Path(AUDIO_FILE)

if not audio_path.exists():
    print(f"❌ Файл не найден: {audio_path}")
    print("\n💡 Создайте тестовый файл:")
    print("   ffmpeg -i temp/Rick_Astley*.mp3 -t 10 temp/test.mp3")
    print("\nИли используйте полный файл:")
    print("   AUDIO_FILE = 'temp/Rick_Astley_-_Never_Gonna_Give_You_Up_(Official_Video)_(4K_Remaster).mp3'")
    sys.exit(1)

print(f"✅ Файл найден: {audio_path}")
print(f"📏 Размер: {audio_path.stat().st_size / 1024 / 1024:.2f} MB")
print()

# ============================================================================
# ШАГ 1: ТРАНСКРИБИРОВАНИЕ
# ============================================================================

print("=" * 70)
print("📝 ШАГ 1: ТРАНСКРИБИРОВАНИЕ")
print("=" * 70)

print(f"Метод: {TRANSCRIBE_METHOD}")
print("Загрузка модели Whisper...")

transcriber = Transcriber(method=TRANSCRIBE_METHOD)

print(f"Устройство: {transcriber.device}")
print("Начинаем транскрибирование...\n")

segments = transcriber.transcribe(
    audio_path,
    language=None,  # Автоопределение
    with_speakers=False
)

print(f"\n✅ Транскрибирование завершено!")
print(f"   Создано сегментов: {len(segments)}")

# Получаем текст
original_text = transcriber.segments_to_text(segments)
original_text_with_timestamps = transcriber.segments_to_text_with_timestamps(segments)

# Определяем язык
detected_lang = detect_language(original_text)
print(f"   Определенный язык: {detected_lang}")

# Выводим первые 3 сегмента
print("\n📄 Первые 3 сегмента:")
print("-" * 70)
for seg in segments[:3]:
    print(f"[{seg.start:6.1f}s - {seg.end:6.1f}s] {seg.text}")

# Статистика
total_duration = segments[-1].end if segments else 0
word_count = len(original_text.split())
print("\n📊 Статистика транскрипции:")
print(f"   Длительность: {total_duration:.1f} секунд")
print(f"   Количество слов: {word_count}")
print(f"   Количество символов: {len(original_text)}")

# ============================================================================
# ШАГ 2: ПЕРЕВОД
# ============================================================================

print("\n" + "=" * 70)
print("🌍 ШАГ 2: ПЕРЕВОД")
print("=" * 70)

if detected_lang == "ru":
    print("⚠️  Текст на русском языке, перевод не требуется")
    translation_segments = None
    translated_text = None
    translated_text_with_timestamps = None
else:
    print(f"Метод: {TRANSLATE_METHOD}")
    print(f"Направление: {detected_lang} → ru")
    print("Загрузка модели NLLB...")
    
    translator = Translator(method=TRANSLATE_METHOD)
    
    print("Начинаем перевод...\n")
    
    # Переводим сегменты
    translation_segments = translator.translate_segments(
        segments,
        source_lang=detected_lang,
        target_lang="ru"
    )
    
    print(f"\n✅ Перевод завершен!")
    
    # Получаем переведенный текст
    translated_text = transcriber.segments_to_text(translation_segments)
    translated_text_with_timestamps = transcriber.segments_to_text_with_timestamps(
        translation_segments
    )
    
    # Выводим первые 3 переведенных сегмента
    print("\n📄 Первые 3 переведенных сегмента:")
    print("-" * 70)
    for seg in translation_segments[:3]:
        print(f"[{seg.start:6.1f}s - {seg.end:6.1f}s] {seg.text}")
    
    # Сравнение
    print("\n📊 Сравнение:")
    print(f"   Оригинал слов: {word_count}")
    print(f"   Перевод слов: {len(translated_text.split())}")
    print(f"   Оригинал символов: {len(original_text)}")
    print(f"   Перевод символов: {len(translated_text)}")

# ============================================================================
# ШАГ 3: СОЗДАНИЕ ДОКУМЕНТОВ
# ============================================================================

if CREATE_DOCUMENTS:
    print("\n" + "=" * 70)
    print("📄 ШАГ 3: СОЗДАНИЕ ДОКУМЕНТОВ")
    print("=" * 70)
    
    writer = DocumentWriter()
    
    # Название для документа
    doc_title = f"TEST_{audio_path.stem}"
    
    print(f"Создание документов: {doc_title}")
    
    docx_path, md_path = writer.create_from_segments(
        title=doc_title,
        transcription_segments=segments,
        translation_segments=translation_segments,
        transcribe_method=TRANSCRIBE_METHOD,
        translate_method=TRANSLATE_METHOD if translation_segments else "",
        with_timestamps=True
    )
    
    print(f"\n✅ Документы созданы:")
    print(f"   📗 DOCX: {docx_path}")
    print(f"   📘 MD:   {md_path}")
    
    # Показываем размеры
    print(f"\n📏 Размеры файлов:")
    print(f"   DOCX: {docx_path.stat().st_size / 1024:.1f} KB")
    print(f"   MD:   {md_path.stat().st_size / 1024:.1f} KB")

# ============================================================================
# ИТОГОВЫЙ ОТЧЕТ
# ============================================================================

print("\n" + "=" * 70)
print("✅ ТЕСТ ЗАВЕРШЕН УСПЕШНО!")
print("=" * 70)

print("\n📋 Что было сделано:")
print(f"   ✅ Транскрибировано сегментов: {len(segments)}")
if translation_segments:
    print(f"   ✅ Переведено сегментов: {len(translation_segments)}")
if CREATE_DOCUMENTS:
    print(f"   ✅ Создано документов: 2 (DOCX + MD)")

print("\n🎯 Проверьте результаты:")
if CREATE_DOCUMENTS:
    print(f"   - Откройте: {docx_path}")
    print(f"   - Или:      {md_path}")

print("\n" + "=" * 70)