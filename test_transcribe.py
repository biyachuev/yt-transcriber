"""
Быстрый тест транскрибирования без скачивания
"""
from pathlib import Path
from src.transcriber import Transcriber
from src.logger import logger

# Путь к уже скачанному файлу
audio_path = Path("temp/Rick_Astley_-_Never_Gonna_Give_You_Up_(Official_Video)_(4K_Remaster).mp3")
# audio_path = Path("temp/NEW_FIDE_HIKARULE_DRAMA!!.mp3")

if not audio_path.exists():
    print(f"❌ Файл не найден: {audio_path}")
    print("Скачайте видео один раз командой:")
    print('python -m src.main --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --transcribe whisper_base')
    exit(1)

print(f"✅ Используем файл: {audio_path}")
print("=" * 60)

# Транскрибирование
# transcriber = Transcriber(method="whisper_base")
transcriber = Transcriber(method="whisper_small")
segments = transcriber.transcribe(audio_path, language=None)

# Вывод результата
print("\n" + "=" * 60)
print("РЕЗУЛЬТАТ ТРАНСКРИБИРОВАНИЯ:")
print("=" * 60)
text = transcriber.segments_to_text_with_timestamps(segments)
print(text[:500])  # Первые 500 символов
print(f"\n... (всего {len(segments)} сегментов)")