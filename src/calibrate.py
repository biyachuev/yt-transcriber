"""
Калибровка оценки времени обработки для разных моделей
Запускайте: python calibrate.py [whisper_base|whisper_small]
"""
import time
from pathlib import Path
import subprocess
import sys

# Добавляем родительскую директорию в путь для импорта
script_dir = Path(__file__).parent
project_root = script_dir if (script_dir / "src").exists() else script_dir.parent
sys.path.insert(0, str(project_root))

from src.transcriber import Transcriber
from src.translator import Translator
from src.utils import detect_language
from src.config import settings, TranscribeOptions

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

# Определяем модель из аргументов командной строки
if len(sys.argv) > 1:
    MODEL = sys.argv[1]
    if MODEL not in ["whisper_base", "whisper_small"]:
        print(f"❌ Неизвестная модель: {MODEL}")
        print("💡 Используйте: python calibrate.py [whisper_base|whisper_small]")
        sys.exit(1)
else:
    MODEL = "whisper_base"  # По умолчанию

print("=" * 70)
print("⏱️  КАЛИБРОВКА ОЦЕНКИ ВРЕМЕНИ")
print("=" * 70)
print(f"🎯 Модель: {MODEL}")
print()

# ============================================================================
# Поиск тестового файла
# ============================================================================

temp_dir = project_root / "temp"

if not temp_dir.exists():
    print(f"❌ Директория temp/ не найдена: {temp_dir}")
    print("\n💡 Создайте тестовый файл:")
    print("   ffmpeg -i temp/your_audio.mp3 -t 30 temp/test.mp3")
    sys.exit(1)

# Ищем test.mp3 или любой другой mp3
test_files = list(temp_dir.glob("test*.mp3"))
if not test_files:
    test_files = list(temp_dir.glob("*.mp3"))

if not test_files:
    print(f"❌ Не найдено mp3 файлов в {temp_dir}")
    print("\n💡 Скачайте видео или создайте тестовый файл:")
    print("   python -m src.main --url 'YOUTUBE_URL' --transcribe whisper_base")
    sys.exit(1)

audio_path = test_files[0]
print(f"✅ Найден файл: {audio_path.name}")
print(f"📁 Путь: {audio_path}")

# ============================================================================
# Получение длительности
# ============================================================================

try:
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 
         'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
         str(audio_path)],
        capture_output=True,
        text=True,
        check=True
    )
    duration = float(result.stdout.strip())
except subprocess.CalledProcessError:
    print("❌ Ошибка при определении длительности")
    print("💡 Проверьте что ffmpeg установлен: brew install ffmpeg")
    sys.exit(1)
except ValueError:
    print("❌ Не удалось получить длительность файла")
    sys.exit(1)

print(f"📏 Длительность аудио: {duration:.1f} секунд ({duration/60:.1f} минут)")
print()

# ============================================================================
# Тест транскрибирования
# ============================================================================

print("=" * 70)
print(f"📝 ТЕСТ ТРАНСКРИБИРОВАНИЯ ({MODEL})")
print("=" * 70)
print()

print(f"Загрузка модели {MODEL}...")
transcriber = Transcriber(method=MODEL)
print(f"Устройство: {transcriber.device}")

# Информация о модели
model_info = {
    "whisper_base": {
        "params": "74M параметров",
        "size": "~150 MB",
        "quality": "хорошее",
        "speed": "быстро"
    },
    "whisper_small": {
        "params": "244M параметров", 
        "size": "~500 MB",
        "quality": "очень хорошее",
        "speed": "медленнее"
    }
}

info = model_info.get(MODEL, {})
if info:
    print(f"   Параметры: {info['params']}")
    print(f"   Размер: {info['size']}")
    print(f"   Качество: {info['quality']}")
    print(f"   Скорость: {info['speed']}")
print()

print("🎵 Начинаем транскрибирование...")
print("   (Первый запуск может быть медленнее из-за загрузки модели)")
start_time = time.time()

try:
    segments = transcriber.transcribe(audio_path, language=None)
    transcribe_time = time.time() - start_time
except Exception as e:
    print(f"❌ Ошибка при транскрибировании: {e}")
    sys.exit(1)

transcribe_ratio = transcribe_time / duration

print()
print(f"✅ Транскрибирование завершено!")
print(f"   Время обработки: {transcribe_time:.1f} секунд")
print(f"   Создано сегментов: {len(segments)}")
print(f"   📊 МНОЖИТЕЛЬ: {transcribe_ratio:.2f}x")
print()

if transcribe_ratio < 0.5:
    print("   🚀 Очень быстро! (меньше половины реального времени)")
elif transcribe_ratio < 1.0:
    print("   ✨ Быстро! (быстрее реального времени)")
elif transcribe_ratio < 1.5:
    print("   👍 Нормально (примерно в реальном времени)")
elif transcribe_ratio < 2.0:
    print("   🐢 Медленновато (1.5-2x реального времени)")
else:
    print("   🐌 Медленно (больше 2x реального времени)")

# ============================================================================
# Тест перевода
# ============================================================================

print()
print("=" * 70)
print("🌍 ТЕСТ ПЕРЕВОДА")
print("=" * 70)
print()

original_text = transcriber.segments_to_text(segments)
detected_lang = detect_language(original_text)

print(f"Определенный язык: {detected_lang}")
print()

if detected_lang == "ru":
    print("⚠️  Текст на русском языке, перевод не требуется")
    print("   Пропускаем тест перевода")
    translate_ratio = 0.0
else:
    print("Загрузка модели NLLB...")
    translator = Translator(method="NLLB")
    print()
    
    print("🔄 Начинаем перевод...")
    start_time = time.time()
    
    try:
        translation_segments = translator.translate_segments(
            segments, 
            source_lang=detected_lang, 
            target_lang="ru"
        )
        translate_time = time.time() - start_time
    except Exception as e:
        print(f"❌ Ошибка при переводе: {e}")
        translate_time = 0
        translate_ratio = 0.0
    else:
        translate_ratio = translate_time / duration
        
        print()
        print(f"✅ Перевод завершен!")
        print(f"   Время обработки: {translate_time:.1f} секунд")
        print(f"   📊 МНОЖИТЕЛЬ: {translate_ratio:.2f}x")
        print()
        
        if translate_ratio < 0.1:
            print("   🚀 Очень быстро!")
        elif translate_ratio < 0.3:
            print("   ✨ Быстро!")
        else:
            print("   👍 Нормально")

# ============================================================================
# Сравнение с другими моделями
# ============================================================================

print()
print("=" * 70)
print("📊 СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 70)
print()

# Типичные множители для разных моделей на CPU (M1)
typical_ratios = {
    "whisper_base": {
        "transcribe": 0.8,
        "quality": "★★★☆☆",
        "speed": "★★★★★"
    },
    "whisper_small": {
        "transcribe": 1.5,
        "quality": "★★★★☆",
        "speed": "★★★☆☆"
    }
}

print("Модель          | Качество    | Скорость    | Множитель")
print("-" * 70)
for model_name, data in typical_ratios.items():
    current = "← ВЫ ТЕСТИРУЕТЕ" if model_name == MODEL else ""
    actual = f"(у вас: {transcribe_ratio:.2f}x)" if model_name == MODEL else ""
    print(f"{model_name:15} | {data['quality']:11} | {data['speed']:11} | "
          f"{data['transcribe']:.2f}x {actual:20} {current}")

print()
print("💡 Рекомендации:")
print(f"   • whisper_base  - для быстрой обработки большого объема")
print(f"   • whisper_small - для лучшего качества (точность выше ~5%)")

# ============================================================================
# Итоговые рекомендации
# ============================================================================

print()
print("=" * 70)
print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("=" * 70)
print()

print(f"Модель: {MODEL}")
print(f"Устройство: {transcriber.device}")
print(f"Аудио: {duration:.1f} сек")
print()
print("Множители для вашего железа:")
print(f"  • Транскрибирование ({MODEL}): {transcribe_ratio:.2f}x")
if translate_ratio > 0:
    print(f"  • Перевод: {translate_ratio:.2f}x")
print()

print("=" * 70)
print("🔧 КАК ОБНОВИТЬ КОД")
print("=" * 70)
print()
print("Откройте файл: src/utils.py")
print("Найдите функцию: estimate_processing_time")
print()
print("Обновите на версию с поддержкой разных моделей:")
print()
print("```python")
print("def estimate_processing_time(")
print("    duration_seconds: float,")
print("    operation: str = \"transcribe\",")
print("    model: str = \"whisper_base\"  # ← Добавьте параметр")
print(") -> str:")
print("    if operation == \"transcribe\":")
print("        # Множители для разных моделей")
print("        multipliers = {")
print(f"            \"whisper_base\": {typical_ratios['whisper_base']['transcribe']:.2f},")
print(f"            \"whisper_small\": {typical_ratios['whisper_small']['transcribe']:.2f},")
print("        }")
print(f"        # Или используйте ваши измеренные значения:")
print(f"        # multipliers = {{")
print(f"        #     \"whisper_base\": {transcribe_ratio:.2f},  # ← Ваше значение")
print(f"        #     \"whisper_small\": X.XX,  # ← Измерьте отдельно")
print(f"        # }}")
print("        estimated = duration_seconds * multipliers.get(model, 1.0)")
print("    else:  # translate")
if translate_ratio > 0:
    print(f"        estimated = duration_seconds * {translate_ratio:.2f}")
else:
    print(f"        estimated = duration_seconds * 0.15")
print("    ")
print("    minutes = int(estimated // 60)")
print("    if estimated < 60:")
print("        return \"менее 1 минуты\"")
print("    elif minutes == 1:")
print("        return \"около 1 минуты\"")
print("    else:")
print("        return f\"около {minutes} минут\"")
print("```")
print()

# ============================================================================
# Примеры оценок
# ============================================================================

print("=" * 70)
print(f"📈 ПРИМЕРЫ ОЦЕНОК ({MODEL})")
print("=" * 70)
print()

test_durations = [
    (30, "30 секунд"),
    (60, "1 минута"),
    (180, "3 минуты"),
    (600, "10 минут"),
    (1800, "30 минут"),
    (3600, "1 час"),
    (7200, "2 часа")
]

print("Для видео длительностью:")
print()
for seconds, label in test_durations:
    trans_est = seconds * transcribe_ratio
    trans_min = int(trans_est // 60)
    trans_sec = int(trans_est % 60)
    
    if translate_ratio > 0:
        transl_est = seconds * translate_ratio
        transl_min = int(transl_est // 60)
        transl_sec = int(transl_est % 60)
        total_est = trans_est + transl_est
        total_min = int(total_est // 60)
        total_sec = int(total_est % 60)
        
        print(f"{label:>10} → транскрибирование: {trans_min}:{trans_sec:02d}, "
              f"перевод: {transl_min}:{transl_sec:02d}, "
              f"ИТОГО: {total_min}:{total_sec:02d}")
    else:
        print(f"{label:>10} → транскрибирование: {trans_min}:{trans_sec:02d}")

print()
print("=" * 70)
print("✅ Калибровка завершена!")
print("=" * 70)

# ============================================================================
# Таблица для документации
# ============================================================================

print()
print("=" * 70)
print("📋 ТАБЛИЦА ДЛЯ README")
print("=" * 70)
print()
print("Скопируйте эту таблицу в README.md:")
print()

print("| Длительность | Транскрибирование | Перевод | Итого |")
print("|--------------|-------------------|---------|-------|")

for seconds, label in test_durations:
    trans_est = seconds * transcribe_ratio
    
    # Форматирование времени
    def format_time(secs):
        m = int(secs // 60)
        s = int(secs % 60)
        if secs < 60:
            return f"~{s} сек"
        elif m < 5:
            return f"~{m}.{s//6} мин"
        else:
            return f"~{m} мин"
    
    trans_str = format_time(trans_est)
    
    if translate_ratio > 0:
        transl_est = seconds * translate_ratio
        transl_str = format_time(transl_est)
        total_est = trans_est + transl_est
        total_str = format_time(total_est)
        
        print(f"| {label:12} | {trans_str:17} | {transl_str:7} | {total_str:5} |")
    else:
        print(f"| {label:12} | {trans_str:17} | - | - |")

print()
print(f"Множители: транскрибирование={transcribe_ratio:.2f}x, перевод={translate_ratio:.2f}x")

print()
print("💡 Хотите протестировать другую модель?")
if MODEL == "whisper_base":
    print(f"   python calibrate.py whisper_small")
else:
    print(f"   python calibrate.py whisper_base")