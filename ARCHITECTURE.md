# Архитектура проекта

Детальное описание архитектурных решений, технического стека и дизайна системы.

## 🏗️ Общая архитектура

### Компонентная диаграмма

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Interface                        │
│                         (main.py)                           │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Downloader  │  │ Transcriber  │  │  Translator  │
│              │  │              │  │              │
│  - yt-dlp    │  │  - Whisper   │  │  - NLLB      │
│  - FFmpeg    │  │  - PyTorch   │  │  - HuggingF. │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       └─────────────────┼──────────────────┘
                         │
                         ▼
                 ┌──────────────┐
                 │ Text Refiner │
                 │              │
                 │  - Ollama    │
                 │  - LLM API   │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │Document      │
                 │Writer        │
                 │              │
                 │- python-docx │
                 │- markdown    │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │  Output      │
                 │  (.docx, .md)│
                 └──────────────┘
```

---

## 📦 Модульная структура

### 1. **config.py** - Конфигурация

**Назначение:** Централизованное управление настройками

**Ключевые компоненты:**
```python
class Settings(BaseSettings):
    # Пути к директориям
    OUTPUT_DIR, TEMP_DIR, LOGS_DIR
    
    # Модели
    WHISPER_MODEL_DIR, NLLB_MODEL_DIR
    
    # API ключи
    OPENAI_API_KEY
    
    # Устройство
    WHISPER_DEVICE (auto-detect)
    
    # Поддерживаемые языки
    SUPPORTED_LANGUAGES: list[str] = ["ru", "en"]
```

**Технические решения:**
- Использование `pydantic-settings` для валидации
- Автоопределение устройства (CPU/CUDA/MPS)
- Поддержка `.env` файлов
- Автосоздание необходимых директорий
- Оптимизация для Apple Silicon (M1/M2)

---

### 2. **downloader.py** - Загрузка с YouTube

**Назначение:** Скачивание аудио с YouTube

**Зависимости:**
- `yt-dlp` - загрузка видео
- `FFmpeg` - конвертация в MP3

**Процесс:**
```
URL → yt-dlp → Video → FFmpeg → MP3 → temp/
                ↓
            Metadata (title, duration)
```

**Особенности:**
- Прогресс-бар для скачивания
- Извлечение только аудио (экономия трафика)
- Очистка имен файлов
- Обработка ошибок сети

---

### 3. **transcriber.py** - Транскрибирование

**Назначение:** Преобразование аудио в текст

**Модели:** OpenAI Whisper
- Base: 74M параметров, быстро
- Small: 244M параметров, лучше качество
- Medium: 769M параметров, высокое качество

**Процесс:**
```
MP3 → Whisper Model → Segments
                         ↓
                    [timestamp, text]
```

**Класс TranscriptionSegment:**
```python
class TranscriptionSegment:
    start: float      # Начало в секундах
    end: float        # Конец в секундах
    text: str         # Текст сегмента
    speaker: str      # Спикер (опционально)
```

**Новые возможности:**
- Поддержка пользовательских промптов
- Автогенерация промптов из метаданных YouTube
- Метод `update_segments_from_text()` для LLM улучшения

**Оптимизации:**
- Использование CPU на M1/M2 (MPS имеет проблемы с Whisper)
- Batch обработка сегментов
- FP16 для GPU (экономия памяти)
- Ленивая загрузка модели

---

### 4. **translator.py** - Перевод

**Назначение:** Перевод текста

**Модель:** Meta NLLB-200
- Модель: `facebook/nllb-200-distilled-600M`
- 200+ языков
- 600M параметров (distilled версия)

**Процесс:**
```
Text → Chunking → NLLB → Translation
         ↓
    [chunk1, chunk2, ...]
```

**Chunking стратегия:**
- Разбивка по параграфам
- Максимум 500 токенов на чанк
- Сохранение контекста

**Особенности:**
- Автоопределение языка источника
- Пропуск перевода если язык совпадает
- Обработка длинных текстов (до 100+ страниц)

---

### 5. **text_refiner.py** - Улучшение текста

**Назначение:** Улучшение транскрипции с помощью локальных LLM

**Модели:** Ollama (локальные LLM)
- qwen2.5:3b - быстрая, хорошее качество
- llama3.2:3b - быстрая, хорошее качество
- llama3:8b - медленнее, высокое качество
- mistral:7b - хороший баланс

**Процесс:**
```
Text → Chunking → LLM → Refined Text
         ↓
    [chunk1, chunk2, ...]
```

**Класс TextRefiner:**
```python
class TextRefiner:
    def __init__(self, model_name: str, ollama_url: str)
    def refine_text(self, text: str, context: str) -> str
    def refine_chunk(self, chunk: str, context: str) -> str
```

**Возможности:**
- Автоопределение языка текста
- Определение тематики
- Разбивка на чанки по предложениям
- Исправление пунктуации
- Удаление слов-паразитов
- Улучшение структуры текста

**Особенности:**
- Работает через HTTP API Ollama
- Поддержка русского и английского
- Обработка ошибок с fallback на оригинал
- Настраиваемые параметры модели

---

### 6. **document_writer.py** - Создание документов

**Назначение:** Экспорт результатов

**Форматы:**
1. **DOCX** - через `python-docx`
   - Форматирование (шрифты, размеры)
   - Стили для секций
   - Кириллица поддерживается
   
2. **Markdown** - текстовый формат
   - Совместимость с Obsidian, Notion
   - Легко редактировать
   - Git-friendly

**Структура документа:**
```markdown
# Название видео

## Перевод
*Метод: NLLB*

[00:15] Переведенный текст...
[01:30] Продолжение...

## Расшифровка
*Метод: whisper_base*

[00:15] Original text...
[01:30] Continuation...
```

---

### 7. **utils.py** - Утилиты

**Функции:**

1. `sanitize_filename()` - Очистка имен файлов
   - Удаление недопустимых символов
   - Сохранение кириллицы
   - Ограничение длины

2. `format_timestamp()` - Форматирование времени
   - MM:SS для коротких видео
   - HH:MM:SS для длинных

3. `detect_language()` - Определение языка
   - Подсчет кириллических/латинских символов
   - Простой но эффективный метод

4. `chunk_text()` - Разбивка текста
   - По параграфам
   - Лимит токенов
   - Сохранение структуры

5. `create_whisper_prompt()` - Создание промптов
   - Из метаданных YouTube
   - Ограничение длины (800 символов)
   - Извлечение ключевых слов

---

### 8. **logger.py** - Логирование

**Уровни логирования:**
- DEBUG - детальная отладка
- INFO - основные события (по умолчанию)
- WARNING - предупреждения
- ERROR - ошибки

**Выводы:**
- Консоль: INFO и выше
- Файл: все уровни
- Формат: `timestamp - module - level - message`

---

## 🔧 Технические решения

### Выбор моделей Whisper

**Whisper Base (основная):**
1. Баланс скорость/качество
2. Работает на CPU
3. Умеренные требования к RAM (~2GB)
4. 90%+ точность на чистом аудио

**Whisper Small:**
- Лучше качество, медленнее
- Требует больше RAM (~3GB)
- Рекомендуется для важных материалов

**Whisper Medium:**
- Высокое качество, медленно
- Требует много RAM (~5GB)
- Рекомендуется с LLM улучшением

**Альтернативы:**
- Tiny: быстрее, но хуже качество
- Large: требует очень много ресурсов

---

### Выбор NLLB для перевода

**Преимущества:**
1. ✅ Работает локально (приватность)
2. ✅ Бесплатно
3. ✅ 200+ языков (расширяемость)
4. ✅ Хорошее качество

**Альтернативы:**
- OpenAI API: лучше качество, но платно (v2.0)
- Google Translate API: платно, требует ключ
- MarianMT: меньше языков

---

### Выбор Ollama для LLM улучшения

**Преимущества:**
1. ✅ Работает локально (приватность)
2. ✅ Бесплатно
3. ✅ Поддержка множества моделей
4. ✅ Простая установка и настройка
5. ✅ HTTP API для интеграции

**Рекомендуемые модели:**
- qwen2.5:3b - быстрая, отлично для русского и английского
- llama3.2:3b - быстрая, хорошее качество
- llama3:8b - медленнее, но качественнее

**Альтернативы:**
- OpenAI API: лучше качество, но платно (v2.0)
- HuggingFace Transformers: сложнее настройка
- Локальные модели без Ollama: требует больше кода

---

### Архитектура хранения данных

```
youtube-transcriber/
├── output/           # Финальные результаты
│   ├── *.docx       # Документы Word
│   ├── *.md         # Markdown файлы
│   ├── * (original).* # Оригинальные версии
│   └── * (refined).*  # Улучшенные версии
│
├── temp/            # Временные файлы (НЕ удаляются)
│   ├── *.mp3        # Скачанное аудио
│   └── *.json       # Промежуточные данные
│
├── models/          # Кешированные модели
│   ├── whisper/     # Whisper модели
│   └── nllb/        # NLLB модели
│
├── logs/            # Логи работы
│   └── app_*.log    # По файлу на запуск
│
└── prompts/         # Пользовательские промпты
    └── *.txt        # Файлы с промптами
```

**Решение:** Не удалять промежуточные файлы
- Полезно для отладки
- Можно переиспользовать аудио
- Пользователь контролирует очистку

---

### Управление памятью

**Проблема:** Модели требуют много RAM

**Решения:**

1. **Ленивая загрузка:**
```python
def _load_model(self):
    if self.model is None:
        self.model = whisper.load_model(...)
```

2. **Chunking для перевода:**
```python
chunks = chunk_text(text, max_tokens=500)
for chunk in chunks:
    translate(chunk)  # Обрабатываем по частям
```

3. **Chunking для LLM улучшения:**
```python
chunks = self._split_text_into_chunks(text, max_chunk_size=2000)
for chunk in chunks:
    refined_chunk = self.refine_chunk(chunk)
```

4. **Освобождение памяти:**
```python
del model
torch.cuda.empty_cache()  # Для GPU
```

5. **Оптимизация для Apple Silicon:**
```python
# Используем CPU вместо MPS для Whisper
if os.uname().machine == 'arm64':
    self.WHISPER_DEVICE = "cpu"
```

---

### Обработка ошибок

**Стратегия:** Fail-fast с информативными сообщениями

**Уровни обработки:**

1. **Валидация входных данных**
```python
if not validate_args(args):
    logger.error("...")
    sys.exit(1)
```

2. **Try-catch в критических местах**
```python
try:
    download_audio(url)
except Exception as e:
    logger.error(f"Ошибка: {e}", exc_info=True)
    sys.exit(1)
```

3. **Graceful degradation**
```python
if with_speakers:
    logger.warning("Speaker diarization будет в v2.0")
    # Продолжаем без speakers

# LLM улучшение с fallback
try:
    refined_text = refiner.refine_text(original_text)
except Exception as e:
    logger.error(f"Ошибка при улучшении: {e}")
    logger.warning("Продолжаем с оригинальной транскрипцией")
    refined_text = original_text
```

---

## 🚀 Оптимизации производительности

### 1. Автоопределение устройства

```python
def _get_device(self):
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"
```

### 2. FP16 для GPU

```python
transcribe_options = {
    'fp16': False if device == "cpu" else True
}
```

**Преимущества:**
- 2x меньше памяти
- 1.5x быстрее
- Минимальная потеря точности

### 3. Batch обработка

```python
# Обрабатываем сегменты батчами
for chunk in tqdm(chunks, desc="Перевод"):
    translate_batch(chunk)

# LLM улучшение с прогресс-баром
for chunk in tqdm(chunks, desc="Улучшение текста"):
    refined_chunk = refiner.refine_chunk(chunk)
```

### 4. Оптимизация для Apple Silicon

```python
# Автоопределение архитектуры
if os.uname().machine == 'arm64':
    # Используем CPU для Whisper (MPS имеет проблемы)
    WHISPER_DEVICE = "cpu"
    # Ollama автоматически использует Neural Engine
```

---

## 🔐 Безопасность и приватность

### Приватность данных

✅ **Что НЕ отправляется:**
- Аудио файлы
- Текст расшифровки
- Переводы
- Любые пользовательские данные
- Промпты и контекст

✅ **Что отправляется:**
- YouTube URL (для скачивания)
- Запросы на скачивание моделей (один раз)
- HTTP запросы к Ollama (локально, если настроен)

### Безопасность кода

1. **Валидация путей:**
```python
def sanitize_filename(filename):
    # Удаляем опасные символы
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    return re.sub(invalid_chars, '_', filename)
```

2. **Изоляция временных файлов:**
```python
TEMP_DIR = BASE_DIR / "temp"  # Контролируемая папка
```

3. **Переменные окружения для ключей:**
```python
OPENAI_API_KEY = Field(default="", env="OPENAI_API_KEY")
```

---

## 📈 Масштабируемость

### Текущие ограничения

| Параметр | Ограничение |
|----------|-------------|
| Длина видео | Нет ограничений* |
| Размер файла | ~5GB аудио |
| Параллельность | 1 видео |
| Языки | RU/EN |
| LLM модели | Зависит от RAM |
| Промпт длина | 800 символов |

*При достаточной RAM и времени

### Планы расширения (v2.0)

1. **Параллельная обработка:**
```python
from multiprocessing import Pool

with Pool(processes=2) as pool:
    pool.map(process_video, urls)
```

2. **Streaming обработка:**
```python
# Обработка по мере скачивания
for chunk in download_stream(url):
    transcribe(chunk)
```

3. **Распределенная обработка:**
- Celery для очередей
- Redis для состояния
- Docker для изоляции

4. **Улучшенная LLM интеграция:**
- Поддержка больше моделей
- Batch обработка для LLM
- Кеширование результатов

---

## 🧪 Тестирование

### Структура тестов

```
tests/
├── test_utils.py         # Unit тесты утилит
├── test_transcriber.py   # Unit тесты транскрайбера
├── test_translator.py    # Unit тесты переводчика
├── test_text_refiner.py  # Unit тесты улучшения текста
└── test_integration.py   # Интеграционные тесты
```

### Стратегия тестирования

1. **Unit тесты:** Отдельные функции
2. **Integration тесты:** Полный pipeline (skip по умолчанию)
3. **Mocking:** Внешние зависимости

```python
@pytest.mark.integration
@pytest.mark.skip(reason="Requires test audio")
def test_full_pipeline():
    # Тест полного процесса
    pass
```

---

## 🐳 Docker архитектура

### Dockerfile структура

```dockerfile
FROM python:3.11-slim

# Системные зависимости
RUN apt-get update && apt-get install -y ffmpeg

# Python зависимости
COPY requirements.txt .
RUN pip install -r requirements.txt

# Код приложения
COPY src/ ./src/

# Volumes для данных
VOLUME ["/app/output", "/app/models"]
```

### Docker-compose для удобства

```yaml
services:
  transcriber:
    build: .
    volumes:
      - ./output:/app/output  # Результаты
      - ./models:/app/models  # Кеш моделей
```

**Преимущества:**
- Изолированное окружение
- Одинаковое поведение на всех ОС
- Простое развертывание

---

## 📊 Метрики и мониторинг

### Логируемые события

1. **Начало/конец операций:**
```python
logger.info("Начало транскрибирования")
logger.info("Транскрибирование завершено за X мин")
```

2. **Прогресс:**
```python
tqdm(total=100, desc="Обработка")
```

3. **Ошибки:**
```python
logger.error(f"Ошибка: {e}", exc_info=True)
```

### Будущее: Prometheus метрики (v2.0)

```python
# Пример метрик
videos_processed_total = Counter(...)
processing_duration_seconds = Histogram(...)
model_memory_usage = Gauge(...)
```

---

## 🔄 CI/CD (Планируется)

### GitHub Actions workflow

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      - name: Run tests
        run: pytest tests/
      - name: Check code style
        run: black --check src/
```

---

## 📚 Документация архитектуры

Этот документ описывает **текущую архитектуру (MVP v1.0)**.

Для будущих версий см.:
- [Roadmap в README](README.md#roadmap)
- [GitHub Issues](https://github.com/yourusername/youtube-transcriber/issues)

---

**Последнее обновление:** Январь 2025