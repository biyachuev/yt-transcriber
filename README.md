# YouTube Transcriber & Translator

Универсальный инструмент для транскрибирования и перевода видео с YouTube, аудиофайлов и текстовых документов.

## 🎯 Возможности

### Версия 1.2 (текущая)
- ✅ **Обработка текстовых файлов** (.docx, .md, .txt)
  - Чтение готовых транскрипций
  - Улучшение текста с помощью LLM
  - Перевод существующих документов
  - Автоматическое определение языка

### Версия 1.1
- ✅ **Оптимизированные промпты для LLM-улучшения**
  - Удаление слов-паразитов (um, uh, эм, ну)
  - Конвертация чисел ("twenty eight" → "28")
  - Сохранение ВСЕХ деталей и примеров
  - Поддержка русского и английского

### Версия 1.0
- ✅ Скачивание и обработка видео с YouTube
- ✅ Обработка локальных аудиофайлов (mp3, wav и др.)
- ✅ Транскрибирование через Whisper (base, small, medium)
- ✅ Улучшение транскрипции с помощью локальных LLM через Ollama (qwen2.5, llama3, и др.)
- ✅ Автоматическое определение языка (русский/английский)
- ✅ Перевод через NLLB от Meta
- ✅ Экспорт в форматы .docx и .md
- ✅ Пользовательские промпты для Whisper (из файла)
- ✅ Автоматическое создание промптов из метаданных YouTube
- ✅ Логирование и прогресс-бары
- ✅ Оптимизация для Apple M1/M2

### В разработке
- 🔄 Whisper через OpenAI API
- 🔄 Перевод через OpenAI API
- 🔄 Speaker diarization
- 🔄 Docker поддержка

## 📋 Требования

### Системные требования
- Python 3.9+
- FFmpeg (для обработки аудио)
- Ollama (опционально, для улучшения транскрипций)
- 8GB RAM (минимум), 16GB рекомендуется
- ~5GB свободного места на диске (для моделей Whisper и NLLB)
- +3-7GB для моделей Ollama (если используется улучшение транскрипций)

### Поддерживаемые платформы
- macOS (включая Apple Silicon M1/M2)
- Linux
- Windows

## 🚀 Установка

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd youtube-transcriber
```

### 2. Создание виртуального окружения

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Установка FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Скачайте с [ffmpeg.org](https://ffmpeg.org/download.html) и добавьте в PATH

### 4. Установка зависимостей Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Установка Ollama (опционально, для улучшения транскрипций)

**macOS/Linux:**
```bash
# Установка Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Загрузка рекомендуемых моделей
ollama pull qwen2.5:3b    # Быстрая, хорошее качество (3GB)
ollama pull qwen2.5:7b    # Медленнее, лучше качество (7GB)

# Запуск сервера (если не запущен автоматически)
ollama serve
```

**Windows:**
Скачайте установщик с [ollama.com](https://ollama.com/download)

### 6. Настройка переменных окружения (опционально)

Создайте файл `.env` в корне проекта:

```bash
# Для использования OpenAI API (в разработке)
OPENAI_API_KEY=your_api_key_here

# Уровень логирования
LOG_LEVEL=INFO
```

## 📖 Использование

### Базовые примеры

#### 1. Транскрибирование YouTube видео

```bash
python -m src.main --url "https://youtube.com/watch?v=dQw4w9WgXcQ" --transcribe whisper_base
```

#### 2. Транскрибирование и перевод

```bash
python -m src.main \
    --url "https://youtube.com/watch?v=dQw4w9WgXcQ" \
    --transcribe whisper_base \
    --translate NLLB
```

#### 3. Обработка локального аудиофайла

```bash
python -m src.main \
    --input_audio audio.mp3 \
    --transcribe whisper_medium \
    --translate NLLB
```

#### 4. Улучшение транскрипции с помощью LLM

```bash
python -m src.main \
    --input_audio audio.mp3 \
    --transcribe whisper_medium \
    --refine-model qwen2.5:7b \
    --translate NLLB
```

Это создаст два документа:
- `audio (original).docx/md` - оригинальная транскрипция без перевода
- `audio (refined).docx/md` - улучшенная транскрипция с переводом

#### 5. Использование пользовательского промпта для Whisper

```bash
# Создайте файл prompt.txt с ключевыми словами:
# FIDE, Hikaru Nakamura, Magnus Carlsen, chess tournament

python -m src.main \
    --url "https://youtube.com/watch?v=..." \
    --transcribe whisper_base \
    --prompt prompt.txt
```



#### 6. Обработка текстовых файлов (v1.2)

```bash
# Улучшение существующей транскрипции
python -m src.main \
    --input_text output/document.md \
    --refine-model qwen2.5:7b

# Перевод документа
python -m src.main \
    --input_text transcription.docx \
    --translate NLLB

# Улучшение и перевод
python -m src.main \
    --input_text document.txt \
    --refine-model qwen2.5:7b \
    --translate NLLB
```

Поддерживаемые форматы: `.md`, `.docx`, `.txt`


#### 7. Просмотр справки

```bash
python -m src.main --help
```

### Параметры командной строки

| Параметр | Описание | Пример |
|----------|----------|--------|
| `--url` | URL видео на YouTube | `--url "https://youtube.com/..."` |
| `--input_audio` | Путь к аудиофайлу (mp3, wav и др.) | `--input_audio audio.mp3` |
| `--input_text` | Путь к текстовому файлу (.docx, .md, .txt) | `--input_text doc.docx` |
| `--transcribe` | Метод транскрибирования | `--transcribe whisper_medium` |
| `--translate` | Метод перевода | `--translate NLLB` |
| `--refine-model` | Модель Ollama для улучшения | `--refine-model qwen2.5:7b` |
| `--prompt` | Файл с промптом для Whisper | `--prompt prompt.txt` |
| `--speakers` | Определение спикеров (в разработке) | `--speakers` |
| `--help` | Показать справку | `--help` |

### Доступные методы

**Транскрибирование:**
- `whisper_base` - Whisper Base (быстро, хорошее качество)
- `whisper_small` - Whisper Small (медленнее, лучше)
- `whisper_medium` - Whisper Medium (медленно, высокое качество)
- `whisper_openai_api` - Whisper API (в разработке)

**Улучшение транскрипций (требует Ollama):**
- `qwen2.5:3b` - Быстрая модель, 3GB (рекомендуется)
- `qwen2.5:7b` - Лучшее качество, 7GB
- `llama3.2:3b` - Быстрая, хорошее качество
- `llama3:8b` - Медленнее, но качественнее
- `mistral:7b` - Хороший баланс
- Любая другая модель из [библиотеки Ollama](https://ollama.com/library)

**Перевод:**
- `NLLB` - NLLB от Meta (локально, бесплатно)
- `openai_api` - OpenAI API (в разработке)

## 📁 Структура проекта

```
youtube-transcriber/
├── src/                      # Исходный код
│   ├── main.py              # Точка входа
│   ├── config.py            # Конфигурация
│   ├── downloader.py        # Загрузка с YouTube
│   ├── transcriber.py       # Транскрибирование
│   ├── text_reader.py        # Чтение текстовых файлов

│   ├── translator.py        # Перевод
│   ├── text_refiner.py      # Улучшение транскрипций через LLM
│   ├── document_writer.py   # Создание документов
│   ├── utils.py             # Утилиты
│   └── logger.py            # Логирование
├── tests/                   # Тесты
├── output/                  # Результаты обработки
├── temp/                    # Временные файлы
├── logs/                    # Логи
├── requirements.txt         # Зависимости
├── .env.example            # Пример конфигурации
└── README.md               # Документация
```

**Примечание:** Модели Whisper и NLLB скачиваются автоматически в `~/.cache/` при первом использовании.

## 🔧 Конфигурация

Настройки находятся в `src/config.py`. Основные параметры:

```python
# Пути
OUTPUT_DIR = "output"        # Папка для результатов
TEMP_DIR = "temp"           # Временные файлы
LOGS_DIR = "logs"           # Логи

# Модели
WHISPER_DEVICE = "mps"      # cpu/cuda/mps (авто для M1)
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Логирование
LOG_LEVEL = "INFO"          # DEBUG/INFO/WARNING/ERROR
```

## 📊 Производительность

### Примерное время обработки (MacBook Air M1, 16GB, CPU)

| Длительность видео | Транскрибирование<br>(whisper_base) | Транскрибирование<br>(whisper_small) | Перевод<br>(NLLB) | Итого<br>(base + перевод) | Итого<br>(small + перевод) |
|-------------------|-------------------------------------|--------------------------------------|-------------------|---------------------------|----------------------------|
| 3 минуты | ~11 сек | ~34 сек | ~1.5 мин | ~2 мин | ~3 мин |
| 10 минут | ~36 сек | ~2 мин | ~5 мин | ~5.5 мин | ~7 мин |
| 30 минут | ~1.8 мин | ~5.7 мин | ~14 мин | ~16 мин | ~20 мин |
| 1 час | ~3.6 мин | ~11 мин | ~28 мин | ~32 мин | ~39 мин |
| 2 часа | ~7 мин | ~23 мин | ~56 мин | ~63 мин | ~79 мин |

**Множители обработки:**
- Whisper Base: 0.06x (в 16 раз быстрее реального времени!) 🚀
- Whisper Small: 0.19x (в 5 раз быстрее реального времени)
- NLLB: 0.47x (в 2 раза быстрее реального времени)


## 🐛 Решение проблем

### Ошибки установки

**Проблема:** `torch` не устанавливается на M1 Mac
```bash
# Решение: используйте версию для Apple Silicon
pip install --upgrade torch torchvision torchaudio
```

**Проблема:** FFmpeg не найден
```bash
# Проверьте установку
ffmpeg -version

# Если не установлен, установите через brew (macOS)
brew install ffmpeg
```

**Проблема:** Недостаточно памяти
```bash
# Используйте меньшую модель Whisper
python -m src.main --url "..." --transcribe whisper_base  # вместо whisper_small
```

### Ошибки выполнения

**Проблема:** "Model not found"
- Модели загружаются автоматически при первом запуске
- Убедитесь, что есть подключение к интернету
- Проверьте, что папка `models/` доступна для записи

**Проблема:** Медленная работа
- Используйте `whisper_base` вместо `whisper_small`
- Убедитесь, что используется GPU/MPS (проверьте логи)
- Закройте другие ресурсоемкие приложения

## 🧪 Тестирование

```bash
# Установка зависимостей для разработки
pip install -r requirements-dev.txt

# Запуск тестов
pytest tests/

# С покрытием кода
pytest --cov=src tests/
```

## 📝 Примеры вывода

### Формат .docx
```
# Название видео

## Перевод
Метод: NLLB

[00:15] Привет всем! Сегодня мы поговорим о...

[01:32] Первая важная тема - это...

## Расшифровка
Метод: whisper_base

[00:15] Hello everyone! Today we'll talk about...

[01:32] The first important topic is...
```

### Формат .md
Аналогичный формат с markdown разметкой

## 🛣️ Roadmap

### v1.0 - ✅ Готово
- ✅ YouTube + локальные аудиофайлы
- ✅ Whisper (base, small, medium)
- ✅ Улучшение транскрипций через Ollama
- ✅ NLLB перевод
- ✅ Пользовательские промпты
- ✅ Автоопределение языка

### v2.0 - Планируется
- [ ] Обработка текстовых файлов (docx, md)
- [ ] OpenAI API интеграция
- [ ] Speaker diarization
- [ ] Unit-тесты и CI/CD
- [ ] Docker образ
- [ ] Web интерфейс
- [ ] Batch обработка

## 🤝 Вклад в проект

Приветствуются pull requests! Для больших изменений сначала откройте issue для обсуждения.

### Процесс разработки

1. Форкните репозиторий
2. Создайте ветку (`git checkout -b feature/amazing-feature`)
3. Закоммитьте изменения (`git commit -m 'Add amazing feature'`)
4. Запушьте ветку (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## 📄 Лицензия

MIT License - см. файл LICENSE

## 🙏 Благодарности

- [OpenAI Whisper](https://github.com/openai/whisper) - Транскрибирование
- [Meta NLLB](https://github.com/facebookresearch/fairseq/tree/nllb) - Перевод
- [Ollama](https://ollama.com) - Локальные LLM для улучшения транскрипций
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Загрузка с YouTube

## 📞 Контакты

Если у вас есть вопросы или предложения, откройте issue в репозитории.

---

## 💡 Советы по использованию

### Для лучшего качества транскрипции:
1. Используйте `whisper_medium` для важных видео
2. Создавайте промпт-файлы с ключевыми терминами и именами
3. Для YouTube промпт создаётся автом��тически из метаданных

### Для улучшения транскрипций:
1. Установите Ollama и модель `qwen2.5:7b` для лучшего качества
2. Модель автоматически определит язык (русский/английский)
3. Используйте `--refine-model` для получения чистой транскрипции
4. **Новое в v1.1:** LLM использует оптимизированные промпты:
   - Удаляет слова-паразиты ("um", "uh", "эм", "ну", "вот")
   - Убирает метакомментарии ("let me scroll", "сейчас открою экран")
   - Конвертирует числа: "twenty eight sixteen" → "2816", "ноль восемь" → "0.8"
   - **Сохраняет ВСЕ детали**: примеры, факты, рассуждения
   - Не суммаризирует - только очищает от шума
   - Исправляет пунктуацию и структурирует текст

### Оптимизация скорости:
- `whisper_base` - для быстрой обработки больших объёмов
- `whisper_medium` - для важных материалов
- `qwen2.5:3b` - быстрое улучшение
- `qwen2.5:7b` - качественное улучшение

### Кэш моделей:
- Whisper: `~/.cache/whisper/` (~140MB-1.5GB)
- Ollama: управляется через `ollama list` и `ollama rm <model>`