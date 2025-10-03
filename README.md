# YouTube Transcriber & Translator

Универсальный инструмент для транскрибирования и перевода видео с YouTube, аудиофайлов и текстовых документов.

## 🎯 Возможности

### MVP (v1.0)
- ✅ Скачивание и обработка видео с YouTube
- ✅ Транскрибирование через Whisper (base модель)
- ✅ Перевод через NLLB от Meta
- ✅ Экспорт в форматы .docx и .md
- ✅ Таймкоды для каждого сегмента
- ✅ Автоопределение языка
- ✅ Логирование и прогресс-бары
- ✅ Оптимизация для Apple M1/M2

### Расширенная версия (v2.0) - В разработке
- 🔄 Обработка локальных аудиофайлов
- 🔄 Обработка текстовых файлов (docx, md)
- 🔄 Whisper Small и OpenAI API
- 🔄 Перевод через OpenAI API
- 🔄 Speaker diarization
- 🔄 Docker поддержка

## 📋 Требования

### Системные требования
- Python 3.9+
- FFmpeg (для обработки аудио)
- 8GB RAM (минимум), 16GB рекомендуется
- ~5GB свободного места на диске (для моделей)

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

### 5. Настройка переменных окружения (опционально)

Создайте файл `.env` в корне проекта:

```bash
# Для использования OpenAI API (расширенная версия)
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

#### 3. Просмотр справки

```bash
python -m src.main --help
```

### Параметры командной строки

| Параметр | Описание | Пример |
|----------|----------|--------|
| `--url` | URL видео на YouTube | `--url "https://youtube.com/..."` |
| `--input_audio` | Путь к аудиофайлу | `--input_audio audio.mp3` |
| `--input_text` | Путь к текстовому файлу | `--input_text doc.docx` |
| `--transcribe` | Метод транскрибирования | `--transcribe whisper_base` |
| `--translate` | Метод перевода | `--translate NLLB` |
| `--speakers` | Определение спикеров | `--speakers` |
| `--help` | Показать справку | `--help` |

### Доступные методы

**Транскрибирование:**
- `whisper_base` - Whisper Base (быстро, хорошее качество)
- `whisper_small` - Whisper Small (медленнее, лучше качество) - В разработке
- `whisper_openai_api` - Whisper API - В разработке

**Перевод:**
- `NLLB` - NLLB от Meta (локально, бесплатно)
- `openai_api` - OpenAI API (требует ключ) - В разработке

## 📁 Структура проекта

```
youtube-transcriber/
├── src/                      # Исходный код
│   ├── main.py              # Точка входа
│   ├── config.py            # Конфигурация
│   ├── downloader.py        # Загрузка с YouTube
│   ├── transcriber.py       # Транскрибирование
│   ├── translator.py        # Перевод
│   ├── document_writer.py   # Создание документов
│   ├── utils.py             # Утилиты
│   └── logger.py            # Логирование
├── tests/                   # Тесты
├── output/                  # Результаты обработки
├── temp/                    # Временные файлы
├── logs/                    # Логи
├── models/                  # Загруженные модели
├── requirements.txt         # Зависимости
├── .env.example            # Пример конфигурации
└── README.md               # Документация
```

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

### v1.0 (MVP) - ✅ Готово
- Базовая функциональность YouTube + Whisper + NLLB

### v1.1 - В разработке
- [ ] Обработка локальных аудиофайлов
- [ ] Обработка текстовых файлов
- [ ] Whisper Small модель
- [ ] Unit-тесты

### v2.0 - Планируется
- [ ] OpenAI API интеграция
- [ ] Speaker diarization
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
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Загрузка с YouTube

## 📞 Контакты

Если у вас есть вопросы или предложения, откройте issue в репозитории.

---

**Примечание:** Это MVP версия. Расширенный функционал будет добавлен в следующих версиях.