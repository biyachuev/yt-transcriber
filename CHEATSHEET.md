# 🚀 Шпаргалка по YouTube Transcriber

Быстрый справочник по всем командам и функциям.

## ⚡ Установка (3 команды)

```bash
# 1. Создать venv и активировать
python -m venv venv && source venv/bin/activate

# 2. Установить FFmpeg (macOS)
brew install ffmpeg

# 3. Установить зависимости
pip install -r requirements.txt
```

---

## 📝 Основные команды

### Показать справку
```bash
python -m src.main --help
```

### Транскрибирование YouTube
```bash
python -m src.main --url "YOUTUBE_URL" --transcribe whisper_base
```

### Транскрибирование + перевод
```bash
python -m src.main --url "YOUTUBE_URL" --transcribe whisper_base --translate NLLB
```

### Обработка аудиофайла
```bash
python -m src.main --input_audio file.mp3 --transcribe whisper_base
```

### Обработка с улучшением LLM
```bash
python -m src.main --input_audio file.mp3 --transcribe whisper_medium --refine-model qwen2.5:3b
```

### Обработка с пользовательским промптом
```bash
python -m src.main --url "..." --transcribe whisper_base --prompt prompt.txt
```

---

## 🔧 Параметры командной строки

| Параметр | Описание | Пример |
|----------|----------|--------|
| `--url` | YouTube URL | `--url "https://youtube.com/..."` |
| `--input_audio` | Аудиофайл | `--input_audio audio.mp3` |
| `--input_text` | Текстовый файл | `--input_text doc.docx` |
| `--transcribe` | Метод транскрибирования | `--transcribe whisper_base` |
| `--translate` | Метод перевода | `--translate NLLB` |
| `--prompt` | Промпт для Whisper | `--prompt prompt.txt` |
| `--refine-model` | Модель Ollama | `--refine-model qwen2.5:3b` |
| `--speakers` | Определение спикеров (v2.0) | `--speakers` |
| `--help` | Справка | `--help` |

---

## 🎯 Методы обработки

### Транскрибирование
- `whisper_base` - быстро, хорошо
- `whisper_small` - медленнее, лучше
- `whisper_medium` - медленно, высокое качество
- `whisper_openai_api` - через API (v2.0)

### Перевод
- `NLLB` - локально, бесплатно
- `openai_api` - через API (v2.0)

### LLM улучшение
- `qwen2.5:3b` - быстрая, хорошее качество
- `llama3.2:3b` - быстрая, хорошее качество
- `llama3:8b` - медленнее, но качественнее
- `mistral:7b` - хороший баланс

---

## 📁 Структура проекта

```
youtube-transcriber/
├── src/              # Исходный код
├── tests/            # Тесты
├── output/           # Результаты ← ЗДЕСЬ!
│   ├── * (original).* # Оригинальные версии
│   └── * (refined).*  # Улучшенные версии
├── temp/             # Временные файлы
├── logs/             # Логи
├── models/           # Модели ML
└── prompts/          # Пользовательские промпты
```

---

## 🐍 Python API

### Импорты
```python
from src.downloader import YouTubeDownloader
from src.transcriber import Transcriber
from src.translator import Translator
from src.text_refiner import TextRefiner
from src.document_writer import DocumentWriter
```

### Скачивание
```python
downloader = YouTubeDownloader()
audio_path, title, duration = downloader.download_audio(url)
```

### Транскрибирование
```python
transcriber = Transcriber(method="whisper_base")
segments = transcriber.transcribe(audio_path)
text = transcriber.segments_to_text(segments)
```

### Перевод
```python
translator = Translator(method="NLLB")
translated = translator.translate_text(text, source_lang="en", target_lang="ru")
```

### Улучшение текста
```python
refiner = TextRefiner(model_name="qwen2.5:3b")
refined_text = refiner.refine_text(original_text, context="prompt")
```

### Создание документов
```python
writer = DocumentWriter()
docx_path, md_path = writer.create_from_segments(
    title="My Video",
    transcription_segments=segments,
    translation_segments=translated_segments
)
```

---

## 🔍 Отладка

### Включить DEBUG логи
```bash
# В .env
LOG_LEVEL=DEBUG
```

### Посмотреть логи
```bash
tail -f logs/app_*.log
```

### Запустить тесты
```bash
# Все тесты
pytest tests/

# С выводом
pytest -v tests/

# С покрытием
pytest --cov=src tests/
```

---

## 🛠️ Полезные команды

### Очистка временных файлов
```bash
rm -rf temp/*
rm logs/app_*.log
```

### Проверка размера моделей
```bash
du -sh models/*
```

### Обновление зависимостей
```bash
pip install --upgrade -r requirements.txt
```

### Проверка FFmpeg
```bash
ffmpeg -version
```

---

## 🐳 Docker команды

### Билд
```bash
docker build -t youtube-transcriber .
```

### Запуск
```bash
docker run -v $(pwd)/output:/app/output \
  youtube-transcriber \
  --url "YOUTUBE_URL" \
  --transcribe whisper_base
```

### Docker Compose
```bash
# Запуск
docker-compose up

# В фоне
docker-compose up -d

# Остановка
docker-compose down
```

---

## ⚙️ Переменные окружения (.env)

```bash
# API ключи
OPENAI_API_KEY=sk-...

# Логирование
LOG_LEVEL=INFO

# Устройство
WHISPER_DEVICE=mps  # cpu, cuda, mps
```

---

## 📊 Время обработки (M1, 16GB)

**Базовая обработка (whisper_base + NLLB):**
| Видео | Обработка |
|-------|-----------|
| 10 мин | ~18 мин |
| 30 мин | ~55 мин |
| 1 час | ~110 мин |
| 2 часа | ~220 мин |

**С улучшением LLM (whisper_medium + qwen2.5:3b + NLLB):**
| Видео | Обработка |
|-------|-----------|
| 10 мин | ~53 мин |
| 30 мин | ~160 мин |
| 1 час | ~320 мин |

---

## 🎨 Форматы вывода

### DOCX
- Форматированный документ Word
- Стили и шрифты
- Кириллица

### Markdown
- Простой текст
- Git-friendly
- Совместим с Obsidian/Notion

### Структура файлов
- `название.docx/md` - оригинальная версия
- `название (original).docx/md` - оригинальная версия (с LLM)
- `название (refined).docx/md` - улучшенная версия

---

## 🚨 Частые ошибки

### FFmpeg not found
```bash
# Установите FFmpeg
brew install ffmpeg  # macOS
sudo apt install ffmpeg  # Linux
```

### Out of memory
```bash
# Используйте меньшую модель
--transcribe whisper_base
```

### Model download timeout
```bash
# Проверьте интернет и попробуйте снова
# Модели кешируются в models/
```

### Ollama connection error
```bash
# Убедитесь что Ollama запущена
ollama serve

# Проверьте доступность
curl http://localhost:11434/api/tags
```

---

## 🔗 Быстрые ссылки

### Документация
- `README.md` - основная документация
- `QUICKSTART.md` - быстрый старт
- `EXAMPLES.md` - примеры
- `FAQ.md` - вопросы и ответы

### Разработка
- `ARCHITECTURE.md` - архитектура
- `CURSOR_DEVELOPMENT_GUIDE.md` - гайд для разработки
- `PROJECT_SUMMARY.md` - общая сводка

---

## 💻 Git команды

### Первый коммит
```bash
git init
git add .
git commit -m "Initial commit: MVP v1.0"
```

### Создание ветки для фичи
```bash
git checkout -b feature/v1.1-audio-files
```

### Коммит изменений
```bash
git add .
git commit -m "Add audio file support"
```

---

## 🧪 Тестирование

### Запуск всех тестов
```bash
pytest tests/
```

### Конкретный тест
```bash
pytest tests/test_utils.py::test_sanitize_filename
```

### С покрытием кода
```bash
pytest --cov=src --cov-report=html tests/
```

### Пропустить медленные тесты
```bash
pytest -m "not integration" tests/
```

---

## 📦 Установка как пакет

```bash
# Для разработки
pip install -e .

# Использование
youtube-transcriber --url "..." --transcribe whisper_base
```

---

## 🔥 Быстрые рецепты

### Обработать плейлист
```bash
yt-dlp --flat-playlist --print url "PLAYLIST_URL" | while read url; do
  python -m src.main --url "$url" --transcribe whisper_base --translate NLLB
done
```

### Batch обработка
```bash
for url in url1 url2 url3; do
  python -m src.main --url "$url" --transcribe whisper_base
done
```

### С уведомлением (macOS)
```bash
python -m src.main --url "..." --transcribe whisper_base && \
  osascript -e 'display notification "Готово!" with title "Transcriber"'
```

---

## 🎯 Примеры сценариев

### Студент: Лекция на английском
```bash
python -m src.main \
  --url "LECTURE_URL" \
  --transcribe whisper_medium \
  --translate NLLB \
  --refine-model qwen2.5:3b
```

### Разработчик: Техническое интервью
```bash
python -m src.main \
  --url "INTERVIEW_URL" \
  --transcribe whisper_medium \
  --translate NLLB \
  --refine-model qwen2.5:3b \
  --prompt tech_prompt.txt
```

### Исследователь: Конспект нескольких видео
```bash
./batch_process.sh urls.txt
```

---

## 📌 Горячие клавиши (в разработке)

### В Cursor IDE
- `F5` - Запустить с отладкой
- `F9` - Поставить breakpoint
- `Ctrl+Shift+P` - Командная палитра
- `Ctrl+`` - Открыть терминал

---

## 🎓 Полезные алиасы

Добавьте в `~/.bashrc` или `~/.zshrc`:

```bash
# Быстрая транскрипция
alias yt-trans='python -m src.main --transcribe whisper_base'

# Транскрипция + перевод
alias yt-full='python -m src.main --transcribe whisper_base --translate NLLB'

# Высокое качество с улучшением
alias yt-premium='python -m src.main --transcribe whisper_medium --translate NLLB --refine-model qwen2.5:3b'

# Очистка temp
alias yt-clean='rm -rf temp/* logs/*.log'

# Использование:
# yt-trans --url "..."
# yt-full --url "..."
# yt-premium --url "..."
```

---

## 📱 Мобильные устройства

### Termux (Android)
```bash
# Установка
pkg install python ffmpeg
pip install -r requirements.txt

# Использование
python -m src.main --url "..."
```

---

## 🌐 Поддерживаемые языки

- 🇷🇺 Русский
- 🇬🇧 Английский

**Транскрибирование:** RU/EN (автоопределение)
**Перевод:** EN → RU
**LLM улучшение:** RU/EN

**v2.0:** 200+ языков через NLLB

---

## ⚡ Производительность

### Ускорение обработки
1. Используйте GPU (если есть)
2. Закройте другие приложения
3. Используйте `whisper_base` вместо `medium`
4. Используйте быстрые LLM модели (`qwen2.5:3b`)

### Экономия памяти
1. Обрабатывайте по одному видео
2. Очищайте temp/ регулярно
3. Используйте SSD для temp/
4. Для LLM: используйте модели 3B вместо 8B

---

## 🔐 Безопасность

### Приватность
- ✅ Все локально
- ✅ Данные не уходят
- ✅ Open-source модели
- ✅ LLM улучшение через локальный Ollama

### API ключи
```bash
# Храните в .env
OPENAI_API_KEY=sk-...

# Не коммитьте!
# .env уже в .gitignore
```

---

## 📞 Получить помощь

1. **Документация:** `README.md`, `FAQ.md`
2. **Логи:** `logs/app_*.log`
3. **Debug режим:** `LOG_LEVEL=DEBUG`
4. **GitHub Issues:** Создайте issue с описанием

---

## ✅ Чеклист готовности

Перед использованием проверьте:

- [ ] Python 3.9+ установлен
- [ ] FFmpeg установлен
- [ ] Venv создан и активирован
- [ ] Зависимости установлены
- [ ] Тесты проходят
- [ ] Первый тест выполнен успешно

**Для LLM улучшения:**
- [ ] Ollama установлен
- [ ] Модель загружена (`ollama pull qwen2.5:3b`)
- [ ] Ollama сервер запущен (`ollama serve`)

---

## 🎉 Готово!

**Всё что нужно знать - на одной странице!**

Для деталей см. полную документацию:
- README.md
- QUICKSTART.md
- EXAMPLES.md

**Happy transcribing! 🚀**