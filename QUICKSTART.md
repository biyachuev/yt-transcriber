# Быстрый старт

Пошаговое руководство для начала работы с YouTube Transcriber за 5 минут.

## ⚡ Установка за 3 шага

### Шаг 1: Клонирование и создание окружения

```bash
# Клонируем репозиторий
git clone <repository-url>
cd youtube-transcriber

# Создаем виртуальное окружение
python -m venv venv

# Активируем (macOS/Linux)
source venv/bin/activate

# Активируем (Windows)
# venv\Scripts\activate
```

### Шаг 2: Установка FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu):**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows:**
- Скачайте с [ffmpeg.org](https://ffmpeg.org/download.html)
- Добавьте в PATH

### Шаг 3: Установка зависимостей Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

⏱️ **Это займет 5-10 минут**

---

## 🚀 Первый запуск

### Простой тест

```bash
python -m src.main \
    --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
    --transcribe whisper_base
```

**Что произойдет:**
1. ⬇️ Скачается аудио с YouTube
2. 📝 Создастся расшифровка
3. 💾 Результат сохранится в `output/`

⏱️ **Первый запуск займет дольше** (загрузка моделей ~2-3GB)

---

### С переводом

```bash
python -m src.main \
    --url "https://www.youtube.com/watch?v=YOUR_VIDEO" \
    --transcribe whisper_base \
    --translate NLLB
```

**Результат:**
- `output/Video_Title.docx` - документ Word
- `output/Video_Title.md` - Markdown файл

Оба содержат:
1. Перевод на русский (если видео на английском)
2. Оригинальную расшифровку
3. Таймкоды для каждого абзаца

---

## 📋 Основные команды

### Просмотр справки

```bash
python -m src.main --help
```

### Только транскрибирование

```bash
python -m src.main --url "URL" --transcribe whisper_base
```

### Транскрибирование + перевод

```bash
python -m src.main --url "URL" --transcribe whisper_base --translate NLLB
```

---

## 📁 Где найти результаты?

```
youtube-transcriber/
├── output/              # ← Здесь ваши результаты!
│   ├── Video_Title.docx
│   └── Video_Title.md
├── temp/                # Временные файлы (аудио)
└── logs/                # Логи работы
```

---

## 🎯 Типичные сценарии

### Сценарий 1: Английская лекция

```bash
python -m src.main \
    --url "https://youtube.com/watch?v=..." \
    --transcribe whisper_base \
    --translate NLLB
```

**Получите:** Расшифровку + русский перевод

---

### Сценарий 2: Русское видео

```bash
python -m src.main \
    --url "https://youtube.com/watch?v=..." \
    --transcribe whisper_base
```

**Получите:** Только расшифровку (перевод не нужен)

---

### Сценарий 3: Несколько видео

Создайте скрипт `process_videos.sh`:

```bash
#!/bin/bash

python -m src.main --url "https://youtube.com/watch?v=VIDEO1" --transcribe whisper_base --translate NLLB
python -m src.main --url "https://youtube.com/watch?v=VIDEO2" --transcribe whisper_base --translate NLLB
python -m src.main --url "https://youtube.com/watch?v=VIDEO3" --transcribe whisper_base --translate NLLB

echo "Готово!"
```

Запустите:
```bash
chmod +x process_videos.sh
./process_videos.sh
```

---

## ⚙️ Настройка (опционально)

### Создание .env файла

```bash
cp .env.example .env
```

Отредактируйте `.env`:
```bash
# Уровень детализации логов
LOG_LEVEL=INFO  # или DEBUG для отладки

# Устройство для вычислений
WHISPER_DEVICE=mps  # mps (M1/M2), cuda (NVIDIA), cpu
```

---

## 🐛 Решение проблем

### Ошибка: "FFmpeg not found"

```bash
# Проверьте установку
ffmpeg -version

# Если не установлен, установите (см. Шаг 2 выше)
```

---

### Ошибка: "Out of memory"

**Решение:** Закройте другие приложения или используйте:
```bash
# Более легкая модель
--transcribe whisper_base  # вместо whisper_small
```

---

### Модели долго загружаются

**Это нормально!** При первом запуске загружаются:
- Whisper Base: ~150MB
- NLLB: ~2.5GB

Они кешируются в `models/` и больше не загружаются.

---

### Медленная работа

**Это нормально для CPU!** Транскрибирование идет примерно в реальном времени:
- 1 час видео = 1-1.5 часа обработки

**Ускорить можно:**
1. Использовать GPU (если есть)
2. Запускать на ночь
3. Обрабатывать видео меньшей длины

---

## 📊 Ожидаемое время

| Видео | Обработка (M1, 16GB) |
|-------|----------------------|
| 10 мин | ~18 мин |
| 30 мин | ~55 мин |
| 1 час | ~110 мин |
| 2 часа | ~220 мин |

---

## 🎓 Следующие шаги

Теперь вы готовы! Вот что можно изучить дальше:

1. 📖 [README.md](README.md) - Полная документация
2. 💡 [EXAMPLES.md](EXAMPLES.md) - Больше примеров использования
3. ❓ [FAQ.md](FAQ.md) - Ответы на частые вопросы
4. 🐳 Docker - Запуск в контейнере (см. README)

---

## 💬 Нужна помощь?

- 📚 Проверьте [FAQ.md](FAQ.md)
- 🐛 Откройте [issue на GitHub](https://github.com/yourusername/youtube-transcriber/issues)
- 📧 Свяжитесь с разработчиками

---

## ✅ Чеклист готовности

- [ ] Python 3.9+ установлен
- [ ] FFmpeg установлен и работает
- [ ] Виртуальное окружение создано и активировано
- [ ] Зависимости установлены (`pip install -r requirements.txt`)
- [ ] Первый тестовый запуск выполнен успешно
- [ ] Результаты найдены в папке `output/`

**Все готово? Отлично! Начинайте обрабатывать свои видео! 🎉**