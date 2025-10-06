# Примеры использования

## Базовые примеры

### 1. Простое транскрибирование YouTube видео

Транскрибирование без перевода:

```bash
python -m src.main \
    --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
    --transcribe whisper_base
```

**Результат:**
- Файл `output/Video_Title.docx` с расшифровкой
- Файл `output/Video_Title.md` с расшифровкой

---

### 2. Транскрибирование с переводом

Транскрибирование английского видео с переводом на русский:

```bash
python -m src.main \
    --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
    --transcribe whisper_base \
    --translate NLLB
```

**Результат:**
Документ с двумя секциями:
1. Перевод на русский язык
2. Оригинальная расшифровка

---

### 3. Обработка локального аудиофайла

Транскрибирование аудиофайла с переводом:

```bash
python -m src.main \
    --input_audio audio.mp3 \
    --transcribe whisper_base \
    --translate NLLB
```

**Поддерживаемые форматы:** mp3, wav, m4a, flac, ogg

---

### 4. Транскрибирование с улучшением качества

Использование более качественной модели Whisper:

```bash
python -m src.main \
    --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
    --transcribe whisper_medium
```

**Доступные модели:**
- `whisper_base` - быстро, хорошо (по умолчанию)
- `whisper_small` - медленнее, лучше качество
- `whisper_medium` - медленно, высокое качество

---

### 5. Использование пользовательского промпта

Промпт помогает правильно распознавать имена и термины:

```bash
python -m src.main \
    --url "https://www.youtube.com/watch?v=CHESS_VIDEO" \
    --transcribe whisper_base \
    --prompt prompt.txt
```

**Содержимое prompt.txt:**
```
FIDE, Hikaru Nakamura, Magnus Carlsen, chess tournament, bongcloud
```

---

### 6. Улучшение транскрипции с помощью LLM

После транскрипции текст можно улучшить с помощью локальной языковой модели:

```bash
python -m src.main \
    --input_audio interview.mp3 \
    --transcribe whisper_medium \
    --refine-model qwen2.5:3b
```

**Результат:** Создаются два документа:
- `interview (original).docx/md` - оригинальная транскрипция
- `interview (refined).docx/md` - улучшенная версия

**Требования:**
1. Установленный Ollama: https://ollama.ai
2. Загруженная модель: `ollama pull qwen2.5:3b`
3. Запущенный сервер: `ollama serve`

---

### 7. Полная обработка с улучшением и переводом

```bash
python -m src.main \
    --input_audio lecture.mp3 \
    --transcribe whisper_medium \
    --translate NLLB \
    --refine-model qwen2.5:3b \
    --prompt lecture_prompt.txt
```

**Результат:** Создаются три документа:
- `lecture (original).docx/md` - оригинальная транскрипция
- `lecture (refined).docx/md` - улучшенная версия
- `lecture (translated).docx/md` - перевод улучшенной версии

---

## Продвинутые примеры

### 8. Обработка образовательного контента

Длинное интервью (2 часа) с улучшением качества:

```bash
python -m src.main \
    --url "https://www.youtube.com/watch?v=LONG_INTERVIEW" \
    --transcribe whisper_medium \
    --translate NLLB \
    --refine-model qwen2.5:3b
```

**Ожидаемое время:** ~4-5 часов на MacBook Air M1

---

### 9. Batch обработка (скрипт)

Создайте файл `process_multiple.sh`:

```bash
#!/bin/bash

# Список URL для обработки
URLS=(
    "https://youtube.com/watch?v=VIDEO1"
    "https://youtube.com/watch?v=VIDEO2"
    "https://youtube.com/watch?v=VIDEO3"
)

# Обработка каждого видео
for url in "${URLS[@]}"; do
    echo "Обработка: $url"
    python -m src.main \
        --url "$url" \
        --transcribe whisper_medium \
        --translate NLLB \
        --refine-model qwen2.5:3b
    
    echo "Завершено: $url"
    echo "---"
done

echo "Все видео обработаны!"
```

Запуск:
```bash
chmod +x process_multiple.sh
./process_multiple.sh
```

---

### 10. Работа с плейлистами

Сначала извлеките URL из плейлиста:

```bash
# Установите yt-dlp если еще не установлен
pip install yt-dlp

# Извлеките URLs
yt-dlp --flat-playlist --print url "PLAYLIST_URL" > urls.txt

# Обработайте каждое видео
while read url; do
    python -m src.main \
        --url "$url" \
        --transcribe whisper_medium \
        --translate NLLB \
        --refine-model qwen2.5:3b
done < urls.txt
```

---

## Сценарии использования

### Сценарий 1: Студент изучает английский

**Задача:** Смотрю английские лекции, хочу качественную расшифровку с переводом.

```bash
python -m src.main \
    --url "https://youtube.com/watch?v=LECTURE_VIDEO" \
    --transcribe whisper_medium \
    --translate NLLB \
    --refine-model qwen2.5:3b \
    --prompt lecture_prompt.txt
```

**Содержимое lecture_prompt.txt:**
```
MIT, Stanford, machine learning, neural networks, deep learning, artificial intelligence
```

**Результат:** Три документа с оригинальным текстом, улучшенной версией и русским переводом.

---

### Сценарий 2: Разработчик смотрит техническое интервью

**Задача:** 2-часовое интервью с техническим специалистом на английском.

```bash
python -m src.main \
    --url "https://youtube.com/watch?v=TECH_INTERVIEW" \
    --transcribe whisper_medium \
    --translate NLLB \
    --refine-model qwen2.5:3b \
    --prompt tech_prompt.txt
```

**Содержимое tech_prompt.txt:**
```
OpenAI, ChatGPT, GPT-4, API, Python, JavaScript, React, Node.js, Docker, Kubernetes
```

**Польза:** 
- Высокое качество транскрипции благодаря whisper_medium
- Улучшенная структура текста благодаря LLM
- Перевод помогает понять сложные термины
- Можно сохранить как справочный материал

---

### Сценарий 3: Создание конспектов

**Задача:** Несколько образовательных видео нужно законспектировать.

```bash
# Создайте файл process_lectures.sh
for i in {1..5}; do
    python -m src.main \
        --url "https://youtube.com/watch?v=LECTURE_${i}" \
        --transcribe whisper_medium \
        --translate NLLB \
        --refine-model qwen2.5:3b
done
```

**Результат:** 15 документов (по 3 на каждую лекцию) с расшифровками, улучшенными версиями и переводами.

---

### Сценарий 4: Обработка подкастов

**Задача:** Локальные аудиофайлы подкастов нужно транскрибировать.

```bash
python -m src.main \
    --input_audio podcast_episode.mp3 \
    --transcribe whisper_medium \
    --refine-model qwen2.5:3b \
    --prompt podcast_prompt.txt
```

**Содержимое podcast_prompt.txt:**
```
Joe Rogan, Elon Musk, SpaceX, Tesla, Neuralink, artificial intelligence, cryptocurrency
```

**Результат:** Два документа с оригинальной и улучшенной транскрипцией.

---
## ⏱️ Реальная производительность

### На MacBook Air M1 (16GB RAM)

**3-минутное видео (базовая обработка):**
```bash
python -m src.main --url "..." --transcribe whisper_base --translate NLLB
# Транскрибирование: ~11 секунд
# Перевод: ~1.5 минуты
# Итого: ~2 минуты
```

**3-минутное видео (с улучшением):**
```bash
python -m src.main --url "..." --transcribe whisper_medium --translate NLLB --refine-model qwen2.5:3b
# Транскрибирование: ~45 секунд
# Улучшение: ~2 минуты
# Перевод: ~1.5 минуты
# Итого: ~4 минуты

---

## Оптимизация производительности

### Совет 1: Выберите подходящую модель Whisper

Для быстрой обработки:
```bash
--transcribe whisper_base  # Быстрее, хорошее качество
```

Для лучшего качества:
```bash
--transcribe whisper_small  # Медленнее, лучше качество
--transcribe whisper_medium  # Медленно, высокое качество
```

### Совет 2: Используйте LLM улучшение разумно

Для важных материалов:
```bash
--refine-model qwen2.5:3b  # Быстрая модель
--refine-model llama3:8b   # Качественная модель
```

Для быстрой обработки:
```bash
# Без --refine-model
```

---

### Совет 3: Обработка во время сна

Запустите обработку перед сном:

```bash
# Добавьте уведомление при завершении (macOS)
python -m src.main \
    --url "https://youtube.com/watch?v=LONG_VIDEO" \
    --transcribe whisper_medium \
    --translate NLLB \
    --refine-model qwen2.5:3b && \
    osascript -e 'display notification "Обработка завершена!" with title "YouTube Transcriber"'
```

---

### Совет 4: Мониторинг прогресса

Следите за логами в реальном времени:

```bash
# В одном терминале
python -m src.main --url "..." --transcribe whisper_medium --translate NLLB --refine-model qwen2.5:3b

# В другом терминале
tail -f logs/app_*.log
```

---

## Работа с результатами

### Пример структуры выходного документа (без улучшения)

```markdown
# Название видео с YouTube

## Перевод
Метод: NLLB

[00:15] Привет всем! Сегодня мы обсудим важную тему...

[01:32] Первый пункт, который я хочу затронуть...

[03:45] Давайте рассмотрим конкретный пример...

## Расшифровка
Метод: whisper_base

[00:15] Hello everyone! Today we'll discuss an important topic...

[01:32] The first point I want to address...

[03:45] Let's look at a specific example...
```

### Пример структуры с улучшением LLM

**Файл: название (original).md**
```markdown
# Название видео с YouTube (original)

## Расшифровка
Метод: whisper_medium

[00:15] hello everyone um today we will discuss an important topic...

[01:32] the first point i want to address is um...
```

**Файл: название (refined).md**
```markdown
# Название видео с YouTube (refined)

## Перевод
Метод: NLLB

[00:15] Привет всем! Сегодня мы обсудим важную тему...

[01:32] Первый пункт, который я хочу затронуть...

## Расшифровка
Метод: whisper_medium + qwen2.5:3b

[00:15] Hello everyone! Today we will discuss an important topic...

[01:32] The first point I want to address is...
```

---

## Поиск по таймкодам

Используйте таймкоды для навигации:

1. Откройте .md файл в редакторе
2. Используйте Ctrl+F для поиска `[MM:SS]`
3. Найдите нужный момент в видео

Или используйте скрипт для извлечения конкретного момента:

```python
# extract_timestamp.py
import sys
import re

def find_text_at_timestamp(md_file, target_time):
    """Найти текст около указанного таймкода"""
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'\[(\d{2}:\d{2}(?::\d{2})?)\] (.*?)(?=\n\[|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for timestamp, text in matches:
        if timestamp >= target_time:
            print(f"[{timestamp}] {text[:200]}...")
            break

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_timestamp.py <file.md> <MM:SS>")
        sys.exit(1)
    
    find_text_at_timestamp(sys.argv[1], sys.argv[2])
```

Использование:
```bash
python extract_timestamp.py output/Video_Title.md 15:30
```

---

## Интеграция с другими инструментами

### Экспорт в Notion

1. Скопируйте содержимое .md файла
2. Вставьте в Notion
3. Notion автоматически отформатирует markdown

### Экспорт в Obsidian

```bash
# Скопируйте .md файлы в vault Obsidian
cp output/*.md ~/ObsidianVault/YouTube/
```

### Создание PDF

```bash
# Используйте pandoc для конвертации
pandoc output/Video_Title.md -o output/Video_Title.pdf
```

---

## Устранение проблем

### Проблема: Видео слишком длинное

**Решение:** Обрабатывайте по частям

```bash
# Скачайте только аудио вручную
yt-dlp -x --audio-format mp3 "VIDEO_URL" -o "audio.mp3"

# Разделите аудио (например, на 2 части по часу)
ffmpeg -i audio.mp3 -ss 00:00:00 -t 01:00:00 part1.mp3
ffmpeg -i audio.mp3 -ss 01:00:00 -t 01:00:00 part2.mp3

# Обработайте каждую часть
python -m src.main --input_audio part1.mp3 --transcribe whisper_medium --refine-model qwen2.5:3b
python -m src.main --input_audio part2.mp3 --transcribe whisper_medium --refine-model qwen2.5:3b
```

---

### Проблема: Плохое качество аудио

**Решение:** Улучшите качество перед обработкой

```bash
# Усиление громкости
ffmpeg -i input.mp3 -filter:a "volume=2.0" output.mp3

# Шумоподавление
ffmpeg -i input.mp3 -af "highpass=f=200, lowpass=f=3000" output.mp3

# Затем обработайте улучшенный файл
python -m src.main --input_audio output.mp3 --transcribe whisper_medium --refine-model qwen2.5:3b
```

---

### Проблема: Специфическая терминология

**Решение:** Используйте пользовательский промпт и LLM улучшение:

```bash
# Создайте промпт с терминами
echo "OpenAI, ChatGPT, GPT-4, API, Python, JavaScript, React, Node.js, Docker, Kubernetes" > tech_prompt.txt

# Обработайте с промптом и улучшением
python -m src.main \
    --input_audio tech_lecture.mp3 \
    --transcribe whisper_medium \
    --refine-model qwen2.5:3b \
    --prompt tech_prompt.txt
```

**Совет:** LLM улучшение автоматически исправляет терминологию и добавляет правильную пунктуацию.

---

## Автоматизация с Cron (Linux/macOS)

Автоматическая обработка видео из списка каждый день в 2 AM:

```bash
# Откройте crontab
crontab -e

# Добавьте задание
0 2 * * * cd /path/to/youtube-transcriber && python -m src.main --url "$(head -n 1 queue.txt)" --transcribe whisper_medium --translate NLLB --refine-model qwen2.5:3b && sed -i '' '1d' queue.txt
```

Создайте файл `queue.txt` со списком URL:
```
https://youtube.com/watch?v=VIDEO1
https://youtube.com/watch?v=VIDEO2
https://youtube.com/watch?v=VIDEO3
```

---

## Полезные алиасы

Добавьте в `~/.bashrc` или `~/.zshrc`:

```bash
# Быстрая транскрипция
alias yt-transcribe='python -m src.main --transcribe whisper_base'

# Транскрипция + перевод
alias yt-translate='python -m src.main --transcribe whisper_base --translate NLLB'

# Высокое качество с улучшением
alias yt-premium='python -m src.main --transcribe whisper_medium --translate NLLB --refine-model qwen2.5:3b'

# Пример использования:
# yt-transcribe --url "https://youtube.com/..."
# yt-translate --url "https://youtube.com/..."
# yt-premium --url "https://youtube.com/..."
```

---

## Продвинутые сценарии

### Создание базы знаний

Обработайте серию видео и создайте индекс:

```bash
#!/bin/bash
# build_knowledge_base.sh

VIDEOS=(
    "https://youtube.com/watch?v=VIDEO1|Topic: AI Basics"
    "https://youtube.com/watch?v=VIDEO2|Topic: Machine Learning"
    "https://youtube.com/watch?v=VIDEO3|Topic: Deep Learning"
)

echo "# База знаний" > knowledge_base_index.md
echo "" >> knowledge_base_index.md

for entry in "${VIDEOS[@]}"; do
    IFS='|' read -r url topic <<< "$entry"
    echo "Обработка: $topic"
    
    python -m src.main \
        --url "$url" \
        --transcribe whisper_medium \
        --translate NLLB \
        --refine-model qwen2.5:3b
    
    # Добавляем в индекс
    title=$(yt-dlp --get-title "$url")
    echo "## $topic" >> knowledge_base_index.md
    echo "- [$title (refined)](output/${title// /_} (refined).md)" >> knowledge_base_index.md
    echo "- [$title (original)](output/${title// /_} (original).md)" >> knowledge_base_index.md
    echo "" >> knowledge_base_index.md
done
```

---

## Дополнительные ресурсы

- **Официальная документация Whisper:** https://github.com/openai/whisper
- **NLLB от Meta:** https://github.com/facebookresearch/fairseq/tree/nllb
- **yt-dlp документация:** https://github.com/yt-dlp/yt-dlp
- **Ollama:** https://ollama.ai (для LLM улучшения)
- **Рекомендуемые модели Ollama:**
  - `qwen2.5:3b` - быстрая, хорошее качество
  - `llama3.2:3b` - быстрая, хорошее качество
  - `llama3:8b` - медленнее, но качественнее
  - `mistral:7b` - хороший баланс

---

## Обратная связь

Если вы нашли полезный сценарий использования, поделитесь им через issue или pull request!