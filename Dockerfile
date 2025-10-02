FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование requirements
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY src/ ./src/
COPY .env.example .env

# Создание необходимых директорий
RUN mkdir -p output temp logs models/whisper models/nllb

# Установка переменных окружения
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Точка входа
ENTRYPOINT ["python", "-m", "src.main"]

# По умолчанию показываем справку
CMD ["--help"]