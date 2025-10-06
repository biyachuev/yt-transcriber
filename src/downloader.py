"""
Модуль для скачивания видео с YouTube
"""
from pathlib import Path
from typing import Tuple, Optional
import yt_dlp
from tqdm import tqdm

from .config import settings
from .logger import logger
from .utils import sanitize_filename


class YouTubeDownloader:
    """Класс для скачивания аудио с YouTube"""
    
    def __init__(self):
        self.temp_dir = settings.TEMP_DIR
        self.progress_bar: Optional[tqdm] = None
    
    def _progress_hook(self, d: dict):
        """Хук для отображения прогресса скачивания"""
        if d['status'] == 'downloading':
            if self.progress_bar is None:
                total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
                self.progress_bar = tqdm(
                    total=total,
                    unit='B',
                    unit_scale=True,
                    desc="Скачивание"
                )
            
            downloaded = d.get('downloaded_bytes', 0)
            if self.progress_bar.n < downloaded:
                self.progress_bar.update(downloaded - self.progress_bar.n)
        
        elif d['status'] == 'finished':
            if self.progress_bar:
                self.progress_bar.close()
                self.progress_bar = None
            logger.info("Скачивание завершено, начинается обработка...")
    
    def download_audio(self, url: str) -> Tuple[Path, str, float, dict]:
        """
        Скачивание аудиодорожки с YouTube

        Args:
            url: URL видео на YouTube

        Returns:
            Кортеж (путь к аудиофайлу, название видео, длительность в секундах, метаданные)
        """
        logger.info(f"Начинается скачивание с YouTube: {url}")

        # Получаем полную информацию о видео
        metadata = self.extract_metadata(url)

        video_title = metadata['title']
        duration = metadata['duration']

        logger.info(f"Название видео: {video_title}")
        logger.info(f"Длительность: {duration // 60} мин {duration % 60} сек")
        
        # Очищаем название для использования в имени файла
        clean_title = sanitize_filename(video_title)
        output_path = self.temp_dir / f"{clean_title}.%(ext)s"
        
        # Опции для скачивания
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': str(output_path),
            'progress_hooks': [self._progress_hook],
            'quiet': False,
            'no_warnings': True,
        }
        
        # Скачиваем
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        audio_file = self.temp_dir / f"{clean_title}.mp3"

        if not audio_file.exists():
            raise FileNotFoundError(f"Аудиофайл не найден: {audio_file}")

        logger.info(f"Аудио сохранено: {audio_file}")

        return audio_file, video_title, duration, metadata
    
    def extract_metadata(self, url: str) -> dict:
        """
        Извлечение метаданных и субтитров из YouTube видео

        Args:
            url: URL видео на YouTube

        Returns:
            Словарь с метаданными (title, description, tags, subtitles и т.д.)
        """
        logger.info("Извлечение метаданных видео...")

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'ru'],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        # Извлекаем основные метаданные
        metadata = {
            'title': info.get('title', 'untitled'),
            'duration': info.get('duration', 0),
            'description': info.get('description', ''),
            'uploader': info.get('uploader', ''),
            'upload_date': info.get('upload_date', ''),
            'tags': info.get('tags', []),
            'categories': info.get('categories', []),
            'channel': info.get('channel', ''),
        }

        # Логируем метаданные
        logger.info(f"Метаданные видео:")
        logger.info(f"  Название: {metadata['title']}")
        logger.info(f"  Автор: {metadata['uploader']}")
        logger.info(f"  Канал: {metadata['channel']}")
        if metadata['tags']:
            logger.info(f"  Теги: {', '.join(metadata['tags'][:10])}")  # Первые 10 тегов
        if metadata['categories']:
            logger.info(f"  Категории: {', '.join(metadata['categories'])}")

        # Извлекаем субтитры (если есть)
        subtitles_info = info.get('subtitles', {})
        automatic_captions = info.get('automatic_captions', {})

        subtitle_sample = ""
        if subtitles_info or automatic_captions:
            logger.info("Найдены субтитры:")

            # Пробуем получить текст субтитров для промпта
            subtitle_url = None

            if 'en' in subtitles_info:
                logger.info("  - Английские субтитры (ручные)")
                # Берём первый доступный формат
                if subtitles_info['en']:
                    subtitle_url = subtitles_info['en'][0].get('url')
            elif 'en' in automatic_captions:
                logger.info("  - Английские субтитры (автоматические)")
                if automatic_captions['en']:
                    subtitle_url = automatic_captions['en'][0].get('url')
            elif 'ru' in subtitles_info:
                logger.info("  - Русские субтитры (ручные)")
                if subtitles_info['ru']:
                    subtitle_url = subtitles_info['ru'][0].get('url')
            elif 'ru' in automatic_captions:
                logger.info("  - Русские субтитры (автоматические)")
                if automatic_captions['ru']:
                    subtitle_url = automatic_captions['ru'][0].get('url')

            # Загружаем образец субтитров (первые ~2000 символов)
            if subtitle_url:
                try:
                    import requests
                    response = requests.get(subtitle_url, timeout=10)
                    if response.status_code == 200:
                        # Простое извлечение текста из субтитров (удаляем XML теги)
                        import re
                        subtitle_text = response.text
                        # Убираем XML теги
                        subtitle_text = re.sub(r'<[^>]+>', '', subtitle_text)
                        # Убираем временные метки
                        subtitle_text = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3}', '', subtitle_text)
                        # Берём первые 2000 символов
                        subtitle_sample = subtitle_text[:2000].strip()
                        logger.info(f"  Загружен образец субтитров ({len(subtitle_sample)} символов)")
                except Exception as e:
                    logger.warning(f"  Не удалось загрузить субтитры: {e}")

            metadata['has_subtitles'] = True
            metadata['subtitles_sample'] = subtitle_sample
        else:
            logger.info("Субтитры не найдены")
            metadata['has_subtitles'] = False
            metadata['subtitles_sample'] = ""

        return metadata

    def get_video_info(self, url: str) -> dict:
        """
        Получение информации о видео без скачивания

        Args:
            url: URL видео на YouTube

        Returns:
            Словарь с информацией о видео
        """
        return self.extract_metadata(url)