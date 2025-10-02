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
    
    def download_audio(self, url: str) -> Tuple[Path, str, float]:
        """
        Скачивание аудиодорожки с YouTube
        
        Args:
            url: URL видео на YouTube
            
        Returns:
            Кортеж (путь к аудиофайлу, название видео, длительность в секундах)
        """
        logger.info(f"Начинается скачивание с YouTube: {url}")
        
        # Сначала получаем информацию о видео
        ydl_opts_info = {
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'untitled')
            duration = info.get('duration', 0)
            
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
        
        return audio_file, video_title, duration
    
    def get_video_info(self, url: str) -> dict:
        """
        Получение информации о видео без скачивания
        
        Args:
            url: URL видео на YouTube
            
        Returns:
            Словарь с информацией о видео
        """
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
        return {
            'title': info.get('title', 'untitled'),
            'duration': info.get('duration', 0),
            'description': info.get('description', ''),
            'uploader': info.get('uploader', ''),
            'upload_date': info.get('upload_date', ''),
        }