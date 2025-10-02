"""
Модуль для создания выходных документов (docx и markdown)
"""
from pathlib import Path
from typing import List, Optional
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

from .config import settings
from .logger import logger
from .utils import sanitize_filename


class DocumentWriter:
    """Класс для создания документов"""
    
    def __init__(self):
        self.output_dir = settings.OUTPUT_DIR
    
    def create_documents(
        self,
        title: str,
        sections: List[dict],
        original_text: Optional[str] = None
    ) -> tuple[Path, Path]:
        """
        Создание docx и markdown документов
        
        Args:
            title: Название документа (от видео)
            sections: Список секций для документа
                     [{
                         'title': 'Название секции',
                         'method': 'метод обработки',
                         'content': 'текст',
                         'with_timestamps': bool
                     }]
            original_text: Оригинальный текст (опционально)
            
        Returns:
            Кортеж (путь к docx, путь к md)
        """
        clean_title = sanitize_filename(title)
        
        docx_path = self.output_dir / f"{clean_title}.docx"
        md_path = self.output_dir / f"{clean_title}.md"
        
        logger.info(f"Создание документов: {clean_title}")
        
        # Создаем docx
        self._create_docx(docx_path, title, sections)
        
        # Создаем markdown
        self._create_markdown(md_path, title, sections)
        
        logger.info(f"Документы сохранены:\n  - {docx_path}\n  - {md_path}")
        
        return docx_path, md_path
    
    def _create_docx(self, path: Path, title: str, sections: List[dict]):
        """Создание docx документа"""
        doc = Document()
        
        # Настройка стилей по умолчанию
        style = doc.styles['Normal']
        font = style.font
        font.name = settings.DEFAULT_FONT
        font.size = Pt(settings.DEFAULT_FONT_SIZE)
        
        # Заголовок документа
        heading = doc.add_heading(title, level=0)
        heading.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        
        # Добавляем секции
        for section in sections:
            # Заголовок секции
            section_heading = doc.add_heading(section['title'], level=1)
            
            # Подзаголовок с методом
            if 'method' in section and section['method']:
                method_para = doc.add_paragraph()
                method_run = method_para.add_run(f"Метод: {section['method']}")
                method_run.italic = True
                method_run.font.size = Pt(10)
                method_run.font.color.rgb = RGBColor(128, 128, 128)
            
            # Контент
            content = section.get('content', '')
            if content:
                # Разбиваем на параграфы
                paragraphs = content.split('\n\n')
                for para_text in paragraphs:
                    if para_text.strip():
                        para = doc.add_paragraph(para_text.strip())
                        para.style = 'Normal'
            
            # Добавляем разделитель между секциями
            doc.add_paragraph()
        
        doc.save(str(path))
    
    def _create_markdown(self, path: Path, title: str, sections: List[dict]):
        """Создание markdown документа"""
        lines = []
        
        # Заголовок документа
        lines.append(f"# {title}\n")
        
        # Добавляем секции
        for section in sections:
            # Заголовок секции
            lines.append(f"## {section['title']}\n")
            
            # Подзаголовок с методом
            if 'method' in section and section['method']:
                lines.append(f"*Метод: {section['method']}*\n")
            
            # Контент
            content = section.get('content', '')
            if content:
                lines.append(content)
                lines.append("")  # Пустая строка после секции
        
        # Записываем в файл
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def create_from_segments(
        self,
        title: str,
        transcription_segments: List,
        translation_segments: Optional[List] = None,
        transcribe_method: str = "",
        translate_method: str = "",
        with_timestamps: bool = True
    ) -> tuple[Path, Path]:
        """
        Создание документов из сегментов транскрипции и перевода
        
        Args:
            title: Название документа
            transcription_segments: Сегменты транскрипции
            translation_segments: Сегменты перевода (опционально)
            transcribe_method: Метод транскрибирования
            translate_method: Метод перевода
            with_timestamps: Включить таймкоды
            
        Returns:
            Кортеж (путь к docx, путь к md)
        """
        from .transcriber import Transcriber
        
        transcriber = Transcriber()
        sections = []
        
        # Секция с переводом (если есть)
        if translation_segments:
            if with_timestamps:
                translation_text = transcriber.segments_to_text_with_timestamps(
                    translation_segments
                )
            else:
                translation_text = transcriber.segments_to_text(translation_segments)
            
            sections.append({
                'title': 'Перевод',
                'method': translate_method,
                'content': translation_text
            })
        
        # Секция с транскрипцией
        if with_timestamps:
            transcription_text = transcriber.segments_to_text_with_timestamps(
                transcription_segments
            )
        else:
            transcription_text = transcriber.segments_to_text(transcription_segments)
        
        sections.append({
            'title': 'Расшифровка',
            'method': transcribe_method,
            'content': transcription_text
        })
        
        return self.create_documents(title, sections)