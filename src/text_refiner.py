"""
Модуль для улучшения транскрибированного текста с помощью LLM
"""
from typing import List, Optional
import requests
import json
from tqdm import tqdm

from .logger import logger


class TextRefiner:
    """Класс для улучшения текста транскрипции с помощью локальной LLM через Ollama"""

    def __init__(self, model_name: str = "qwen2.5:3b", ollama_url: str = "http://localhost:11434"):
        """
        Инициализация

        Args:
            model_name: Название модели в Ollama
            ollama_url: URL сервера Ollama
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_endpoint = f"{ollama_url}/api/generate"

        # Проверяем доступность Ollama
        self._check_ollama_available()

    def _check_ollama_available(self):
        """Проверка доступности Ollama сервера"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"Ollama сервер доступен: {self.ollama_url}")

                # Проверяем наличие нужной модели
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]

                if self.model_name not in model_names:
                    error_msg = f"\n{'='*60}\n"
                    error_msg += f"❌ ОШИБКА: Модель '{self.model_name}' не найдена в Ollama\n"
                    error_msg += f"{'='*60}\n\n"

                    if model_names:
                        error_msg += f"Доступные модели:\n"
                        for name in model_names:
                            error_msg += f"  - {name}\n"
                        error_msg += f"\n"
                    else:
                        error_msg += f"В Ollama нет загруженных моделей\n\n"

                    error_msg += f"Для загрузки модели выполните:\n"
                    error_msg += f"  ollama pull {self.model_name}\n\n"
                    error_msg += f"Рекомендуемые модели:\n"
                    error_msg += f"  - qwen2.5:3b  (быстрая, ~2GB)\n"
                    error_msg += f"  - qwen2.5:7b  (лучше качество, ~4.7GB)\n"
                    error_msg += f"  - llama3:8b   (хорошая альтернатива, ~4.7GB)\n"
                    error_msg += f"{'='*60}"

                    logger.error(error_msg)
                    raise RuntimeError(f"Модель '{self.model_name}' не найдена в Ollama. Выполните: ollama pull {self.model_name}")
                else:
                    logger.info(f"Модель '{self.model_name}' найдена")
            else:
                raise Exception("Ollama не отвечает")
        except requests.exceptions.RequestException as e:
            error_msg = f"\n{'='*60}\n"
            error_msg += f"❌ ОШИБКА: Не удалось подключиться к Ollama серверу\n"
            error_msg += f"{'='*60}\n\n"
            error_msg += f"Причина: {e}\n\n"
            error_msg += f"Решение:\n"
            error_msg += f"1. Проверьте, что Ollama установлена:\n"
            error_msg += f"   https://ollama.com/download\n\n"
            error_msg += f"2. Запустите Ollama сервер:\n"
            error_msg += f"   ollama serve\n\n"
            error_msg += f"3. Или просто запустите Ollama приложение\n"
            error_msg += f"{'='*60}"

            logger.error(error_msg)
            raise RuntimeError(f"Не удалось подключиться к Ollama: {e}")

    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 2000) -> List[str]:
        """
        Разбивка текста на чанки по предложениям

        Args:
            text: Исходный текст
            max_chunk_size: Максимальный размер чанка в символах

        Returns:
            Список чанков текста
        """
        # Разбиваем по предложениям (простой вариант)
        sentences = text.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence)

            # Если добавление предложения превысит лимит и чанк не пустой
            if current_length + sentence_length > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Добавляем последний чанк
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        logger.info(f"Текст разбит на {len(chunks)} частей для обработки")
        return chunks

    def _call_ollama(self, prompt: str) -> str:
        """
        Вызов Ollama API

        Args:
            prompt: Промпт для модели

        Returns:
            Ответ модели
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Очень низкая температура
                "top_p": 0.9,
                "num_predict": 1000,  # Ограничение длины ответа
                "stop": ["###", "RAW TRANSCRIPTION:", "EXAMPLE"]  # Стоп-последовательности для предотвращения переобъяснений
            }
        }

        try:
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=300  # 5 минут таймаут
            )
            response.raise_for_status()

            result = response.json()
            return result.get('response', '').strip()
        except Exception as e:
            logger.error(f"Ошибка при вызове Ollama: {e}")
            raise

    def _detect_topic(self, text_sample: str) -> str:
        """
        Определение тематики текста

        Args:
            text_sample: Образец текста (начало)

        Returns:
            Определенная тематика
        """
        prompt = f"""Определи тематику текста. Ответь ТОЛЬКО одним словом, БЕЗ объяснений.

Примеры: шахматы, программирование, политика, бизнес, спорт

Текст: {text_sample}

Тематика (одно слово):"""

        try:
            topic = self._call_ollama(prompt)

            # Очищаем ответ от thinking tags и берём только первое слово
            topic = topic.replace("<think>", "").replace("</think>", "")
            topic = topic.strip().split()[0] if topic.strip() else "общая"

            logger.info(f"Определена тематика: {topic}")
            return topic
        except Exception as e:
            logger.warning(f"Не удалось определить тематику: {e}")
            return "общая"

    def _detect_language(self, text: str) -> str:
        """
        Определение языка текста (простое эвристическое определение)

        Args:
            text: Текст для определения языка

        Returns:
            'ru' для русского, 'en' для английского
        """
        # Простое определение: считаем кириллицу
        cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        total_chars = sum(1 for c in text if c.isalpha())

        if total_chars == 0:
            return 'en'

        # Если больше 30% кириллицы - считаем русским
        return 'ru' if (cyrillic_chars / total_chars) > 0.3 else 'en'

    def refine_chunk(self, chunk: str, context: Optional[str] = None, topic: Optional[str] = None, language: Optional[str] = None) -> str:
        """
        Улучшение одного чанка текста

        Args:
            chunk: Чанк текста для улучшения
            context: Контекст (промпт от пользователя или из метаданных)
            topic: Тематика текста
            language: Язык текста ('en' или 'ru'), если None - определяется автомати��ески

        Returns:
            Улучшенный текст
        """
        # Определяем язык если не передан
        if language is None:
            language = self._detect_language(chunk)

        # Формируем промпт
        context_info = f"\nКонтекст: {context}" if context else ""
        topic_info = topic if topic else "общая"

        # Выбираем промпт в зависимости от языка
        if language == 'ru':
            prompt = f"""Очистите эту расшифровку речи. Удаляйте ТОЛЬКО слова-паразиты и метакомментарии. Сохраняйте ВСЁ остальное.

УДАЛЯЙТЕ ТОЛЬКО:
1. Слова-паразиты: эм, э-э, как бы, ну, вот, короче, типа, в общем, слушайте, значит, это самое
2. Явные заикания: "в в", "я-я-я"
3. Метакомментарии о процессе записи: "сейчас открою экран", "секунду", "вот так", "давайте", "смотрите"

ОБЯЗАТЕЛЬНО СОХРАНЯЙТЕ ВСЁ ОСТАЛЬНОЕ - НЕ УДАЛЯЙТЕ:
- Каждое предложение, даже если кажется повторяющимся
- ВСЕ примеры: "если у тебя рейтинг выше 2650 и ты играешь с 1800 или 1700"
- ВСЕ цифры и рейтинги
- ВСЕ аргументы, рассуждения и объяснения
- ВСЕ мнения и мысли
- Повторяющиеся утверждения (говорящий может специально подчёркивать)
- Фразы вроде "я расскажу об этом позже" или "извините если..."

ФОРМАТИРОВАНИЕ:
- Исправляйте ошибки распознавания: "фиде" → "ФИДЕ", "епл" → "Apple"
- Конвертируйте произнесённые числа в цифры: "две тысячи восемьсот шестнадцать" → "2816", "ноль восемь" → "0.8", "сорок партий" → "40 партий"
- Исправляйте пунктуацию
- Объединяйте фрагменты предложений

КРИТИЧНО: Ваша ЕДИНСТВЕННАЯ задача — удалить слова-паразиты и исправить форматирование. Если вы удалите ЛЮБОЙ содержательный контент, примеры или рассуждения, вы провалите задачу.

Возвращайте ТОЛЬКО очищенный текст.

ПРИМЕР:
ИСХОДНЫЙ: "Слушайте ребят ну вот э-э значит главная новость о которой все хотели чтобы я рассказал ну я щас об этом расскажу короче вот прямо сейчас так. Давайте сразу к делу. Ну вот это вот что началось началось вся эта драма главная тема последних суток которая началась вчера с этого твита от Эмиля Сутовского который э-э я думаю сейчас гендир фиде ну то есть главной шахматной организации."

ИСПРАВЛЕННЫЙ: "Главная новость, о которой все хотели чтобы я рассказал, я расскажу об этом прямо сейчас. Сразу к делу. Это то, что началось, вся эта драма, главная тема последних суток, которая началась вчера с этого твита от Эмиля Сутовского, который, я думаю, сейчас генеральный директор ФИДЕ, то есть главной шахматной организации."

---

Теперь очистите этот текст. Возвращайте ТОЛЬКО очищенный текст:

{chunk}"""
        else:
            prompt = f"""Clean up this speech-to-text transcript. Remove ONLY filler words and meta-commentary. Keep EVERYTHING else.

REMOVE ONLY:
1. Filler words: um, uh, like, you know, so, well, basically, actually, alright, you guys
2. Obvious stutters: "the the", "I-I-I"
3. Meta-commentary about the recording process: "let me resize", "I'll scroll", "there we go", "here we go", "let me adjust this"

KEEP ABSOLUTELY EVERYTHING ELSE - DO NOT REMOVE:
- Every sentence, even if it seems repetitive
- ALL examples: "if you're above 2650 and play someone like 1800 or 1700"
- ALL numbers and ratings
- ALL arguments, reasoning, and explanations
- ALL opinions and thoughts
- Repetitive statements (the speaker may be emphasizing)
- Phrases like "I'll talk about this later" or "I apologize if..."

FORMATTING:
- Fix speech recognition errors: "feed it" → "FIDE", "feeding" → "FIDE"
- Convert spoken numbers to digits: "twenty eight sixteen" → "2816", "zero point eight" → "0.8", "point eight" → "0.8"
- Clean up punctuation
- Combine sentence fragments

CRITICAL: Your ONLY job is to remove filler words and fix formatting. If you delete ANY substantive content, examples, or reasoning, you fail.

Return ONLY the cleaned text.

EXAMPLE:
RAW: "All right, you guys, so the big news that everybody's been wanting me to talk about, I will talk about it right now. And let's dive right into the meat. So here we go. This is what kicked off the kicked off all the drama, the big topic of the last 24 hours that began yesterday with this tweet from Emil Sotovsky, who I believe is currently the feed it the CEO of FIDE, the governing body of chess. So this was the tweet. And the tweet says, no more farming. If you're a 2650 plus player, please, or if you're a 2650 plus player, do prove your skill versus opponents of comparable strength. Why 2650 plus?"

CORRECTED: "The big news everyone's been wanting me to discuss—I'll talk about it right now. Let's dive right into the meat. This is what kicked off all the drama, the big topic of the last 24 hours that began yesterday with this tweet from Emil Sotovsky, who I believe is currently the CEO of FIDE, the governing body of chess.

The tweet says: "No more farming. If you're a 2650+ player, please prove your skill versus opponents of comparable strength. Why 2650+?""

---

Now clean this text. Return ONLY the cleaned text:

{chunk}"""


        try:
            refined_text = self._call_ollama(prompt)

            # Очистка ответа от артефактов
            # Удаляем thinking tags если есть
            refined_text = refined_text.replace("<think>", "").replace("</think>", "").strip()

            # Удаляем артефакты, которые может добавить модель (английские)
            if refined_text.startswith("Corrected transcription:"):
                refined_text = refined_text.replace("Corrected transcription:", "", 1).strip()

            # Удаляем артефакты на русском
            if refined_text.startswith("Исправленная транскрипция:"):
                refined_text = refined_text.replace("Исправленная транскрипция:", "", 1).strip()

            # Удаляем маркеры в конце
            refined_text = refined_text.replace("### END OF TRANSCRIPTION###", "").strip()

            # Удаляем возможные артефакты в начале (```text, ##, и т.д.)
            refined_text = refined_text.lstrip('`#\n ')

            # Если после очистки текст стал слишком коротким - вернуть оригинал
            if len(refined_text.strip()) < len(chunk) * 0.3:
                logger.warning("Улучшенный текст слишком короткий, возвращаю оригинал")
                return chunk

            return refined_text.strip()
        except Exception as e:
            logger.error(f"Ошибка при улучшении чанка: {e}")
            # В случае ошибки возвращаем оригинал
            return chunk

    def refine_text(self, text: str, context: Optional[str] = None) -> str:
        """
        Улучшение всего текста транскрипции

        Args:
            text: Полный текст транскрипции
            context: Контекст (промпт от пользователя или из метаданных)

        Returns:
            Улучшенный текст
        """
        logger.info("Начало улучшения транскрипции с помощью LLM...")

        # Определяем язык текста
        language = self._detect_language(text[:500])
        logger.info(f"Определён язык текста: {'русский' if language == 'ru' else 'английский'}")

        # Определяем тематику по началу текста
        topic = self._detect_topic(text[:500])

        # Разбиваем на чанки
        chunks = self._split_text_into_chunks(text, max_chunk_size=2000)

        # Обрабатываем каждый чанк
        refined_chunks = []
        for chunk in tqdm(chunks, desc="Улучшение текста"):
            refined_chunk = self.refine_chunk(chunk, context=context, topic=topic, language=language)
            refined_chunks.append(refined_chunk)

        # Объединяем результаты
        refined_text = '\n\n'.join(refined_chunks)

        logger.info("Улучшение транскрипции завершено")
        return refined_text

    def refine_translation(self, translated_text: str, context: Optional[str] = None) -> str:
        """
        Улучшение перевода с помощью LLM для большей естественности

        Args:
            translated_text: Переведённый текст для улучшения
            context: Контекст (промпт от пользователя или из метаданных)

        Returns:
            Улучшенный перевод
        """
        logger.info("Начало улучшения перевода с помощью LLM...")

        # Промпт для улучшения перевода
        TRANSLATION_REFINEMENT_PROMPT = """Ты - эксперт по адаптации и улучшению переводов. Твоя задача - переработать машинный перевод, сделав его естественным и понятным для русскоязычного читателя. Главное - передать смысл и основные идеи, а не копировать структуру предложений.

ИСХОДНЫЙ ПЕРЕВОД:
{text}

КОНТЕКСТ: {context}

ПРИНЦИПЫ РАБОТЫ:
1. СМЫСЛ ВАЖНЕЕ ФОРМЫ: Не бойся перестроить предложения, если это сделает текст понятнее
2. Убирай избыточность: Если автор повторяется без необходимости или говорит очевидные вещи - сократи до сути
3. Объединяй разрозненные мысли: Если несколько предложений говорят об одном - объедини их в цельный абзац
4. Замени неестественные конструкции на простые русские фразы
5. Удаляй слова-паразиты, вводные обороты и лишнюю "воду"
6. Делай текст более структурированным: выделяй ключевые мысли, группируй связанные идеи
7. Упрощай сложные формулировки, сохраняя точность смысла

ЧТО СОХРАНЯТЬ:
- Все ключевые идеи и аргументы (но можешь выразить их лаконичнее)
- Имена собственные, термины, цифры, рейтинги
- Временные метки [MM:SS]
- Общий тон автора (формальный/неформальный)
- Примеры и конкретные случаи (но без лишних деталей)

ЧТО МОЖНО ИЗМЕНЯТЬ/СОКРАЩАТЬ:
- Повторяющиеся объяснения одного и того же
- Вводные фразы типа "я хочу сказать", "ну вот", "в общем"
- Избыточные уточнения и оговорки
- Длинные запутанные конструкции (перефразируй короче и яснее)
- Порядок предложений (если логичнее изложить иначе)

ВАЖНО:
- Фокус на понятности и читабельности
- Текст должен быть кратким, но информативным
- Сохраняй ВСЕ важные факты, цифры и аргументы
- НЕ добавляй информацию, которой нет в оригинале
- Пиши так, как писал бы носитель русского языка

Верни ТОЛЬКО переработанный текст без пояснений и комментариев."""

        # Разбиваем перевод на чанки
        chunks = self._split_text_into_chunks(translated_text, max_chunk_size=2000)

        # Обрабатываем каждый чанк
        refined_chunks = []
        context_str = context if context else "не указан"

        for chunk in tqdm(chunks, desc="Улучшение перевода"):
            prompt = TRANSLATION_REFINEMENT_PROMPT.format(
                text=chunk,
                context=context_str
            )

            try:
                refined_chunk = self._call_ollama(prompt)

                # Очистка ответа от артефактов
                refined_chunk = refined_chunk.replace("<think>", "").replace("</think>", "").strip()

                # Удаляем возможные артефакты
                if refined_chunk.startswith("Улучшенный перевод:"):
                    refined_chunk = refined_chunk.replace("Улучшенный перевод:", "", 1).strip()

                # Если результат слишком короткий - вернуть оригинал
                if len(refined_chunk.strip()) < len(chunk) * 0.3:
                    logger.warning("Улучшенный перевод слишком короткий, возвращаю оригинал чанка")
                    refined_chunks.append(chunk)
                else:
                    refined_chunks.append(refined_chunk.strip())
            except Exception as e:
                logger.error(f"Ошибка при улучшении чанка перевода: {e}")
                # В случае ошибки возвращаем оригинал
                refined_chunks.append(chunk)

        # Объединяем результаты
        refined_translation = '\n\n'.join(refined_chunks)

        logger.info("Улучшение перевода завершено")
        return refined_translation
