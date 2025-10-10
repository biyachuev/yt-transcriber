"""
Text translation utilities (Meta NLLB and future backends).
"""
from typing import List, Optional

import torch
from tqdm import tqdm

from .config import settings, TranslateOptions
from .logger import logger
from .utils import chunk_text, detect_language


class Translator:
    """Translate text using the configured backend (NLLB by default)."""

    def __init__(self, method: str = TranslateOptions.NLLB, model_name: Optional[str] = None):
        self.method = method
        self.model_name = model_name if model_name else settings.NLLB_MODEL_NAME
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = self._get_device()
        logger.info("Translator using device: %s", self.device)
        if model_name:
            logger.info("Custom NLLB model specified: %s", model_name)

    def _get_device(self) -> str:
        """Pick the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_nllb_model(self):
        """Lazy-load the NLLB model and tokenizer."""
        if self.model is not None:
            return

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        logger.info("Loading NLLB model: %s", self.model_name)
        logger.info("Initial download may take a few minutes...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=str(settings.NLLB_MODEL_DIR),
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            cache_dir=str(settings.NLLB_MODEL_DIR),
        )

        if self.device != "cpu":
            self.model = self.model.to(self.device)

        logger.info("NLLB model loaded successfully")

    def _get_nllb_language_code(self, lang: str) -> str:
        """
        Map a language code to the NLLB representation.

        Args:
            lang: Language code ('ru' or 'en').

        Returns:
            NLLB-compatible code.
        """
        mapping = {"ru": "rus_Cyrl", "en": "eng_Latn"}
        return mapping.get(lang, "eng_Latn")

    def translate_text(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: str = "ru",
    ) -> str:
        """
        Translate text into the target language.

        Args:
            text: Input text to translate.
            source_lang: Source language code. Auto-detected if None.
            target_lang: Target language code.

        Returns:
            Translated text.
        """
        if not text.strip():
            return text

        if source_lang is None:
            source_lang = detect_language(text)
            logger.info("Detected source language: %s", source_lang)

        if source_lang == target_lang:
            logger.info("Source and target language match; skipping translation")
            return text

        logger.info("Translating from %s to %s", source_lang, target_lang)

        if self.method == TranslateOptions.NLLB:
            return self._translate_with_nllb(text, source_lang, target_lang)
        if self.method == TranslateOptions.OPENAI_API:
            return self._translate_with_openai(text, source_lang, target_lang)

        raise ValueError(f"Unsupported translation method: {self.method}")

    def _translate_with_nllb(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """
        Translate using Meta's NLLB model.

        Args:
            text: Text to translate.
            source_lang: Source language code.
            target_lang: Target language code.

        Returns:
            Translated text.
        """
        self._load_nllb_model()

        chunks = chunk_text(text, max_tokens=700)
        logger.info("Split text into %d chunks for translation", len(chunks))

        translated_chunks: List[str] = []

        src_lang_code = self._get_nllb_language_code(source_lang)
        tgt_lang_code = self._get_nllb_language_code(target_lang)

        for chunk in tqdm(chunks, desc="Translating chunks"):
            self.tokenizer.src_lang = src_lang_code

            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )

            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang_code)

            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=1024,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

            translated_text = self.tokenizer.batch_decode(
                translated_tokens,
                skip_special_tokens=True,
            )[0]

            translated_chunks.append(translated_text)

        result = "\n\n".join(translated_chunks)
        logger.info("Translation completed successfully")

        return result

    def _translate_with_openai(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """
        Translate using OpenAI's GPT models.

        Args:
            text: Text to translate.
            source_lang: Source language code.
            target_lang: Target language code.

        Returns:
            Translated text.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. Install it with: pip install openai>=1.6.0"
            )

        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file or environment."
            )

        logger.info("Using OpenAI API for translation")
        client = OpenAI(api_key=api_key)

        # Map language codes to full names
        lang_names = {
            "ru": "Russian",
            "en": "English",
        }

        source_lang_name = lang_names.get(source_lang, source_lang)
        target_lang_name = lang_names.get(target_lang, target_lang)

        # Split text into chunks (GPT has token limits)
        chunks = chunk_text(text, max_tokens=2000)
        logger.info("Split text into %d chunks for translation", len(chunks))

        translated_chunks: List[str] = []

        system_prompt = f"""You are a professional translator. Translate the following text from {source_lang_name} to {target_lang_name}.

IMPORTANT RULES:
1. Preserve all timestamps in format [MM:SS] or [HH:MM:SS]
2. Preserve all speaker labels like [Speaker 1], [Speaker 2]
3. Maintain paragraph structure and formatting
4. Translate accurately while keeping the tone and style
5. Keep technical terms accurate
6. Do NOT add explanations or comments
7. Return ONLY the translated text"""

        for chunk in tqdm(chunks, desc="Translating chunks"):
            try:
                response = client.chat.completions.create(
                    model="gpt-4",  # Can be configured
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": chunk}
                    ],
                    temperature=0.3,
                    max_tokens=3000,
                )

                translated_text = response.choices[0].message.content.strip()
                translated_chunks.append(translated_text)

            except Exception as e:
                logger.error(f"Error translating chunk: {e}")
                # Fall back to original text on error
                translated_chunks.append(chunk)

        result = "\n\n".join(translated_chunks)
        logger.info("Translation completed successfully")

        return result

    def translate_segments(
        self,
        segments: List,
        source_lang: Optional[str] = None,
        target_lang: str = "ru",
    ) -> List:
        """
        Translate a list of transcription segments.

        Args:
            segments: Iterable of TranscriptionSegment objects or dictionaries.
            source_lang: Source language code.
            target_lang: Target language code.

        Returns:
            List of translated TranscriptionSegment instances.
        """
        from .transcriber import TranscriptionSegment

        logger.info("Translating %d segments...", len(segments))

        translated_segments: List[TranscriptionSegment] = []

        for seg in tqdm(segments, desc="Translating segments"):
            if hasattr(seg, "text"):
                seg_text = seg.text
                seg_start = seg.start
                seg_end = seg.end
                seg_speaker = seg.speaker
            elif isinstance(seg, dict):
                seg_text = seg.get("text", "")
                seg_start = seg.get("start")
                seg_end = seg.get("end")
                seg_speaker = seg.get("speaker")
            else:
                continue

            translated_text = self.translate_text(seg_text, source_lang, target_lang)

            translated_seg = TranscriptionSegment(
                start=seg_start,
                end=seg_end,
                text=translated_text,
                speaker=seg_speaker,
            )
            translated_segments.append(translated_seg)

        logger.info("Segment translation finished")
        return translated_segments
