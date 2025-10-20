"""
Text translation utilities (Meta NLLB and future backends).
"""
from typing import List, Optional

import torch
from tqdm import tqdm

from .config import settings, TranslateOptions
from .logger import logger
from .utils import chunk_text, detect_language
from .api_cache import get_cache, get_openai_rate_limiter
from .retry_handler import retry_api_call
from .cost_tracker import get_cost_tracker


class Translator:
    """Translate text using the configured backend (NLLB by default)."""

    def __init__(self, method: str = TranslateOptions.NLLB, model_name: Optional[str] = None, use_cache: bool = True):
        self.method = method
        self.model_name = model_name if model_name else settings.NLLB_MODEL_NAME
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = self._get_device()
        self.use_cache = use_cache
        self.cache = get_cache() if use_cache else None
        self.rate_limiter = get_openai_rate_limiter() if method == TranslateOptions.OPENAI_API else None
        logger.info("Translator using device: %s", self.device)
        if model_name:
            logger.info("Custom NLLB model specified: %s", model_name)
        if use_cache:
            logger.info("Translation caching enabled")

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
        Translate using OpenAI's GPT models with caching and rate limiting.

        Args:
            text: Text to translate.
            source_lang: Source language code.
            target_lang: Target language code.

        Returns:
            Translated text.
        """
        # Check cache first
        if self.use_cache:
            cache_key = {
                "text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "model": "gpt-4",
                "method": "openai_translation"
            }
            cached_result = self.cache.get("translation", cache_key)
            if cached_result is not None:
                logger.info("Using cached translation")
                return cached_result

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
            translated_text = self._translate_chunk_with_retry(
                client, chunk, system_prompt
            )
            translated_chunks.append(translated_text)

        result = "\n\n".join(translated_chunks)
        logger.info("Translation completed successfully")

        # Cache the result
        if self.use_cache:
            self.cache.set("translation", cache_key, result)

        return result

    @retry_api_call(max_retries=5)
    def _translate_chunk_with_retry(self, client, chunk: str, system_prompt: str) -> str:
        """
        Translate a single chunk with retry logic.

        Args:
            client: OpenAI client instance
            chunk: Text chunk to translate
            system_prompt: System prompt for translation

        Returns:
            Translated text
        """
        try:
            # Apply rate limiting
            if self.rate_limiter:
                self.rate_limiter.wait_if_needed()

            response = client.chat.completions.create(
                model="gpt-4",  # Can be configured
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk}
                ],
                temperature=0.3,
                max_tokens=3000,
            )

            # Track token usage
            if response.usage:
                cost_tracker = get_cost_tracker()
                cost_tracker.add_translation(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error translating chunk: {e}")
            # Fall back to original text on final error
            return chunk

    def translate_segments(
        self,
        segments: List,
        source_lang: Optional[str] = None,
        target_lang: str = "ru",
        batch_size: int = 50,
    ) -> List:
        """
        Translate a list of transcription segments with batching to reduce API costs.

        Args:
            segments: Iterable of TranscriptionSegment objects or dictionaries.
            source_lang: Source language code.
            target_lang: Target language code.
            batch_size: Number of segments to combine per translation request (default: 50).

        Returns:
            List of translated TranscriptionSegment instances.
        """
        from .transcriber import TranscriptionSegment

        logger.info("Translating %d segments with batch size %d...", len(segments), batch_size)

        # Extract segment data
        segment_data = []
        for seg in segments:
            if hasattr(seg, "text"):
                segment_data.append({
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end,
                    "speaker": seg.speaker,
                })
            elif isinstance(seg, dict):
                segment_data.append({
                    "text": seg.get("text", ""),
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "speaker": seg.get("speaker"),
                })
            else:
                continue

        # Process in batches
        translated_segments: List[TranscriptionSegment] = []
        total_batches = (len(segment_data) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(0, len(segment_data), batch_size), desc="Translating segments", total=total_batches):
            batch = segment_data[batch_idx:batch_idx + batch_size]

            # Combine batch segments with markers
            batch_text_parts = []
            for i, seg_data in enumerate(batch):
                # Use unique markers to split later
                marker = f"<<<SEG_{i}>>>"
                batch_text_parts.append(f"{marker}\n{seg_data['text']}")

            combined_text = "\n\n".join(batch_text_parts)

            # Translate the entire batch at once
            translated_combined = self.translate_text(combined_text, source_lang, target_lang)

            # Split back into individual segments
            translated_parts = []
            for i in range(len(batch)):
                marker = f"<<<SEG_{i}>>>"
                if marker in translated_combined:
                    # Find this segment's translation
                    start_idx = translated_combined.find(marker)
                    next_marker = f"<<<SEG_{i+1}>>>"

                    if next_marker in translated_combined:
                        end_idx = translated_combined.find(next_marker)
                        segment_translation = translated_combined[start_idx:end_idx]
                    else:
                        segment_translation = translated_combined[start_idx:]

                    # Remove marker and clean up
                    segment_translation = segment_translation.replace(marker, "").strip()
                    translated_parts.append(segment_translation)
                else:
                    # Fallback: use original text if marker not found
                    logger.warning(f"Marker not found for segment {i} in batch {batch_idx // batch_size}, using original")
                    translated_parts.append(batch[i]["text"])

            # Create TranscriptionSegment objects
            for seg_data, translated_text in zip(batch, translated_parts):
                translated_seg = TranscriptionSegment(
                    start=seg_data["start"],
                    end=seg_data["end"],
                    text=translated_text,
                    speaker=seg_data["speaker"],
                )
                translated_segments.append(translated_seg)

        logger.info("Segment translation finished (%d batches processed)", total_batches)
        return translated_segments
