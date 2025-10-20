"""
Text summarization module supporting Ollama and OpenAI backends.
"""
from typing import Optional
import requests
from tqdm import tqdm

from .logger import logger
from .config import SummarizeOptions, settings
from .utils import chunk_text
from .api_cache import get_cache, get_openai_rate_limiter
from .retry_handler import retry_api_call
from .cost_tracker import get_cost_tracker


class Summarizer:
    """Summarize text using LLMs (Ollama or OpenAI)."""

    def __init__(
        self,
        backend: str = SummarizeOptions.OLLAMA,
        model_name: str = "qwen2.5:3b",
        ollama_url: str = "http://localhost:11434",
        use_cache: bool = True
    ):
        """
        Initialize the summarizer.

        Args:
            backend: Backend to use (ollama or openai_api)
            model_name: Model name (for Ollama: qwen2.5:3b, for OpenAI: gpt-4, gpt-3.5-turbo)
            ollama_url: Ollama server URL (only for Ollama backend)
            use_cache: Enable caching (default True)
        """
        self.backend = backend
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_endpoint = f"{ollama_url}/api/generate"
        self.use_cache = use_cache
        self.cache = get_cache() if use_cache else None
        self.rate_limiter = get_openai_rate_limiter() if backend == SummarizeOptions.OPENAI_API else None

        # Check backend availability
        if self.backend == SummarizeOptions.OLLAMA:
            self._check_ollama_available()
        elif self.backend == SummarizeOptions.OPENAI_API:
            self._check_openai_available()
        else:
            raise ValueError(f"Unsupported summarization backend: {self.backend}")

        if use_cache:
            logger.info("Summarization caching enabled")

    def _check_ollama_available(self):
        """Check if Ollama server is available."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"Ollama server available at {self.ollama_url}")

                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]

                if self.model_name not in model_names:
                    error_msg = f"\n{'='*60}\n"
                    error_msg += f"❌ ERROR: Model '{self.model_name}' not found in Ollama\n"
                    error_msg += f"{'='*60}\n\n"

                    if model_names:
                        error_msg += f"Available models:\n"
                        for name in model_names:
                            error_msg += f"  - {name}\n"
                        error_msg += f"\n"
                    else:
                        error_msg += f"No models loaded in Ollama\n\n"

                    error_msg += f"To download the model, run:\n"
                    error_msg += f"  ollama pull {self.model_name}\n\n"
                    error_msg += f"Recommended models:\n"
                    error_msg += f"  - qwen2.5:3b  (fast, ~2GB)\n"
                    error_msg += f"  - qwen2.5:7b  (better quality, ~4.7GB)\n"
                    error_msg += f"  - llama3:8b   (good alternative, ~4.7GB)\n"
                    error_msg += f"{'='*60}"

                    logger.error(error_msg)
                    raise RuntimeError(f"Model '{self.model_name}' not found in Ollama")
                else:
                    logger.info(f"Model '{self.model_name}' found")
            else:
                raise Exception("Ollama not responding")
        except requests.exceptions.RequestException as e:
            error_msg = f"\n{'='*60}\n"
            error_msg += f"❌ ERROR: Cannot connect to Ollama server\n"
            error_msg += f"{'='*60}\n\n"
            error_msg += f"Reason: {e}\n\n"
            error_msg += f"Solution:\n"
            error_msg += f"1. Check if Ollama is installed:\n"
            error_msg += f"   https://ollama.com/download\n\n"
            error_msg += f"2. Start the Ollama server:\n"
            error_msg += f"   ollama serve\n\n"
            error_msg += f"3. Or just launch the Ollama application\n"
            error_msg += f"{'='*60}"

            logger.error(error_msg)
            raise RuntimeError(f"Cannot connect to Ollama: {e}")

    def _check_openai_available(self):
        """Check if OpenAI API is available."""
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            error_msg = f"\n{'='*60}\n"
            error_msg += f"❌ ERROR: OPENAI_API_KEY not found\n"
            error_msg += f"{'='*60}\n\n"
            error_msg += f"To use OpenAI API:\n"
            error_msg += f"1. Get API key at https://platform.openai.com/api-keys\n"
            error_msg += f"2. Add it to .env file:\n"
            error_msg += f"   OPENAI_API_KEY=your-api-key-here\n"
            error_msg += f"{'='*60}"

            logger.error(error_msg)
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            logger.info("OpenAI API available")
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. Install it with: pip install openai>=1.6.0"
            )
        except Exception as e:
            logger.error(f"Error checking OpenAI API: {e}")
            raise

    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API.

        Args:
            prompt: Prompt for the model

        Returns:
            Model response
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 4000,
            }
        }

        try:
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=600  # 10 minute timeout for summarization
            )
            response.raise_for_status()

            result = response.json()
            return result.get('response', '').strip()
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            raise

    @retry_api_call(max_retries=5)
    def _call_openai(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call OpenAI API with caching and rate limiting.

        Args:
            prompt: Prompt for the model
            system_prompt: System prompt (optional)

        Returns:
            Model response
        """
        # Check cache first
        if self.use_cache:
            cache_key = {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "model": self.model_name,
                "method": "openai_summarize"
            }
            cached_result = self.cache.get("summarization", cache_key)
            if cached_result is not None:
                logger.debug("Using cached summarization result")
                return cached_result

        try:
            from openai import OpenAI

            client = OpenAI(api_key=settings.OPENAI_API_KEY)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Apply rate limiting
            if self.rate_limiter:
                self.rate_limiter.wait_if_needed()

            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=4000,
            )

            result = response.choices[0].message.content.strip()

            # Track token usage
            if response.usage:
                cost_tracker = get_cost_tracker()
                cost_tracker.add_summarization(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )

            # Cache the result
            if self.use_cache:
                self.cache.set("summarization", cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise

    def summarize(
        self,
        text: str,
        language: str = "ru",
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        Summarize text.

        Args:
            text: Text to summarize
            language: Target language for summary ('ru' or 'en')
            custom_prompt: Custom instructions for summarization (optional)

        Returns:
            Summary text
        """
        logger.info(f"Starting summarization with {self.backend} backend...")

        # Prepare the prompt based on language
        if language == "ru":
            base_prompt = """Сделай подробный пересказ на русском языке следующего текста.

ТРЕБОВАНИЯ:
1. Структурируй пересказ по разделам с заголовками
2. Сохрани все ключевые идеи и аргументы
3. Включи важные примеры и цифры
4. Используй понятный русский язык
5. Объём: примерно 20-30% от оригинала

"""
        else:
            base_prompt = """Provide a detailed summary of the following text in English.

REQUIREMENTS:
1. Structure the summary with section headings
2. Preserve all key ideas and arguments
3. Include important examples and numbers
4. Use clear English
5. Length: approximately 20-30% of the original

"""

        if custom_prompt:
            base_prompt += f"\nДОПОЛНИТЕЛЬНЫЕ ИНСТРУКЦИИ:\n{custom_prompt}\n\n"

        base_prompt += f"ТЕКСТ:\n\n{text}\n\nПЕРЕСКАЗ:"

        # Call the appropriate backend
        try:
            if self.backend == SummarizeOptions.OPENAI_API:
                system_prompt = "You are a professional summarizer who creates structured, detailed summaries."
                summary = self._call_openai(base_prompt, system_prompt)
            else:
                summary = self._call_ollama(base_prompt)

            logger.info("Summarization completed successfully")
            return summary.strip()

        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            raise

    def summarize_long_text(
        self,
        text: str,
        language: str = "ru",
        custom_prompt: Optional[str] = None,
        max_chunk_tokens: int = 8000
    ) -> str:
        """
        Summarize long text by splitting it into chunks and summarizing each chunk.

        Args:
            text: Text to summarize
            language: Target language for summary ('ru' or 'en')
            custom_prompt: Custom instructions for summarization (optional)
            max_chunk_tokens: Maximum tokens per chunk (words for estimation)

        Returns:
            Combined summary
        """
        logger.info("Summarizing long text...")

        # Adjust chunk size based on model context window
        # For OpenAI backend, check model context limits
        if self.backend == SummarizeOptions.OPENAI_API:
            # GPT-4 (8k context): ~4000 input tokens + 4000 output = safe limit ~3000 words
            # GPT-4-turbo/GPT-4o (128k): can handle larger chunks
            if "gpt-4-turbo" in self.model_name.lower() or "gpt-4o" in self.model_name.lower():
                # Large context models - can use larger chunks
                safe_chunk_size = min(max_chunk_tokens, 20000)
            elif "gpt-3.5" in self.model_name.lower():
                # GPT-3.5-turbo (16k context) - moderate chunks
                safe_chunk_size = min(max_chunk_tokens, 6000)
            else:
                # Standard GPT-4 (8k context) - conservative chunks
                # ~1 word ≈ 1.3 tokens, so 3000 words ≈ 4000 tokens input + 4000 output
                safe_chunk_size = min(max_chunk_tokens, 3000)
                if max_chunk_tokens > 3000:
                    logger.warning(
                        "Reducing chunk size from %d to %d words for GPT-4 8k context window. "
                        "Consider using gpt-4-turbo or gpt-4o for larger chunks.",
                        max_chunk_tokens, safe_chunk_size
                    )
        else:
            # Ollama - use configured chunk size
            safe_chunk_size = max_chunk_tokens

        # Split into chunks
        chunks = chunk_text(text, max_tokens=safe_chunk_size)
        logger.info(f"Split text into {len(chunks)} chunks (max {safe_chunk_size} words per chunk)")

        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(tqdm(chunks, desc="Summarizing chunks"), 1):
            logger.info(f"Summarizing chunk {i}/{len(chunks)}")
            summary = self.summarize(chunk, language=language, custom_prompt=custom_prompt)
            chunk_summaries.append(summary)

        # If we have multiple summaries, combine them
        if len(chunk_summaries) > 1:
            logger.info("Combining chunk summaries...")
            combined_text = "\n\n".join(chunk_summaries)

            # Create final summary from chunk summaries
            if language == "ru":
                final_prompt = f"""Объедини следующие части пересказа в один связный текст на русском языке.
Сохрани всю важную информацию из всех частей.

ЧАСТИ ПЕРЕСКАЗА:

{combined_text}

ИТОГОВЫЙ ПЕРЕСКАЗ:"""
            else:
                final_prompt = f"""Combine the following summary parts into one coherent text in English.
Preserve all important information from all parts.

SUMMARY PARTS:

{combined_text}

FINAL SUMMARY:"""

            if self.backend == SummarizeOptions.OPENAI_API:
                system_prompt = "You are a professional editor combining summaries into a coherent whole."
                final_summary = self._call_openai(final_prompt, system_prompt)
            else:
                final_summary = self._call_ollama(final_prompt)

            return final_summary.strip()
        else:
            return chunk_summaries[0]
