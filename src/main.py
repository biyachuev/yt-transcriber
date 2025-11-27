"""
Main application module.
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

# Suppress third-party library deprecation warnings early
# Must be set before importing modules that trigger warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyannote")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchaudio")

from .config import (
    settings,
    TranscribeOptions,
    TranslateOptions,
    RefineOptions,
    SummarizeOptions,
)
from .logger import logger, format_warning
from .downloader import YouTubeDownloader
from .transcriber import Transcriber
from .document_writer import DocumentWriter
from .utils import (
    detect_language,
    sanitize_filename,
    create_whisper_prompt,
    create_whisper_prompt_with_llm,
    format_log_preview,
)
from .text_reader import TextReader
from .video_processor import VideoProcessor
from .cost_tracker import get_cost_tracker
from .api_cache import get_cache
from .yt_dlp_updater import check_and_update_yt_dlp
from .api_validator import validate_api_keys_for_operation


def _print_session_summary():
    """Print cost tracking and cache statistics summary."""
    # Print cost tracking summary
    tracker = get_cost_tracker()
    tracker.print_summary()

    # Print cache statistics
    cache = get_cache()
    stats = cache.get_stats()
    logger.info("\n" + "=" * 60)
    logger.info("Cache Statistics")
    logger.info("=" * 60)
    logger.info(
        f"Total entries:    {stats['total_entries']}/{stats['max_entries']} ({stats['usage_percentage']}%)"
    )
    logger.info(f"Cache size:       {stats['total_size_mb']} MB")
    logger.info(f"TTL:              {stats['ttl_days']} days")
    logger.info("=" * 60)


def load_prompt_from_file(prompt_file_path: str) -> str:
    """
    Load a Whisper prompt from a text file.

    Args:
        prompt_file_path: Path to the prompt file.

    Returns:
        Prompt text trimmed to Whisper limits.
    """
    try:
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        # Limit prompt length to the Whisper constraint.
        MAX_PROMPT_LENGTH = 800
        if len(prompt) > MAX_PROMPT_LENGTH:
            logger.warning(
                "Prompt loaded from file is too long (%d chars), trimming to %d",
                len(prompt),
                MAX_PROMPT_LENGTH,
            )
            prompt = prompt[:MAX_PROMPT_LENGTH]

        logger.info("Loaded custom prompt from file (%d chars)", len(prompt))
        logger.debug("Prompt preview (first 80 chars): %s", format_log_preview(prompt))

        return prompt
    except FileNotFoundError:
        logger.error("Prompt file not found: %s", prompt_file_path)
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to read prompt file: %s", e)
        sys.exit(1)


def print_help():
    """Print CLI usage instructions."""
    help_text = """
YouTube Transcriber & Translator
================================

Usage:
    python -m src.main <command> [options]

Commands:
    youtube         Process a YouTube video
    audio           Process a local audio file
    video           Process a local video file
    text            Process a text document (.docx, .md, .txt)

Run 'python -m src.main <command> --help' for command-specific options.

Examples:
    # Transcribe and translate a YouTube video
    python -m src.main youtube --url "https://youtube.com/watch?v=..." --transcribe whisper-base --translate nllb
    python -m src.main youtube --url "https://youtube.com/watch?v=..." --transcribe gigaam-e2e-rnnt --translate nllb

    # Process a local audio file with refinement
    python -m src.main audio --input audio.mp3 --transcribe whisper-medium --refine-model qwen2.5:3b

    # Process a video file
    python -m src.main video --input video.mp4 --transcribe whisper-base --translate nllb

    # Translate an existing document
    python -m src.main text --input document.docx --translate nllb

    # Full pipeline with summarization
    python -m src.main youtube --url "..." --transcribe whisper-medium --translate nllb \\
        --refine-model qwen2.5:3b --summarize-model qwen2.5:7b

Global Options:
    --help, -h      Show this message

Common Options (available for all commands):
    --transcribe METHOD         Transcription method (whisper-base, whisper-small, whisper-medium, whisper-openai-api, gigaam-e2e-rnnt, gigaam-e2e-ctc)
    --language CODE             Force source language (e.g., ru or en). Defaults to auto-detect.
    --translate METHOD          Translation method (nllb, openai-api)
    --prompt-file PATH          Custom Whisper prompt file
    --refine-model MODEL        Model for transcript refinement (e.g. qwen2.5:3b, gpt-4)
    --refine-backend BACKEND    Backend for transcript refinement (ollama, openai-api)
    --refine-translation MODEL  Model for translation refinement (Ollama)
    --summarize-model MODEL     Model for summarization
    --summarize-backend BACKEND Backend for summarization (ollama, openai-api)
    --nllb-model MODEL          NLLB model (default: facebook/nllb-200-distilled-1.3B)
    --speakers                  Enable speaker diarization

Notes:
    - Results are stored in 'output/' (.docx and .md)
    - Temporary files go to 'temp/'
    - Logs are written to 'logs/'
    - First run downloads models (~2-5 GB)
    - YouTube processing automatically checks for yt-dlp updates before downloading
      (keeps yt-dlp up-to-date to prevent HTTP 403 errors)
    """
    print(help_text)


def _generate_summary(
    title: str, segments: list, summarize_model: str, summarize_backend: str
) -> None:
    """
    Generate and save a summary of the text in both MD and DOCX formats.

    Args:
        title: Base title for the output file.
        segments: List of text segments to summarize (TranscriptionSegment objects or dicts).
        summarize_model: Model to use for summarization.
        summarize_backend: Backend to use (ollama or openai_api).
    """
    logger.info("\n[Final] Generating summary with %s...", summarize_model)
    try:
        from .summarizer import Summarizer
        from .config import settings
        from .translator import detect_language

        summarizer = Summarizer(backend=summarize_backend, model_name=summarize_model)

        # Combine text from segments
        # Handle both TranscriptionSegment objects and dict segments
        text_to_summarize = "\n\n".join(
            [seg.text if hasattr(seg, "text") else seg["text"] for seg in segments]
        )

        # Detect language for summary
        detected_lang = detect_language(text_to_summarize)
        summary_lang = "ru" if detected_lang in ["ru", "uk", "be"] else "en"

        summary = summarizer.summarize_long_text(
            text_to_summarize, language=summary_lang
        )

        # Save summary as both MD and DOCX
        from .document_writer import DocumentWriter

        writer = DocumentWriter()
        summary_title = f"{title}_summary"

        # Create sections for document writer
        sections = [
            {
                "title": "Summary",
                "method": f"Generated by {summarize_model} ({summarize_backend})",
                "content": summary,
            }
        ]

        docx_path, md_path = writer.create_documents(
            title=summary_title, sections=sections
        )

        logger.info("Summary saved:")
        logger.info("  - %s", docx_path)
        logger.info("  - %s", md_path)
    except Exception as e:
        logger.error("Failed to generate summary: %s", e)


def process_text_file(
    text_path: str,
    translate_methods: Optional[list[str]] = None,
    refine_model: Optional[str] = None,
    refine_translation_model: Optional[str] = None,
    translate_model: Optional[str] = None,
    openai_translate_model: Optional[str] = None,
    refine_backend: str = "ollama",
    summarize: bool = False,
    summarize_model: Optional[str] = None,
    summarize_backend: str = "ollama",
):
    """
    Process an existing text document (.docx, .md, .txt).

    Args:
        text_path: Path to the source document.
        translate_methods: Translation backends to run.
        refine_model: Model for text refinement.
        refine_translation_model: Model for translation refinement.
        refine_backend: Backend for refinement (ollama or openai_api).
        summarize: Whether to generate a summary.
        summarize_model: Model for summarization.
        summarize_backend: Backend for summarization (ollama or openai_api).
    """
    logger.info("=" * 60)
    logger.info("Starting text document processing")
    logger.info("=" * 60)

    text_path_obj = Path(text_path)
    text_title = sanitize_filename(text_path_obj.stem)

    # 1. Read source text.
    logger.info(f"\n[1/3] Reading document: {text_path}")

    text_reader = TextReader()
    try:
        text_content = text_reader.read_file(text_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Unable to read document: %s", e)
        return

    # Detect input language.
    detected_language = text_reader.detect_language(text_content)
    logger.info("Detected language: %s", format_warning(str(detected_language).upper()))

    # Build pseudo-segments to reuse the existing pipeline.
    paragraphs = [p.strip() for p in text_content.split("\n\n") if p.strip()]

    original_segments = []
    for i, para in enumerate(paragraphs):
        if para:
            original_segments.append(
                {"text": para, "start": None, "end": None, "speaker": None}
            )

    logger.info("Document split into %d paragraphs", len(original_segments))

    # 1.5. Optional LLM-based refinement.
    refined_segments = None
    if refine_model:
        logger.info("\n[1.5/3] Refining text with %s...", refine_model)

        try:
            from .text_refiner import TextRefiner

            refiner = TextRefiner(backend=refine_backend, model_name=refine_model)
            refined_text = refiner.refine_text(text_content)
            refined_paragraphs = [
                p.strip() for p in refined_text.split("\n\n") if p.strip()
            ]

            refined_segments = []
            for para in refined_paragraphs:
                if para:
                    refined_segments.append(
                        {"text": para, "start": None, "end": None, "speaker": None}
                    )

            logger.info("Refinement complete (%d paragraphs)", len(refined_segments))

        except ImportError:
            logger.warning("text_refiner module is not available, skipping refinement")
        except Exception as e:
            logger.error("Failed to refine document: %s", e)
            logger.warning("Continuing with the original text")

    # 2. Translation.
    translated_segments_dict = {}
    refined_translation_segments_dict = {}
    if translate_methods:
        logger.info("\n[2/3] Translating text...")

        for method in translate_methods:
            logger.info("\n  Translation method: %s", method)
            from .translator import Translator

            # Use appropriate model based on translation method
            model_override = (
                openai_translate_model
                if method == TranslateOptions.OPENAI_API
                else translate_model
            )
            translator = Translator(method=method, model_name=model_override)
            method_key = method
            if method == "NLLB":
                method_key = f"{method} ({translator.model_name})"
            elif method == TranslateOptions.OPENAI_API:
                method_key = f"OpenAI API ({translator.model_name})"
            segments_to_translate = (
                refined_segments if refined_segments else original_segments
            )

            try:
                translated_segments = translator.translate_segments(
                    segments_to_translate,
                    source_lang=detected_language,
                    target_lang="ru" if detected_language == "en" else "en",
                )

                translated_segments_dict[method_key] = translated_segments

            except Exception as e:
                logger.error("Translation failed with %s: %s", method, e)
                continue

            # Optional LLM post-translation polish.
            if refine_translation_model and method_key in translated_segments_dict:
                logger.info(
                    "  Refining translation with %s...", refine_translation_model
                )

                try:
                    from .text_refiner import TextRefiner
                    from .transcriber import TranscriptionSegment

                    translation_refiner = TextRefiner(
                        model_name=refine_translation_model
                    )
                    translated_text = "\n\n".join(
                        [seg.text for seg in translated_segments_dict[method_key]]
                    )
                    refined_translation = translation_refiner.refine_translation(
                        translated_text
                    )
                    refined_translation_paragraphs = [
                        p.strip()
                        for p in refined_translation.split("\n\n")
                        if p.strip()
                    ]

                    # Rebuild segments for the refined translation.
                    refined_translated_segments = []
                    for i, para in enumerate(refined_translation_paragraphs):
                        if i < len(translated_segments_dict[method_key]):
                            refined_translated_segments.append(
                                TranscriptionSegment(
                                    text=para,
                                    start=translated_segments_dict[method_key][i].start,
                                    end=translated_segments_dict[method_key][i].end,
                                    speaker=translated_segments_dict[method_key][
                                        i
                                    ].speaker,
                                )
                            )

                    refined_translation_segments_dict[method_key] = (
                        refined_translated_segments
                    )
                    logger.info("  Translation refinement complete")

                except Exception as e:
                    logger.error("Failed to refine translation: %s", e)
                    logger.warning("Falling back to the unrefined translation")

    # 3. Export documents.
    logger.info("\n[3/3] Generating documents...")

    writer = DocumentWriter()

    # Create a refined-only document if available.
    if refined_segments:
        logger.info("  Creating refined document...")
        docx_path_refined, md_path_refined = writer.create_from_segments(
            title=f"{text_title}_refined",
            transcription_segments=refined_segments,
            translation_segments=None,
            transcribe_method=f"Refined with {refine_model}",
            translate_method="",
            with_timestamps=False,
            with_speakers=False,
            description="Улучшенный текст",
        )
        logger.info("  Saved refined markdown: %s", md_path_refined)

    # Create translated documents.
    if translated_segments_dict:
        for method, translated_segs in translated_segments_dict.items():
            logger.info("  Creating translated document (%s)...", method)

            trans_desc = (
                "Улучшенный текст + перевод"
                if refined_segments
                else "Оригинальный текст + перевод"
            )
            docx_path_trans, md_path_trans = writer.create_from_segments(
                title=f"{text_title}_translated_{method}",
                transcription_segments=(
                    refined_segments if refined_segments else original_segments
                ),
                translation_segments=translated_segs,
                transcribe_method=f"Loaded from {text_path_obj.suffix}"
                + (f" + {refine_model}" if refine_model else ""),
                translate_method=method,
                with_timestamps=False,
                with_speakers=False,
                description=trans_desc,
            )
            logger.info("  Saved translation markdown: %s", md_path_trans)

    # Create refined translation documents.
    if refined_translation_segments_dict:
        for (
            method,
            refined_translated_segs,
        ) in refined_translation_segments_dict.items():
            logger.info("  Creating refined translation document (%s)...", method)

            trans_refined_desc = (
                "Улучшенный текст + улучшенный перевод (дополнительно отполированный через LLM)"
                if refined_segments
                else "Оригинальный текст + улучшенный перевод (дополнительно отполированный через LLM)"
            )
            docx_path_trans_refined, md_path_trans_refined = (
                writer.create_from_segments(
                    title=f"{text_title}_translated_{method}_refined",
                    transcription_segments=(
                        refined_segments if refined_segments else original_segments
                    ),
                    translation_segments=refined_translated_segs,
                    transcribe_method=f"Loaded from {text_path_obj.suffix}"
                    + (f" + {refine_model}" if refine_model else ""),
                    translate_method=f"{method} + {refine_translation_model}",
                    with_timestamps=False,
                    with_speakers=False,
                    description=trans_refined_desc,
                )
            )
            logger.info(
                "  Saved refined translation markdown: %s", md_path_trans_refined
            )

    # Generate summary if requested.
    if summarize and summarize_model:
        _generate_summary(
            title=text_title,
            segments=refined_segments if refined_segments else original_segments,
            summarize_model=summarize_model,
            summarize_backend=summarize_backend,
        )

    # Warn if no output was produced.
    if (
        not refined_segments
        and not translated_segments_dict
        and not (summarize and summarize_model)
    ):
        logger.warning(
            "No --refine-model, --translate, or --summarize options were provided"
        )
        logger.info(
            "The source file is unchanged. Use --refine-model, --translate, or --summarize to generate output."
        )

    logger.info("\n" + "=" * 60)
    logger.info("Text document processing complete!")
    logger.info("=" * 60)

    # Print session summary
    _print_session_summary()


def validate_args(command: str, args) -> bool:
    """
    Validate CLI arguments for a specific command.

    Args:
        command: The subcommand being executed (youtube, audio, video, text).
        args: Parsed argparse namespace.

    Returns:
        True if validation succeeded.
    """
    # Validate input file exists for file-based commands
    if command == "audio" and hasattr(args, "input") and args.input:
        if not Path(args.input).exists():
            logger.error("Audio file not found: %s", args.input)
            return False

    if command == "video" and hasattr(args, "input") and args.input:
        if not Path(args.input).exists():
            logger.error("Video file not found: %s", args.input)
            return False

    if command == "text" and hasattr(args, "input") and args.input:
        if not Path(args.input).exists():
            logger.error("Text file not found: %s", args.input)
            return False

    # Audio/YouTube/Video sources require a transcription backend
    if command in ["youtube", "audio", "video"] and not args.transcribe:
        logger.error("%s command requires --transcribe to be set", command.capitalize())
        return False

    # Validate language override
    if hasattr(args, "language") and args.language:
        if args.language.lower() not in ("ru", "en"):
            logger.error("Unsupported language code '%s'. Use 'ru' or 'en'.", args.language)
            return False

    # Validate refinement parameters early (before expensive operations)
    if hasattr(args, "refine_model") and args.refine_model:
        if args.refine_backend == "openai_api":
            # Check OpenAI API key.
            if not settings.OPENAI_API_KEY:
                logger.error(
                    "--refine-backend openai_api requires OPENAI_API_KEY in environment"
                )
                logger.error("Set it in .env file: OPENAI_API_KEY=your-key-here")
                return False

            # Validate OpenAI library availability.
            try:
                import openai
            except ImportError:
                logger.error(
                    "OpenAI library not installed. Install it with: pip install openai>=1.6.0"
                )
                return False

        elif args.refine_backend == "ollama":
            # Check Ollama server availability and model existence.
            try:
                import requests

                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code != 200:
                    logger.error(
                        "Cannot connect to Ollama server at http://localhost:11434"
                    )
                    logger.error("Please start Ollama: ollama serve")
                    return False

                # Check if the model exists.
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]

                if args.refine_model not in model_names:
                    logger.error("Model '%s' not found in Ollama", args.refine_model)
                    logger.error(
                        "Available models: %s",
                        ", ".join(model_names) if model_names else "none",
                    )
                    logger.error("To download: ollama pull %s", args.refine_model)
                    return False

            except requests.exceptions.RequestException as e:
                logger.error("Cannot connect to Ollama server: %s", e)
                logger.error("Please start Ollama: ollama serve")
                return False
            except Exception as e:
                logger.error("Error checking Ollama: %s", e)
                return False

    # Validate translation refinement parameters.
    if args.refine_translation:
        # Translation refinement currently only supports Ollama.
        try:
            import requests

            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                logger.error("Cannot connect to Ollama server for --refine-translation")
                logger.error("Please start Ollama: ollama serve")
                return False

            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]

            if args.refine_translation not in model_names:
                logger.error("Model '%s' not found in Ollama", args.refine_translation)
                logger.error(
                    "Available models: %s",
                    ", ".join(model_names) if model_names else "none",
                )
                logger.error("To download: ollama pull %s", args.refine_translation)
                return False

        except requests.exceptions.RequestException as e:
            logger.error("Cannot connect to Ollama server: %s", e)
            return False

    # Validate summarization parameters.
    if hasattr(args, "summarize_model") and args.summarize_model:
        if args.summarize_backend == "openai_api":
            if not settings.OPENAI_API_KEY:
                logger.error(
                    "--summarize-backend openai_api requires OPENAI_API_KEY in environment"
                )
                return False

            try:
                import openai
            except ImportError:
                logger.error(
                    "OpenAI library not installed. Install it with: pip install openai>=1.6.0"
                )
                return False

        elif args.summarize_backend == "ollama":
            try:
                import requests

                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code != 200:
                    logger.error("Cannot connect to Ollama server for --summarize")
                    return False

                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]

                if args.summarize_model not in model_names:
                    logger.error("Model '%s' not found in Ollama", args.summarize_model)
                    logger.error(
                        "Available models: %s",
                        ", ".join(model_names) if model_names else "none",
                    )
                    logger.error("To download: ollama pull %s", args.summarize_model)
                    return False

            except requests.exceptions.RequestException as e:
                logger.error("Cannot connect to Ollama server: %s", e)
                return False

    # Validate translation backend if specified
    if hasattr(args, "translate") and args.translate:
        translate_methods = [m.strip() for m in args.translate.split(",")]

        for method in translate_methods:
            # Normalize method names (both nllb and NLLB are acceptable)
            method_lower = method.lower()
            if method_lower not in ["nllb", "openai_api", "openai-api"]:
                logger.error("Unknown translation method: %s", method)
                logger.error("Available methods: nllb, openai-api")
                return False

            if method_lower in ["openai_api", "openai-api"]:
                if not settings.OPENAI_API_KEY:
                    logger.error(
                        "Translation method 'openai-api' requires OPENAI_API_KEY"
                    )
                    return False

                try:
                    import openai
                except ImportError:
                    logger.error(
                        "OpenAI library not installed. Install it with: pip install openai>=1.6.0"
                    )
                    return False

    # Validate transcription backend if specified
    if hasattr(args, "transcribe") and args.transcribe:
        # Normalize method names (accept both whisper_base and whisper-base)
        method = args.transcribe.replace("-", "_")
        valid_transcribe_methods = [
            "whisper_base",
            "whisper_small",
            "whisper_medium",
            "whisper_openai_api",
            "gigaam_e2e_rnnt",
            "gigaam_e2e_ctc",
        ]
        if method not in valid_transcribe_methods:
            logger.error("Unknown transcription method: %s", args.transcribe)
            logger.error(
                "Available methods: whisper-base, whisper-small, whisper-medium, whisper-openai-api, gigaam-e2e-rnnt, gigaam-e2e-ctc"
            )
            return False

        if method == "whisper_openai_api":
            if not settings.OPENAI_API_KEY:
                logger.error(
                    "Transcription method 'whisper-openai-api' requires OPENAI_API_KEY"
                )
                return False

            try:
                import openai
            except ImportError:
                logger.error(
                    "OpenAI library not installed. Install it with: pip install openai>=1.6.0"
                )
                return False

    # Validate API keys with actual API calls early
    # Extract parameters from args
    transcribe_method = getattr(args, "transcribe", None)
    if transcribe_method:
        transcribe_method = transcribe_method.replace("-", "_")

    translate_methods = None
    if hasattr(args, "translate") and args.translate:
        translate_methods = [
            m.strip().replace("-", "_") for m in args.translate.split(",")
        ]

    refine_backend = getattr(args, "refine_backend", None)
    summarize_backend = getattr(args, "summarize_backend", None)

    # Validate all required API keys
    if not validate_api_keys_for_operation(
        transcribe_method=transcribe_method,
        translate_methods=translate_methods,
        refine_backend=refine_backend,
        summarize_backend=summarize_backend,
    ):
        logger.error(
            "API key validation failed. Please fix the issues above before continuing."
        )
        return False

    return True


def process_youtube_video(
    url: str,
    transcribe_method: str,
    language: Optional[str] = None,
    translate_methods: Optional[list[str]] = None,
    with_speakers: bool = False,
    custom_prompt: Optional[str] = None,
    refine_model: Optional[str] = None,
    refine_translation_model: Optional[str] = None,
    translate_model: Optional[str] = None,
    openai_translate_model: Optional[str] = None,
    refine_backend: str = "ollama",
    summarize: bool = False,
    summarize_model: Optional[str] = None,
    summarize_backend: str = "ollama",
):
    """
    Process a YouTube video end-to-end.

    Args:
        url: Video URL.
        transcribe_method: Whisper backend to use.
        language: Force source language (ru/en); default is auto-detect.
        translate_methods: Translation backends to run.
        with_speakers: Whether to enable speaker diarisation.
        custom_prompt: Optional custom Whisper prompt.
        refine_model: Model for transcript refinement.
        refine_translation_model: Model for translation refinement.
        refine_backend: Backend for refinement (ollama or openai_api).
        summarize: Whether to generate a summary.
        summarize_model: Model for summarization.
        summarize_backend: Backend for summarization (ollama or openai_api).
    """
    logger.info("=" * 60)
    logger.info("Starting YouTube processing")
    logger.info("=" * 60)

    # 1. Download audio and metadata.
    logger.info("\n[1/4] Downloading audio from YouTube...")
    downloader = YouTubeDownloader()
    audio_path, video_title, duration, metadata = downloader.download_audio(url)

    # 2. Transcription.
    logger.info("\n[2/4] Transcribing audio...")

    # Decide whether to use a custom prompt or generate one.
    if custom_prompt:
        whisper_prompt = custom_prompt
        logger.info("Using custom prompt supplied by user")
    else:
        # Try to use LLM for prompt generation if available
        llm_available = False

        if refine_model:
            if refine_backend == "openai_api":
                # OpenAI is always available if API key is set
                from .config import settings

                llm_available = bool(settings.OPENAI_API_KEY)
            else:
                # Check if Ollama is available
                try:
                    import requests

                    llm_available = (
                        requests.get(
                            "http://localhost:11434/api/tags", timeout=2
                        ).status_code
                        == 200
                    )
                except:
                    llm_available = False

        if llm_available and refine_model:
            # Reuse the refinement model to build the prompt.
            whisper_prompt = create_whisper_prompt_with_llm(
                metadata,
                use_ollama=(refine_backend != "openai_api"),
                model=refine_model,
                backend=refine_backend,
            )
            logger.info("Prompt generated from metadata via LLM (%s)", refine_backend)
        else:
            whisper_prompt = create_whisper_prompt(metadata)
            logger.info("Prompt generated from metadata (standard method)")

    transcriber = Transcriber(method=transcribe_method)
    transcription_segments = transcriber.transcribe(
        audio_path,
        language=language,  # Auto-detect when None
        with_speakers=with_speakers,
        initial_prompt=whisper_prompt,
    )

    # Retain original segments for comparison/export.
    original_transcription_segments = transcription_segments

    # 2.5. Optional LLM refinement.
    refined_transcription_segments = None
    if refine_model:
        logger.info("\n[2.5/4] Refining transcript with %s...", refine_model)
        try:
            from .text_refiner import TextRefiner

            refiner = TextRefiner(backend=refine_backend, model_name=refine_model)

            # Extract plain text.
            original_text = transcriber.segments_to_text(transcription_segments)

            # Improve text (use prompt as context).
            refined_text = refiner.refine_text(original_text, context=whisper_prompt)

            # Rebuild segments from the refined text.
            refined_transcription_segments = transcriber.update_segments_from_text(
                transcription_segments, refined_text
            )

            logger.info("Transcript refined")
        except Exception as e:
            logger.error("Failed to refine transcript: %s", e)
            logger.warning("Continuing with the original transcript")
            refined_transcription_segments = None

    # 3. Translation (optional).
    translation_segments = None
    translation_segments_refined = None
    translate_method_str = ""

    if translate_methods:
        logger.info("\n[3/4] Translating text...")

        # Determine source language.
        original_text = transcriber.segments_to_text(original_transcription_segments)
        source_lang = detect_language(original_text)

        # Use the first translation backend (MVP approach).
        translate_method = translate_methods[0]

        translate_method_str = translate_method

        from .translator import Translator

        if source_lang == "en":
            # Use appropriate model based on translation method
            model_override = (
                openai_translate_model
                if translate_method == TranslateOptions.OPENAI_API
                else translate_model
            )
            translator = Translator(method=translate_method, model_name=model_override)

            if translate_method == "NLLB":
                translate_method_str = f"{translate_method} ({translator.model_name})"
            elif translate_method == TranslateOptions.OPENAI_API:
                translate_method_str = f"OpenAI API ({translator.model_name})"

            # Prefer translating the refined transcript when present.
            if refined_transcription_segments:
                logger.info("Translating refined transcript...")
                translation_segments_refined = translator.translate_segments(
                    refined_transcription_segments, source_lang="en", target_lang="ru"
                )
                logger.info("Refined transcript translation complete")
            else:
                # Otherwise translate the original transcript.
                translation_segments = translator.translate_segments(
                    original_transcription_segments, source_lang="en", target_lang="ru"
                )
                logger.info("Original transcript translation complete")
        else:
            logger.info("Source audio is Russian; translation skipped")
    else:
        logger.info("\n[3/4] Translation not requested")

    # 3.5. Optional translation refinement via LLM.
    translation_segments_refined_llm = None
    if refine_translation_model and (
        translation_segments_refined or translation_segments
    ):
        logger.info(
            "\n[3.5/4] Refining translation with %s...", refine_translation_model
        )
        try:
            from .text_refiner import TextRefiner

            refiner = TextRefiner(model_name=refine_translation_model)

            # Pick which translation output to refine.
            segments_to_refine = (
                translation_segments_refined
                if translation_segments_refined
                else translation_segments
            )

            # Convert segments to text.
            translated_text = transcriber.segments_to_text(segments_to_refine)

            # Apply refinement.
            refined_translation_text = refiner.refine_translation(
                translated_text, context=whisper_prompt
            )

            # Rebuild refined translation segments.
            translation_segments_refined_llm = transcriber.update_segments_from_text(
                segments_to_refine, refined_translation_text
            )

            logger.info("Translation refinement complete")
        except Exception as e:
            logger.error("Failed to refine translation: %s", e)
            logger.warning("Continuing with the original translation")
            translation_segments_refined_llm = None

    # 4. Document generation.
    logger.info("\n[4/4] Generating output documents...")
    writer = DocumentWriter()

    # If a refined transcript exists.
    if refined_transcription_segments:
        # Original transcript without translation.
        logger.info("Creating document with original transcript...")
        docx_path_orig, md_path_orig = writer.create_from_segments(
            title=f"{video_title}_original",
            transcription_segments=original_transcription_segments,
            translation_segments=None,
            transcribe_method=transcribe_method,
            translate_method="",
            with_timestamps=False,
            with_speakers=with_speakers,
            description="Оригинальная транскрипция без улучшений",
        )

        # Refined transcript (with translation if available, not LLM-polished).
        logger.info("Creating document with refined transcript...")
        refine_desc = "Улучшенная транскрипция"
        if translation_segments_refined:
            refine_desc += " + перевод"
        docx_path_refined, md_path_refined = writer.create_from_segments(
            title=f"{video_title}_refined",
            transcription_segments=refined_transcription_segments,
            translation_segments=translation_segments_refined,
            transcribe_method=f"{transcribe_method} + {refine_model}",
            translate_method=translate_method_str,
            with_timestamps=False,
            with_speakers=with_speakers,
            description=refine_desc,
        )

        # Add refined translation document if available.
        if translation_segments_refined_llm:
            logger.info("Creating document with refined translation...")
            docx_path_trans_refined, md_path_trans_refined = (
                writer.create_from_segments(
                    title=f"{video_title}_translated_refined",
                    transcription_segments=(
                        refined_transcription_segments
                        if refined_transcription_segments
                        else original_transcription_segments
                    ),
                    translation_segments=translation_segments_refined_llm,
                    transcribe_method=(
                        f"{transcribe_method} + {refine_model}"
                        if refine_model
                        else transcribe_method
                    ),
                    translate_method=f"{translate_method_str} + {refine_translation_model}",
                    with_timestamps=False,
                    with_speakers=with_speakers,
                    description="Улучшенная транскрипция + улучшенный перевод (дополнительно отполированный через LLM)",
                )
            )

            logger.info("\n" + "=" * 60)
            logger.info("Processing finished successfully!")
            logger.info("Results saved:")
            logger.info("\nOriginal transcript:")
            logger.info(f"  - {docx_path_orig}")
            logger.info(f"  - {md_path_orig}")
            logger.info("\nRefined transcript:")
            logger.info(f"  - {docx_path_refined}")
            logger.info(f"  - {md_path_refined}")
            logger.info("\nRefined translation:")
            logger.info(f"  - {docx_path_trans_refined}")
            logger.info(f"  - {md_path_trans_refined}")
            logger.info("=" * 60)
        else:
            logger.info("\n" + "=" * 60)
            logger.info("Processing finished successfully!")
            logger.info("Results saved:")
            logger.info("\nOriginal transcript:")
            logger.info(f"  - {docx_path_orig}")
            logger.info(f"  - {md_path_orig}")
            logger.info("\nRefined transcript:")
            logger.info(f"  - {docx_path_refined}")
            logger.info(f"  - {md_path_refined}")
            logger.info("=" * 60)
    else:
        # Only original transcript available.
        # Still check for refined translation (without refined transcript).
        if translation_segments_refined_llm:
            # Original translation document.
            logger.info("Creating document with original translation...")
            docx_path_orig, md_path_orig = writer.create_from_segments(
                title=f"{video_title}_translated",
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments,
                transcribe_method=transcribe_method,
                translate_method=translate_method_str,
                with_timestamps=False,
                with_speakers=with_speakers,
                description="Оригинальная транскрипция + перевод",
            )

            # Translation refined via LLM.
            logger.info("Creating document with refined translation...")
            docx_path_refined, md_path_refined = writer.create_from_segments(
                title=f"{video_title}_translated_refined",
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments_refined_llm,
                transcribe_method=transcribe_method,
                translate_method=f"{translate_method_str} + {refine_translation_model}",
                with_timestamps=False,
                with_speakers=with_speakers,
                description="Оригинальная транскрипция + улучшенный перевод (дополнительно отполированный через LLM)",
            )

            logger.info("\n" + "=" * 60)
            logger.info("Processing finished successfully!")
            logger.info("Results saved:")
            logger.info("\nOriginal translation:")
            logger.info(f"  - {docx_path_orig}")
            logger.info(f"  - {md_path_orig}")
            logger.info("\nRefined translation:")
            logger.info(f"  - {docx_path_refined}")
            logger.info(f"  - {md_path_refined}")
            logger.info("=" * 60)
        else:
            # Single document (with or without translation).
            single_desc = "Транскрипция"
            if translation_segments:
                single_desc += " + перевод"
            docx_path, md_path = writer.create_from_segments(
                title=video_title,
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments,
                transcribe_method=transcribe_method,
                translate_method=translate_method_str,
                with_timestamps=False,
                with_speakers=with_speakers,
                description=single_desc,
            )

            logger.info("\n" + "=" * 60)
            logger.info("Processing finished successfully!")
            logger.info("Results saved:")
            logger.info(f"  - {docx_path}")
            logger.info(f"  - {md_path}")
            logger.info("=" * 60)

    # Generate summary if requested.
    if summarize and summarize_model:
        # Use the best available segments
        segments_for_summary = (
            refined_transcription_segments or original_transcription_segments
        )
        _generate_summary(
            title=video_title,
            segments=segments_for_summary,
            summarize_model=summarize_model,
            summarize_backend=summarize_backend,
        )

    # Print session summary
    _print_session_summary()


def process_local_video(
    video_path: str,
    transcribe_method: str,
    language: Optional[str] = None,
    translate_methods: Optional[list[str]] = None,
    with_speakers: bool = False,
    custom_prompt: Optional[str] = None,
    refine_model: Optional[str] = None,
    refine_translation_model: Optional[str] = None,
    translate_model: Optional[str] = None,
    openai_translate_model: Optional[str] = None,
    refine_backend: str = "ollama",
    summarize: bool = False,
    summarize_model: Optional[str] = None,
    summarize_backend: str = "ollama",
):
    """
    Process a local video file by extracting audio and transcribing.

    Args:
        video_path: Path to the video file.
        transcribe_method: Transcription backend to use.
        language: Force source language (ru/en); default is auto-detect.
        translate_methods: Translation backends to apply.
        with_speakers: Enable speaker diarisation (not yet supported).
        custom_prompt: Optional custom Whisper prompt.
        refine_model: Model for transcript refinement.
        refine_backend: Backend for refinement (ollama or openai_api).
        refine_translation_model: Model for translation refinement.
        translate_model: NLLB model override.
        summarize: Whether to generate a summary.
        summarize_model: Model for summarization.
        summarize_backend: Backend for summarization (ollama or openai_api).
    """
    logger.info("=" * 60)
    logger.info("Starting local video processing")
    logger.info("=" * 60)

    video_path_obj = Path(video_path)

    # 1. Extract audio from video using FFmpeg.
    logger.info("\n[1/4] Extracting audio from video...")
    video_processor = VideoProcessor()

    try:
        audio_path = video_processor.extract_audio(video_path_obj)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error("Failed to extract audio from video: %s", e)
        return

    # 2. Continue with audio processing (same as process_local_audio).
    logger.info("\n[2/4] Transcribing audio...")

    audio_title = sanitize_filename(video_path_obj.stem)

    if custom_prompt:
        logger.info("Using custom prompt supplied by user")
    else:
        logger.info("No prompt provided (Whisper auto-detection will be used)")

    transcriber = Transcriber(method=transcribe_method)
    transcription_segments = transcriber.transcribe(
        audio_path,
        language=language,  # Auto-detect when None
        with_speakers=with_speakers,
        initial_prompt=custom_prompt,
    )

    # Keep original segments to generate multiple document variants.
    original_transcription_segments = transcription_segments

    # 2.5. Optional transcript refinement via LLM.
    refined_transcription_segments = None
    if refine_model:
        logger.info("\n[2.5/4] Refining transcript with %s...", refine_model)
        try:
            from .text_refiner import TextRefiner

            refiner = TextRefiner(backend=refine_backend, model_name=refine_model)

            # Convert segments to plain text.
            original_text = transcriber.segments_to_text(transcription_segments)

            # Improve text (using custom prompt as optional context).
            refined_text = refiner.refine_text(original_text, context=custom_prompt)

            # Rebuild segments from refined text.
            refined_transcription_segments = transcriber.update_segments_from_text(
                transcription_segments, refined_text
            )

            logger.info("Transcript refined successfully")
        except Exception as e:
            logger.error("Failed to refine transcript: %s", e)
            logger.warning("Continuing with the original transcript")
            refined_transcription_segments = None

    # 3. Translation (optional).
    translation_segments = None
    translation_segments_refined = None
    translate_method_str = ""

    if translate_methods:
        logger.info("\n[3/4] Translating text...")

        # Detect source language.
        original_text = transcriber.segments_to_text(transcription_segments)
        source_lang = detect_language(original_text)

        # Use the first translation backend (MVP approach).
        translate_method = translate_methods[0]

        translate_method_str = translate_method

        from .translator import Translator

        if source_lang == "en":
            # Use appropriate model based on translation method
            model_override = (
                openai_translate_model
                if translate_method == TranslateOptions.OPENAI_API
                else translate_model
            )
            translator = Translator(method=translate_method, model_name=model_override)

            if translate_method == "NLLB":
                translate_method_str = f"{translate_method} ({translator.model_name})"
            elif translate_method == TranslateOptions.OPENAI_API:
                translate_method_str = f"OpenAI API ({translator.model_name})"

            # Translate the refined transcript when present.
            if refined_transcription_segments:
                logger.info("Translating refined transcript...")
                translation_segments_refined = translator.translate_segments(
                    refined_transcription_segments, source_lang="en", target_lang="ru"
                )
                logger.info("Refined transcript translation complete")
            else:
                # Otherwise translate the original transcript.
                translation_segments = translator.translate_segments(
                    transcription_segments, source_lang="en", target_lang="ru"
                )
                logger.info("Original transcript translation complete")
        else:
            logger.info("Audio is in Russian; translation skipped")
    else:
        logger.info("\n[3/4] Translation not requested")

    # 3.5. Optional translation refinement via LLM.
    translation_segments_refined_llm = None
    if refine_translation_model and (
        translation_segments_refined or translation_segments
    ):
        logger.info(
            "\n[3.5/4] Refining translation with %s...", refine_translation_model
        )
        try:
            from .text_refiner import TextRefiner

            refiner = TextRefiner(model_name=refine_translation_model)

            # Choose which translation output to refine.
            segments_to_refine = (
                translation_segments_refined
                if translation_segments_refined
                else translation_segments
            )

            # Convert segments to text.
            translated_text = transcriber.segments_to_text(segments_to_refine)

            # Refine translation.
            refined_translation_text = refiner.refine_translation(
                translated_text, context=custom_prompt
            )

            # Build refined translation segments.
            translation_segments_refined_llm = transcriber.update_segments_from_text(
                segments_to_refine, refined_translation_text
            )

            logger.info("Translation refinement complete")
        except Exception as e:
            logger.error("Failed to refine translation: %s", e)
            logger.warning("Continuing with the original translation")
            translation_segments_refined_llm = None

    # 4. Document generation.
    logger.info("\n[4/4] Generating output documents...")
    writer = DocumentWriter()

    # If a refined transcript exists.
    if refined_transcription_segments:
        # Original transcript without translation.
        logger.info("Creating document with original transcript...")
        docx_path_orig, md_path_orig = writer.create_from_segments(
            title=f"{audio_title}_original",
            transcription_segments=original_transcription_segments,
            translation_segments=None,
            transcribe_method=transcribe_method,
            translate_method="",
            with_timestamps=False,
            with_speakers=with_speakers,
            description="Оригинальная транскрипция без улучшений",
        )

        # Refined transcript (with translation if available).
        logger.info("Creating document with refined transcript...")
        refine_desc = "Улучшенная транскрипция"
        if translation_segments_refined:
            refine_desc += " + перевод"
        docx_path_refined, md_path_refined = writer.create_from_segments(
            title=f"{audio_title}_refined",
            transcription_segments=refined_transcription_segments,
            translation_segments=translation_segments_refined,
            transcribe_method=f"{transcribe_method} + {refine_model}",
            translate_method=translate_method_str,
            with_timestamps=False,
            with_speakers=with_speakers,
            description=refine_desc,
        )

        # Create refined translation document if available.
        if translation_segments_refined_llm:
            logger.info("Creating document with refined translation...")
            docx_path_trans_refined, md_path_trans_refined = (
                writer.create_from_segments(
                    title=f"{audio_title}_translated_refined",
                    transcription_segments=(
                        refined_transcription_segments
                        if refined_transcription_segments
                        else original_transcription_segments
                    ),
                    translation_segments=translation_segments_refined_llm,
                    transcribe_method=(
                        f"{transcribe_method} + {refine_model}"
                        if refine_model
                        else transcribe_method
                    ),
                    translate_method=f"{translate_method_str} + {refine_translation_model}",
                    with_timestamps=False,
                    with_speakers=with_speakers,
                    description="Улучшенная транскрипция + улучшенный перевод (дополнительно отполированный через LLM)",
                )
            )

            logger.info("\n" + "=" * 60)
            logger.info("Processing finished successfully!")
            logger.info("Results saved:")
            logger.info("\nOriginal transcript:")
            logger.info(f"  - {docx_path_orig}")
            logger.info(f"  - {md_path_orig}")
            logger.info("\nRefined transcript:")
            logger.info(f"  - {docx_path_refined}")
            logger.info(f"  - {md_path_refined}")
            logger.info("\nRefined translation:")
            logger.info(f"  - {docx_path_trans_refined}")
            logger.info(f"  - {md_path_trans_refined}")
            logger.info("=" * 60)
        else:
            logger.info("\n" + "=" * 60)
            logger.info("Processing finished successfully!")
            logger.info("Results saved:")
            logger.info("\nOriginal transcript:")
            logger.info(f"  - {docx_path_orig}")
            logger.info(f"  - {md_path_orig}")
            logger.info("\nRefined transcript:")
            logger.info(f"  - {docx_path_refined}")
            logger.info(f"  - {md_path_refined}")
            logger.info("=" * 60)
    else:
        # Only original transcript available.
        # Still create refined translation documents if present.
        if translation_segments_refined_llm:
            # Original translation document.
            logger.info("Creating document with original translation...")
            docx_path_orig, md_path_orig = writer.create_from_segments(
                title=f"{audio_title}_translated",
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments,
                transcribe_method=transcribe_method,
                translate_method=translate_method_str,
                with_timestamps=False,
                with_speakers=with_speakers,
                description="Оригинальная транскрипция + перевод",
            )

            # LLM-refined translation.
            logger.info("Creating document with refined translation...")
            docx_path_refined, md_path_refined = writer.create_from_segments(
                title=f"{audio_title}_translated_refined",
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments_refined_llm,
                transcribe_method=transcribe_method,
                translate_method=f"{translate_method_str} + {refine_translation_model}",
                with_timestamps=False,
                with_speakers=with_speakers,
                description="Улучшенная транскрипция + улучшенный перевод (дополнительно отполированный через LLM)",
            )

            logger.info("\n" + "=" * 60)
            logger.info("Processing finished successfully!")
            logger.info("Results saved:")
            logger.info("\nOriginal translation:")
            logger.info(f"  - {docx_path_orig}")
            logger.info(f"  - {md_path_orig}")
            logger.info("\nRefined translation:")
            logger.info(f"  - {docx_path_refined}")
            logger.info(f"  - {md_path_refined}")
            logger.info("=" * 60)
        else:
            # Single document (with or without translation).
            docx_path, md_path = writer.create_from_segments(
                title=audio_title,
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments,
                transcribe_method=transcribe_method,
                translate_method=translate_method_str,
                with_timestamps=False,
                with_speakers=with_speakers,
                description="Транскрипция"
                + (" + перевод" if translation_segments else ""),
            )

            logger.info("\n" + "=" * 60)
            logger.info("Processing finished successfully!")
            logger.info("Results saved:")
            logger.info(f"  - {docx_path}")
            logger.info(f"  - {md_path}")
            logger.info("=" * 60)

    # Generate summary if requested.
    if summarize and summarize_model:
        # Use the best available segments
        segments_for_summary = (
            refined_transcription_segments or original_transcription_segments
        )
        _generate_summary(
            title=audio_title,
            segments=segments_for_summary,
            summarize_model=summarize_model,
            summarize_backend=summarize_backend,
        )

    # Print session summary
    _print_session_summary()


def process_local_audio(
    audio_path: str,
    transcribe_method: str,
    language: Optional[str] = None,
    translate_methods: Optional[list[str]] = None,
    with_speakers: bool = False,
    custom_prompt: Optional[str] = None,
    refine_model: Optional[str] = None,
    refine_translation_model: Optional[str] = None,
    translate_model: Optional[str] = None,
    openai_translate_model: Optional[str] = None,
    refine_backend: str = "ollama",
    summarize: bool = False,
    summarize_model: Optional[str] = None,
    summarize_backend: str = "ollama",
):
    """
    Process a local audio file end-to-end.

    Args:
        audio_path: Path to the audio file.
        transcribe_method: Transcription backend to use.
        language: Force source language (ru/en); default is auto-detect.
        translate_methods: Translation backends to apply.
        with_speakers: Enable speaker diarisation (not yet supported).
        custom_prompt: Optional custom Whisper prompt.
        refine_model: Model for transcript refinement.
        refine_backend: Backend for refinement (ollama or openai_api).
        refine_translation_model: Model for translation refinement.
        summarize: Whether to generate a summary.
        summarize_model: Model for summarization.
        summarize_backend: Backend for summarization (ollama or openai_api).
    """
    logger.info("=" * 60)
    logger.info("Starting local audio processing")
    logger.info("=" * 60)

    audio_path_obj = Path(audio_path)
    audio_title = sanitize_filename(audio_path_obj.stem)  # File-safe stem

    # 1. Transcription.
    logger.info("\n[1/3] Transcribing audio...")

    if custom_prompt:
        logger.info("Using custom prompt supplied by user")
    else:
        logger.info("No prompt provided (Whisper auto-detection will be used)")

    transcriber = Transcriber(method=transcribe_method)
    transcription_segments = transcriber.transcribe(
        audio_path_obj,
        language=language,  # Auto-detect when None
        with_speakers=with_speakers,
        initial_prompt=custom_prompt,
    )

    # Keep original segments to generate multiple document variants.
    original_transcription_segments = transcription_segments

    # 1.5. Optional transcript refinement via LLM.
    refined_transcription_segments = None
    if refine_model:
        logger.info("\n[1.5/3] Refining transcript with %s...", refine_model)
        try:
            from .text_refiner import TextRefiner

            refiner = TextRefiner(backend=refine_backend, model_name=refine_model)

            # Convert segments to plain text.
            original_text = transcriber.segments_to_text(transcription_segments)

            # Improve text (using custom prompt as optional context).
            refined_text = refiner.refine_text(original_text, context=custom_prompt)

            # Rebuild segments from refined text.
            refined_transcription_segments = transcriber.update_segments_from_text(
                transcription_segments, refined_text
            )

            logger.info("Transcript refined successfully")
        except Exception as e:
            logger.error("Failed to refine transcript: %s", e)
            logger.warning("Continuing with the original transcript")
            refined_transcription_segments = None

    # 2. Translation (optional).
    translation_segments = None
    translation_segments_refined = None
    translate_method_str = ""

    if translate_methods:
        logger.info("\n[2/3] Translating text...")

        # Detect source language.
        original_text = transcriber.segments_to_text(transcription_segments)
        source_lang = detect_language(original_text)

        # Use the first translation backend (MVP approach).
        translate_method = translate_methods[0]

        translate_method_str = translate_method

        from .translator import Translator

        if source_lang == "en":
            # Use appropriate model based on translation method
            model_override = (
                openai_translate_model
                if translate_method == TranslateOptions.OPENAI_API
                else translate_model
            )
            translator = Translator(method=translate_method, model_name=model_override)

            if translate_method == "NLLB":
                translate_method_str = f"{translate_method} ({translator.model_name})"
            elif translate_method == TranslateOptions.OPENAI_API:
                translate_method_str = f"OpenAI API ({translator.model_name})"

            # Translate the refined transcript when present.
            if refined_transcription_segments:
                logger.info("Translating refined transcript...")
                translation_segments_refined = translator.translate_segments(
                    refined_transcription_segments, source_lang="en", target_lang="ru"
                )
                logger.info("Refined transcript translation complete")
            else:
                # Otherwise translate the original transcript.
                translation_segments = translator.translate_segments(
                    transcription_segments, source_lang="en", target_lang="ru"
                )
                logger.info("Original transcript translation complete")
        else:
            logger.info("Audio is in Russian; translation skipped")
    else:
        logger.info("\n[2/3] Translation not requested")

    # 2.5. Optional translation refinement via LLM.
    translation_segments_refined_llm = None
    if refine_translation_model and (
        translation_segments_refined or translation_segments
    ):
        logger.info(
            "\n[2.5/3] Refining translation with %s...", refine_translation_model
        )
        try:
            from .text_refiner import TextRefiner

            refiner = TextRefiner(model_name=refine_translation_model)

            # Choose which translation output to refine.
            segments_to_refine = (
                translation_segments_refined
                if translation_segments_refined
                else translation_segments
            )

            # Convert segments to text.
            translated_text = transcriber.segments_to_text(segments_to_refine)

            # Refine translation.
            refined_translation_text = refiner.refine_translation(
                translated_text, context=custom_prompt
            )

            # Build refined translation segments.
            translation_segments_refined_llm = transcriber.update_segments_from_text(
                segments_to_refine, refined_translation_text
            )

            logger.info("Translation refinement complete")
        except Exception as e:
            logger.error("Failed to refine translation: %s", e)
            logger.warning("Continuing with the original translation")
            translation_segments_refined_llm = None

    # 3. Document generation.
    logger.info("\n[3/3] Generating output documents...")
    writer = DocumentWriter()

    # If a refined transcript exists.
    if refined_transcription_segments:
        # Original transcript without translation.
        logger.info("Creating document with original transcript...")
        docx_path_orig, md_path_orig = writer.create_from_segments(
            title=f"{audio_title}_original",
            transcription_segments=original_transcription_segments,
            translation_segments=None,
            transcribe_method=transcribe_method,
            translate_method="",
            with_timestamps=False,
            with_speakers=with_speakers,
            description="Оригинальная транскрипция без улучшений",
        )

        # Refined transcript (with translation if available).
        logger.info("Creating document with refined transcript...")
        refine_desc = "Улучшенная транскрипция"
        if translation_segments_refined:
            refine_desc += " + перевод"
        docx_path_refined, md_path_refined = writer.create_from_segments(
            title=f"{audio_title}_refined",
            transcription_segments=refined_transcription_segments,
            translation_segments=translation_segments_refined,
            transcribe_method=f"{transcribe_method} + {refine_model}",
            translate_method=translate_method_str,
            with_timestamps=False,
            with_speakers=with_speakers,
            description=refine_desc,
        )

        # Create refined translation document if available.
        if translation_segments_refined_llm:
            logger.info("Creating document with refined translation...")
            docx_path_trans_refined, md_path_trans_refined = (
                writer.create_from_segments(
                    title=f"{audio_title}_translated_refined",
                    transcription_segments=(
                        refined_transcription_segments
                        if refined_transcription_segments
                        else original_transcription_segments
                    ),
                    translation_segments=translation_segments_refined_llm,
                    transcribe_method=(
                        f"{transcribe_method} + {refine_model}"
                        if refine_model
                        else transcribe_method
                    ),
                    translate_method=f"{translate_method_str} + {refine_translation_model}",
                    with_timestamps=False,
                    with_speakers=with_speakers,
                    description="Улучшенная транскрипция + улучшенный перевод (дополнительно отполированный через LLM)",
                )
            )

            logger.info("\n" + "=" * 60)
            logger.info("Processing finished successfully!")
            logger.info("Results saved:")
            logger.info("\nOriginal transcript:")
            logger.info(f"  - {docx_path_orig}")
            logger.info(f"  - {md_path_orig}")
            logger.info("\nRefined transcript:")
            logger.info(f"  - {docx_path_refined}")
            logger.info(f"  - {md_path_refined}")
            logger.info("\nRefined translation:")
            logger.info(f"  - {docx_path_trans_refined}")
            logger.info(f"  - {md_path_trans_refined}")
            logger.info("=" * 60)
        else:
            logger.info("\n" + "=" * 60)
            logger.info("Processing finished successfully!")
            logger.info("Results saved:")
            logger.info("\nOriginal transcript:")
            logger.info(f"  - {docx_path_orig}")
            logger.info(f"  - {md_path_orig}")
            logger.info("\nRefined transcript:")
            logger.info(f"  - {docx_path_refined}")
            logger.info(f"  - {md_path_refined}")
            logger.info("=" * 60)
    else:
        # Only original transcript available.
        # Still create refined translation documents if present.
        if translation_segments_refined_llm:
            # Original translation document.
            logger.info("Creating document with original translation...")
            docx_path_orig, md_path_orig = writer.create_from_segments(
                title=f"{audio_title}_translated",
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments,
                transcribe_method=transcribe_method,
                translate_method=translate_method_str,
                with_timestamps=False,
                with_speakers=with_speakers,
                description="Оригинальная транскрипция + перевод",
            )

            # LLM-refined translation.
            logger.info("Creating document with refined translation...")
            docx_path_refined, md_path_refined = writer.create_from_segments(
                title=f"{audio_title}_translated_refined",
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments_refined_llm,
                transcribe_method=transcribe_method,
                translate_method=f"{translate_method_str} + {refine_translation_model}",
                with_timestamps=False,
                with_speakers=with_speakers,
                description="Улучшенная транскрипция + улучшенный перевод (дополнительно отполированный через LLM)",
            )

            logger.info("\n" + "=" * 60)
            logger.info("Processing finished successfully!")
            logger.info("Results saved:")
            logger.info("\nOriginal translation:")
            logger.info(f"  - {docx_path_orig}")
            logger.info(f"  - {md_path_orig}")
            logger.info("\nRefined translation:")
            logger.info(f"  - {docx_path_refined}")
            logger.info(f"  - {md_path_refined}")
            logger.info("=" * 60)
        else:
            # Single document (with or without translation).
            docx_path, md_path = writer.create_from_segments(
                title=audio_title,
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments,
                transcribe_method=transcribe_method,
                translate_method=translate_method_str,
                with_timestamps=False,
                with_speakers=with_speakers,
                description="Транскрипция"
                + (" + перевод" if translation_segments else ""),
            )

            logger.info("\n" + "=" * 60)
            logger.info("Processing finished successfully!")
            logger.info("Results saved:")
            logger.info(f"  - {docx_path}")
            logger.info(f"  - {md_path}")
            logger.info("=" * 60)

    # Generate summary if requested.
    if summarize and summarize_model:
        # Use the best available segments
        segments_for_summary = (
            refined_transcription_segments or original_transcription_segments
        )
        _generate_summary(
            title=audio_title,
            segments=segments_for_summary,
            summarize_model=summarize_model,
            summarize_backend=summarize_backend,
        )

    # Print session summary
    _print_session_summary()


def normalize_method_names(args):
    """Normalize method names to internal format (underscore-based)."""
    if hasattr(args, "transcribe") and args.transcribe:
        args.transcribe = args.transcribe.replace("-", "_")
    if hasattr(args, "language") and args.language:
        args.language = args.language.lower()
    if hasattr(args, "translate") and args.translate:
        # Normalize each method in comma-separated list
        methods = [
            (
                m.strip().replace("-", "_").upper()
                if m.strip().lower() == "nllb"
                else m.strip().replace("-", "_")
            )
            for m in args.translate.split(",")
        ]
        args.translate = ",".join(methods)
    return args


def main():
    """Application entry point."""
    parser = argparse.ArgumentParser(
        description="YouTube Transcriber & Translator", add_help=False
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Common arguments shared across commands
    def add_common_args(subparser):
        """Add common arguments to a subparser."""
        subparser.add_argument(
            "--transcribe",
            type=str,
            help="Transcription method (whisper-base, whisper-small, whisper-medium, whisper-openai-api, gigaam-e2e-rnnt, gigaam-e2e-ctc)",
        )
        subparser.add_argument(
            "--language",
            "-l",
            type=str,
            dest="language",
            help="Force source language code (e.g., ru or en). Defaults to auto-detect.",
        )
        subparser.add_argument(
            "--translate", type=str, help="Translation method (nllb, openai-api)"
        )
        subparser.add_argument(
            "--prompt-file",
            type=str,
            dest="prompt_file",
            help="Path to a Whisper prompt file",
        )
        subparser.add_argument(
            "--refine-model",
            type=str,
            dest="refine_model",
            help="Model for transcript refinement (e.g. qwen2.5:3b, gpt-4)",
        )
        subparser.add_argument(
            "--refine-backend",
            type=str,
            dest="refine_backend",
            choices=["ollama", "openai-api", "openai_api"],
            default="ollama",
            help="Backend for transcript refinement (ollama, openai-api)",
        )
        subparser.add_argument(
            "--refine-translation",
            type=str,
            dest="refine_translation",
            help="Model for translation refinement",
        )
        subparser.add_argument(
            "--summarize-model",
            type=str,
            dest="summarize_model",
            help="Model for summarization",
        )
        subparser.add_argument(
            "--summarize-backend",
            type=str,
            dest="summarize_backend",
            choices=["ollama", "openai-api", "openai_api"],
            default="ollama",
            help="Backend for summarization (ollama, openai-api)",
        )
        subparser.add_argument(
            "--nllb-model",
            type=str,
            dest="nllb_model",
            help="NLLB model (default: facebook/nllb-200-distilled-1.3B)",
        )
        subparser.add_argument(
            "--openai-translate-model",
            type=str,
            dest="openai_translate_model",
            help="OpenAI translation model (default: gpt-4o-mini, options: gpt-4o, gpt-4, gpt-3.5-turbo)",
        )
        subparser.add_argument(
            "--speakers", action="store_true", help="Enable speaker diarization"
        )

    # YouTube command
    youtube_parser = subparsers.add_parser("youtube", help="Process a YouTube video")
    youtube_parser.add_argument(
        "--url", type=str, required=True, help="YouTube video URL"
    )
    add_common_args(youtube_parser)

    # Audio command
    audio_parser = subparsers.add_parser("audio", help="Process a local audio file")
    audio_parser.add_argument(
        "--input", type=str, required=True, help="Path to audio file"
    )
    add_common_args(audio_parser)

    # Video command
    video_parser = subparsers.add_parser("video", help="Process a local video file")
    video_parser.add_argument(
        "--input", type=str, required=True, help="Path to video file"
    )
    add_common_args(video_parser)

    # Text command
    text_parser = subparsers.add_parser("text", help="Process a text document")
    text_parser.add_argument(
        "--input", type=str, required=True, help="Path to text file (.docx, .md, .txt)"
    )
    text_parser.add_argument(
        "--translate", type=str, help="Translation method (nllb, openai-api)"
    )
    text_parser.add_argument(
        "--refine-model",
        type=str,
        dest="refine_model",
        help="Model for text refinement",
    )
    text_parser.add_argument(
        "--refine-backend",
        type=str,
        dest="refine_backend",
        choices=["ollama", "openai-api", "openai_api"],
        default="ollama",
        help="Backend for transcript refinement (not translation)",
    )
    text_parser.add_argument(
        "--refine-translation",
        type=str,
        dest="refine_translation",
        help="Model for translation refinement",
    )
    text_parser.add_argument(
        "--summarize-model",
        type=str,
        dest="summarize_model",
        help="Model for summarization",
    )
    text_parser.add_argument(
        "--summarize-backend",
        type=str,
        dest="summarize_backend",
        choices=["ollama", "openai-api", "openai_api"],
        default="ollama",
        help="Backend for summarization",
    )
    text_parser.add_argument(
        "--nllb-model", type=str, dest="nllb_model", help="NLLB model override"
    )
    text_parser.add_argument(
        "--openai-translate-model",
        type=str,
        dest="openai_translate_model",
        help="OpenAI translation model (default: gpt-4o-mini, options: gpt-4o, gpt-4, gpt-3.5-turbo)",
    )

    # Global help
    parser.add_argument("--help", "-h", action="store_true", help="Show help message")

    # Parse arguments
    args = parser.parse_args()

    # Show help when explicitly requested or no command provided
    if args.help or not args.command:
        print_help()
        sys.exit(0)

    # Normalize method names (convert dashes to underscores for internal use)
    args = normalize_method_names(args)

    # Normalize backend names
    if hasattr(args, "refine_backend"):
        args.refine_backend = args.refine_backend.replace("-", "_")
    if hasattr(args, "summarize_backend"):
        args.summarize_backend = args.summarize_backend.replace("-", "_")

    # Validate arguments
    if not validate_args(args.command, args):
        sys.exit(1)

    # Check and update yt-dlp before processing YouTube videos
    if args.command == "youtube":
        logger.info("=" * 60)
        logger.info("Checking yt-dlp version...")
        logger.info("=" * 60)
        check_and_update_yt_dlp()
        logger.info("=" * 60)

    try:
        # Parse translation methods
        translate_methods = None
        if hasattr(args, "translate") and args.translate:
            translate_methods = [m.strip() for m in args.translate.split(",")]

        # Load a custom prompt if supplied
        custom_prompt = None
        if hasattr(args, "prompt_file") and args.prompt_file:
            custom_prompt = load_prompt_from_file(args.prompt_file)

        # Determine if summarization is requested (model presence implies request)
        summarize = bool(getattr(args, "summarize_model", None))

        # Get NLLB model override
        nllb_model = getattr(args, "nllb_model", None)

        # Get OpenAI translate model override
        openai_translate_model = getattr(args, "openai_translate_model", None)

        # Dispatch based on command
        if args.command == "youtube":
            process_youtube_video(
                url=args.url,
                transcribe_method=args.transcribe,
                language=getattr(args, "language", None),
                translate_methods=translate_methods,
                with_speakers=getattr(args, "speakers", False),
                custom_prompt=custom_prompt,
                refine_model=getattr(args, "refine_model", None),
                refine_translation_model=getattr(args, "refine_translation", None),
                translate_model=nllb_model,
                openai_translate_model=openai_translate_model,
                refine_backend=getattr(args, "refine_backend", "ollama"),
                summarize=summarize,
                summarize_model=getattr(args, "summarize_model", None),
                summarize_backend=getattr(args, "summarize_backend", "ollama"),
            )
        elif args.command == "audio":
            process_local_audio(
                audio_path=args.input,
                transcribe_method=args.transcribe,
                language=getattr(args, "language", None),
                translate_methods=translate_methods,
                with_speakers=getattr(args, "speakers", False),
                custom_prompt=custom_prompt,
                refine_model=getattr(args, "refine_model", None),
                refine_translation_model=getattr(args, "refine_translation", None),
                translate_model=nllb_model,
                openai_translate_model=openai_translate_model,
                refine_backend=getattr(args, "refine_backend", "ollama"),
                summarize=summarize,
                summarize_model=getattr(args, "summarize_model", None),
                summarize_backend=getattr(args, "summarize_backend", "ollama"),
            )
        elif args.command == "video":
            process_local_video(
                video_path=args.input,
                transcribe_method=args.transcribe,
                language=getattr(args, "language", None),
                translate_methods=translate_methods,
                with_speakers=getattr(args, "speakers", False),
                custom_prompt=custom_prompt,
                refine_model=getattr(args, "refine_model", None),
                refine_translation_model=getattr(args, "refine_translation", None),
                translate_model=nllb_model,
                openai_translate_model=openai_translate_model,
                refine_backend=getattr(args, "refine_backend", "ollama"),
                summarize=summarize,
                summarize_model=getattr(args, "summarize_model", None),
                summarize_backend=getattr(args, "summarize_backend", "ollama"),
            )
        elif args.command == "text":
            process_text_file(
                text_path=args.input,
                translate_methods=translate_methods,
                refine_model=getattr(args, "refine_model", None),
                refine_translation_model=getattr(args, "refine_translation", None),
                translate_model=nllb_model,
                openai_translate_model=openai_translate_model,
                refine_backend=getattr(args, "refine_backend", "ollama"),
                summarize=summarize,
                summarize_model=getattr(args, "summarize_model", None),
                summarize_backend=getattr(args, "summarize_backend", "ollama"),
            )

    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("An error occurred: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
