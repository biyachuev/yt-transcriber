"""
Main application module.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

from .config import settings, TranscribeOptions, TranslateOptions, RefineOptions, SummarizeOptions
from .logger import logger
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


def load_prompt_from_file(prompt_file_path: str) -> str:
    """
    Load a Whisper prompt from a text file.

    Args:
        prompt_file_path: Path to the prompt file.

    Returns:
        Prompt text trimmed to Whisper limits.
    """
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()

        # Limit prompt length to the Whisper constraint.
        MAX_PROMPT_LENGTH = 800
        if len(prompt) > MAX_PROMPT_LENGTH:
            logger.warning("Prompt loaded from file is too long (%d chars), trimming to %d", len(prompt), MAX_PROMPT_LENGTH)
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
    python -m src.main [OPTIONS]

Examples:
    # Transcribe and translate a YouTube video
    python -m src.main --url "https://youtube.com/watch?v=..." --transcribe whisper_base --translate NLLB

    # Transcribe only
    python -m src.main --url "https://youtube.com/watch?v=..." --transcribe whisper_base

    # Process a local audio file
    python -m src.main --input_audio audio.mp3 --transcribe whisper_base --translate NLLB

    # Process a local video file (MP4, MKV, AVI, etc.)
    python -m src.main --input_video video.mp4 --transcribe whisper_base --translate NLLB

    # Apply a custom Whisper prompt
    python -m src.main --url "https://youtube.com/watch?v=..." --transcribe whisper_base --prompt prompt.txt

    # Refine a transcript with a local LLM
    python -m src.main --input_audio audio.mp3 --transcribe whisper_medium --refine-model llama3.2:3b

    # Refine and translate (creates original, refined, translated documents)
    python -m src.main --input_audio audio.mp3 --transcribe whisper_medium --translate NLLB --refine-model llama3.2:3b

Options:
    --url URL                   YouTube video URL
    --input_audio PATH          Path to an audio file (mp3, wav, etc.)
    --input_video PATH          Path to a video file (mp4, mkv, avi, etc.)
    --input_text PATH           Path to a text document (.docx, .md)

    --transcribe METHOD         Transcription backend
    --translate METHOD          Translation backend (comma separated list)

    --prompt PATH               Custom Whisper prompt file
                                Helps capture domain-specific names and terms
                                For YouTube: generated automatically if omitted
                                For audio files: recommended to supply manually

    --refine-model MODEL        Ollama model used to refine the transcript
                                (e.g. qwen2.5:3b, llama3:8b). Requires a running Ollama server.

    --refine-translation MODEL  Ollama model used to polish translation output
                                (e.g. qwen2.5:3b, llama3:8b). Produces more natural phrasing.
                                Requires a running Ollama server.

    --speakers                  Enable speaker diarisation (experimental)

    --translate-model MODEL     NLLB translation model.
                                Default: facebook/nllb-200-distilled-1.3B
                                Other options: facebook/nllb-200-distilled-600M (faster),
                                facebook/nllb-200-3.3B (best quality, slowest)

    --help, -h                  Show this message

Available transcription methods:
    - whisper_base              Fast local model
    - whisper_small             Balanced local model
    - whisper_medium            Best quality local model
    - whisper_openai_api        OpenAI Whisper API (requires OPENAI_API_KEY)

Available translation methods:
    - NLLB                      Local NLLB inference
    - openai_api                OpenAI GPT API (requires OPENAI_API_KEY)

Available refinement backends:
    - ollama (default)          Local LLM via Ollama
    - openai_api                OpenAI GPT API (requires OPENAI_API_KEY)

Available summarization backends:
    - ollama (default)          Local LLM via Ollama
    - openai_api                OpenAI GPT API (requires OPENAI_API_KEY)

Custom Whisper prompts:
    Prompts help Whisper recognise proper names, brands, and technical jargon.
    Format: plain text file, comma-separated keywords. Example prompt.txt:
        FIDE, Hikaru Nakamura, Magnus Carlsen, chess tournament, bongcloud

Refinement with LLM (--refine-model):
    After transcription, you can polish the text with a local LLM to:
    - Fix terminology and proper nouns
    - Improve punctuation
    - Remove filler words
    - Split text into paragraphs

    Output files:
    - Without --refine-model: name.docx, name.md
    - With --refine-model: name (original).docx/md, name (refined).docx/md
    - With --refine-model and --translate: name (translated).docx/md

    Requirements:
    1. Install Ollama: https://ollama.ai
    2. Pull a model: ollama pull qwen2.5:3b
    3. Start the server: ollama serve

    Recommended models:
    - llama3.2:3b   — fast, good quality
    - qwen2.5:3b    — fast, excellent for Russian & English
    - llama3:8b     — slower, higher quality
    - mistral:7b    — balanced option

Notes:
    - Results are stored in 'output/' (.docx and .md)
    - Temporary artefacts go to 'temp/'
    - Logs are written to 'logs/'
    """
    print(help_text)


def _generate_summary(
    title: str,
    segments: list[dict],
    summarize_model: str,
    summarize_backend: str
) -> None:
    """
    Generate and save a summary of the text.

    Args:
        title: Base title for the output file.
        segments: List of text segments to summarize.
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
        text_to_summarize = '\n\n'.join([seg['text'] for seg in segments])

        # Detect language for summary
        detected_lang = detect_language(text_to_summarize)
        summary_lang = "ru" if detected_lang in ["ru", "uk", "be"] else "en"

        summary = summarizer.summarize_long_text(text_to_summarize, language=summary_lang)

        # Save summary as a separate document
        summary_path = settings.OUTPUT_DIR / f"{title}_summary.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"# Summary: {title}\n\n")
            f.write(summary)

        logger.info("Summary saved: %s", summary_path)
    except Exception as e:
        logger.error("Failed to generate summary: %s", e)


def process_text_file(
    text_path: str,
    translate_methods: Optional[list[str]] = None,
    refine_model: Optional[str] = None,
    refine_translation_model: Optional[str] = None,
    translate_model: Optional[str] = None,
    refine_backend: str = "ollama",
    summarize: bool = False,
    summarize_model: Optional[str] = None,
    summarize_backend: str = "ollama"
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
    logger.info("Detected language: %s", "Russian" if detected_language == "ru" else "English")

    # Build pseudo-segments to reuse the existing pipeline.
    paragraphs = [p.strip() for p in text_content.split("\n\n") if p.strip()]

    original_segments = []
    for i, para in enumerate(paragraphs):
        if para:
            original_segments.append({
                "text": para,
                "start": None,
                "end": None,
                "speaker": None
            })

    logger.info("Document split into %d paragraphs", len(original_segments))

    # 1.5. Optional LLM-based refinement.
    refined_segments = None
    if refine_model:
        logger.info("\n[1.5/3] Refining text with %s...", refine_model)

        try:
            from .text_refiner import TextRefiner

            refiner = TextRefiner(backend=refine_backend, model_name=refine_model)
            refined_text = refiner.refine_text(text_content)
            refined_paragraphs = [p.strip() for p in refined_text.split("\n\n") if p.strip()]

            refined_segments = []
            for para in refined_paragraphs:
                if para:
                    refined_segments.append({
                        "text": para,
                        "start": None,
                        "end": None,
                        "speaker": None
                    })

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

            translator = Translator(method=method, model_name=translate_model)
            method_key = method
            if method == "NLLB":
                method_key = f"{method} ({translator.model_name})"
            segments_to_translate = refined_segments if refined_segments else original_segments

            try:
                translated_segments = translator.translate_segments(
                    segments_to_translate,
                    source_lang=detected_language,
                    target_lang="ru" if detected_language == "en" else "en"
                )

                translated_segments_dict[method_key] = translated_segments

            except Exception as e:
                logger.error("Translation failed with %s: %s", method, e)
                continue

            # Optional LLM post-translation polish.
            if refine_translation_model and method_key in translated_segments_dict:
                logger.info("  Refining translation with %s...", refine_translation_model)

                try:
                    from .text_refiner import TextRefiner
                    from .transcriber import TranscriptionSegment

                    translation_refiner = TextRefiner(model_name=refine_translation_model)
                    translated_text = "\n\n".join([seg.text for seg in translated_segments_dict[method_key]])
                    refined_translation = translation_refiner.refine_translation(translated_text)
                    refined_translation_paragraphs = [p.strip() for p in refined_translation.split("\n\n") if p.strip()]

                    # Rebuild segments for the refined translation.
                    refined_translated_segments = []
                    for i, para in enumerate(refined_translation_paragraphs):
                        if i < len(translated_segments_dict[method_key]):
                            refined_translated_segments.append(TranscriptionSegment(
                                text=para,
                                start=translated_segments_dict[method_key][i].start,
                                end=translated_segments_dict[method_key][i].end,
                                speaker=translated_segments_dict[method_key][i].speaker
                            ))

                    refined_translation_segments_dict[method_key] = refined_translated_segments
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
            with_timestamps=False
        )
        logger.info("  Saved refined markdown: %s", md_path_refined)

    # Create translated documents.
    if translated_segments_dict:
        for method, translated_segs in translated_segments_dict.items():
            logger.info("  Creating translated document (%s)...", method)

            docx_path_trans, md_path_trans = writer.create_from_segments(
                title=f"{text_title}_translated_{method}",
                transcription_segments=refined_segments if refined_segments else original_segments,
                translation_segments=translated_segs,
                transcribe_method=f"Loaded from {text_path_obj.suffix}" + (f" + {refine_model}" if refine_model else ""),
                translate_method=method,
                with_timestamps=False
            )
            logger.info("  Saved translation markdown: %s", md_path_trans)

    # Create refined translation documents.
    if refined_translation_segments_dict:
        for method, refined_translated_segs in refined_translation_segments_dict.items():
            logger.info("  Creating refined translation document (%s)...", method)

            docx_path_trans_refined, md_path_trans_refined = writer.create_from_segments(
                title=f"{text_title}_translated_{method}_refined",
                transcription_segments=refined_segments if refined_segments else original_segments,
                translation_segments=refined_translated_segs,
                transcribe_method=f"Loaded from {text_path_obj.suffix}" + (f" + {refine_model}" if refine_model else ""),
                translate_method=f"{method} + {refine_translation_model}",
                with_timestamps=False
            )
            logger.info("  Saved refined translation markdown: %s", md_path_trans_refined)

    # Generate summary if requested.
    if summarize and summarize_model:
        _generate_summary(
            title=text_title,
            segments=refined_segments if refined_segments else original_segments,
            summarize_model=summarize_model,
            summarize_backend=summarize_backend
        )

    # Warn if no output was produced.
    if not refined_segments and not translated_segments_dict and not (summarize and summarize_model):
        logger.warning("No --refine-model, --translate, or --summarize options were provided")
        logger.info("The source file is unchanged. Use --refine-model, --translate, or --summarize to generate output.")

    logger.info("\n" + "=" * 60)
    logger.info("Text document processing complete!")
    logger.info("=" * 60)



def validate_args(args) -> bool:
    """
    Validate CLI arguments.

    Args:
        args: Parsed argparse namespace.

    Returns:
        True if validation succeeded.
    """
    # Ensure only one input source is provided.
    input_count = sum([
        bool(args.url),
        bool(args.input_audio),
        bool(args.input_video),
        bool(args.input_text)
    ])

    if input_count == 0:
        logger.error("You must specify one input source: --url, --input_audio, --input_video, or --input_text")
        return False

    if input_count > 1:
        logger.error("Only one input source may be specified at a time")
        return False

    # Audio/YouTube/Video sources require a transcription backend.
    if (args.url or args.input_audio or args.input_video) and not args.transcribe:
        logger.error("Audio/YouTube/Video processing requires --transcribe to be set")
        return False


    # Validate referenced files.
    if args.input_audio:
        if not Path(args.input_audio).exists():
            logger.error("Audio file not found: %s", args.input_audio)
            return False

    if args.input_video:
        if not Path(args.input_video).exists():
            logger.error("Video file not found: %s", args.input_video)
            return False

    if args.input_text:
        if not Path(args.input_text).exists():
            logger.error("Text file not found: %s", args.input_text)
            return False

    # Validate refinement parameters early (before expensive operations).
    if args.refine_model:
        if args.refine_backend == "openai_api":
            # Check OpenAI API key.
            if not settings.OPENAI_API_KEY:
                logger.error("--refine-backend openai_api requires OPENAI_API_KEY in environment")
                logger.error("Set it in .env file: OPENAI_API_KEY=your-key-here")
                return False

            # Validate OpenAI library availability.
            try:
                import openai
            except ImportError:
                logger.error("OpenAI library not installed. Install it with: pip install openai>=1.6.0")
                return False

        elif args.refine_backend == "ollama":
            # Check Ollama server availability and model existence.
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code != 200:
                    logger.error("Cannot connect to Ollama server at http://localhost:11434")
                    logger.error("Please start Ollama: ollama serve")
                    return False

                # Check if the model exists.
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]

                if args.refine_model not in model_names:
                    logger.error("Model '%s' not found in Ollama", args.refine_model)
                    logger.error("Available models: %s", ', '.join(model_names) if model_names else 'none')
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

            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]

            if args.refine_translation not in model_names:
                logger.error("Model '%s' not found in Ollama", args.refine_translation)
                logger.error("Available models: %s", ', '.join(model_names) if model_names else 'none')
                logger.error("To download: ollama pull %s", args.refine_translation)
                return False

        except requests.exceptions.RequestException as e:
            logger.error("Cannot connect to Ollama server: %s", e)
            return False

    # Validate summarization parameters.
    if args.summarize and args.summarize_model:
        if args.summarize_backend == "openai_api":
            if not settings.OPENAI_API_KEY:
                logger.error("--summarize-backend openai_api requires OPENAI_API_KEY in environment")
                return False

            try:
                import openai
            except ImportError:
                logger.error("OpenAI library not installed. Install it with: pip install openai>=1.6.0")
                return False

        elif args.summarize_backend == "ollama":
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code != 200:
                    logger.error("Cannot connect to Ollama server for --summarize")
                    return False

                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]

                if args.summarize_model not in model_names:
                    logger.error("Model '%s' not found in Ollama", args.summarize_model)
                    logger.error("Available models: %s", ', '.join(model_names) if model_names else 'none')
                    logger.error("To download: ollama pull %s", args.summarize_model)
                    return False

            except requests.exceptions.RequestException as e:
                logger.error("Cannot connect to Ollama server: %s", e)
                return False

    # Validate translation backend if specified.
    if args.translate:
        translate_methods = [m.strip() for m in args.translate.split(',')]

        for method in translate_methods:
            if method not in ["NLLB", "openai_api"]:
                logger.error("Unknown translation method: %s", method)
                logger.error("Available methods: NLLB, openai_api")
                return False

            if method == "openai_api":
                if not settings.OPENAI_API_KEY:
                    logger.error("Translation method 'openai_api' requires OPENAI_API_KEY")
                    return False

                try:
                    import openai
                except ImportError:
                    logger.error("OpenAI library not installed. Install it with: pip install openai>=1.6.0")
                    return False

    # Validate transcription backend if specified.
    if args.transcribe:
        valid_transcribe_methods = ["whisper_base", "whisper_small", "whisper_medium", "whisper_openai_api"]
        if args.transcribe not in valid_transcribe_methods:
            logger.error("Unknown transcription method: %s", args.transcribe)
            logger.error("Available methods: %s", ', '.join(valid_transcribe_methods))
            return False

        if args.transcribe == "whisper_openai_api":
            if not settings.OPENAI_API_KEY:
                logger.error("Transcription method 'whisper_openai_api' requires OPENAI_API_KEY")
                return False

            try:
                import openai
            except ImportError:
                logger.error("OpenAI library not installed. Install it with: pip install openai>=1.6.0")
                return False

    return True


def process_youtube_video(
    url: str,
    transcribe_method: str,
    translate_methods: Optional[list[str]] = None,
    with_speakers: bool = False,
    custom_prompt: Optional[str] = None,
    refine_model: Optional[str] = None,
    refine_translation_model: Optional[str] = None,
    translate_model: Optional[str] = None,
    refine_backend: str = "ollama",
    summarize: bool = False,
    summarize_model: Optional[str] = None,
    summarize_backend: str = "ollama"
):
    """
    Process a YouTube video end-to-end.

    Args:
        url: Video URL.
        transcribe_method: Whisper backend to use.
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
        # Check if Ollama is available to build a prompt.
        try:
            import requests
            ollama_available = requests.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
        except:
            ollama_available = False

        if ollama_available and refine_model:
            # Reuse the refinement model to build the prompt.
            whisper_prompt = create_whisper_prompt_with_llm(metadata, use_ollama=True, model=refine_model)
            logger.info("Prompt generated from metadata via LLM")
        else:
            whisper_prompt = create_whisper_prompt(metadata)
            logger.info("Prompt generated from metadata (standard method)")

    transcriber = Transcriber(method=transcribe_method)
    transcription_segments = transcriber.transcribe(
        audio_path,
        language=None,  # Auto-detect
        with_speakers=with_speakers,
        initial_prompt=whisper_prompt
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
                transcription_segments,
                refined_text
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
            translator = Translator(method=translate_method, model_name=translate_model)

            if translate_method == "NLLB":
                translate_method_str = f"{translate_method} ({translator.model_name})"

            # Prefer translating the refined transcript when present.
            if refined_transcription_segments:
                logger.info("Translating refined transcript...")
                translation_segments_refined = translator.translate_segments(
                    refined_transcription_segments,
                    source_lang="en",
                    target_lang="ru"
                )
                logger.info("Refined transcript translation complete")
            else:
                # Otherwise translate the original transcript.
                translation_segments = translator.translate_segments(
                    original_transcription_segments,
                    source_lang="en",
                    target_lang="ru"
                )
                logger.info("Original transcript translation complete")
        else:
            logger.info("Source audio is Russian; translation skipped")
    else:
        logger.info("\n[3/4] Translation not requested")

    # 3.5. Optional translation refinement via LLM.
    translation_segments_refined_llm = None
    if refine_translation_model and (translation_segments_refined or translation_segments):
        logger.info("\n[3.5/4] Refining translation with %s...", refine_translation_model)
        try:
            from .text_refiner import TextRefiner

            refiner = TextRefiner(model_name=refine_translation_model)

            # Pick which translation output to refine.
            segments_to_refine = translation_segments_refined if translation_segments_refined else translation_segments

            # Convert segments to text.
            translated_text = transcriber.segments_to_text(segments_to_refine)

            # Apply refinement.
            refined_translation_text = refiner.refine_translation(translated_text, context=whisper_prompt)

            # Rebuild refined translation segments.
            translation_segments_refined_llm = transcriber.update_segments_from_text(
                segments_to_refine,
                refined_translation_text
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
            title=f"{video_title} (original)",
            transcription_segments=original_transcription_segments,
            translation_segments=None,
            transcribe_method=transcribe_method,
            translate_method="",
            with_timestamps=False
        )

        # Refined transcript (with translation if available, not LLM-polished).
        logger.info("Creating document with refined transcript...")
        docx_path_refined, md_path_refined = writer.create_from_segments(
            title=f"{video_title} (refined)",
            transcription_segments=refined_transcription_segments,
            translation_segments=translation_segments_refined,
            transcribe_method=f"{transcribe_method} + {refine_model}",
            translate_method=translate_method_str,
            with_timestamps=False
        )

        # Add refined translation document if available.
        if translation_segments_refined_llm:
            logger.info("Creating document with refined translation...")
            docx_path_trans_refined, md_path_trans_refined = writer.create_from_segments(
                title=f"{video_title} (translated refined)",
                transcription_segments=refined_transcription_segments if refined_transcription_segments else original_transcription_segments,
                translation_segments=translation_segments_refined_llm,
                transcribe_method=f"{transcribe_method} + {refine_model}" if refine_model else transcribe_method,
                translate_method=f"{translate_method_str} + {refine_translation_model}",
                with_timestamps=False
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
                title=f"{video_title} (translated)",
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments,
                transcribe_method=transcribe_method,
                translate_method=translate_method_str,
                with_timestamps=False
            )

            # Translation refined via LLM.
            logger.info("Creating document with refined translation...")
            docx_path_refined, md_path_refined = writer.create_from_segments(
                title=f"{video_title} (translated refined)",
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments_refined_llm,
                transcribe_method=transcribe_method,
                translate_method=f"{translate_method_str} + {refine_translation_model}",
                with_timestamps=False
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
                title=video_title,
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments,
                transcribe_method=transcribe_method,
                translate_method=translate_method_str,
                with_timestamps=False
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
        segments_for_summary = refined_transcription_segments or original_transcription_segments
        _generate_summary(
            title=video_title,
            segments=segments_for_summary,
            summarize_model=summarize_model,
            summarize_backend=summarize_backend
        )


def process_local_video(
    video_path: str,
    transcribe_method: str,
    translate_methods: Optional[list[str]] = None,
    with_speakers: bool = False,
    custom_prompt: Optional[str] = None,
    refine_model: Optional[str] = None,
    refine_translation_model: Optional[str] = None,
    translate_model: Optional[str] = None,
    refine_backend: str = "ollama",
    summarize: bool = False,
    summarize_model: Optional[str] = None,
    summarize_backend: str = "ollama"
):
    """
    Process a local video file by extracting audio and transcribing.

    Args:
        video_path: Path to the video file.
        transcribe_method: Transcription backend to use.
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
        language=None,  # Auto-detect
        with_speakers=with_speakers,
        initial_prompt=custom_prompt
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
                transcription_segments,
                refined_text
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
            translator = Translator(method=translate_method, model_name=translate_model)

            if translate_method == "NLLB":
                translate_method_str = f"{translate_method} ({translator.model_name})"

            # Translate the refined transcript when present.
            if refined_transcription_segments:
                logger.info("Translating refined transcript...")
                translation_segments_refined = translator.translate_segments(
                    refined_transcription_segments,
                    source_lang="en",
                    target_lang="ru"
                )
                logger.info("Refined transcript translation complete")
            else:
                # Otherwise translate the original transcript.
                translation_segments = translator.translate_segments(
                    transcription_segments,
                    source_lang="en",
                    target_lang="ru"
                )
                logger.info("Original transcript translation complete")
        else:
            logger.info("Audio is in Russian; translation skipped")
    else:
        logger.info("\n[3/4] Translation not requested")

    # 3.5. Optional translation refinement via LLM.
    translation_segments_refined_llm = None
    if refine_translation_model and (translation_segments_refined or translation_segments):
        logger.info("\n[3.5/4] Refining translation with %s...", refine_translation_model)
        try:
            from .text_refiner import TextRefiner

            refiner = TextRefiner(model_name=refine_translation_model)

            # Choose which translation output to refine.
            segments_to_refine = translation_segments_refined if translation_segments_refined else translation_segments

            # Convert segments to text.
            translated_text = transcriber.segments_to_text(segments_to_refine)

            # Refine translation.
            refined_translation_text = refiner.refine_translation(translated_text, context=custom_prompt)

            # Build refined translation segments.
            translation_segments_refined_llm = transcriber.update_segments_from_text(
                segments_to_refine,
                refined_translation_text
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
            title=f"{audio_title} (original)",
            transcription_segments=original_transcription_segments,
            translation_segments=None,
            transcribe_method=transcribe_method,
            translate_method="",
            with_timestamps=False
        )

        # Refined transcript (with translation if available).
        logger.info("Creating document with refined transcript...")
        docx_path_refined, md_path_refined = writer.create_from_segments(
            title=f"{audio_title} (refined)",
            transcription_segments=refined_transcription_segments,
            translation_segments=translation_segments_refined,
            transcribe_method=f"{transcribe_method} + {refine_model}",
            translate_method=translate_method_str,
            with_timestamps=False
        )

        # Create refined translation document if available.
        if translation_segments_refined_llm:
            logger.info("Creating document with refined translation...")
            docx_path_trans_refined, md_path_trans_refined = writer.create_from_segments(
                title=f"{audio_title} (translated refined)",
                transcription_segments=refined_transcription_segments if refined_transcription_segments else original_transcription_segments,
                translation_segments=translation_segments_refined_llm,
                transcribe_method=f"{transcribe_method} + {refine_model}" if refine_model else transcribe_method,
                translate_method=f"{translate_method_str} + {refine_translation_model}",
                with_timestamps=False
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
                title=f"{audio_title} (translated)",
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments,
                transcribe_method=transcribe_method,
                translate_method=translate_method_str,
                with_timestamps=False
            )

            # LLM-refined translation.
            logger.info("Creating document with refined translation...")
            docx_path_refined, md_path_refined = writer.create_from_segments(
                title=f"{audio_title} (translated refined)",
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments_refined_llm,
                transcribe_method=transcribe_method,
                translate_method=f"{translate_method_str} + {refine_translation_model}",
                with_timestamps=False
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
                with_timestamps=False
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
        segments_for_summary = refined_transcription_segments or original_transcription_segments
        _generate_summary(
            title=audio_title,
            segments=segments_for_summary,
            summarize_model=summarize_model,
            summarize_backend=summarize_backend
        )


def process_local_audio(
    audio_path: str,
    transcribe_method: str,
    translate_methods: Optional[list[str]] = None,
    with_speakers: bool = False,
    custom_prompt: Optional[str] = None,
    refine_model: Optional[str] = None,
    refine_translation_model: Optional[str] = None,
    translate_model: Optional[str] = None,
    refine_backend: str = "ollama",
    summarize: bool = False,
    summarize_model: Optional[str] = None,
    summarize_backend: str = "ollama"
):
    """
    Process a local audio file end-to-end.

    Args:
        audio_path: Path to the audio file.
        transcribe_method: Transcription backend to use.
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
        language=None,  # Auto-detect
        with_speakers=with_speakers,
        initial_prompt=custom_prompt
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
                transcription_segments,
                refined_text
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
            translator = Translator(method=translate_method, model_name=translate_model)

            if translate_method == "NLLB":
                translate_method_str = f"{translate_method} ({translator.model_name})"

            # Translate the refined transcript when present.
            if refined_transcription_segments:
                logger.info("Translating refined transcript...")
                translation_segments_refined = translator.translate_segments(
                    refined_transcription_segments,
                    source_lang="en",
                    target_lang="ru"
                )
                logger.info("Refined transcript translation complete")
            else:
                # Otherwise translate the original transcript.
                translation_segments = translator.translate_segments(
                    transcription_segments,
                    source_lang="en",
                    target_lang="ru"
                )
                logger.info("Original transcript translation complete")
        else:
            logger.info("Audio is in Russian; translation skipped")
    else:
        logger.info("\n[2/3] Translation not requested")

    # 2.5. Optional translation refinement via LLM.
    translation_segments_refined_llm = None
    if refine_translation_model and (translation_segments_refined or translation_segments):
        logger.info("\n[2.5/3] Refining translation with %s...", refine_translation_model)
        try:
            from .text_refiner import TextRefiner

            refiner = TextRefiner(model_name=refine_translation_model)

            # Choose which translation output to refine.
            segments_to_refine = translation_segments_refined if translation_segments_refined else translation_segments

            # Convert segments to text.
            translated_text = transcriber.segments_to_text(segments_to_refine)

            # Refine translation.
            refined_translation_text = refiner.refine_translation(translated_text, context=custom_prompt)

            # Build refined translation segments.
            translation_segments_refined_llm = transcriber.update_segments_from_text(
                segments_to_refine,
                refined_translation_text
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
            title=f"{audio_title} (original)",
            transcription_segments=original_transcription_segments,
            translation_segments=None,
            transcribe_method=transcribe_method,
            translate_method="",
            with_timestamps=False
        )

        # Refined transcript (with translation if available).
        logger.info("Creating document with refined transcript...")
        docx_path_refined, md_path_refined = writer.create_from_segments(
            title=f"{audio_title} (refined)",
            transcription_segments=refined_transcription_segments,
            translation_segments=translation_segments_refined,
            transcribe_method=f"{transcribe_method} + {refine_model}",
            translate_method=translate_method_str,
            with_timestamps=False
        )

        # Create refined translation document if available.
        if translation_segments_refined_llm:
            logger.info("Creating document with refined translation...")
            docx_path_trans_refined, md_path_trans_refined = writer.create_from_segments(
                title=f"{audio_title} (translated refined)",
                transcription_segments=refined_transcription_segments if refined_transcription_segments else original_transcription_segments,
                translation_segments=translation_segments_refined_llm,
                transcribe_method=f"{transcribe_method} + {refine_model}" if refine_model else transcribe_method,
                translate_method=f"{translate_method_str} + {refine_translation_model}",
                with_timestamps=False
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
                title=f"{audio_title} (translated)",
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments,
                transcribe_method=transcribe_method,
                translate_method=translate_method_str,
                with_timestamps=False
            )

            # LLM-refined translation.
            logger.info("Creating document with refined translation...")
            docx_path_refined, md_path_refined = writer.create_from_segments(
                title=f"{audio_title} (translated refined)",
                transcription_segments=original_transcription_segments,
                translation_segments=translation_segments_refined_llm,
                transcribe_method=transcribe_method,
                translate_method=f"{translate_method_str} + {refine_translation_model}",
                with_timestamps=False
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
                with_timestamps=False
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
        segments_for_summary = refined_transcription_segments or original_transcription_segments
        _generate_summary(
            title=audio_title,
            segments=segments_for_summary,
            summarize_model=summarize_model,
            summarize_backend=summarize_backend
        )


def main():
    """Application entry point."""
    parser = argparse.ArgumentParser(
        description="YouTube Transcriber & Translator",
        add_help=False
    )

    # Input sources.
    parser.add_argument('--url', type=str, help='YouTube video URL')
    parser.add_argument('--input_audio', type=str, help='Path to a local audio file')
    parser.add_argument('--input_video', type=str, help='Path to a local video file')
    parser.add_argument('--input_text', type=str, help='Path to a text document')

    # Processing backends.
    parser.add_argument('--transcribe', type=str, help='Transcription backend name')
    parser.add_argument('--translate', type=str, help='Translation backends (comma separated)')

    # Additional options.
    parser.add_argument('--prompt', type=str, help='Path to a Whisper prompt file')
    parser.add_argument('--refine-model', type=str, help='Model used to refine the transcript (e.g. qwen2.5:3b for Ollama, gpt-4 for OpenAI)')
    parser.add_argument('--refine-backend', type=str, choices=['ollama', 'openai_api'], default='ollama',
                        help='Backend for refinement: ollama (default) or openai_api')
    parser.add_argument('--refine-translation', type=str, help='Model used to refine the translation (e.g. qwen2.5:3b)')
    parser.add_argument('--speakers', action='store_true', help='Enable speaker diarisation (experimental)')
    parser.add_argument('--translate-model', type=str, help='NLLB model override (default: facebook/nllb-200-distilled-1.3B)')

    # Summarization options.
    parser.add_argument('--summarize', action='store_true', help='Generate a summary of the content')
    parser.add_argument('--summarize-model', type=str, help='Model for summarization (e.g. qwen2.5:7b for Ollama, gpt-4 for OpenAI)')
    parser.add_argument('--summarize-backend', type=str, choices=['ollama', 'openai_api'], default='ollama',
                        help='Backend for summarization: ollama (default) or openai_api')

    parser.add_argument('--help', '-h', action='store_true', help='Show this help message')

    args = parser.parse_args()

    # Show help when explicitly requested or no arguments were provided.
    if args.help or len(sys.argv) == 1:
        print_help()
        sys.exit(0)

    # Validate arguments.
    if not validate_args(args):
        sys.exit(1)

    try:
        # Parse translation methods.
        translate_methods = None
        if args.translate:
            translate_methods = [m.strip() for m in args.translate.split(',')]

        # Load a custom prompt if supplied.
        custom_prompt = None
        if args.prompt:
            custom_prompt = load_prompt_from_file(args.prompt)

        # Dispatch based on the selected input source.
        if args.url:
            process_youtube_video(
                url=args.url,
                transcribe_method=args.transcribe,
                translate_methods=translate_methods,
                with_speakers=args.speakers,
                custom_prompt=custom_prompt,
                refine_model=args.refine_model,
                refine_translation_model=args.refine_translation,
                translate_model=args.translate_model,
                refine_backend=args.refine_backend,
                summarize=args.summarize,
                summarize_model=args.summarize_model,
                summarize_backend=args.summarize_backend
            )
        elif args.input_audio:
            process_local_audio(
                audio_path=args.input_audio,
                transcribe_method=args.transcribe,
                translate_methods=translate_methods,
                with_speakers=args.speakers,
                custom_prompt=custom_prompt,
                refine_model=args.refine_model,
                refine_translation_model=args.refine_translation,
                translate_model=args.translate_model,
                refine_backend=args.refine_backend,
                summarize=args.summarize,
                summarize_model=args.summarize_model,
                summarize_backend=args.summarize_backend
            )
        elif args.input_video:
            process_local_video(
                video_path=args.input_video,
                transcribe_method=args.transcribe,
                translate_methods=translate_methods,
                with_speakers=args.speakers,
                custom_prompt=custom_prompt,
                refine_model=args.refine_model,
                refine_translation_model=args.refine_translation,
                translate_model=args.translate_model,
                refine_backend=args.refine_backend,
                summarize=args.summarize,
                summarize_model=args.summarize_model,
                summarize_backend=args.summarize_backend
            )
        elif args.input_text:
            process_text_file(
                text_path=args.input_text,
                translate_methods=translate_methods,
                refine_model=args.refine_model,
                refine_translation_model=args.refine_translation,
                translate_model=args.translate_model,
                refine_backend=args.refine_backend,
                summarize=args.summarize,
                summarize_model=args.summarize_model,
                summarize_backend=args.summarize_backend
            )


    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("An error occurred: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
