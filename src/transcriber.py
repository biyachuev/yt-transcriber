"""
Module responsible for audio transcription via Whisper or GigaAM.
"""

import atexit
import hashlib
import inspect
import re
import warnings
from functools import wraps
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

import torch
import whisper
from tqdm import tqdm

from src.config import settings, TranscribeOptions
from src.logger import logger, format_warning
from src.utils import format_timestamp, estimate_processing_time, format_log_preview
from src.api_cache import get_cache

# Suppress deprecation warnings from third-party libraries
# These are coming from pyannote.audio and speechbrain internals that we don't control
warnings.filterwarnings(
    "ignore",
    message=".*torchaudio._backend.list_audio_backends has been deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*torchaudio.load.*will be changed to use.*torchcodec.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Lightning automatically upgraded your loaded checkpoint.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Model was trained with.*yours is.*Bad things might happen.*",
    category=UserWarning,
)

# Global registry for cleanup of temporary chunk files
_temp_chunk_files: Set[Path] = set()


def _cleanup_temp_chunks():
    """Clean up any remaining temporary chunk files on program exit."""
    if _temp_chunk_files:
        logger.debug("Cleaning up %d temporary chunk files...", len(_temp_chunk_files))
        for chunk_path in list(_temp_chunk_files):
            try:
                if chunk_path.exists():
                    chunk_path.unlink()
                    logger.debug("Cleaned up: %s", chunk_path.name)
            except Exception as e:
                logger.warning("Failed to clean up %s: %s", chunk_path, e)
            finally:
                _temp_chunk_files.discard(chunk_path)


# Register cleanup handler
atexit.register(_cleanup_temp_chunks)

_pyannote_revision_patch_applied = False


def _patch_pyannote_revision_support():
    """
    Allow legacy 'repo@revision' checkpoint notation with newer pyannote.audio.
    """
    global _pyannote_revision_patch_applied
    if _pyannote_revision_patch_applied:
        return

    try:
        from pyannote.audio.core.model import Model
        from pyannote.audio.core.pipeline import Pipeline as PyannotePipeline
    except ImportError:
        logger.debug(
            "pyannote.audio not installed; skipping revision compatibility patch."
        )
        return

    def _patch_classmethod(cls, method_name: str) -> bool:
        descriptor = cls.__dict__.get(method_name)
        if not isinstance(descriptor, classmethod):
            return False

        original = descriptor.__func__
        if getattr(original, "_yt_revision_patch", False):
            return False

        @wraps(original)
        def wrapper(inner_cls, checkpoint, *args, **kwargs):
            new_checkpoint = checkpoint
            if isinstance(checkpoint, str):
                base, sep, revision = checkpoint.partition("@")
                if sep and "revision" not in kwargs:
                    new_checkpoint = base
                    kwargs["revision"] = revision
            return original(inner_cls, new_checkpoint, *args, **kwargs)

        wrapper._yt_revision_patch = True  # type: ignore[attr-defined]
        setattr(cls, method_name, classmethod(wrapper))
        return True

    patched = False
    for target_cls in (Model, PyannotePipeline):
        patched |= _patch_classmethod(target_cls, "from_pretrained")

    if patched:
        logger.debug("Enabled legacy '@revision' checkpoints for pyannote.audio.")

    _pyannote_revision_patch_applied = True


class TranscriptionSegment:
    """Lightweight wrapper representing a single transcription segment."""

    def __init__(
        self, start: float, end: float, text: str, speaker: Optional[str] = None
    ):
        self.start = start
        self.end = end
        self.text = text.strip()
        self.speaker = speaker

    def __repr__(self):
        speaker_prefix = f"[{self.speaker}] " if self.speaker else ""
        return f"[{format_timestamp(self.start)}] {speaker_prefix}{self.text}"

    def to_dict(self) -> dict:
        """Convert the segment to a dictionary representation."""
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "timestamp": format_timestamp(self.start),
            "speaker": self.speaker,
        }


def _compute_audio_hash(audio_path: Path, chunk_size: int = 8192) -> str:
    """
    Compute SHA-256 hash of an audio file for caching purposes.

    We only hash the first 1MB to speed up the process for large files,
    as transcription parameters also contribute to cache key uniqueness.

    Args:
        audio_path: Path to the audio file
        chunk_size: Size of chunks to read

    Returns:
        Hex digest of the file hash
    """
    hasher = hashlib.sha256()
    bytes_read = 0
    max_bytes = 1024 * 1024  # 1MB

    with open(audio_path, "rb") as f:
        while bytes_read < max_bytes:
            chunk = f.read(min(chunk_size, max_bytes - bytes_read))
            if not chunk:
                break
            hasher.update(chunk)
            bytes_read += len(chunk)

    return hasher.hexdigest()


class Transcriber:
    """ASR transcriber supporting Whisper and GigaAM."""

    def __init__(
        self, method: str = TranscribeOptions.WHISPER_BASE, use_cache: bool = True
    ):
        self.method = method
        self.model = None
        self.device = self._get_device()
        self.use_cache = use_cache
        self.cache = get_cache() if use_cache else None
        logger.info("Using device: %s", self.device)
        if use_cache:
            logger.info("Transcription caching enabled")

    def _is_gigaam_method(self) -> bool:
        """Return True if selected backend is a GigaAM model."""
        return self.method in (
            TranscribeOptions.GIGAAM_E2E_RNNT,
            TranscribeOptions.GIGAAM_E2E_CTC,
        )

    def _get_device(self) -> str:
        """Detect the best available device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        """Load the selected ASR model if it has not been loaded yet."""
        if self.model is not None:
            return

        if self._is_gigaam_method():
            self._load_gigaam_model()
            return

        logger.info("Loading Whisper model (%s)...", self.method)

        if self.method == TranscribeOptions.WHISPER_BASE:
            model_name = "base"
        elif self.method == TranscribeOptions.WHISPER_SMALL:
            model_name = "small"
        elif self.method == TranscribeOptions.WHISPER_MEDIUM:
            model_name = "medium"
        else:
            raise ValueError(f"Unsupported transcription method: {self.method}")

        try:
            self.model = whisper.load_model(
                model_name,
                device=self.device,
                download_root=str(settings.WHISPER_MODEL_DIR),
            )
        except NotImplementedError as exc:
            message = str(exc)
            if "SparseMPS" in message or "_sparse_coo_tensor" in message:
                logger.warning(
                    "Sparse operations are not supported on MPS. Falling back to CPU..."
                )
                self.device = "cpu"
                self.model = whisper.load_model(
                    model_name,
                    device=self.device,
                    download_root=str(settings.WHISPER_MODEL_DIR),
                )
            else:
                raise
        except RuntimeError as exc:
            message = str(exc)
            if "MPS" in message and (
                "sparse" in message.lower() or "_sparse_coo_tensor" in message
            ):
                logger.warning(
                    "Encountered an MPS issue while loading the model. Switching to CPU..."
                )
                self.device = "cpu"
                self.model = whisper.load_model(
                    model_name,
                    device=self.device,
                    download_root=str(settings.WHISPER_MODEL_DIR),
                )
            else:
                raise

        logger.info("Model loaded successfully")

    def _load_gigaam_model(self):
        """Load a GigaAM ASR model."""
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - transformers is already a dependency
            raise ValueError(
                "GigaAM backend requires transformers>=4.57.1. "
                "Install transformers or update your environment."
            ) from exc

        revision = None
        if self.method == TranscribeOptions.GIGAAM_E2E_RNNT:
            revision = "e2e_rnnt"
        elif self.method == TranscribeOptions.GIGAAM_E2E_CTC:
            revision = "e2e_ctc"

        if not revision:
            raise ValueError(f"Unsupported GigaAM model: {self.method}")

        logger.info("Loading GigaAM model (revision: %s)...", revision)

        cache_dir = settings.CACHE_DIR / "gigaam"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # trust_remote_code is required for custom GigaAM classes with .transcribe()
        model = AutoModel.from_pretrained(
            "ai-sage/GigaAM-v3",
            revision=revision,
            trust_remote_code=True,
            cache_dir=str(cache_dir),
        )

        try:
            model = model.to(self.device)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not move GigaAM model to %s: %s", self.device, exc)

        self.model = model
        logger.info("GigaAM model loaded successfully")

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        with_speakers: bool = False,
        initial_prompt: Optional[str] = None,
    ) -> List[TranscriptionSegment]:
        """
        Transcribe an audio file and return a list of transcription segments.

        Args:
            audio_path: Path to the audio file.
            language: Language code ('ru' or 'en'), or None to auto-detect.
            with_speakers: Whether to perform speaker diarisation (not yet supported).
            initial_prompt: Optional Whisper prompt to improve recognition.

        Returns:
            List of TranscriptionSegment instances.
        """
        logger.info("Starting transcription: %s", audio_path.name)

        if initial_prompt:
            if self._is_gigaam_method():
                logger.info(
                    "Initial prompt provided but GigaAM models ignore prompts; proceeding without it."
                )
            else:
                logger.info(
                    "Using initial prompt (length: %d chars)", len(initial_prompt)
                )
                logger.debug(
                    "Prompt preview (first 80 chars): %s",
                    format_log_preview(initial_prompt),
                )
        else:
            if not self._is_gigaam_method():
                logger.warning(
                    "No initial prompt provided. Consider using --whisper-prompt for better accuracy."
                )

        # Check cache first
        if self.use_cache:
            audio_hash = _compute_audio_hash(audio_path)
            cache_key = {
                "audio_hash": audio_hash,
                "method": self.method,
                "language": language,
                "with_speakers": with_speakers,
                "initial_prompt": initial_prompt or "",
            }
            cached_result = self.cache.get("transcription", cache_key)
            if cached_result is not None:
                logger.info("Using cached transcription result")
                # Reconstruct TranscriptionSegment objects from cached dict
                segments = [
                    TranscriptionSegment(
                        start=seg["start"],
                        end=seg["end"],
                        text=seg["text"],
                        speaker=seg.get("speaker"),
                    )
                    for seg in cached_result
                ]
                return segments

        # Use OpenAI API if specified
        if self.method == TranscribeOptions.WHISPER_OPENAI_API:
            segments = self._transcribe_with_openai_api(
                audio_path, language, initial_prompt
            )

            # Apply speaker diarization if requested
            if with_speakers:
                segments = self._perform_speaker_diarization(audio_path, segments)

            # Cache the result for OpenAI API transcriptions
            if self.use_cache:
                segments_dict = [seg.to_dict() for seg in segments]
                self.cache.set("transcription", cache_key, segments_dict)

            return segments

        if self._is_gigaam_method():
            self._load_model()
            segments = self._transcribe_with_gigaam(
                audio_path, language=language, initial_prompt=initial_prompt
            )

            if with_speakers:
                segments = self._perform_speaker_diarization(audio_path, segments)

            if self.use_cache:
                segments_dict = [seg.to_dict() for seg in segments]
                self.cache.set("transcription", cache_key, segments_dict)

            return segments

        self._load_model()

        import subprocess

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
            )
            duration = float(result.stdout.strip())
            estimate = estimate_processing_time(duration, "transcribe", self.method)
            logger.info("Estimated processing time: %s", estimate)
        except Exception as exc:  # pragma: no cover
            logger.debug("Unable to determine audio duration: %s", exc)

        transcribe_options: Dict[str, Optional[str]] = {
            "language": language,
            "task": "transcribe",
            "verbose": False,
            "fp16": True if self.device == "cuda" else False,
        }

        language_set = False
        pre_language = None
        pre_language_confidence = None
        first_speech_start = None

        if language is None and not self._is_gigaam_method():
            (
                pre_language,
                pre_language_confidence,
                first_speech_start,
            ) = self._detect_language_from_first_speech(audio_path)

            # Helper to infer a hint from filename/prompt scripts.
            def _lang_hint_from_text(text: str) -> Optional[str]:
                cyr = len(re.findall(r"[а-яА-ЯёЁ]", text))
                lat = len(re.findall(r"[a-zA-Z]", text))
                if cyr >= 4 and cyr > lat:
                    return "ru"
                if lat >= 4 and lat > cyr:
                    return "en"
                return None

            # 1) Trust high-confidence VAD-based detection.
            if pre_language and (pre_language_confidence or 0.0) >= 0.7:
                transcribe_options["language"] = pre_language
                language_set = True
                logger.info(
                    "Language auto-detected from first speech (%.1fs in): %s (p=%.2f)",
                    first_speech_start or 0.0,
                    format_warning(pre_language.upper()),
                    pre_language_confidence if pre_language_confidence else 0.0,
                )
            else:
                # 2) If low confidence, fall back to script hints from filename/prompt.
                hint_source = None
                filename_hint = _lang_hint_from_text(audio_path.stem)
                prompt_hint = (
                    _lang_hint_from_text(initial_prompt) if initial_prompt else None
                )

                if filename_hint:
                    transcribe_options["language"] = filename_hint
                    language_set = True
                    hint_source = "filename"
                elif prompt_hint:
                    transcribe_options["language"] = prompt_hint
                    language_set = True
                    hint_source = "prompt"

                if language_set:
                    logger.info(
                        "Language hint from %s: %s (first-speech detect: %s, p=%.2f)",
                        hint_source,
                        format_warning(transcribe_options["language"].upper()),
                        format_warning((pre_language or "unknown").upper()),
                        pre_language_confidence if pre_language_confidence else 0.0,
                    )
                elif pre_language:
                    logger.info(
                        "Language auto-detect from first speech low confidence (p=%.2f): %s; keeping Whisper auto mode",
                        pre_language_confidence if pre_language_confidence else 0.0,
                        format_warning(pre_language.upper()),
                    )

        if initial_prompt:
            transcribe_options["initial_prompt"] = initial_prompt

        logger.info("Transcription in progress...")

        def _run_transcription(options: Dict[str, Optional[str]]):
            res = self.model.transcribe(str(audio_path), **options)
            segs: List[TranscriptionSegment] = []
            for segment in tqdm(res["segments"], desc="Processing segments"):
                segs.append(
                    TranscriptionSegment(
                        start=segment["start"],
                        end=segment["end"],
                        text=segment["text"],
                    )
                )
            return res, segs

        result, segments = _run_transcription(transcribe_options)

        if initial_prompt and self._transcription_dominated_by_prompt(
            initial_prompt, segments
        ):
            logger.warning(
                "Initial prompt appears to dominate the transcription output; retrying without the prompt."
            )
            transcribe_options.pop("initial_prompt", None)
            result, segments = _run_transcription(transcribe_options)

        detected_language = result.get("language", "unknown")
        logger.info("Detected language: %s", format_warning(str(detected_language).upper()))
        logger.info("Transcription finished. Generated %d segments", len(segments))

        # Clean up hallucinations
        logger.info("Cleaning up potential hallucinations...")
        segments = self._clean_hallucinations(
            segments, expected_language=detected_language
        )
        logger.info("After cleanup: %d segments remain", len(segments))

        # Apply speaker diarization if requested
        if with_speakers:
            segments = self._perform_speaker_diarization(audio_path, segments)

        # Cache the result
        if self.use_cache:
            segments_dict = [seg.to_dict() for seg in segments]
            self.cache.set("transcription", cache_key, segments_dict)

        return segments

    def _get_audio_duration(self, audio_path: Path) -> Optional[float]:
        """Return audio duration in seconds using ffprobe."""
        import subprocess

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return float(result.stdout.strip())
        except Exception as exc:  # pragma: no cover
            logger.debug("Unable to determine audio duration: %s", exc)
            return None

    def _transcribe_with_gigaam(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> List[TranscriptionSegment]:
        """Transcribe audio using a GigaAM model with local chunking."""
        duration = self._get_audio_duration(audio_path)
        if duration:
            estimate = estimate_processing_time(duration, "transcribe", self.method)
            logger.info("Estimated processing time: %s", estimate)

        # Decide whether to chunk (GigaAM transcribe() supports ~25s max).
        if duration is not None and duration <= 24.0:
            chunks: List[Tuple[Path, float, float]] = [
                (audio_path, 0.0, duration)
            ]
        else:
            chunks = self._split_audio_for_gigaam(audio_path)
            logger.info("Prepared %d chunks for GigaAM", len(chunks))

        cleanup_after = any(chunk_path != audio_path for chunk_path, _, _ in chunks)

        segments: List[TranscriptionSegment] = []
        try:
            for chunk_path, start, end in chunks:
                text = self.model.transcribe(str(chunk_path))
                chunk_duration = end - start if end > start else 0.0
                if chunk_duration == 0.0:
                    detected_chunk_duration = self._get_audio_duration(chunk_path)
                    if detected_chunk_duration:
                        chunk_duration = detected_chunk_duration
                end_time = start + chunk_duration
                segments.append(
                    TranscriptionSegment(start=start, end=end_time, text=text)
                )
        finally:
            if cleanup_after:
                for chunk_path, _, _ in chunks:
                    if chunk_path == audio_path:
                        continue
                    try:
                        if chunk_path.exists():
                            chunk_path.unlink()
                        _temp_chunk_files.discard(chunk_path)
                    except Exception as exc:  # pragma: no cover
                        logger.debug("Failed to delete temp chunk %s: %s", chunk_path, exc)

        # Clean up common hallucinations lightly using expected language hint (default ru).
        cleaned_segments = self._clean_hallucinations(
            segments, expected_language=language or "ru"
        )
        return cleaned_segments

    def _get_vad_pipeline(self):
        """
        Get or create VAD pipeline for speech detection.

        Returns:
            pyannote VAD pipeline or None if not available.
        """
        if hasattr(self, "_vad_pipeline"):
            logger.debug("Returning cached VAD pipeline")
            return self._vad_pipeline

        logger.debug("Initializing VAD pipeline...")
        try:
            logger.debug("Attempting to import pyannote.audio...")
            from pyannote.audio import Pipeline

            logger.debug("Successfully imported Pipeline from pyannote.audio")
            import os

            _patch_pyannote_revision_support()
            logger.debug("Revision patch applied")

            # Check if HuggingFace token is available
            hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
            vad_revision = os.environ.get("PYANNOTE_VAD_REVISION") or "main"

            if not hf_token:
                logger.warning(
                    "HUGGINGFACE_TOKEN not found. VAD-based splitting disabled. "
                    "Set HF_TOKEN in .env to enable smart chunking."
                )
                self._vad_pipeline = None
                return None

            load_kwargs = self._build_pipeline_load_kwargs(
                Pipeline,
                hf_token,
                vad_revision,
            )

            # pyannote.audio 4.0+ requires explicit revision parameter
            if "revision" not in load_kwargs and vad_revision:
                load_kwargs["revision"] = vad_revision

            token_param = None
            if "token" in load_kwargs:
                token_param = "token"
            elif "use_auth_token" in load_kwargs:
                token_param = "use_auth_token"
            if token_param:
                logger.debug(
                    "Passing HuggingFace token via '%s' parameter.", token_param
                )

            logger.info(
                "Loading pyannote VAD pipeline (revision: %s)...",
                load_kwargs.get("revision", "default"),
            )
            logger.debug(
                "Note: Version compatibility warnings from pyannote.audio and PyTorch are expected and can be safely ignored"
            )

            try:
                # Note: pyannote.audio may print version warnings to stderr during model loading.
                # These are harmless compatibility warnings from the upstream library.
                self._vad_pipeline = Pipeline.from_pretrained(
                    "pyannote/voice-activity-detection", **load_kwargs
                )
            except TypeError as type_error:
                error_msg = str(type_error)
                logger.debug(
                    "Retrying VAD pipeline load due to signature mismatch: %s",
                    type_error,
                )

                fallback_kwargs = load_kwargs.copy()

                # Try different combinations of parameters
                # 1. First, try swapping token parameter name
                if "unexpected keyword argument 'token'" in error_msg:
                    # Version 3.x uses use_auth_token
                    if "token" in fallback_kwargs:
                        fallback_kwargs["use_auth_token"] = fallback_kwargs.pop("token")
                elif "unexpected keyword argument 'use_auth_token'" in error_msg:
                    # Version 4.x uses token
                    if "use_auth_token" in fallback_kwargs:
                        fallback_kwargs["token"] = fallback_kwargs.pop("use_auth_token")

                # 2. Try removing revision if it's the issue
                if "unexpected keyword argument 'revision'" in error_msg:
                    fallback_kwargs.pop("revision", None)

                if fallback_kwargs == load_kwargs:
                    raise

                self._vad_pipeline = Pipeline.from_pretrained(
                    "pyannote/voice-activity-detection", **fallback_kwargs
                )

            logger.info("VAD pipeline loaded successfully")
            return self._vad_pipeline

        except ImportError as import_error:
            # Check if it's missing omegaconf specifically (common issue)
            error_msg = str(import_error)
            if "omegaconf" in error_msg.lower():
                logger.warning(
                    "Missing required dependency 'omegaconf' for pyannote.audio. "
                    "Install with: pip install omegaconf>=2.3.0"
                )
            else:
                logger.warning(
                    "pyannote.audio not properly installed (%s). "
                    "Falling back to simple time-based splitting. "
                    "Install with: pip install pyannote.audio omegaconf",
                    error_msg,
                )
            self._vad_pipeline = None
            return None
        except Exception as e:
            logger.warning(
                "Failed to load VAD pipeline: %s. Using simple splitting.", e
            )
            logger.debug("VAD pipeline error details:", exc_info=True)
            self._vad_pipeline = None
            return None

    def _get_diarization_pipeline(self):
        """
        Get or create speaker diarization pipeline.

        Returns:
            pyannote diarization pipeline or None if not available.
        """
        if hasattr(self, "_diarization_pipeline"):
            return self._diarization_pipeline

        try:
            from pyannote.audio import Pipeline
            import os

            _patch_pyannote_revision_support()

            # Check if HuggingFace token is available
            hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
            diarization_revision = (
                os.environ.get("PYANNOTE_DIARIZATION_REVISION") or "main"
            )

            if not hf_token:
                logger.warning(
                    "HUGGINGFACE_TOKEN not found. Speaker diarization disabled. "
                    "To enable:\n"
                    "  1. Get token: https://huggingface.co/settings/tokens\n"
                    "  2. Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                    "  3. Set HF_TOKEN in .env"
                )
                self._diarization_pipeline = None
                return None

            load_kwargs = self._build_pipeline_load_kwargs(
                Pipeline,
                hf_token,
                diarization_revision,
            )

            token_param = None
            if "token" in load_kwargs:
                token_param = "token"
            elif "use_auth_token" in load_kwargs:
                token_param = "use_auth_token"
            if token_param:
                logger.debug(
                    "Passing HuggingFace token via '%s' parameter for diarization pipeline.",
                    token_param,
                )

            logger.info(
                "Loading pyannote speaker diarization pipeline (revision: %s)...",
                load_kwargs.get("revision") or diarization_revision or "default",
            )
            logger.debug(
                "Note: Version compatibility warnings from pyannote.audio are expected and can be safely ignored"
            )
            try:
                # Note: pyannote.audio may print version warnings to stderr during model loading.
                # These are harmless compatibility warnings from the upstream library.
                self._diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", **load_kwargs
                )
            except TypeError as type_error:
                error_msg = str(type_error)
                logger.debug(
                    "Retrying speaker diarization pipeline load due to signature mismatch: %s",
                    type_error,
                )

                fallback_kwargs = load_kwargs.copy()

                # Try different combinations of parameters
                # 1. First, try swapping token parameter name
                if "unexpected keyword argument 'token'" in error_msg:
                    # Version 3.x uses use_auth_token
                    if "token" in fallback_kwargs:
                        fallback_kwargs["use_auth_token"] = fallback_kwargs.pop("token")
                elif "unexpected keyword argument 'use_auth_token'" in error_msg:
                    # Version 4.x uses token
                    if "use_auth_token" in fallback_kwargs:
                        fallback_kwargs["token"] = fallback_kwargs.pop("use_auth_token")

                # 2. Try removing revision if it's the issue
                if "unexpected keyword argument 'revision'" in error_msg:
                    fallback_kwargs.pop("revision", None)

                if fallback_kwargs == load_kwargs:
                    raise

                self._diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", **fallback_kwargs
                )
            logger.info("Speaker diarization pipeline loaded successfully")
            return self._diarization_pipeline

        except ImportError:
            logger.warning(
                "pyannote.audio not installed. Speaker diarization disabled. "
                "Install with: pip install pyannote.audio"
            )
            self._diarization_pipeline = None
            return None
        except Exception as e:
            logger.warning("Failed to load diarization pipeline: %s", e)
            self._diarization_pipeline = None
            return None

    def _preprocess_audio_for_diarization(
        self, audio_path: Path, apply_noise_reduction: bool = False
    ) -> tuple:
        """
        Preprocess audio for better speaker diarization quality.

        Applies:
        1. Conversion to mono 16kHz
        2. Volume normalization
        3. Optional noise reduction

        Args:
            audio_path: Path to the audio file.
            apply_noise_reduction: Whether to apply noise reduction (slower but better quality).

        Returns:
            Tuple of (waveform_tensor, sample_rate)

        Raises:
            ImportError: If required libraries (torch, numpy, soundfile/librosa) are not available.
        """
        # Import required libraries (will raise ImportError if not available)
        # This is intentional - caller should handle ImportError and fallback to direct file loading
        import torch as torch_lib
        import numpy as np

        logger.debug("Preprocessing audio for speaker diarization...")

        # Load audio
        try:
            import soundfile as sf

            waveform_np, sample_rate = sf.read(str(audio_path), dtype="float32")
        except ImportError:
            import librosa

            waveform_np, sample_rate = librosa.load(
                str(audio_path), sr=None, mono=False
            )

        # Convert to mono if stereo
        if len(waveform_np.shape) > 1:
            logger.debug("Converting stereo to mono...")
            waveform_np = waveform_np.mean(axis=1)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            logger.debug("Resampling from %d Hz to 16000 Hz...", sample_rate)
            from scipy import signal

            num_samples = int(len(waveform_np) * 16000 / sample_rate)
            waveform_np = signal.resample(waveform_np, num_samples)
            sample_rate = 16000

        # Normalize volume (RMS normalization to -20 dBFS)
        # This prevents quiet sections from being classified as different speakers
        logger.debug("Normalizing volume...")
        rms = np.sqrt(np.mean(waveform_np**2))
        if rms > 0:
            target_rms = 0.1  # -20 dBFS roughly
            waveform_np = waveform_np * (target_rms / rms)

        # Clip to prevent distortion
        waveform_np = np.clip(waveform_np, -1.0, 1.0)

        # Optional: Apply noise reduction
        if apply_noise_reduction:
            try:
                import importlib

                nr = importlib.import_module("noisereduce")
                logger.debug("Applying noise reduction...")
                # Use stationary noise reduction (assumes consistent background noise)
                waveform_np = nr.reduce_noise(
                    y=waveform_np,
                    sr=sample_rate,
                    stationary=True,
                    prop_decrease=0.8,  # Reduce noise by 80%
                )
            except ImportError:
                logger.warning(
                    "noisereduce not installed. Skipping noise reduction. "
                    "Install with: pip install noisereduce"
                )
            except Exception as e:
                logger.warning("Noise reduction failed: %s. Continuing without it.", e)

        # Convert to torch tensor with correct shape (channel, time)
        # Ensure float32 dtype without creating extra copy if already float32
        if waveform_np.dtype != np.float32:
            waveform_np = waveform_np.astype(np.float32)

        waveform = torch_lib.from_numpy(waveform_np).unsqueeze(0)

        logger.debug("Audio preprocessing complete")
        return waveform, sample_rate

    def _perform_speaker_diarization(
        self, audio_path: Path, segments: List[TranscriptionSegment]
    ) -> List[TranscriptionSegment]:
        """
        Perform speaker diarization and assign speaker labels to segments.

        Args:
            audio_path: Path to the audio file.
            segments: Transcription segments without speaker labels.

        Returns:
            Updated segments with speaker labels.
        """
        diarization_pipeline = self._get_diarization_pipeline()
        if not diarization_pipeline:
            logger.warning(
                "Speaker diarization not available. Segments will not have speaker labels."
            )
            return segments

        logger.info("Performing speaker diarization...")

        try:
            # Preprocess audio for better diarization quality
            # Includes: mono conversion, 16kHz resampling, volume normalization
            diarization = None
            preprocessing_failed = False

            try:
                waveform, sample_rate = self._preprocess_audio_for_diarization(
                    audio_path,
                    apply_noise_reduction=False,  # TODO: Make this configurable via CLI
                )

                # Create audio dict that pyannote expects
                audio_dict = {"waveform": waveform, "sample_rate": sample_rate}

                logger.debug("Running diarization with preprocessed audio...")
                # Run diarization with preprocessed audio
                diarization = diarization_pipeline(audio_dict)

            except ImportError as e:
                logger.warning(
                    "Audio preprocessing libraries not available (%s). Using direct file loading...",
                    e,
                )
                preprocessing_failed = True
            except Exception as e:
                logger.warning(
                    "Failed to preprocess audio (%s: %s). Using direct file loading...",
                    type(e).__name__,
                    str(e),
                )
                preprocessing_failed = True

            # Fallback to direct file path if preprocessing failed
            if preprocessing_failed:
                logger.debug("Loading audio directly from file...")
                # pyannote.audio pipeline can handle file paths directly
                # This uses the pipeline's internal audio loading
                diarization = diarization_pipeline(audio_path)

            # Build a mapping of time ranges to speakers
            # Format: list of (start, end, speaker_label)
            # Note: In pyannote.audio 4.0+, diarization returns a DiarizeOutput object
            # with speaker_diarization and exclusive_speaker_diarization attributes
            speaker_timeline = []

            # Use exclusive_speaker_diarization if available (4.0+), otherwise use speaker_diarization
            # exclusive_speaker_diarization ensures only one speaker is active at any time,
            # which simplifies reconciliation with transcription timestamps
            if hasattr(diarization, "exclusive_speaker_diarization"):
                annotation = diarization.exclusive_speaker_diarization
            elif hasattr(diarization, "speaker_diarization"):
                annotation = diarization.speaker_diarization
            else:
                # Fallback for older versions (3.x) that return Annotation directly
                annotation = diarization

            for turn, _, speaker in annotation.itertracks(yield_label=True):
                speaker_timeline.append((turn.start, turn.end, speaker))

            # Sort speaker timeline by start time for potential optimization
            # (though pyannote usually returns sorted results)
            speaker_timeline.sort(key=lambda x: x[0])

            logger.info("Found %d speaker turns", len(speaker_timeline))

            # Assign speakers to transcription segments
            # Optimized approach: for each transcription segment,
            # find overlapping speaker turns and select the one with maximum overlap
            updated_segments = []
            for seg in segments:
                best_speaker = None
                max_overlap = 0.0

                # Binary search could be used here for large timelines,
                # but for typical use cases (< 1000 speaker turns),
                # linear search with early termination is sufficient
                for spk_start, spk_end, spk_label in speaker_timeline:
                    # Early termination: if speaker turn starts after segment ends
                    if spk_start >= seg.end:
                        break

                    # Skip if speaker turn ends before segment starts
                    if spk_end <= seg.start:
                        continue

                    # Calculate overlap between segment and speaker turn
                    overlap_start = max(seg.start, spk_start)
                    overlap_end = min(seg.end, spk_end)
                    overlap_duration = overlap_end - overlap_start

                    if overlap_duration > max_overlap:
                        max_overlap = overlap_duration
                        best_speaker = spk_label

                # Create updated segment with speaker label
                updated_seg = TranscriptionSegment(
                    start=seg.start, end=seg.end, text=seg.text, speaker=best_speaker
                )
                updated_segments.append(updated_seg)

            # Count unique speakers
            unique_speakers = set(
                seg.speaker for seg in updated_segments if seg.speaker
            )
            logger.info("Identified %d unique speakers", len(unique_speakers))

            return updated_segments

        except Exception as e:
            logger.error("Speaker diarization failed: %s", e)
            logger.warning("Continuing without speaker labels")
            return segments

    def _build_pipeline_load_kwargs(
        self,
        pipeline_cls,
        hf_token: Optional[str],
        revision: Optional[str],
    ) -> Dict[str, str]:
        """
        Build keyword arguments for Pipeline.from_pretrained across pyannote versions.
        """
        kwargs: Dict[str, str] = {}

        try:
            signature = inspect.signature(pipeline_cls.from_pretrained)
            parameters = signature.parameters
        except (TypeError, ValueError):
            # Fallback if signature inspection fails; assume modern interface.
            parameters = {}

        if hf_token:
            if "token" in parameters:
                kwargs["token"] = hf_token
            elif "use_auth_token" in parameters:
                kwargs["use_auth_token"] = hf_token
            else:
                logger.warning(
                    "pyannote.audio Pipeline.from_pretrained does not accept a HuggingFace "
                    "token parameter. Make sure credentials are stored via `huggingface-cli login`."
                )

        if revision:
            if not parameters or "revision" in parameters:
                kwargs["revision"] = revision
            else:
                logger.debug(
                    "pyannote.audio version does not support the 'revision' kwarg; "
                    "default model revision will be used."
                )

        return kwargs

    def _validate_audio_path(self, audio_path: Path) -> None:
        """
        Validate audio file path for security and existence.

        Args:
            audio_path: Path to validate.

        Raises:
            ValueError: If path is invalid or unsafe.
            FileNotFoundError: If file doesn't exist.
        """
        # Check if file exists
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Check if it's actually a file (not a directory or special file)
        if not audio_path.is_file():
            raise ValueError(f"Path is not a regular file: {audio_path}")

        # Check for suspicious characters that could indicate command injection attempts
        # Note: subprocess.run with list args is safe, but this is defense in depth
        suspicious_chars = [";", "&", "|", "`", "$", "\n", "\r"]
        path_str = str(audio_path)
        for char in suspicious_chars:
            if char in path_str:
                logger.warning(
                    "Suspicious character '%s' found in path: %s", char, path_str
                )
                # Don't raise error - just log warning as Path objects are safe with subprocess

        # Check file size is reasonable (< 10 GB)
        max_size = 10 * 1024 * 1024 * 1024  # 10 GB
        file_size = audio_path.stat().st_size
        if file_size > max_size:
            raise ValueError(
                f"Audio file too large: {file_size / (1024**3):.2f} GB. "
                f"Maximum supported: {max_size / (1024**3):.0f} GB"
            )

    def _get_silero_vad(self):
        """
        Get or create Silero VAD model (lightweight alternative to pyannote).

        Returns:
            Silero VAD model or None if not available.
        """
        if hasattr(self, "_silero_vad_model"):
            return self._silero_vad_model

        try:
            import torch

            logger.debug("Attempting to load Silero VAD model...")

            # Load Silero VAD from torch hub (1.8MB model)
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )

            # Cache the model and utilities
            self._silero_vad_model = model
            self._silero_vad_utils = utils

            logger.info("Silero VAD model loaded successfully (1.8MB)")
            return model

        except Exception as e:
            logger.debug("Could not load Silero VAD: %s", e)
            self._silero_vad_model = None
            return None

    def _find_speech_boundaries(self, audio_path: Path) -> List[tuple]:
        """
        Use VAD to find speech segment boundaries.
        Tries Silero VAD first (lightweight), falls back to pyannote if unavailable.

        Args:
            audio_path: Path to audio file.

        Returns:
            List of (start, end) tuples in seconds for speech segments.
        """
        # Try Silero VAD first (faster, lighter)
        silero_model = self._get_silero_vad()
        if silero_model is not None:
            try:
                return self._find_speech_boundaries_silero(audio_path, silero_model)
            except Exception as e:
                logger.warning("Silero VAD failed: %s. Trying pyannote...", e)

        # Fall back to pyannote VAD
        vad_pipeline = self._get_vad_pipeline()
        if not vad_pipeline:
            return []

        logger.info("Detecting speech boundaries with pyannote VAD...")

        try:
            # Run VAD
            vad_result = vad_pipeline(str(audio_path))

            # Extract speech segments
            speech_segments = []
            for speech in vad_result.get_timeline().support():
                speech_segments.append((speech.start, speech.end))

            logger.info("Found %d speech segments", len(speech_segments))
            return speech_segments

        except Exception as e:
            logger.warning("VAD failed: %s. Falling back to simple splitting.", e)
            return []

    def _find_speech_boundaries_silero(self, audio_path: Path, model) -> List[tuple]:
        """
        Use Silero VAD to find speech boundaries (faster alternative to pyannote).

        Args:
            audio_path: Path to audio file.
            model: Silero VAD model.

        Returns:
            List of (start, end) tuples in seconds for speech segments.
        """
        import torch
        import torchaudio

        logger.info("Detecting speech boundaries with Silero VAD...")

        # Load audio
        wav, sample_rate = torchaudio.load(str(audio_path))

        # Convert to mono if needed
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        # Resample to 16kHz if needed (Silero VAD requirement)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            wav = resampler(wav)
            sample_rate = 16000

        # Get VAD utilities
        utils = self._silero_vad_utils
        get_speech_timestamps = utils[0]

        # Run VAD with optimized parameters
        # Slightly more sensitive settings to surface more pauses for chunk splitting.
        speech_timestamps = get_speech_timestamps(
            wav.squeeze(0),
            model,
            sampling_rate=sample_rate,
            threshold=0.45,  # Lower threshold → more segments detected
            min_speech_duration_ms=220,  # Allow shorter speech fragments
            min_silence_duration_ms=220,  # Treat shorter gaps as splits
            window_size_samples=512,  # Processing window
            speech_pad_ms=80,  # Add padding to avoid cutting words
        )

        # Convert to seconds
        speech_segments = []
        for ts in speech_timestamps:
            start_sec = ts["start"] / sample_rate
            end_sec = ts["end"] / sample_rate
            speech_segments.append((start_sec, end_sec))

        logger.info("Found %d speech segments with Silero VAD", len(speech_segments))
        return speech_segments

    def _detect_language_from_first_speech(
        self, audio_path: Path
    ) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        """
        Detect language using the first speech fragment instead of the file start.

        Returns:
            Tuple of (language_code, confidence, speech_start_seconds).
        """
        try:
            speech_segments = self._find_speech_boundaries(audio_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "Failed to detect speech boundaries for language guess: %s", exc
            )
            return None, None, None

        if not speech_segments:
            logger.debug("No speech segments found; falling back to Whisper auto-detect")
            return None, None, None

        first_start, first_end = speech_segments[0]

        # Build a richer speech window (helps when the very first words are off-language).
        target_speech_seconds = 25.0  # aim for this much voiced audio
        max_window_seconds = 180.0  # absolute cap to keep detection quick

        speech_covered = max(0.0, first_end - first_start)
        window_end = first_end

        for seg_start, seg_end in speech_segments[1:]:
            # Stop expanding if we already have enough speech or went too far in time.
            if speech_covered >= target_speech_seconds:
                break
            if seg_start - first_start > max_window_seconds:
                break

            speech_covered += max(0.0, seg_end - seg_start)
            window_end = max(window_end, seg_end)

        window_end = min(window_end, first_start + max_window_seconds)
        if window_end <= first_start:
            logger.debug("Speech window is empty; skipping language pre-detection")
            return None, None, None

        try:
            audio = whisper.load_audio(str(audio_path))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to load audio for language detection: %s", exc)
            return None, None, None

        sample_rate = whisper.audio.SAMPLE_RATE
        start_sample = int(max(0.0, first_start) * sample_rate)
        end_sample = int(min(len(audio), window_end * sample_rate))

        if start_sample >= len(audio):
            logger.debug("Speech start is beyond audio length; skipping pre-detection")
            return None, None, None

        if end_sample - start_sample < sample_rate * 2:
            # Ensure we have a few seconds of actual speech for detection.
            end_sample = min(len(audio), start_sample + sample_rate * 10)

        audio_chunk = audio[start_sample:end_sample]
        if audio_chunk.size == 0:
            logger.debug("Empty audio chunk for language detection")
            return None, None, None

        mel = whisper.log_mel_spectrogram(
            whisper.pad_or_trim(audio_chunk)
        ).to(self.model.device)
        _, probs = self.model.detect_language(mel)

        language = max(probs, key=probs.get)
        confidence = float(probs[language])

        return language, confidence, first_start

    def _create_chunks_from_points(
        self, audio_path: Path, split_points: List[tuple], prefix: str = "chunk"
    ) -> List[Tuple[Path, float, float]]:
        """
        Export chunks based on provided split points.
        """
        import subprocess

        chunks: List[Tuple[Path, float, float]] = []
        temp_dir = audio_path.parent
        base_name = audio_path.stem
        original_extension = audio_path.suffix

        for i, (start_time, end_time) in enumerate(split_points):
            chunk_path = temp_dir / f"{base_name}_{prefix}_{i}{original_extension}"
            duration = end_time - start_time

            # Try stream copy first (fast, no re-encoding)
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        str(audio_path),
                        "-ss",
                        str(start_time),
                        "-t",
                        str(duration),
                        "-c",
                        "copy",
                        "-y",
                        str(chunk_path),
                    ],
                    capture_output=True,
                    check=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                # Stream copy failed, try re-encoding
                logger.warning(
                    "Stream copy failed for chunk %d (error: %s). Re-encoding...",
                    i,
                    e.stderr[:100] if e.stderr else "unknown",
                )

                # Use MP3 encoding as fallback (widely compatible)
                chunk_path = temp_dir / f"{base_name}_{prefix}_{i}.mp3"
                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-i",
                            str(audio_path),
                            "-ss",
                            str(start_time),
                            "-t",
                            str(duration),
                            "-c:a",
                            "libmp3lame",
                            "-b:a",
                            "192k",
                            "-y",
                            str(chunk_path),
                        ],
                        capture_output=True,
                        check=True,
                        text=True,
                    )
                    logger.debug("Successfully re-encoded chunk %d to MP3", i)
                except subprocess.CalledProcessError as e2:
                    logger.error(
                        "Failed to create chunk %d even with re-encoding: %s",
                        i,
                        e2.stderr[:200] if e2.stderr else "unknown error",
                    )
                    raise

            # Register chunk for cleanup
            _temp_chunk_files.add(chunk_path)

            chunks.append((chunk_path, start_time, end_time))
            logger.debug(
                "Created chunk %d/%d: %s (%.1f-%.1f sec, %.1f sec duration)",
                i + 1,
                len(split_points),
                chunk_path.name,
                start_time,
                end_time,
                duration,
            )

        return chunks

    def _split_audio_for_gigaam(
        self, audio_path: Path, target_chunk_duration: float = 20.0
    ) -> List[Tuple[Path, float, float]]:
        """
        Split audio into ~target_chunk_duration pieces for GigaAM (needs <=25s).
        """
        import math

        duration = self._get_audio_duration(audio_path)
        if duration is None:
            logger.warning(
                "Could not determine audio duration; proceeding without pre-chunking."
            )
            return [(audio_path, 0.0, 0.0)]

        logger.info(
            "Splitting audio (%.1fs) into ~%.0fs chunks for GigaAM long-form",
            duration,
            target_chunk_duration,
        )

        speech_segments = self._find_speech_boundaries(audio_path)
        split_points = self._calculate_split_points(
            duration, target_chunk_duration, speech_segments
        )
        if not split_points:
            # Edge case: extremely short files.
            return [(audio_path, 0.0, duration)]

        # Guardrail: ensure we won't exceed the transcribe() 25s limit.
        max_chunk = max(end - start for start, end in split_points)
        if max_chunk > 25.0:
            logger.warning(
                "Calculated chunk duration %.1fs exceeds 25s limit; increasing count.",
                max_chunk,
            )
            num_chunks = math.ceil(duration / 20.0)
            adjusted_points = []
            chunk_duration = duration / num_chunks
            for i in range(num_chunks):
                start = i * chunk_duration
                end = min((i + 1) * chunk_duration, duration)
                adjusted_points.append((start, end))
            split_points = adjusted_points

        return self._create_chunks_from_points(audio_path, split_points, prefix="giga")

    def _split_audio_file(self, audio_path: Path, max_size_mb: int = 24) -> List[tuple]:
        """
        Split audio file into chunks that fit within size limit.
        Uses VAD to find optimal split points at speech boundaries.

        Args:
            audio_path: Path to the audio file.
            max_size_mb: Maximum size per chunk in MB (default 24 MB to leave margin).

        Returns:
            List of tuples (chunk_path, start_time, end_time) for each chunk.
        """
        import subprocess

        # Validate audio path for security
        self._validate_audio_path(audio_path)

        logger.info("Splitting audio file into chunks...")

        # Get audio duration
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        total_duration = float(result.stdout.strip())

        # Get actual bitrate for accurate chunk size estimation
        bitrate_result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=bit_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=False,  # Don't fail if bitrate is unavailable
        )

        # Calculate target chunk duration more accurately
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)

        if bitrate_result.returncode == 0 and bitrate_result.stdout.strip():
            try:
                # Bitrate in bits per second
                bitrate_bps = float(bitrate_result.stdout.strip())
                # Calculate MB per second
                mb_per_second = (bitrate_bps / 8) / (1024 * 1024)
                # Calculate duration that fits in max_size_mb
                target_chunk_duration = (
                    max_size_mb / mb_per_second if mb_per_second > 0 else total_duration
                )
                num_chunks = int(total_duration / target_chunk_duration) + 1

                logger.debug(
                    "Using actual bitrate: %.0f kbps (%.3f MB/sec)",
                    bitrate_bps / 1000,
                    mb_per_second,
                )
            except (ValueError, ZeroDivisionError):
                # Fallback to file size based calculation
                num_chunks = int(file_size_mb / max_size_mb) + 1
                target_chunk_duration = total_duration / num_chunks
                logger.debug("Could not parse bitrate, using file size estimation")
        else:
            # Fallback: assume constant bytes/sec based on total file size
            num_chunks = int(file_size_mb / max_size_mb) + 1
            target_chunk_duration = total_duration / num_chunks
            logger.debug("Bitrate unavailable, using file size estimation")

        logger.info(
            "Splitting %.2f MB file (%.1f sec) into ~%d chunks (target: %.1f sec each)",
            file_size_mb,
            total_duration,
            num_chunks,
            target_chunk_duration,
        )

        # Try to get speech boundaries from VAD
        speech_segments = self._find_speech_boundaries(audio_path)

        # Inform user about chunking strategy
        if not speech_segments:
            logger.warning(
                "Using simple time-based splitting. For better quality, enable VAD-based smart chunking:\n"
                "  1. Get HuggingFace token: https://huggingface.co/settings/tokens\n"
                "  2. Accept model terms: https://huggingface.co/pyannote/voice-activity-detection\n"
                "  3. Export token: export HF_TOKEN=your_token_here"
            )

        # Calculate split points
        split_points = self._calculate_split_points(
            total_duration, target_chunk_duration, speech_segments
        )

        # Create chunks based on split points
        return self._create_chunks_from_points(
            audio_path, split_points, prefix="chunk"
        )

    def _calculate_split_points(
        self,
        total_duration: float,
        target_chunk_duration: float,
        speech_segments: List[tuple],
    ) -> List[tuple]:
        """
        Calculate optimal split points for audio chunks.

        If VAD data is available, splits at speech boundaries.
        Otherwise, splits at regular intervals.

        Args:
            total_duration: Total audio duration in seconds.
            target_chunk_duration: Target duration for each chunk.
            speech_segments: List of (start, end) speech segments from VAD.

        Returns:
            List of (start, end) tuples for each chunk.
        """
        if not speech_segments:
            # Fallback: simple time-based splitting
            logger.info("Using simple time-based splitting")
            num_chunks = int(total_duration / target_chunk_duration) + 1
            chunk_duration = total_duration / num_chunks

            split_points = []
            for i in range(num_chunks):
                start = i * chunk_duration
                end = min((i + 1) * chunk_duration, total_duration)
                split_points.append((start, end))

            return split_points

        # VAD-based splitting: find optimal boundaries
        logger.info("Using VAD-based smart splitting at speech boundaries")

        # Precompute gaps between speech segments once (O(n) instead of O(n²))
        gaps = []
        min_gap_duration = 0.3  # Minimum gap duration to consider (300ms)
        gap_padding = (
            0.1  # Padding around split points to avoid mid-syllable cuts (100ms)
        )

        for i in range(len(speech_segments) - 1):
            gap_start = speech_segments[i][1]  # End of current segment
            gap_end = speech_segments[i + 1][0]  # Start of next segment
            gap_duration = gap_end - gap_start

            # Only consider gaps longer than minimum duration
            if gap_duration >= min_gap_duration:
                # Use the middle of the gap as split point, with padding
                gap_middle = (gap_start + gap_end) / 2
                gaps.append(gap_middle)

        logger.debug(
            "Found %d suitable gaps (>= %.1fs) for splitting",
            len(gaps),
            min_gap_duration,
        )

        if not gaps:
            # No suitable gaps found, fall back to simple splitting
            logger.warning("No suitable speech gaps found, using time-based splitting")
            num_chunks = int(total_duration / target_chunk_duration) + 1
            chunk_duration = total_duration / num_chunks

            split_points = []
            for i in range(num_chunks):
                start = i * chunk_duration
                end = min((i + 1) * chunk_duration, total_duration)
                split_points.append((start, end))

            return split_points

        split_points = []
        current_start = 0.0
        gap_index = 0  # Sequential walking through gaps (O(n) instead of O(n²))

        # Calculate expected number of chunks for infinite loop protection
        expected_chunks = int(total_duration / target_chunk_duration) + 1
        max_iterations = (
            expected_chunks * 3
        )  # Allow 3x expected iterations as safety margin
        iteration = 0

        while current_start < total_duration and iteration < max_iterations:
            iteration += 1

            # Target end time for this chunk
            target_end = min(current_start + target_chunk_duration, total_duration)

            # Find the best gap near target_end using sequential search
            # Search window: ±30 seconds around target
            search_window_start = max(target_end - 30, current_start)
            search_window_end = min(target_end + 30, total_duration)

            best_split = target_end  # Default to target if no good gap found
            best_distance = float("inf")

            # Skip gaps that are before our search window (sequential optimization)
            while gap_index < len(gaps) and gaps[gap_index] < search_window_start:
                gap_index += 1

            # Find the best gap within the search window
            temp_index = gap_index
            while temp_index < len(gaps) and gaps[temp_index] <= search_window_end:
                gap_pos = gaps[temp_index]
                distance = abs(gap_pos - target_end)

                if distance < best_distance:
                    best_distance = distance
                    best_split = gap_pos

                temp_index += 1

            # Apply padding to the split point to avoid cutting mid-syllable
            if best_split != target_end:
                # We found a gap - the padding is already implicit in gap_middle calculation
                # But we can be more careful by checking we're not too close to speech
                pass

            # Force progress if stuck at same position
            if best_split <= current_start:
                logger.warning(
                    "Could not find suitable split point, forcing progress at %.1f sec",
                    target_end,
                )
                best_split = target_end

            # Check if this is the last chunk
            # If best_split is very close to the end, extend it to total_duration
            # to avoid leaving a tiny chunk at the end
            if best_split >= total_duration - 1.0:
                # This is the last chunk - extend to the very end
                split_points.append((current_start, total_duration))
                break
            else:
                # Normal chunk - add it and continue
                split_points.append((current_start, best_split))
                current_start = best_split

        # Check if we hit the iteration limit (should never happen in normal operation)
        if iteration >= max_iterations:
            logger.error(
                "Reached maximum iterations (%d) in split point calculation. "
                "This may indicate a bug. Returning partial results.",
                max_iterations,
            )

        return split_points

    def _transcribe_with_openai_api(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> List[TranscriptionSegment]:
        """
        Transcribe audio using OpenAI's Whisper API.

        Args:
            audio_path: Path to the audio file.
            language: Language code ('ru' or 'en'), or None to auto-detect.
            initial_prompt: Optional Whisper prompt to improve recognition.

        Returns:
            List of TranscriptionSegment instances.
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

        logger.info("Using OpenAI Whisper API for transcription")
        client = OpenAI(api_key=api_key)

        # Check file size (OpenAI has 25MB limit)
        file_size = audio_path.stat().st_size
        max_size = 25 * 1024 * 1024  # 25 MB

        # Split file if too large
        if file_size > max_size:
            logger.warning(
                "Audio file size (%.2f MB) exceeds OpenAI limit (25 MB). Splitting into chunks...",
                file_size / (1024 * 1024),
            )
            chunks = self._split_audio_file(audio_path)

            try:
                all_segments = []

                for i, (chunk_path, chunk_start, chunk_end) in enumerate(chunks):
                    logger.info("Processing chunk %d/%d...", i + 1, len(chunks))

                    # Transcribe chunk
                    chunk_segments = self._transcribe_single_file_with_openai(
                        chunk_path, language, initial_prompt, client
                    )

                    # Adjust timestamps to account for chunk position in original file
                    for seg in chunk_segments:
                        seg.start += chunk_start
                        seg.end += chunk_start
                        all_segments.append(seg)

                    # Log if chunk was silent (no segments returned)
                    if not chunk_segments:
                        logger.debug(
                            "Chunk %d/%d (%.1f-%.1f sec) contained no speech",
                            i + 1,
                            len(chunks),
                            chunk_start,
                            chunk_end,
                        )

                logger.info(
                    "All chunks processed. Total segments: %d", len(all_segments)
                )
                return all_segments

            finally:
                # Clean up chunk files
                for chunk_path, _, _ in chunks:
                    try:
                        if chunk_path.exists():
                            chunk_path.unlink()
                        # Remove from global registry
                        _temp_chunk_files.discard(chunk_path)
                    except Exception as e:
                        logger.warning("Failed to delete chunk %s: %s", chunk_path, e)

        # File is small enough, process directly
        return self._transcribe_single_file_with_openai(
            audio_path, language, initial_prompt, client
        )

    def _transcribe_single_file_with_openai(
        self,
        audio_path: Path,
        language: Optional[str],
        initial_prompt: Optional[str],
        client,
    ) -> List[TranscriptionSegment]:
        """
        Transcribe a single audio file using OpenAI's Whisper API.

        Args:
            audio_path: Path to the audio file.
            language: Language code or None.
            initial_prompt: Optional prompt.
            client: OpenAI client instance.

        Returns:
            List of TranscriptionSegment instances.
        """
        from .api_cache import get_openai_rate_limiter
        from .retry_handler import retry_api_call
        from .cost_tracker import get_cost_tracker

        # Check cache first using audio hash
        if self.use_cache:
            audio_hash = _compute_audio_hash(audio_path)
            cache_key = {
                "audio_hash": audio_hash,
                "method": "whisper_openai_api",
                "language": language,
                "initial_prompt": initial_prompt or "",
            }
            cached_result = self.cache.get("transcription", cache_key)
            if cached_result is not None:
                logger.info("Using cached OpenAI Whisper transcription result")
                segments = [
                    TranscriptionSegment(
                        start=seg["start"],
                        end=seg["end"],
                        text=seg["text"],
                        speaker=seg.get("speaker"),
                    )
                    for seg in cached_result
                ]
                return segments

        logger.info("Uploading audio to OpenAI...")

        # Get rate limiter for OpenAI API
        rate_limiter = get_openai_rate_limiter()

        # Prepare API parameters
        api_params = {
            "file": open(audio_path, "rb"),
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }

        if language:
            api_params["language"] = language

        if initial_prompt:
            api_params["prompt"] = initial_prompt

        try:
            # Apply rate limiting
            if rate_limiter:
                rate_limiter.wait_if_needed()

            # Wrap the API call with retry logic
            @retry_api_call(max_retries=3)
            def _call_whisper_api():
                return client.audio.transcriptions.create(**api_params)

            # Call OpenAI API with retry
            response = _call_whisper_api()

            # Track transcription cost (Whisper charges by duration, not tokens)
            # Get audio duration for cost tracking
            try:
                import subprocess

                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        str(audio_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                duration_seconds = float(result.stdout.strip())
                cost_tracker = get_cost_tracker()
                cost_tracker.add_transcription(duration_seconds)
            except Exception as e:
                logger.debug("Could not track transcription cost: %s", e)

            # Extract segments from response
            segments: List[TranscriptionSegment] = []

            if hasattr(response, "segments") and response.segments:
                for seg in tqdm(response.segments, desc="Processing segments"):
                    segments.append(
                        TranscriptionSegment(
                            start=getattr(seg, "start", 0.0),
                            end=getattr(seg, "end", 0.0),
                            text=getattr(seg, "text", ""),
                        )
                    )
            else:
                # Fallback: if no segments, create one segment with full text
                logger.warning("No segments in API response, creating single segment")
                segments.append(
                    TranscriptionSegment(
                        start=0.0,
                        end=0.0,
                        text=response.text if hasattr(response, "text") else "",
                    )
                )

            detected_language = (
                response.language if hasattr(response, "language") else "unknown"
            )
            logger.info("Detected language: %s", format_warning(str(detected_language).upper()))
            logger.info("Transcription finished. Generated %d segments", len(segments))

            # Clean up hallucinations
            logger.info("Cleaning up potential hallucinations...")
            segments = self._clean_hallucinations(
                segments, expected_language=detected_language
            )
            logger.info("After cleanup: %d segments remain", len(segments))

            # Cache the result
            if self.use_cache:
                segments_dict = [seg.to_dict() for seg in segments]
                self.cache.set("transcription", cache_key, segments_dict)

            return segments

        finally:
            # Close the file handle
            if "file" in api_params and hasattr(api_params["file"], "close"):
                api_params["file"].close()

    def _transcription_dominated_by_prompt(
        self,
        prompt: str,
        segments: List[TranscriptionSegment],
    ) -> bool:
        """
        Detect cases where Whisper echoed the initial prompt instead of transcribing audio.

        Args:
            prompt: Prompt text supplied to Whisper.
            segments: Transcribed segments.

        Returns:
            True when the transcript mostly repeats the prompt.
        """
        if not prompt or not segments:
            return False

        prompt_norm = re.sub(r"\s+", " ", prompt).strip().lower()
        if len(prompt_norm) < 40:
            return False

        transcript_text = " ".join(seg.text for seg in segments).strip()
        transcript_norm = re.sub(r"\s+", " ", transcript_text).lower()

        if not transcript_norm:
            return True

        if transcript_norm.startswith(prompt_norm[: min(len(prompt_norm), 160)]):
            return True

        prompt_tokens = re.findall(r"\w+", prompt_norm)
        transcript_tokens = re.findall(r"\w+", transcript_norm)

        if not transcript_tokens:
            return True

        prompt_vocab = set(prompt_tokens)
        overlap = sum(1 for tok in transcript_tokens if tok in prompt_vocab)
        overlap_ratio = overlap / max(len(transcript_tokens), 1)

        if len(transcript_tokens) < 80 and overlap_ratio > 0.7:
            return True

        return False

    def _clean_hallucinations(
        self, segments: List[TranscriptionSegment], expected_language: str = "ru"
    ) -> List[TranscriptionSegment]:
        """
        Clean up common Whisper hallucinations from transcription segments.

        Whisper may generate nonsensical text when encountering silence, noise,
        or unclear audio. This function detects and removes such hallucinations.

        Args:
            segments: Original transcription segments.
            expected_language: Expected language code (ru, en, etc.).

        Returns:
            Cleaned segments with hallucinations removed or fixed.
        """
        import unicodedata

        # Common hallucination patterns
        hallucination_patterns = [
            # Subscription/outro phrases (English)
            r"(?i)(thanks?\s+for\s+watching|please\s+subscribe|like\s+and\s+subscribe)",
            r"(?i)(don't\s+forget\s+to|hit\s+the\s+bell|smash\s+that)",
            # Random names that appear in silence
            r"(?i)(shepherd\s+bettsies?|betsy|campo)",
            # Short nonsensical fragments
            r"^\s*[\w\-]{1,3}\s*$",  # Very short words alone
            # Music/sound effect markers
            r"^\s*\[.*?\]\s*$",  # Already bracketed items
            r"(?i)^\s*(music|applause|laughter)\s*$",
        ]

        # Patterns indicating mixed-language hallucinations
        def has_suspicious_mix(text: str) -> bool:
            """Check if text has suspicious language mixing."""
            # Count Cyrillic, Latin, and other scripts
            cyrillic = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
            latin = sum(1 for c in text if c.isascii() and c.isalpha())
            other = sum(1 for c in text if unicodedata.category(c).startswith("Lo"))

            total_chars = cyrillic + latin + other
            if total_chars == 0:
                return False

            # Non-Latin, non-Cyrillic scripts appearing (e.g., Korean, Chinese, Arabic) in wrong context
            # This is a STRONG indicator of hallucination
            if other > 0 and expected_language in ["ru", "en"]:
                # Even a single Korean/Chinese character is suspicious
                # But only if there are multiple or if it's a significant portion
                if other >= 2 or (other > 0 and other / (total_chars + 1) > 0.03):
                    return True

            # For Russian: check if it's PURE English (no Cyrillic at all) and short
            # This catches segments like "got interesting questionable"
            if expected_language == "ru":
                if cyrillic == 0 and latin > 10 and total_chars < 50:
                    # No Russian characters, but has English - likely hallucination
                    # Exception: very common technical terms
                    text_lower = text.lower().strip()
                    common_terms = [
                        "ai",
                        "genai",
                        "chatgpt",
                        "openai",
                        "api",
                        "rag",
                        "ml",
                        "gpt",
                    ]
                    # If it's ONLY common terms, keep it
                    words = re.findall(r"\b\w+\b", text_lower)
                    if words and all(
                        word in common_terms or len(word) <= 2 for word in words
                    ):
                        return False
                    return True

            return False

        def is_hallucination(text: str) -> bool:
            """Check if text segment is likely a hallucination."""
            text_stripped = text.strip()

            # Empty or very short
            if len(text_stripped) < 3:
                return True

            # Check against known patterns
            for pattern in hallucination_patterns:
                if re.search(pattern, text_stripped):
                    return True

            # Check for suspicious language mixing
            if has_suspicious_mix(text_stripped):
                return True

            # Check for segments with too many special/unknown characters
            special_chars = sum(
                1
                for c in text_stripped
                if not (c.isalnum() or c.isspace() or c in ".,!?;:-—")
            )
            if len(text_stripped) > 0 and special_chars / len(text_stripped) > 0.3:
                return True

            return False

        def clean_segment_text(text: str) -> str:
            """Clean individual segment text from partial hallucinations."""
            # Remove non-standard Unicode characters (e.g., Korean, weird symbols)
            cleaned = ""
            for char in text:
                cat = unicodedata.category(char)
                # Keep Latin, Cyrillic, common punctuation, spaces
                if (
                    cat.startswith("L")
                    or cat.startswith("P")
                    or cat.startswith("Z")
                    or cat.startswith("N")
                    or char in ".,!?;:—–-\"'()[]"
                ):
                    # But filter out scripts we don't expect
                    if expected_language == "ru":
                        # Allow Cyrillic, Latin (for terms), common chars
                        if (
                            "\u0400" <= char <= "\u04ff"  # Cyrillic
                            or char.isascii()  # ASCII (Latin + punct)
                            or char.isspace()
                        ):
                            cleaned += char
                    elif expected_language == "en":
                        if char.isascii() or char.isspace():
                            cleaned += char
                    else:
                        cleaned += char

            # Remove multiple spaces
            cleaned = re.sub(r"\s+", " ", cleaned)

            return cleaned.strip()

        # Process segments
        cleaned_segments = []
        hallucination_count = 0

        for seg in segments:
            # Check if it's a hallucination BEFORE cleaning
            # (cleaning removes foreign characters, which we need to detect)
            if is_hallucination(seg.text):
                hallucination_count += 1
                logger.debug(
                    "Removed hallucination at %.1f-%.1fs: %s",
                    seg.start,
                    seg.end,
                    seg.text[:50],
                )
                continue

            # Clean the text after hallucination check
            cleaned_text = clean_segment_text(seg.text)

            # Keep the segment with cleaned text
            if cleaned_text:  # Only keep non-empty segments
                cleaned_segments.append(
                    TranscriptionSegment(
                        start=seg.start,
                        end=seg.end,
                        text=cleaned_text,
                        speaker=seg.speaker,
                    )
                )

        if hallucination_count > 0:
            logger.info(
                "Cleaned %d hallucination segments (%.1f%% of total)",
                hallucination_count,
                100 * hallucination_count / len(segments) if segments else 0,
            )

        return cleaned_segments

    def segments_to_text(self, segments: List[TranscriptionSegment]) -> str:
        """
        Convert transcription segments into plain text.

        Args:
            segments: Iterable of segments or dicts.

        Returns:
            Newline-delimited string.
        """
        result = []
        for seg in segments:
            if hasattr(seg, "text"):
                result.append(seg.text)
            elif isinstance(seg, dict) and "text" in seg:
                result.append(seg["text"])
        return "\n\n".join(result)

    def update_segments_from_text(
        self,
        segments: List[TranscriptionSegment],
        refined_text: str,
    ) -> List[TranscriptionSegment]:
        """
        Replace segment text with refined content while preserving timestamps.

        Args:
            segments: Original segments.
            refined_text: Improved transcript.

        Returns:
            List of updated segments.
        """
        paragraphs = [p.strip() for p in refined_text.split("\n\n") if p.strip()]

        if len(paragraphs) == len(segments):
            updated_segments = []
            for seg, new_text in zip(segments, paragraphs):
                updated_segments.append(
                    TranscriptionSegment(
                        start=seg.start,
                        end=seg.end,
                        text=new_text,
                        speaker=seg.speaker,
                    )
                )
            return updated_segments

        logger.info(
            "Refined text has %d paragraphs vs %d original segments. "
            "This is expected when LLM groups text into logical paragraphs. Rebuilding segments with evenly distributed timestamps.",
            len(paragraphs),
            len(segments),
        )

        if not segments:
            return []

        def _select_speaker_for_interval(
            original_segments: List[TranscriptionSegment],
            interval_start: float,
            interval_end: float,
        ) -> Optional[str]:
            """
            Pick the speaker that overlaps most with the provided time range.

            Args:
                original_segments: Segments with existing speaker labels.
                interval_start: Start of the interval.
                interval_end: End of the interval.

            Returns:
                Speaker label or None if unavailable.
            """
            best_speaker = None
            best_overlap = 0.0

            for seg in original_segments:
                if not seg.speaker:
                    continue

                overlap_start = max(interval_start, seg.start)
                overlap_end = min(interval_end, seg.end)
                overlap = overlap_end - overlap_start

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = seg.speaker

            if best_speaker:
                return best_speaker

            # Fallback: use the closest preceding speaker label.
            for seg in reversed(original_segments):
                if seg.start <= interval_start and seg.speaker:
                    return seg.speaker

            # Final fallback: return the first available speaker label.
            for seg in original_segments:
                if seg.speaker:
                    return seg.speaker

            return None

        total_duration = segments[-1].end - segments[0].start
        segment_duration = total_duration / len(paragraphs) if paragraphs else 0

        updated_segments: List[TranscriptionSegment] = []
        start_time = segments[0].start

        for index, paragraph in enumerate(paragraphs):
            end_time = (
                start_time + segment_duration
                if index < len(paragraphs) - 1
                else segments[-1].end
            )

            updated_segments.append(
                TranscriptionSegment(
                    start=start_time,
                    end=end_time,
                    text=paragraph,
                    speaker=_select_speaker_for_interval(
                        segments, start_time, end_time
                    ),
                )
            )
            start_time = end_time

        return updated_segments

    def segments_to_text_with_speakers(
        self,
        segments: List[TranscriptionSegment],
    ) -> str:
        """
        Convert transcription segments to text with speaker labels (no timestamps).
        Speaker labels are only shown when the speaker changes.

        Args:
            segments: Transcription segments with speaker information.

        Returns:
            Text with speaker labels in format: [SPEAKER_00] text
        """
        lines = []
        previous_speaker = None

        for seg in segments:
            # Only add speaker label when speaker changes
            if seg.speaker and seg.speaker != previous_speaker:
                speaker = f"[{seg.speaker}] "
                previous_speaker = seg.speaker
            else:
                speaker = ""

            lines.append(f"{speaker}{seg.text}")

        return "\n\n".join(lines)

    def segments_to_text_with_timestamps(
        self,
        segments: List[TranscriptionSegment],
        with_speakers: bool = False,
    ) -> str:
        """
        Convert segments to text including timestamps (and speakers if available).
        Speaker labels are only shown when the speaker changes.

        Args:
            segments: Transcription segments.
            with_speakers: Include speaker labels if present.

        Returns:
            Multiline string with timestamps.
        """
        lines = []
        previous_speaker = None

        for seg in segments:
            timestamp = format_timestamp(seg.start)

            # Only add speaker label when speaker changes
            if with_speakers and seg.speaker and seg.speaker != previous_speaker:
                speaker = f"[{seg.speaker}] "
                previous_speaker = seg.speaker
            else:
                speaker = ""

            lines.append(f"[{timestamp}] {speaker}{seg.text}")

        return "\n\n".join(lines)
