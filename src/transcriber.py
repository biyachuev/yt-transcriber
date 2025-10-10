"""
Module responsible for audio transcription via Whisper.
"""
import re
from pathlib import Path
from typing import List, Dict, Optional

import torch
import whisper
from tqdm import tqdm

from src.config import settings, TranscribeOptions
from src.logger import logger
from src.utils import format_timestamp, estimate_processing_time, format_log_preview


class TranscriptionSegment:
    """Lightweight wrapper representing a single transcription segment."""

    def __init__(self, start: float, end: float, text: str, speaker: Optional[str] = None):
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


class Transcriber:
    """Whisper-based audio transcriber."""

    def __init__(self, method: str = TranscribeOptions.WHISPER_BASE):
        self.method = method
        self.model = None
        self.device = self._get_device()
        logger.info("Using device: %s", self.device)

    def _get_device(self) -> str:
        """Detect the best available device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        """Load the selected Whisper model if it has not been loaded yet."""
        if self.model is not None:
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
                logger.warning("Sparse operations are not supported on MPS. Falling back to CPU...")
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
            if "MPS" in message and ("sparse" in message.lower() or "_sparse_coo_tensor" in message):
                logger.warning("Encountered an MPS issue while loading the model. Switching to CPU...")
                self.device = "cpu"
                self.model = whisper.load_model(
                    model_name,
                    device=self.device,
                    download_root=str(settings.WHISPER_MODEL_DIR),
                )
            else:
                raise

        logger.info("Model loaded successfully")

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
            logger.info("Using initial prompt (length: %d chars)", len(initial_prompt))
            logger.debug("Prompt preview (first 80 chars): %s", format_log_preview(initial_prompt))

        if with_speakers:
            logger.warning("Speaker diarisation is not available in the current release")

        # Use OpenAI API if specified
        if self.method == TranscribeOptions.WHISPER_OPENAI_API:
            return self._transcribe_with_openai_api(audio_path, language, initial_prompt)

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

        if initial_prompt and self._transcription_dominated_by_prompt(initial_prompt, segments):
            logger.warning(
                "Initial prompt appears to dominate the transcription output; retrying without the prompt."
            )
            transcribe_options.pop("initial_prompt", None)
            result, segments = _run_transcription(transcribe_options)

        detected_language = result.get("language", "unknown")
        logger.info("Detected language: %s", detected_language)
        logger.info("Transcription finished. Generated %d segments", len(segments))

        return segments

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

        if file_size > max_size:
            logger.warning(
                "Audio file size (%.2f MB) exceeds OpenAI limit (25 MB). "
                "Consider using local Whisper instead.",
                file_size / (1024 * 1024)
            )
            raise ValueError(
                f"Audio file too large for OpenAI API ({file_size / (1024 * 1024):.2f} MB). "
                "Maximum is 25 MB."
            )

        logger.info("Uploading audio to OpenAI...")

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
            # Call OpenAI API
            response = client.audio.transcriptions.create(**api_params)

            # Extract segments from response
            segments: List[TranscriptionSegment] = []

            if hasattr(response, 'segments') and response.segments:
                for seg in tqdm(response.segments, desc="Processing segments"):
                    segments.append(
                        TranscriptionSegment(
                            start=seg.get('start', 0.0),
                            end=seg.get('end', 0.0),
                            text=seg.get('text', ''),
                        )
                    )
            else:
                # Fallback: if no segments, create one segment with full text
                logger.warning("No segments in API response, creating single segment")
                segments.append(
                    TranscriptionSegment(
                        start=0.0,
                        end=0.0,
                        text=response.text if hasattr(response, 'text') else '',
                    )
                )

            detected_language = response.language if hasattr(response, 'language') else 'unknown'
            logger.info("Detected language: %s", detected_language)
            logger.info("Transcription finished. Generated %d segments", len(segments))

            return segments

        finally:
            # Close the file handle
            if 'file' in api_params and hasattr(api_params['file'], 'close'):
                api_params['file'].close()

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

        logger.warning(
            "Paragraph count (%d) does not match segment count (%d). Rebuilding segments.",
            len(paragraphs),
            len(segments),
        )

        if not segments:
            return []

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
                    speaker=None,
                )
            )
            start_time = end_time

        return updated_segments

    def segments_to_text_with_timestamps(
        self,
        segments: List[TranscriptionSegment],
        with_speakers: bool = False,
    ) -> str:
        """
        Convert segments to text including timestamps (and speakers if available).

        Args:
            segments: Transcription segments.
            with_speakers: Include speaker labels if present.

        Returns:
            Multiline string with timestamps.
        """
        lines = []
        for seg in segments:
            timestamp = format_timestamp(seg.start)
            speaker = f"[{seg.speaker}] " if with_speakers and seg.speaker else ""
            lines.append(f"[{timestamp}] {speaker}{seg.text}")

        return "\n\n".join(lines)
