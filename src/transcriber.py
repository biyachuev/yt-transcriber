"""
Module responsible for audio transcription via Whisper.
"""
import atexit
import re
from pathlib import Path
from typing import List, Dict, Optional, Set

import torch
import whisper
from tqdm import tqdm

from src.config import settings, TranscribeOptions
from src.logger import logger
from src.utils import format_timestamp, estimate_processing_time, format_log_preview

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

    def _get_vad_pipeline(self):
        """
        Get or create VAD pipeline for speech detection.

        Returns:
            pyannote VAD pipeline or None if not available.
        """
        if hasattr(self, '_vad_pipeline'):
            return self._vad_pipeline

        try:
            from pyannote.audio import Pipeline
            import os

            # Check if HuggingFace token is available
            hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

            if not hf_token:
                logger.warning(
                    "HUGGINGFACE_TOKEN not found. VAD-based splitting disabled. "
                    "Set HF_TOKEN in .env to enable smart chunking."
                )
                self._vad_pipeline = None
                return None

            logger.info("Loading pyannote VAD pipeline...")
            self._vad_pipeline = Pipeline.from_pretrained(
                "pyannote/voice-activity-detection",
                token=hf_token
            )
            logger.info("VAD pipeline loaded successfully")
            return self._vad_pipeline

        except ImportError:
            logger.warning(
                "pyannote.audio not installed. Falling back to simple time-based splitting. "
                "Install with: pip install pyannote.audio"
            )
            self._vad_pipeline = None
            return None
        except Exception as e:
            logger.warning("Failed to load VAD pipeline: %s. Using simple splitting.", e)
            self._vad_pipeline = None
            return None

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
        suspicious_chars = [';', '&', '|', '`', '$', '\n', '\r']
        path_str = str(audio_path)
        for char in suspicious_chars:
            if char in path_str:
                logger.warning(
                    "Suspicious character '%s' found in path: %s",
                    char, path_str
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

    def _find_speech_boundaries(self, audio_path: Path) -> List[tuple]:
        """
        Use VAD to find speech segment boundaries.

        Args:
            audio_path: Path to audio file.

        Returns:
            List of (start, end) tuples in seconds for speech segments.
        """
        vad_pipeline = self._get_vad_pipeline()
        if not vad_pipeline:
            return []

        logger.info("Detecting speech boundaries with VAD...")

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
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=True
        )
        total_duration = float(result.stdout.strip())

        # Calculate how many chunks we need
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        num_chunks = int(file_size_mb / max_size_mb) + 1
        target_chunk_duration = total_duration / num_chunks

        logger.info(
            "Splitting %.2f MB file (%.1f sec) into ~%d chunks",
            file_size_mb, total_duration, num_chunks
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
            total_duration,
            target_chunk_duration,
            speech_segments
        )

        # Create chunks based on split points
        chunks = []
        temp_dir = audio_path.parent
        base_name = audio_path.stem
        # Preserve original file extension to match container format
        original_extension = audio_path.suffix

        for i, (start_time, end_time) in enumerate(split_points):
            chunk_path = temp_dir / f"{base_name}_chunk_{i}{original_extension}"
            duration = end_time - start_time

            # Try stream copy first (fast, no re-encoding)
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i", str(audio_path),
                        "-ss", str(start_time),
                        "-t", str(duration),
                        "-c", "copy",
                        "-y",
                        str(chunk_path),
                    ],
                    capture_output=True,
                    check=True,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                # Stream copy failed, try re-encoding
                logger.warning(
                    "Stream copy failed for chunk %d (error: %s). Re-encoding...",
                    i, e.stderr[:100] if e.stderr else "unknown"
                )

                # Use MP3 encoding as fallback (widely compatible)
                chunk_path = temp_dir / f"{base_name}_chunk_{i}.mp3"
                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-i", str(audio_path),
                            "-ss", str(start_time),
                            "-t", str(duration),
                            "-c:a", "libmp3lame",
                            "-b:a", "192k",
                            "-y",
                            str(chunk_path),
                        ],
                        capture_output=True,
                        check=True,
                        text=True
                    )
                    logger.debug("Successfully re-encoded chunk %d to MP3", i)
                except subprocess.CalledProcessError as e2:
                    logger.error(
                        "Failed to create chunk %d even with re-encoding: %s",
                        i, e2.stderr[:200] if e2.stderr else "unknown error"
                    )
                    raise

            # Register chunk for cleanup
            _temp_chunk_files.add(chunk_path)

            chunks.append((chunk_path, start_time, end_time))
            logger.debug(
                "Created chunk %d/%d: %s (%.1f-%.1f sec, %.1f sec duration)",
                i + 1, len(split_points), chunk_path.name, start_time, end_time, duration
            )

        return chunks

    def _calculate_split_points(
        self,
        total_duration: float,
        target_chunk_duration: float,
        speech_segments: List[tuple]
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

        split_points = []
        current_start = 0.0

        # Calculate expected number of chunks for infinite loop protection
        expected_chunks = int(total_duration / target_chunk_duration) + 1
        max_iterations = expected_chunks * 3  # Allow 3x expected iterations as safety margin
        iteration = 0

        while current_start < total_duration and iteration < max_iterations:
            iteration += 1

            # Target end time for this chunk
            target_end = min(current_start + target_chunk_duration, total_duration)

            # Find the best split point near target_end
            # Look in a window of ±30 seconds around target
            search_window_start = max(target_end - 30, current_start)
            search_window_end = min(target_end + 30, total_duration)

            # Find gaps between speech segments in the search window
            best_split = target_end  # Default to target if no good gap found
            min_gap_duration = 0.5  # Minimum gap duration to consider (seconds)

            for i in range(len(speech_segments) - 1):
                gap_start = speech_segments[i][1]  # End of current segment
                gap_end = speech_segments[i + 1][0]  # Start of next segment
                gap_duration = gap_end - gap_start

                # Check if this gap is in our search window and is long enough
                if (search_window_start <= gap_start <= search_window_end and
                    gap_duration >= min_gap_duration):
                    # Use the middle of the gap as split point
                    gap_middle = (gap_start + gap_end) / 2

                    # Prefer gaps closer to the target
                    if abs(gap_middle - target_end) < abs(best_split - target_end):
                        best_split = gap_middle

            # Force progress if stuck at same position
            if best_split <= current_start:
                logger.warning(
                    "Could not find suitable split point, forcing progress at %.1f sec",
                    target_end
                )
                best_split = target_end

            # Add this chunk
            split_points.append((current_start, best_split))
            current_start = best_split

            # Break if we're at the end
            if current_start >= total_duration - 1.0:
                break

        # Check if we hit the iteration limit (should never happen in normal operation)
        if iteration >= max_iterations:
            logger.error(
                "Reached maximum iterations (%d) in split point calculation. "
                "This may indicate a bug. Returning partial results.",
                max_iterations
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
                file_size / (1024 * 1024)
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
                            i + 1, len(chunks), chunk_start, chunk_end
                        )

                logger.info("All chunks processed. Total segments: %d", len(all_segments))
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
        return self._transcribe_single_file_with_openai(audio_path, language, initial_prompt, client)

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
                            start=getattr(seg, 'start', 0.0),
                            end=getattr(seg, 'end', 0.0),
                            text=getattr(seg, 'text', ''),
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
