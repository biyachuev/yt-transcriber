# Changelog

All significant changes to this project are documented here.

## [Unreleased]

### Added
- ✨ **Early API Key Validation**
  - New `api_validator.py` module for early API key validation
  - Validates OpenAI API keys at startup with test API call
  - Fails fast before expensive operations (downloads, processing)
  - Clear error messages with authentication failure details
  - Validates keys once even when multiple operations need them
  - Saves time and bandwidth by catching errors early
  - Comprehensive test coverage with 12 unit tests

- ✨ **Automatic yt-dlp version checking and updates**
  - New `yt_dlp_updater.py` module for automatic version management
  - Checks for yt-dlp updates before processing YouTube videos
  - Auto-updates to latest version if outdated
  - Prevents HTTP 403 Forbidden errors from YouTube API changes
  - Graceful handling of version check failures (non-blocking)
  - Added `packaging>=23.0` dependency for version comparison

## [1.6.0] - 2025-10-21

### 🚀 Performance - VAD Optimizations

Based on Codex code review feedback, this release dramatically improves Voice Activity Detection (VAD) performance for large audio files:

#### Added
- ⚡ **Silero VAD Integration**
  - Lightweight 1.8MB model for speech boundary detection
  - 10-100x faster than pyannote for VAD-only tasks
  - CPU-based inference (~1ms per 30ms chunk)
  - No HuggingFace token required for basic chunking
  - Automatic fallback to pyannote VAD if unavailable
  - New functions: `_get_silero_vad()`, `_find_speech_boundaries_silero()`

- 🎯 **Accurate Bitrate Detection**
  - Uses `ffprobe` to detect actual audio bitrate
  - Calculates optimal chunk duration based on real data rate
  - Prevents over-splitting or under-splitting files
  - Ensures chunks stay under OpenAI's 25MB limit
  - Fallback to file-size estimation if bitrate unavailable

#### Performance Improvements
- 🔥 **O(n²) → O(n) Complexity Reduction**
  - Rewrote boundary search algorithm in `_calculate_split_points()`
  - Sequential gap walking instead of repeated scanning
  - 100-1000x faster for files with many speech segments
  - Handles 1000+ segments in < 1 second

- ✂️ **Gap Quality Filtering**
  - Minimum gap duration filter (300ms) to prevent mid-syllable cuts
  - Filters out very short pauses that would create poor split points
  - Reduces audio artifacts at chunk boundaries
  - Configurable minimum gap threshold

- 📊 **Enhanced Logging**
  - Shows which VAD method is being used (Silero/pyannote)
  - Logs number of suitable gaps found
  - Reports bitrate detection results
  - Better debugging for chunk splitting

#### Documentation
- 📝 **New VAD Performance Optimization section in README**
  - Comparison of Silero vs pyannote VAD
  - Performance characteristics (speed, footprint)
  - Setup instructions and automatic fallback behavior
  - PyTorch Lightning warning documentation

### Fixed
- 🐛 **TextRefiner topic detection now respects backend setting**
  - Fixed hardcoded Ollama call in `_detect_topic()` method
  - Topic detection now correctly uses OpenAI API when `--refine-backend openai_api` is specified
  - Resolves 404 error when using OpenAI backend without Ollama running

### Added
- ✨ **OpenAI support for Whisper prompt generation**
  - Enhanced `create_whisper_prompt_with_llm()` to support both Ollama and OpenAI backends
  - Automatically uses the same backend as refinement (`--refine-backend`) for prompt generation
  - Improves consistency when using OpenAI API throughout the pipeline
- 🎯 **Audio preprocessing for speaker diarization**
  - Automatic conversion to mono 16kHz
  - RMS volume normalization to -20 dBFS
  - Clipping prevention
  - Optional noise reduction support (via noisereduce library)
  - Helps reduce false speaker clusters from volume variations and background noise
  - New function: `_preprocess_audio_for_diarization()`

### Documentation
- 📝 **Added FAQ entries for speaker diarization warnings**
  - Documented torchcodec FFmpeg version warning (safe to ignore)
  - Documented pyannote std() warning (safe to ignore)
  - Explained fallback audio loading mechanism
  - Added quick reference in README troubleshooting section
- 📝 **Added speaker diarization accuracy information**
  - New FAQ section: "How accurate is speaker diarization?"
  - Documented over-segmentation limitation (one speaker → multiple labels)
  - Added accuracy guidelines based on audio quality
  - Recommendation to verify speaker labels manually for critical applications
  - Added warning note in README highlights
  - Documented automatic audio preprocessing features

## [1.3.0] - 2025-10-10

### Added
- 🚀 **Full OpenAI API Integration**
  - OpenAI Whisper API for transcription (cloud-based, fast)
  - GPT-4/GPT-3.5 for translation (better context handling)
  - GPT for text refinement (alternative to Ollama)
  - GPT for summarization (new feature!)

### New Features
- **Summarization Module** (`src/summarizer.py`)
  - Generate detailed summaries of transcripts
  - Support for both Ollama and OpenAI backends
  - Automatic chunking for long documents
  - Structured output with section headings
  - CLI flags: `--summarize`, `--summarize-model`, `--summarize-backend`

- **OpenAI Whisper Transcription**
  - Cloud-based transcription via OpenAI API
  - Faster than local processing (no model download)
  - Supports files up to 25MB
  - Returns timestamped segments
  - Usage: `--transcribe whisper_openai_api`

- **GPT Translation**
  - Translation using GPT-4 or GPT-3.5-turbo
  - Better handling of technical terms and idioms
  - Preserves formatting and timestamps
  - Usage: `--translate openai_api`

- **GPT Text Refinement**
  - Alternative to Ollama for transcript cleanup
  - Removes filler words, improves punctuation
  - Usage: `--refine-model gpt-4 --refine-backend openai_api`

### New CLI Options
- `--refine-backend {ollama,openai_api}` - Choose refinement backend
- `--summarize` - Enable summarization
- `--summarize-model MODEL` - Model for summarization
- `--summarize-backend {ollama,openai_api}` - Choose summarization backend

### Configuration
- Added `RefineOptions` enum to `config.py`
- Added `SummarizeOptions` enum to `config.py`
- Updated help text with OpenAI examples

### Documentation
- Added `OPENAI_INTEGRATION.md` - Complete OpenAI guide
  - Setup instructions
  - Usage examples
  - Cost estimates
  - Troubleshooting

### Technical
- Lazy imports for OpenAI library (optional dependency)
- Proper error handling for missing API keys
- File size validation for OpenAI Whisper (25MB limit)
- Chunking support for long text processing

### Architecture
- `src/summarizer.py` - New module (345 lines)
- `src/transcriber.py` - Added `_transcribe_with_openai_api()` method
- `src/translator.py` - Implemented `_translate_with_openai()` method
- `src/text_refiner.py` - Added OpenAI backend support
- `src/config.py` - New option enums

---

## [1.2.0] - 2025-10-06

### Added
- 📄 **Document ingestion**
  - Read `.docx`, `.md`, `.txt`
  - Improve existing transcripts with an LLM
  - Translate uploaded documents
  - Auto-detect source language

### New modules
- `src/text_reader.py`
  - `read_docx()` — Microsoft Word support
  - `read_markdown()` — Markdown parsing with formatting removal
  - `read_text()` — plain text reader
  - `detect_language()` — Cyrillic/Latin heuristics

### Integration
- Added `process_text_file()` to `src/main.py`
- CLI flag `--input_text path/to/file.md`
- Lazy translation imports to reduce dependencies
- Reuses the existing pipeline (segments, documents)

### Improved
- **NLLB translation quality**
  - Default model upgraded: `facebook/nllb-200-distilled-600M` → `facebook/nllb-200-distilled-1.3B`
  - New `--translate-model` flag to choose another NLLB variant
  - Chunk size increased 500 → 700 tokens (more context)
  - Generation limit increased 512 → 1024 tokens
  - Added `no_repeat_ngram_size=3` to avoid duplicates
  - Result: **~40% higher quality**, ~3× slower
- `segments_to_text()` and `translate_segments()` accept dicts and objects
- Translation no longer required for text documents
- Lazy `transformers` import (skipped unless `--translate` is used)

### Usage examples
```bash
# Improve a document
python -m src.main --input_text doc.md --refine-model qwen2.5:7b

# Translate with a custom NLLB model
python -m src.main --input_text doc.docx --translate NLLB --translate-model facebook/nllb-200-distilled-600M

# Improve + translate (defaults to 1.3B model)
python -m src.main --input_text doc.txt --refine-model qwen2.5:7b --translate NLLB
```

---

## [1.1.0] - 2025-10-06

### Added
- 🎯 **Optimised prompts for transcript polishing**
  - Separate prompts for English and Russian
  - Minimal editing (noise removal only, no summarisation)
  - All facts, examples, and reasoning preserved

### Improved
- ✨ **Transcript clean-up quality**
  - Removes filler words (“um”, “uh”, “like”, “эм”, “ну”, “вот”, “короче”)
  - Drops meta-comments (“let me scroll”, “сейчас открою экран”)
  - Normalises numbers (“twenty eight sixteen” → “2816”, “point eight” → “0.8”)
  - Keeps contextual examples like “if you’re above 2650 ...”

- 📝 **Better structure**
  - Improved punctuation
  - Merges fragmented sentences
  - Proper paragraph formatting

- 🚫 **No summarisation**
  - Explicit instructions to keep meaningful content
  - Repeated statements retained (can be important)
  - All reasoning preserved

### Technical details
- Updated prompts in `src/text_refiner.py`
  - English prompt (lines 242–279)
  - Russian prompt (lines 205–240)
- Added helper scripts:
  - `test_refiner_prompt.py`
  - `test_refiner_russian.py`
  - `update_prompt.py`
- Reference prompt files:
  - `improved_prompt.txt`
  - `improved_prompt_ru.txt`

### Testing results
- **English sample** (`test5min_original.md`)
  - ~28% reduction (only filler removal)
  - All examples retained (“1800 or 1700”, “0.8 points”, “2816 rating”)

- **Russian sample** (`tarn5minru_original.md`)
  - ~24% reduction
  - Technical vocabulary preserved
  - Removed fillers “ну”, “вот”, “э-э”

---

## [1.0.0] - 2025-10-02

### Added
- 🎉 Initial stable release
- ✅ Download and process YouTube videos
- ✅ Process local audio files (mp3, wav, ...)
- ✅ Whisper transcription (base, small, medium)
- ✅ LLM refinement via Ollama
- ✅ Automatic language detection (ru/en)
- ✅ Translation with Meta NLLB
- ✅ DOCX and Markdown export
- ✅ Custom Whisper prompts from file
- ✅ Prompt generation from YouTube metadata
- ✅ Logging and progress bars
- ✅ Apple M1/M2 optimisations

### Performance
- Whisper Base: 0.06× (≈16× faster than realtime)
- Whisper Small: 0.19× (≈5× faster than realtime)
- NLLB: 0.47× (≈2× faster than realtime)

---

## Versioning

The project follows [Semantic Versioning](https://semver.org/):
- MAJOR.MINOR.PATCH (e.g. 1.2.3)
- MAJOR — incompatible API changes
- MINOR — backward-compatible features
- PATCH — backward-compatible bug fixes
