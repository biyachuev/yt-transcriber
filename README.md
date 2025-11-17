# YouTube Transcriber & Translator

A flexible toolkit for transcribing and translating YouTube videos, audio files, and existing documents.

## 🎯 Highlights

### Version 1.6 (current)
- ✅ **Early API Key Validation**
  - Validates OpenAI API keys at startup with test API call
  - Fails fast before expensive operations (downloads, processing)
  - Clear error messages for invalid or missing keys
  - Saves time and bandwidth by catching errors early
- ✅ **VAD Performance Optimizations**
  - Lightweight Silero VAD for speech boundary detection (1.8MB model)
  - O(n²) → O(n) complexity reduction in boundary search
  - Gap quality filtering (300ms minimum) to prevent mid-syllable cuts
  - Accurate bitrate detection for optimal chunk sizing
  - No HuggingFace token required for basic chunking
  - 10-100x faster VAD processing for large files
- ✅ **Automatic yt-dlp Updates**
  - Automatic version checking before YouTube downloads
  - Auto-updates yt-dlp to prevent HTTP 403 errors
  - Keeps up with YouTube API changes

### Version 1.5
- ✅ **Speaker Diarization**
  - Automatic speaker identification using pyannote.audio
  - Speaker labels in transcripts ([SPEAKER_00], [SPEAKER_01], etc.)
  - Works with both local Whisper and OpenAI API
  - Optimal speaker detection using VAD integration
  - Enable with `--speakers` flag
  - ⚠️ Note: May over-segment speakers (one person → multiple labels); manual review recommended for critical use
- ✅ **Enhanced logging with colored output**
  - Color-coded log levels for better visibility
  - WARNING messages in orange for important notices
  - INFO messages in green for successful operations
  - ERROR/CRITICAL messages in red for failures
  - Smart warnings (e.g., missing Whisper prompt suggestions)

### Version 1.4
- ✅ **Video file support**
  - Process local video files (MP4, MKV, AVI, MOV, etc.)
  - Automatic audio extraction using FFmpeg
  - Full pipeline support (transcribe, translate, refine)

### Version 1.3
- ✅ **Document processing** (.docx, .md, .txt, .pdf)
  - Read existing transcripts
  - **PDF support**
  - Post-process text with an LLM
  - Translate uploaded documents
  - Automatic language detection
- ✅ **Quality & testing**
  - 139 automated tests with 49% coverage
  - CI/CD powered by GitHub Actions
  - Pre-commit hooks (black, flake8, mypy)
  - Full type hints across the codebase

### Version 1.1
- ✅ **Optimised prompts for LLM polishing**
  - Removes filler words ("um", "uh", etc.)
  - Normalises numbers ("twenty eight" → "28")
  - Preserves **all** facts and examples
  - Works for both Russian and English content

### Version 1.0
- ✅ Downloading and processing YouTube videos
- ✅ Processing local audio files (mp3, wav, ...)
- ✅ Processing local video files (mp4, mkv, avi, ...)
- ✅ Whisper-based transcription (base, small, medium)
- ✅ LLM-based refinement through Ollama (qwen2.5, llama3, ...)
- ✅ Automatic language detection (ru/en)
- ✅ Translation with Meta NLLB
- ✅ Export to .docx and .md
- ✅ Custom Whisper prompts (from file)
- ✅ Prompt generation from YouTube metadata
- ✅ Rich logging and progress bars
- ✅ Apple M1/M2 optimisations

### In progress
- 🔄 Optimized chunk processing for OpenAI API
- 🔄 Batch processing support
- 🔄 Docker support

## 📋 Requirements

### System
- Python 3.9+
- FFmpeg (audio preprocessing)
- Ollama (optional, for LLM refinement)
- 8 GB RAM minimum, 16 GB recommended
- ~5 GB disk space for Whisper and NLLB models
- Additional 3–7 GB if you use Ollama models

### Supported platforms
- macOS (including Apple Silicon)
- Linux
- Windows

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd yt-transcriber
```

### 2. Create a virtual environment

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install FFmpeg

**macOS**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian)**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows**
Download a build from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your `PATH`.

### 4. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Install Ollama (optional, for refinement)

**macOS/Linux**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Recommended models
ollama pull qwen2.5:3b    # Fast, good quality (~3 GB)
ollama pull qwen2.5:7b    # Slower, higher quality (~7 GB)

# Start the server (if not already running)
ollama serve
```

**Windows**
Download the installer from [ollama.com](https://ollama.com/download).

### 6. Environment variables (optional)

Create a `.env` file in the project root:

```bash
# Enable OpenAI integration (experimental)
OPENAI_API_KEY=your_api_key_here

# Logging level
LOG_LEVEL=INFO
```

## 📖 Usage

### Quick examples

#### 1. Transcribe a YouTube video

```bash
python -m src.main youtube --url "https://youtube.com/watch?v=dQw4w9WgXcQ" --transcribe whisper-base
```

#### 2. Transcribe and translate

```bash
python -m src.main youtube \
    --url "https://youtube.com/watch?v=dQw4w9WgXcQ" \
    --transcribe whisper-base \
    --translate nllb
```

#### 3. Process a local audio file

```bash
python -m src.main audio \
    --input audio.mp3 \
    --transcribe whisper-medium \
    --translate nllb
```

#### 4. Process a local video file

```bash
python -m src.main video \
    --input video.mp4 \
    --transcribe whisper-medium \
    --translate nllb
```

Supported video formats: MP4, MKV, AVI, MOV, and any format supported by FFmpeg.

#### 5. Refine a transcript with an LLM

```bash
python -m src.main audio \
    --input audio.mp3 \
    --transcribe whisper-medium \
    --refine-model qwen2.5:7b \
    --translate nllb
```

Produces two documents:
- `audio_original.docx/md` — raw transcript without translation
- `audio_refined.docx/md` — polished transcript with translation

#### 6. Use a custom Whisper prompt

```bash
# Create prompt.txt with project-specific terms
# FIDE, Hikaru Nakamura, Magnus Carlsen, chess tournament

python -m src.main youtube \
    --url "https://youtube.com/watch?v=YOUR_VIDEO_ID" \
    --transcribe whisper-base \
    --prompt-file prompt.txt
```

#### 7. Enable speaker diarization (v1.5)

```bash
# Transcribe with automatic speaker identification
python -m src.main youtube \
    --url "https://youtube.com/watch?v=YOUR_VIDEO_ID" \
    --transcribe whisper-medium \
    --speakers
```

**Requirements for speaker diarization:**
1. Get HuggingFace token: https://huggingface.co/settings/tokens (create a "Read" token)
2. Accept model terms for all required models:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/voice-activity-detection (optional, for better chunking)
3. Set token in environment: `export HF_TOKEN=your_token_here` (add to `~/.zshrc` or `~/.bashrc`)

Output will include speaker labels:
```
[00:00] [SPEAKER_00] Hello everyone, welcome to the show
[00:05] [SPEAKER_01] Thanks for having me
[00:08] [SPEAKER_00] Let's get started with today's topic
```

## ⚖️ Legal notice
- Make sure you respect YouTube Terms of Service and copyright law before downloading or processing any content. Only use the tool for media you own or have explicit permission to process.
- Output documents and logs may contain fragments of the original content. Store them locally and review licences before sharing.
- The default translation model `facebook/nllb-200-distilled-1.3B` is released under CC BY-NC 4.0 (non-commercial). Use a different model or obtain a licence for commercial scenarios.

#### 8. Process existing documents (v1.2)

```bash
# Improve an existing transcript
python -m src.main text --input output/document.md --refine-model qwen2.5:7b

# Translate a document
python -m src.main text --input transcription.docx --translate nllb

# Refine and translate
python -m src.main text --input document.txt --refine-model qwen2.5:7b --translate nllb
```

Supported formats: `.md`, `.docx`, `.txt`

#### 9. Help screen

```bash
python -m src.main --help
python -m src.main youtube --help
python -m src.main audio --help
```

### CLI structure

```bash
python -m src.main <command> [options]
```

**Commands:**
- `youtube` — Process a YouTube video
- `audio` — Process a local audio file
- `video` — Process a local video file
- `text` — Process a text document

### Common options

| Option | Description | Example |
|--------|-------------|---------|
| `--transcribe` | Transcription method | `--transcribe whisper-base` |
| `--translate` | Translation method | `--translate nllb` |
| `--refine-model` | Model for refinement | `--refine-model qwen2.5:7b` |
| `--refine-backend` | Refinement backend | `--refine-backend ollama` |
| `--prompt-file` | Custom Whisper prompt file | `--prompt-file prompt.txt` |
| `--nllb-model` | NLLB model override | `--nllb-model facebook/nllb-200-distilled-600M` |
| `--speakers` | Enable speaker diarization | `--speakers` |
| `--summarize-model` | Model for summarization | `--summarize-model qwen2.5:7b` |
| `--help` | Show help | `--help` |

### Available methods

**Transcription**
- `whisper-base` — fast, good quality
- `whisper-small` — slower, higher quality
- `whisper-medium` — slowest, best quality
- `whisper-openai-api` — OpenAI Whisper API (requires OPENAI_API_KEY)

**Refinement (requires Ollama or OpenAI API)**
- `qwen2.5:3b` — fast, 3 GB (recommended)
- `qwen2.5:7b` — slower, better quality
- `llama3.2:3b` — fast, solid quality
- `llama3:8b` — slower, higher quality
- `mistral:7b` — balanced
- `gpt-4o-mini` — OpenAI (requires API key)
- Any other model available in the [Ollama library](https://ollama.com/library)

**Translation**
- `nllb` — Meta NLLB (local, free)
- `openai-api` — OpenAI GPT API (requires OPENAI_API_KEY)

## 📁 Project structure

```
yt-transcriber/
├── src/                      # Source code
│   ├── main.py              # Entry point
│   ├── config.py            # Configuration
│   ├── downloader.py        # YouTube downloads
│   ├── transcriber.py       # Transcription
│   ├── text_reader.py       # Text ingestion
│   ├── translator.py        # Translation
│   ├── text_refiner.py      # LLM-based refinement
│   ├── document_writer.py   # Document generation
│   ├── utils.py             # Utilities
│   └── logger.py            # Logging setup
├── tests/                   # Automated tests
├── output/                  # Generated docs
├── temp/                    # Temporary files
├── logs/                    # Logs
├── requirements.txt         # Runtime dependencies
├── .env.example             # Sample configuration
└── README.md                # Documentation
```

**Note:** Whisper and NLLB models are cached in `~/.cache/` on first run.

## 🔧 Configuration

Main settings live in `src/config.py`:

```python
# Paths
OUTPUT_DIR = "output"        # Output folder
TEMP_DIR = "temp"            # Temporary files
LOGS_DIR = "logs"            # Logs

# Models
WHISPER_DEVICE = "mps"       # cpu/cuda/mps (auto-switch for M1)
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Logging
LOG_LEVEL = "INFO"           # DEBUG/INFO/WARNING/ERROR
```

## 📊 Performance

Approximate processing time on a MacBook Air M1 (16 GB, CPU):

| Video length | whisper_base | whisper_small | NLLB translation | Total (base+translate) | Total (small+translate) |
|--------------|--------------|---------------|------------------|------------------------|-------------------------|
| 3 minutes    | ~11 s        | ~34 s         | ~1.5 min         | ~2 min                 | ~3 min                  |
| 10 minutes   | ~36 s        | ~2 min        | ~5 min           | ~5.5 min               | ~7 min                  |
| 30 minutes   | ~1.8 min     | ~5.7 min      | ~14 min          | ~16 min                | ~20 min                 |
| 1 hour       | ~3.6 min     | ~11 min       | ~28 min          | ~32 min                | ~39 min                 |
| 2 hours      | ~7 min       | ~23 min       | ~56 min          | ~63 min                | ~79 min                 |

**Processing factors:**
- Whisper Base: 0.06× (≈16× faster than realtime) 🚀
- Whisper Small: 0.19× (≈5× faster than realtime)
- NLLB: 0.47× (≈2× faster than realtime)

## 🐛 Troubleshooting

### Installation issues

**Problem:** `torch` fails to install on Apple Silicon
```bash
# Use the dedicated Apple Silicon build
pip install --upgrade torch torchvision torchaudio
```

**Problem:** FFmpeg not found
```bash
ffmpeg -version
# If missing, install via Homebrew (macOS)
brew install ffmpeg
```

**Problem:** Out of memory
```bash
# Switch to a smaller Whisper model
python -m src.main --url "..." --transcribe whisper_base
```

### Runtime issues

**Problem:** `Model not found`
- Models download automatically on first run
- Ensure you have an internet connection
- Check that the `models/` directory is writable

**Problem:** Processing is slow
- Use `whisper_base` instead of `whisper_small`
- Confirm that GPU/MPS acceleration is active (see logs)
- Close other resource-heavy applications

**Safe to ignore:** Speaker diarization warnings
- `UserWarning: torchcodec is not installed correctly` — Audio loading uses soundfile/librosa fallback (works correctly)
- `UserWarning: std(): degrees of freedom is <= 0` — Internal pyannote calculation (does not affect results)
- `UserWarning: Lightning automatically upgraded your loaded checkpoint` — PyTorch Lightning version compatibility (does not affect results)
- See [FAQ.md](FAQ.md) for detailed explanations

### VAD Performance Optimization

The tool now uses **Silero VAD** by default for speech boundary detection when splitting large audio files. Silero VAD offers:
- **Faster processing**: ~1ms per 30ms chunk (CPU-based)
- **Smaller footprint**: 1.8MB model vs pyannote's full diarization stack
- **No HuggingFace token required** for basic VAD functionality
- **Automatic fallback** to pyannote VAD if Silero is unavailable

**For optimal performance:**
1. Silero VAD is used automatically (no setup needed)
2. PyAnnote VAD is still available for speaker diarization (`--speakers` flag)
3. To eliminate PyTorch Lightning warnings from pyannote, upgrade checkpoints once:
   ```bash
   python -c "from lightning.pytorch.cli import LightningCLI; LightningCLI(run=False)"
   ```

**Performance improvements in v1.6:**
- ✅ O(n²) → O(n) complexity in boundary search
- ✅ Gap quality filtering (300ms minimum, prevents mid-syllable cuts)
- ✅ Accurate bitrate detection for chunk size estimation
- ✅ Lightweight Silero VAD by default

## 🧪 Testing

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Coverage report
pytest --cov=src tests/
```

## 📝 Sample output

### .docx format
```
# Video title

## Translation
Method: NLLB

[00:15] Hello everyone! Today we will talk about...

[01:32] The first important topic is...

## Transcript
Method: whisper_base

[00:15] Hello everyone! Today we'll talk about...

[01:32] The first important topic is...
```

### .md format
Uses the same layout with Markdown syntax.

## 🛣️ Roadmap

### v1.0 — ✅ Shipped
- ✅ YouTube + local audio ingestion
- ✅ Whisper (base, small, medium)
- ✅ LLM-based refinement via Ollama
- ✅ NLLB translation
- ✅ Custom prompts
- ✅ Automatic language detection

### v2.0 — Planned
- [ ] Extended document ingestion
- [ ] OpenAI API integration
- [ ] Speaker diarisation
- [ ] Enhanced CI/CD + unit tests
- [ ] Docker image
- [ ] Web UI
- [ ] Batch processing helpers

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss what you would like to improve.

### Development flow

1. Fork the repository
2. Create a branch (`git checkout -b feature/amazing-feature`)
3. Commit (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

- Distributed under the MIT License — see `LICENSE` for details.
- The codebase was developed with help from AI-assisted tools (e.g., GitHub Copilot, Codex). All code and docs were reviewed and validated manually before publishing.

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) — transcription
- [Meta NLLB](https://github.com/facebookresearch/fairseq/tree/nllb) — translation
- [Ollama](https://ollama.com) — local LLMs
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube downloads

## 📞 Contact

For questions or suggestions, please open an issue in this repository.

---

## 💡 Usage tips

### Improve transcription quality
1. Use `whisper_medium` for critical content
2. Provide prompt files with key terms and names
3. For YouTube sources, metadata-derived prompts are added automatically

### Improve text quality
1. Install Ollama and pull `qwen2.5:7b` for best results
2. Language detection switches between Russian and English automatically
3. Use `--refine-model` to produce a clean transcript
4. **New in v1.1:** the LLM prompt
   - Removes filler words ("um", "uh", "эм", "ну", "вот")
   - Skips meta commentary ("let me scroll", "сейчас открою экран")
   - Normalises numbers ("twenty eight sixteen" → "2816", "ноль восемь" → "0.8")
   - **Keeps every detail**: examples, facts, reasoning
   - No summarisation — only clean-up and structuring
   - Fixes punctuation and paragraphing

### Optimise speed
- `whisper_base` — high throughput
- `whisper_medium` — best accuracy
- `qwen2.5:3b` — fast refinement
- `qwen2.5:7b` — highest quality

### Model cache locations
- Whisper: `~/.cache/whisper/` (~140 MB – 1.5 GB)
- Ollama: manage via `ollama list` and `ollama rm <model>`
