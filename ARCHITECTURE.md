# Project Architecture

A detailed overview of the system design, tech stack, and major decisions.

## 🏗️ High-level overview

### Component diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Interface                        │
│                            (main.py)                         │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Downloader  │  │ Transcriber  │  │  Translator  │
│              │  │              │  │              │
│  - yt-dlp    │  │  - Whisper   │  │  - NLLB      │
│  - FFmpeg    │  │  - PyTorch   │  │  - HuggingF. │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       └─────────────────┼──────────────────┘
                         │
                         ▼
                 ┌──────────────┐
                 │ Text Refiner │
                 │              │
                 │  - Ollama    │
                 │  - LLM API   │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │ Document     │
                 │ Writer       │
                 │              │
                 │ - python-docx│
                 │ - markdown   │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │   Output     │
                 │ (.docx, .md) │
                 └──────────────┘
```

---

## 📦 Modules

### 1. **config.py** — configuration hub

**Purpose:** single source of truth for application settings.

```python
class Settings(BaseSettings):
    OUTPUT_DIR, TEMP_DIR, LOGS_DIR
    WHISPER_MODEL_DIR, NLLB_MODEL_DIR
    OPENAI_API_KEY
    WHISPER_DEVICE  # auto-detected
    SUPPORTED_LANGUAGES = ["ru", "en"]
```

**Key choices**
- `pydantic-settings` for validation and environment loading
- Automatic device detection (CPU / CUDA / MPS)
- Optional `.env` file support
- Auto-create required folders
- Apple Silicon friendly defaults

---

### 2. **downloader.py** — YouTube ingestion

**Purpose:** download audio plus metadata from YouTube.

**Dependencies**
- `yt-dlp` — fetch media and metadata
- `FFmpeg` — convert to MP3

**Flow**
```
URL → yt-dlp → video → FFmpeg → MP3 → temp/
             ↓
        metadata (title, duration, tags)
```

**Highlights**
- Progress bar for downloads
- Audio-only pipelines to save bandwidth
- Filename sanitising
- Resilient network error handling

---

### 3. **transcriber.py** — speech to text

**Purpose:** convert audio into timestamped segments.

**Models:** OpenAI Whisper
- Base — 74M parameters, fastest
- Small — 244M parameters, balanced
- Medium — 769M parameters, highest quality

**Flow**
```
MP3 → Whisper → segments → [timestamp, text]
```

**TranscriptionSegment**
```python
class TranscriptionSegment:
    start: float   # seconds
    end: float
    text: str
    speaker: str | None
```

**Capabilities**
- Custom initial prompts
- Prompt generation from YouTube metadata
- `update_segments_from_text()` for LLM-refined text

**Optimisations**
- CPU fallback on Apple Silicon (MPS limitations)
- Progress indicators via `tqdm`
- FP16 on CUDA devices
- Lazy model loading

---

### 4. **translator.py** — text translation

**Purpose:** translate transcripts segment-by-segment.

**Default backend:** Meta NLLB-200 (`facebook/nllb-200-distilled-600M`)
- 200+ languages
- Distilled 600M parameter model

**Flow**
```
text → chunking → NLLB → translated text
```

**Features**
- Automatic language identification
- Chunking to respect context limits
- Beam search decoding for quality (5 beams)
- Segment level translation preserving timestamps

---

### 5. **text_refiner.py** — LLM polishing

**Purpose:** clean up transcripts or translations with a local LLM.

**How it works**
- Collects context: original text + optional prompt
- Sends request to an Ollama model
- Reconstructs segments with improved text

**Safeguards**
- Graceful degradation if Ollama is unavailable
- Length checks to prevent prompt overflows

---

### 6. **document_writer.py** — export

**Purpose:** generate DOCX and Markdown reports.

**Responsibilities**
- Compose sections with transcription and translation
- Optional timestamps
- Apply project-wide typography (Arial, 11pt)
- Save to `output/`

---

### 7. **utils.py** — helpers

Provides utilities for:
- Filename sanitising
- Time formatting
- Language detection heuristics
- Prompt generation (metadata + LLM assisted)
- Text chunking and processing time estimates

---

### 8. **logger.py** — logging setup

- Structured logging to console and rotating files
- Timestamped log files in `logs/`
- INFO level for console, DEBUG for files
- Reusable `logger` instance for all modules

---

## 🔄 Data flow

1. **Input acquisition** — YouTube URL, audio file, or text document
2. **Optional metadata prompt** — derived from title, tags, subtitles
3. **Transcription** — Whisper produces timestamped segments
4. **Refinement** — Ollama improves phrasing/punctuation
5. **Translation** — NLLB (and optional LLM refinement)
6. **Export** — DOCX + Markdown written to `output/`

---

## ⚙️ Operational considerations

- **Caching**: models are cached in `~/.cache/` (Whisper/NLLB) and Ollama’s store
- **Resource usage**: medium Whisper requires ~5 GB RAM; NLLB loads on demand
- **Error handling**: downloader retries, transcription exits gracefully, translators skip failing chunks
- **Logging**: every step reports status; prompts are trimmed before logging for privacy
- **Extensibility**: additional translators or diarisation modules can be plugged in via interface-like classes

---

## 🧱 Build vs buy decisions

- **Whisper**: chosen for quality and available open-source weights
- **NLLB**: covers 200 languages locally, avoiding API costs
- **Ollama**: simplifies local LLM management; optional for offline scenarios
- **yt-dlp**: community-maintained, resilient to YouTube updates

---

## 🚧 Future improvements

- Replace hard-coded log strings with localisation support
- Add diarisation once pyannote integration stabilises
- Provide a plugin interface for custom exporters (PDF, HTML)
- Improve batching for large document processing

