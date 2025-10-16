# 🚀 YouTube Transcriber Cheat Sheet

A quick reference covering installation, CLI usage, and handy developer commands.

## ⚡ Installation (3 commands)

```bash
# 1. Create and activate a virtual environment
python -m venv venv && source venv/bin/activate

# 2. Install FFmpeg (macOS example)
brew install ffmpeg

# 3. Install Python dependencies
pip install -r requirements.txt
```

---

## 📝 Core commands

### Show help
```bash
python -m src.main --help
```

### Transcribe a YouTube video
```bash
python -m src.main --url "YOUTUBE_URL" --transcribe whisper_base
```

### Transcribe + translate
```bash
python -m src.main --url "YOUTUBE_URL" --transcribe whisper_base --translate NLLB
```

### Process a local audio file
```bash
python -m src.main --input_audio file.mp3 --transcribe whisper_base
```

### Refine with an LLM
```bash
python -m src.main --input_audio file.mp3 --transcribe whisper_medium --refine-model qwen2.5:3b
```

### Use a custom prompt
```bash
python -m src.main --url "..." --transcribe whisper_base --prompt prompt.txt
```

---

## 🔧 CLI arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--url` | YouTube URL | `--url "https://youtube.com/..."` |
| `--input_audio` | Audio file path | `--input_audio audio.mp3` |
| `--input_text` | Text document path | `--input_text doc.docx` |
| `--transcribe` | Transcription backend | `--transcribe whisper_base` |
| `--translate` | Translation backend | `--translate NLLB` |
| `--prompt` | Whisper prompt file | `--prompt prompt.txt` |
| `--refine-model` | Ollama model | `--refine-model qwen2.5:3b` |
| `--speakers` | Speaker diarisation (experimental) | `--speakers` |
| `--help` | Show help | `--help` |

---

## 🎯 Processing backends

### Transcription
- `whisper_base` — fast, good quality
- `whisper_small` — slower, higher quality
- `whisper_medium` — slowest, best quality
- `whisper_openai_api` — via OpenAI API (planned)

### Translation
- `NLLB` — local, free
- `openai_api` — via OpenAI API (planned)

### LLM refinement
- `qwen2.5:3b` — fast, solid quality
- `llama3.2:3b` — fast, solid quality
- `llama3:8b` — slower, premium quality
- `mistral:7b` — balanced choice

---

## 📁 Project layout

```
yt-transcriber/
├── src/              # Source code
├── tests/            # Automated tests
├── output/           # Results ← start here
│   ├── * (original).*  # Original versions
│   └── * (refined).*   # Polished versions
├── temp/             # Temp files
├── logs/             # Logs
├── models/           # Cached models
└── prompts/          # Custom prompts
```

---

## 🐍 Python API snippets

### Imports
```python
from src.downloader import YouTubeDownloader
from src.transcriber import Transcriber
from src.translator import Translator
from src.text_refiner import TextRefiner
from src.document_writer import DocumentWriter
```

### Download audio
```python
downloader = YouTubeDownloader()
audio_path, title, duration = downloader.download_audio(url)
```

### Transcribe
```python
transcriber = Transcriber(method="whisper_base")
segments = transcriber.transcribe(audio_path)
text = transcriber.segments_to_text(segments)
```

### Translate
```python
translator = Translator(method="NLLB")
translated = translator.translate_text(text, source_lang="en", target_lang="ru")
```

### Refine text
```python
refiner = TextRefiner(model_name="qwen2.5:3b")
refined_text = refiner.refine_text(original_text, context="prompt")
```

### Create documents
```python
writer = DocumentWriter()
docx_path, md_path = writer.create_from_segments(
    title="My Video",
    transcription_segments=segments,
    translation_segments=translated_segments,
)
```

---

## 🔍 Debugging

### Enable DEBUG logs
```bash
# In .env
LOG_LEVEL=DEBUG
```

### Tail logs
```bash
tail -f logs/app_*.log
```

### Run tests
```bash
pytest tests/          # full suite
pytest -v tests/       # verbose output
pytest --cov=src tests/ # coverage
```

---

## 🛠️ Helpful commands

```bash
rm -rf temp/*            # clean temp files
rm logs/app_*.log        # remove old logs

du -sh models/*          # check model sizes

pip install --upgrade -r requirements.txt  # update deps

ffmpeg -version          # verify FFmpeg
```

---

## 🐳 Docker essentials

```bash
docker build -t yt-transcriber .

docker run -v $(pwd)/output:/app/output   yt-transcriber   --url "YOUTUBE_URL"   --transcribe whisper_base

# docker compose
docker-compose up           # foreground
docker-compose up -d        # detached
docker-compose down         # stop
```

---

## ⚙️ Environment variables (.env)

```bash
OPENAI_API_KEY=sk-...   # API keys
LOG_LEVEL=INFO          # logging level
WHISPER_DEVICE=mps      # cpu, cuda, or mps
```

---

## 📊 Processing time (M1, 16 GB)

**Baseline (whisper_base + NLLB)**
| Video | Approx. time |
|-------|--------------|
| 10 min | ~18 min |
| 30 min | ~55 min |
| 1 hour | ~110 min |
| 2 hours | ~220 min |

**With refinement (whisper_medium + qwen2.5:3b + NLLB)**
| Video | Approx. time |
|-------|--------------|
| 10 min | ~53 min |
| 30 min | ~160 min |
| 1 hour | ~320 min |

---

## 🎨 Output formats

### DOCX
- Styled Word document
- Typography presets
- Cyrillic compatible

### Markdown
- Lightweight text
- Git-friendly
- Works with Obsidian/Notion

### File naming
- `name.docx/md` — original version
- `name (original).docx/md` — raw transcript (when refinement is enabled)
- `name (refined).docx/md` — polished version

---

## 🚨 Common issues

### FFmpeg not found
```bash
brew install ffmpeg        # macOS
sudo apt install ffmpeg    # Linux
```

### Out of memory
```bash
--transcribe whisper_base  # switch to a smaller model
```

### Model download timeout
```bash
# Check your connection and retry
# Models are cached in models/
```

### Ollama connection error
```bash
ollama serve
curl http://localhost:11434/api/tags
```
