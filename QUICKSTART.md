# Quick Start

Get up and running with YouTube Transcriber in five minutes.

## ⚡ Install in 3 steps

### Step 1: clone and create a virtual environment

```bash
# Clone the repository
git clone <repository-url>
cd yt-transcriber

# Create a virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
# venv\Scripts\activate
```

### Step 2: install FFmpeg

**macOS**
```bash
brew install ffmpeg
```

**Linux (Ubuntu)**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows**
1. Download a build from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Add the `bin` folder to `PATH`

### Step 3: install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

⏱️ Expect 5–10 minutes on first setup.

---

## 🚀 First run

### Smoke test

```bash
python -m src.main     --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"     --transcribe whisper_base
```

What happens:
1. ⬇️ Audio is downloaded from YouTube
2. 📝 The transcript is generated
3. 💾 Results land in `output/`

⏱️ First run is slower because models (~2–3 GB) are downloaded.

---

### Add translation

```bash
python -m src.main     --url "https://www.youtube.com/watch?v=YOUR_VIDEO"     --transcribe whisper_base     --translate NLLB
```

Output:
- `output/Video_Title.docx` — Word document
- `output/Video_Title.md` — Markdown document

Both contain:
1. Russian translation (if the original is in English)
2. Original transcript
3. Timestamps for each paragraph

---

## 📋 Essential commands

### Show help
```bash
python -m src.main --help
```

### Transcribe only
```bash
python -m src.main --url "URL" --transcribe whisper_base
```

### Transcribe + translate
```bash
python -m src.main --url "URL" --transcribe whisper_base --translate NLLB
```

---

## 📁 Where to find results

```
yt-transcriber/
├── output/              # ← Processed documents
│   ├── Video_Title.docx
│   └── Video_Title.md
├── temp/                # Temporary audio
└── logs/                # Execution logs
```

---

## 🎯 Typical scenarios

### Scenario 1: English lecture
```bash
python -m src.main     --url "https://youtube.com/watch?v=..."     --transcribe whisper_base     --translate NLLB
```
Result: transcript + Russian translation

---

### Scenario 2: Russian video
```bash
python -m src.main     --url "https://youtube.com/watch?v=..."     --transcribe whisper_base
```
Result: transcript only

---

### Scenario 3: Process multiple videos

Create `process_videos.sh`:

```bash
#!/bin/bash

python -m src.main --url "https://youtube.com/watch?v=VIDEO1" --transcribe whisper_base --translate NLLB
python -m src.main --url "https://youtube.com/watch?v=VIDEO2" --transcribe whisper_base --translate NLLB
python -m src.main --url "https://youtube.com/watch?v=VIDEO3" --transcribe whisper_base --translate NLLB

echo "Done!"
```

Run it:
```bash
chmod +x process_videos.sh
./process_videos.sh
```

---

## ⚙️ Optional configuration

### Create a `.env`

```bash
cp .env.example .env
```

Edit `.env`:
```bash
LOG_LEVEL=INFO    # or DEBUG for verbose logging
WHISPER_DEVICE=mps  # mps (M1/M2), cuda (NVIDIA), or cpu
```

---

## 🐛 Troubleshooting

### “FFmpeg not found”
```bash
ffmpeg -version  # should print version info
# If missing, install FFmpeg (see Step 2)
```

---

### “Out of memory”

Use a lighter model:
```bash
--transcribe whisper_base
```

Or split the audio:
```bash
ffmpeg -i long_video.mp3 -ss 00:00:00 -t 01:00:00 part1.mp3
```

---

### Models take a long time to download

Expected on first run:
- Whisper Base ~150 MB
- NLLB ~2.5 GB

They are cached in `models/` afterwards.

---

### Processing feels slow

CPU-only processing runs close to realtime:
- 1 hour of video ≈ 1–1.5 hours processing

Speedups:
1. Use a GPU if available
2. Run overnight for long batches
3. Process shorter clips

---

## 📊 Expected duration (M1, 16 GB)

| Video length | Approx. time |
|--------------|--------------|
| 10 min       | ~18 min |
| 30 min       | ~55 min |
| 1 hour       | ~110 min |
| 2 hours      | ~220 min |

---

## 🎓 Next steps

1. 📖 [README.md](README.md) — full documentation
2. 💡 [EXAMPLES.md](EXAMPLES.md) — more scenarios
3. ❓ [FAQ.md](FAQ.md) — troubleshooting and tips
4. 🐳 Docker instructions in README for containerised runs

---

## 💬 Need help?

- Check the [FAQ](FAQ.md)
- Open an [issue on GitHub](https://github.com/yourusername/yt-transcriber/issues)
- Reach out to the maintainers

---

## ✅ Readiness checklist

- [ ] Python 3.9+ installed
- [ ] FFmpeg installed and on PATH
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] First test run completed
- [ ] Results located in `output/`

All set? Great — start processing your videos! 🎉
