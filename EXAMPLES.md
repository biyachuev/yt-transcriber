# Usage Examples

> ⚠️ Use the tool only for content you have the rights to process. Replace the placeholder IDs below (`YOUR_VIDEO_ID`) with your own links or files.

## Basic scenarios

### 1. Transcribe a YouTube video

```bash
python -m src.main     --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"     --transcribe whisper_base
```

**Output**
- `output/Video_Title.docx` – transcript
- `output/Video_Title.md` – transcript

---

### 2. Transcribe and translate

```bash
python -m src.main     --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"     --transcribe whisper_base     --translate NLLB
```

**Output**: a document with two sections:
1. Russian translation
2. Original transcript

---

### 3. Process a local audio file

```bash
python -m src.main     --input_audio audio.mp3     --transcribe whisper_base     --translate NLLB
```

**Supported formats:** mp3, wav, m4a, flac, ogg

---

### 4. Transcribe with a higher-quality model

```bash
python -m src.main     --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"     --transcribe whisper_medium
```

**Available models:**
- `whisper_base` — fast, solid quality (default)
- `whisper_small` — slower, better quality
- `whisper_medium` — slowest, highest quality

---

### 5. Provide a custom prompt

Custom prompts help Whisper capture domain-specific names and terminology.

```bash
python -m src.main     --url "https://www.youtube.com/watch?v=YOUR_CHESS_VIDEO_ID"     --transcribe whisper_base     --prompt prompt.txt
```

**Example `prompt.txt`:**
```
FIDE, Hikaru Nakamura, Magnus Carlsen, chess tournament, bongcloud
```

---

### 6. Refine a transcript with an LLM

```bash
python -m src.main     --input_audio interview.mp3     --transcribe whisper_medium     --refine-model qwen2.5:3b
```

**Output**
- `interview (original).docx/md` – raw transcript
- `interview (refined).docx/md` – polished transcript

**Requirements**
1. Ollama installed: https://ollama.ai
2. Model pulled: `ollama pull qwen2.5:3b`
3. Ollama server running: `ollama serve`

---

### 7. Full pipeline: refine + translate + prompt

```bash
python -m src.main     --input_audio lecture.mp3     --transcribe whisper_medium     --translate NLLB     --refine-model qwen2.5:3b     --prompt lecture_prompt.txt
```

**Output**
- `lecture (original).docx/md`
- `lecture (refined).docx/md`
- `lecture (translated).docx/md`

---

## Advanced scenarios

### 8. Long-form educational content

```bash
python -m src.main     --url "https://www.youtube.com/watch?v=YOUR_LONG_INTERVIEW_ID"     --transcribe whisper_medium     --translate NLLB     --refine-model qwen2.5:3b
```

**Estimated time:** ~4–5 hours on a MacBook Air M1 for a two-hour interview.

---

### 9. Batch processing script

Create `process_multiple.sh`:

```bash
#!/bin/bash

URLS=(
    "https://youtube.com/watch?v=VIDEO1"
    "https://youtube.com/watch?v=VIDEO2"
    "https://youtube.com/watch?v=VIDEO3"
)

for url in "${URLS[@]}"; do
    echo "Processing: $url"
    python -m src.main \
        --url "$url" \
        --transcribe whisper_medium \
        --translate NLLB \
        --refine-model qwen2.5:3b

    echo "Done: $url"
    echo "---"
done

echo "All videos processed!"
```

Run it:
```bash
chmod +x process_multiple.sh
./process_multiple.sh
```

---

### 10. Work with playlists

Extract video URLs first:

```bash
pip install yt-dlp

yt-dlp --flat-playlist --print url "PLAYLIST_URL" > urls.txt

while read url; do
    python -m src.main \
        --url "$url" \
        --transcribe whisper_medium \
        --translate NLLB \
        --refine-model qwen2.5:3b
done < urls.txt

```
