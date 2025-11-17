# Usage Examples

> ⚠️ Use the tool only for content you have the rights to process. Replace the placeholder IDs below (`YOUR_VIDEO_ID`) with your own links or files.

## Basic scenarios

### 1. Transcribe a YouTube video

```bash
python -m src.main youtube \
    --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" \
    --transcribe whisper-base
```

**Output**
- `output/Video_Title.docx` – transcript
- `output/Video_Title.md` – transcript

---

### 2. Transcribe and translate

```bash
python -m src.main youtube \
    --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" \
    --transcribe whisper-base \
    --translate nllb
```

**Output**: a document with two sections:
1. Russian translation
2. Original transcript

---

### 3. Process a local audio file

```bash
python -m src.main audio \
    --input audio.mp3 \
    --transcribe whisper-base \
    --translate nllb
```

**Supported formats:** mp3, wav, m4a, flac, ogg

---

### 4. Transcribe with a higher-quality model

```bash
python -m src.main youtube \
    --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" \
    --transcribe whisper-medium
```

**Available models:**
- `whisper-base` — fast, solid quality (default)
- `whisper-small` — slower, better quality
- `whisper-medium` — slowest, highest quality

---

### 5. Provide a custom prompt

Custom prompts help Whisper capture domain-specific names and terminology.

```bash
python -m src.main youtube \
    --url "https://www.youtube.com/watch?v=YOUR_CHESS_VIDEO_ID" \
    --transcribe whisper-base \
    --prompt-file prompt.txt
```

**Example `prompt.txt`:**
```
FIDE, Hikaru Nakamura, Magnus Carlsen, chess tournament, bongcloud
```

---

### 6. Refine a transcript with an LLM

```bash
python -m src.main audio \
    --input interview.mp3 \
    --transcribe whisper-medium \
    --refine-model qwen2.5:3b
```

**Output**
- `interview_original.docx/md` – raw transcript
- `interview_refined.docx/md` – polished transcript

**Requirements**
1. Ollama installed: https://ollama.ai
2. Model pulled: `ollama pull qwen2.5:3b`
3. Ollama server running: `ollama serve`

---

### 7. Full pipeline: refine + translate + prompt

```bash
python -m src.main audio \
    --input lecture.mp3 \
    --transcribe whisper-medium \
    --translate nllb \
    --refine-model qwen2.5:3b \
    --prompt-file lecture_prompt.txt
```

**Output**
- `lecture_original.docx/md`
- `lecture_refined.docx/md`
- `lecture_translated.docx/md`

---

## Advanced scenarios

### 8. Long-form educational content

```bash
python -m src.main youtube \
    --url "https://www.youtube.com/watch?v=YOUR_LONG_INTERVIEW_ID" \
    --transcribe whisper-medium \
    --translate nllb \
    --refine-model qwen2.5:3b
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
    python -m src.main youtube \
        --url "$url" \
        --transcribe whisper-medium \
        --translate nllb \
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
    python -m src.main youtube \
        --url "$url" \
        --transcribe whisper-medium \
        --translate nllb \
        --refine-model qwen2.5:3b
done < urls.txt

```

---

## Error Handling

### Early API Key Validation

The tool validates OpenAI API keys **before** starting expensive operations (downloads, processing). This saves time and bandwidth by catching authentication errors early.

**Example with invalid key:**

```bash
# Set invalid API key
export OPENAI_API_KEY="invalid_key"

# Try to transcribe
python -m src.main youtube \
    --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" \
    --transcribe whisper-openai-api
```

**Output:**
```
2025-10-30 16:49:05 - yt - INFO - Validating OpenAI API key...
2025-10-30 16:49:05 - yt - ERROR - ❌ Invalid OpenAI API key
2025-10-30 16:49:05 - yt - ERROR - Error: Error code: 401 - {'error': {'message': 'Incorrect API key provided...'}}
2025-10-30 16:49:05 - yt - ERROR - Please check your OPENAI_API_KEY in .env file
2025-10-30 16:49:05 - yt - ERROR - API key validation failed. Please fix the issues above before continuing.
```

The tool exits immediately without downloading the video or starting transcription.

**Validated operations:**
- `--transcribe whisper-openai-api` (transcription)
- `--translate openai-api` (translation)
- `--refine-backend openai-api` (refinement)
- `--summarize-backend openai-api` (summarization)

If any operation requires OpenAI, the key is validated once at startup.
