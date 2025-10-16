# Frequently Asked Questions (FAQ)

## General

### What does this tool do?

YouTube Transcriber & Translator automatically creates transcripts for YouTube videos and local audio files, optionally polishes them with an LLM, and translates them into Russian. It relies on Whisper for transcription, local LLMs via Ollama for refinement, and Meta NLLB for translation.

---

### Which languages are supported?

**Current release**
- Transcription: Russian and English (auto-detected)
- Translation: English → Russian
- LLM refinement: Russian and English

**Available Whisper models**
- `whisper_base` — fast, good quality
- `whisper_small` — slower, better quality
- `whisper_medium` — slowest, best quality

**Roadmap**
- Broader language pairs via NLLB (200+ languages)

---

### Is the tool free?

Yes. The current version runs fully on your machine. All models (Whisper, NLLB) are downloaded and executed locally.

**Optional extras**
- LLM refinement through Ollama (free, requires installing Ollama)
- Custom prompts for better term recognition

**Note:** Future versions may add optional OpenAI API integration (paid usage).

---

## Technical

### What are the hardware requirements?

**Minimum**
- 8 GB RAM
- 5 GB of free disk space (models)
- Python 3.9+

**Recommended**
- 16 GB RAM
- 10 GB free space
- Apple M1/M2 or an NVIDIA GPU (optional)

**For LLM refinement (Ollama)**
- Additional 4–8 GB RAM, depending on the model
- Ollama installed and running

---

### Does it run on Windows/Linux?

Yes, the tool is cross-platform:
- ✅ macOS (including M1/M2)
- ✅ Linux
- ✅ Windows

---

### Do I need a GPU?

No. The app works on CPU. A GPU simply speeds things up:
- **With GPU:** roughly 3–5× faster
- **Without GPU:** works fine, just slower

On Apple Silicon the MPS backend is used automatically.

---

### How long does processing take?

Approximate timings on a MacBook Air M1 (16 GB):

**Baseline (whisper_base + NLLB)**
| Video length | Transcription | Translation | Total |
|--------------|---------------|-------------|-------|
| 10 min       | ~15 min       | ~3 min      | ~18 min |
| 30 min       | ~45 min       | ~10 min     | ~55 min |
| 1 hour       | ~90 min       | ~20 min     | ~110 min |
| 2 hours      | ~180 min      | ~40 min     | ~220 min |

**With LLM refinement (whisper_medium + qwen2.5:3b + NLLB)**
| Video length | Transcription | Refinement | Translation | Total |
|--------------|---------------|------------|-------------|-------|
| 10 min       | ~45 min       | ~5 min     | ~3 min      | ~53 min |
| 30 min       | ~135 min      | ~15 min    | ~10 min     | ~160 min |
| 1 hour       | ~270 min      | ~30 min    | ~20 min     | ~320 min |

More powerful hardware will reduce these numbers.

---

### How accurate is the transcription?

Accuracy depends on:
- **Audio quality:** cleaner audio produces better results
- **Speaker accent:** neutral accents are easier to transcribe
- **Background noise:** less noise → higher accuracy
- **Model choice:** Base (good) vs Small (better) vs Medium (best)
- **Prompt quality:** custom prompts improve terminology recognition

**Typical accuracy**
- Clean audio, standard accent: 90–95%
- Noisy environments: 70–85%
- Heavy accent: 60–80%

**LLM refinement adds**
- Better punctuation
- Removal of filler words
- Improved structure
- Terminology fixes

---

### How accurate is the translation?

NLLB delivers solid quality:
- ✅ Meaning is preserved
- ⚠️ Technical jargon may need review
- ⚠️ Idioms can translate literally

Always proofread technical translations.

---

### How accurate is speaker diarization?

Speaker diarization (enabled with `--speakers`) identifies different speakers in audio and labels them as SPEAKER_00, SPEAKER_01, etc.

**Accuracy depends on:**
- **Audio quality:** single-channel recordings may reduce accuracy
- **Recording setup:** studio mics (high quality) vs phone recordings (lower quality)
- **Speaker overlap:** people talking over each other causes confusion
- **Voice similarity:** similar-sounding speakers are harder to distinguish
- **Microphone distance changes:** one speaker moving closer/farther may be split into multiple labels

**Typical results:**
- ✅ Clean studio recordings with distinct voices: 85–95% accuracy
- ⚠️ Phone/video calls: 70–85% accuracy
- ⚠️ Noisy environments or overlapping speech: 50–70% accuracy

**Known limitations:**
- ⚠️ **Over-segmentation:** One speaker may be assigned multiple labels (e.g., SPEAKER_00 and SPEAKER_01 for the same person)
  - Common when speaker changes tone, distance from mic, or there are long pauses
  - Manual review recommended for critical applications
- ⚠️ **Under-segmentation:** Multiple speakers may be assigned the same label
  - Less common, happens with very similar voices

**Recommendation:** Use speaker labels as a guide, but verify important segments manually.

**Built-in audio preprocessing:**
The system automatically preprocesses audio before diarization to improve accuracy:
- ✅ Conversion to mono 16kHz (standard for speech models)
- ✅ RMS volume normalization to -20 dBFS (prevents quiet sections from being misclassified)
- ✅ Clipping prevention (avoids distortion)
- 🔄 Optional noise reduction (available with `noisereduce` library - install separately)

This preprocessing helps reduce false speaker clusters caused by:
- Volume variations (one speaker at different distances from mic)
- Background noise (can be classified as separate "speaker")
- Audio quality inconsistencies

---

## Troubleshooting

### “FFmpeg not found”

**Cause:** FFmpeg is missing or not on PATH.

**Fix:**
```bash
# macOS
brew install ffmpeg

# Linux (Ubuntu/Debian)
sudo apt install ffmpeg

# Windows
# Download from ffmpeg.org and add to PATH
```

---

### “Out of memory”

**Cause:** insufficient RAM.

**Fixes:**
1. Close unused applications
2. Switch to `whisper_base`
3. Split long videos into chunks

```bash
ffmpeg -i long_video.mp3 -ss 00:00:00 -t 01:00:00 part1.mp3
```

---

### Models take forever to download

**Cause:** first run downloads large weights (2–3 GB).

**Fix:** expected behaviour — models are cached in `models/`:
- Whisper Base: ~150 MB
- NLLB: ~2.5 GB

Subsequent runs reuse the cache.

---

### “CUDA out of memory”

**Cause:** GPU memory is insufficient.

**Fix:** force CPU usage:
```bash
# .env
WHISPER_DEVICE=cpu
```

---

### YouTube download fails

**Possible reasons**
1. The video is private or geo-blocked
2. Network connectivity issues
3. YouTube changed the API (yt-dlp update needed)

**Fixes**
- Verify the URL in a browser
- Update yt-dlp (`pip install -U yt-dlp`)
- Try again later; the community patch cycle is quick

---

### Whisper errors on Apple Silicon

**Cause:** MPS backend lacks Sparse tensor ops.

**Fix:** the app falls back to CPU automatically. To enforce it:
```bash
WHISPER_DEVICE=cpu
```

---

### Ollama server is unreachable

**Fix:**
```bash
ollama serve
curl http://localhost:11434/api/tags
```

Ensure the desired model is pulled (`ollama pull qwen2.5:3b`).

---

### Warning: "torchcodec is not installed correctly" (Speaker Diarization)

**Message:**
```
UserWarning: torchcodec is not installed correctly so built-in audio decoding will fail.
Could not load libtorchcodec... FFmpeg is not properly installed...
We support versions 4, 5, 6 and 7.
```

**Cause:** FFmpeg 8.0 is installed, but pyannote's torchcodec expects FFmpeg 4-7.

**Is this critical?** ❌ **No, this is safe to ignore.**

The speaker diarization system has built-in fallback audio loaders:
1. First tries `soundfile` (doesn't need FFmpeg)
2. Falls back to `librosa` if needed
3. Only uses direct file loading as last resort

**What happens:**
- ✅ Speaker diarization works correctly
- ✅ Audio is loaded via soundfile/librosa
- ⚠️ Warning appears but can be ignored

**If you want to suppress the warning:**

Option 1: Keep FFmpeg 8 (recommended, everything works)
```bash
# Do nothing - the fallback works perfectly
```

Option 2: Downgrade to FFmpeg 7 (optional, only to remove warning)
```bash
# macOS
brew uninstall ffmpeg
brew install ffmpeg@7
brew link ffmpeg@7
```

**Note:** Downgrading FFmpeg is unnecessary since the fallback mechanism works reliably.

---

### Warning: "std(): degrees of freedom is <= 0" (Speaker Diarization)

**Message:**
```
UserWarning: std(): degrees of freedom is <= 0. Correction should be strictly less than...
```

**Cause:** Internal pyannote.audio calculation during speaker diarization.

**Is this critical?** ❌ **No, this is safe to ignore.**

This warning appears during normal operation of the speaker diarization pipeline and does not affect:
- ✅ Accuracy of speaker detection
- ✅ Quality of diarization results
- ✅ Stability of the process

**What to do:** Nothing - the process will complete successfully and identify speakers correctly.

---

### Where are logs and outputs saved?

- Transcripts & translations: `output/`
- Temporary audio/files: `temp/`
- Logs: `logs/`

---

### Can I use the tool for commercial projects?

The CLI itself is under MIT, but the default NLLB model is CC BY-NC 4.0 (non-commercial). For commercial use either:
- Switch to a commercially licensed translation model, or
- Obtain permission from Meta/third-party providers.

---

Still stuck? Open an issue on GitHub — happy to help.
