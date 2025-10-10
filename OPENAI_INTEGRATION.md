# OpenAI API Integration

## Overview

Version 1.3.0 adds full OpenAI API integration, allowing you to use GPT models for transcription, translation, text refinement, and summarization.

## Features Added

### 1. OpenAI Whisper API for Transcription
- Uses OpenAI's hosted Whisper model
- Faster than local processing (no model download required)
- Supports audio files up to 25MB
- Returns timestamped segments

### 2. GPT for Translation
- Uses GPT-4 or GPT-3.5-turbo for translation
- Better at preserving context and nuance than NLLB
- Handles technical terms and idioms more naturally
- Maintains formatting and timestamps

### 3. GPT for Text Refinement
- Alternative to Ollama for transcript cleanup
- Removes filler words and improves punctuation
- Available for both original transcripts and translations
- Uses GPT-4 or GPT-3.5-turbo

### 4. GPT for Summarization (New Module!)
- Generate detailed summaries of transcripts
- Structured output with section headings
- Supports both Russian and English
- Can process long documents by chunking

## Setup

1. **Get your OpenAI API key:**
   - Visit https://platform.openai.com/api-keys
   - Create a new API key

2. **Add to your `.env` file:**
   ```bash
   OPENAI_API_KEY=sk-your-api-key-here
   ```

3. **Install the OpenAI library (if not already installed):**
   ```bash
   pip install openai>=1.6.0
   ```

## Usage Examples

### Transcription with OpenAI Whisper

```bash
# Basic transcription
python -m src.main \
  --url "https://youtube.com/watch?v=..." \
  --transcribe whisper_openai_api

# With translation
python -m src.main \
  --input_audio audio.mp3 \
  --transcribe whisper_openai_api \
  --translate openai_api
```

**Note:** OpenAI Whisper API has a 25MB file size limit. For larger files, use local Whisper models.

### Translation with GPT

```bash
# Translate using GPT-4
python -m src.main \
  --input_audio audio.mp3 \
  --transcribe whisper_base \
  --translate openai_api

# Combine with local Whisper and GPT translation
python -m src.main \
  --url "https://youtube.com/watch?v=..." \
  --transcribe whisper_medium \
  --translate openai_api
```

### Text Refinement with GPT

```bash
# Refine transcript with GPT-4
python -m src.main \
  --input_audio audio.mp3 \
  --transcribe whisper_base \
  --refine-model gpt-4 \
  --refine-backend openai_api

# Use GPT-3.5-turbo (faster and cheaper)
python -m src.main \
  --input_audio audio.mp3 \
  --transcribe whisper_base \
  --refine-model gpt-3.5-turbo \
  --refine-backend openai_api
```

### Summarization with GPT

```bash
# Generate summary with GPT-4
python -m src.main \
  --input_audio audio.mp3 \
  --transcribe whisper_base \
  --summarize \
  --summarize-model gpt-4 \
  --summarize-backend openai_api

# Full pipeline: transcribe, translate, refine, and summarize
python -m src.main \
  --url "https://youtube.com/watch?v=..." \
  --transcribe whisper_openai_api \
  --translate openai_api \
  --refine-model gpt-4 \
  --refine-backend openai_api \
  --summarize \
  --summarize-model gpt-4 \
  --summarize-backend openai_api
```

### Summarize Existing Document

```bash
# Summarize a text document
python -m src.main \
  --input_text document.docx \
  --summarize \
  --summarize-model gpt-4 \
  --summarize-backend openai_api
```

## Comparison: OpenAI vs Local

| Feature | OpenAI API | Local (Ollama/Whisper/NLLB) |
|---------|------------|------------------------------|
| **Setup** | Just API key | Install models (~5-10GB) |
| **Cost** | Pay per use | Free (compute cost) |
| **Privacy** | Data sent to OpenAI | All local, private |
| **Speed** | Fast (cloud processing) | Depends on hardware |
| **Quality** | Excellent (GPT-4) | Very good (comparable) |
| **Internet** | Required | Not required |
| **File size limit** | 25MB (Whisper) | No limit |

## Cost Estimates (OpenAI)

Approximate costs for a 1-hour video:

| Operation | Model | Est. Tokens | Est. Cost |
|-----------|-------|-------------|-----------|
| Transcription | Whisper | N/A | $0.36 |
| Translation | GPT-4 | ~20k | $0.40 |
| Translation | GPT-3.5 | ~20k | $0.04 |
| Refinement | GPT-4 | ~15k | $0.30 |
| Refinement | GPT-3.5 | ~15k | $0.03 |
| Summarization | GPT-4 | ~15k | $0.30 |
| Summarization | GPT-3.5 | ~15k | $0.03 |

**Full pipeline (GPT-4):** ~$1.36 per hour
**Full pipeline (GPT-3.5):** ~$0.46 per hour

*Prices as of October 2024. Check OpenAI pricing page for current rates.*

## Mixing Backends

You can mix and match backends for optimal cost/quality:

```bash
# Use local Whisper (free) + GPT-4 for translation only
python -m src.main \
  --input_audio audio.mp3 \
  --transcribe whisper_medium \
  --translate openai_api

# Use OpenAI Whisper + local NLLB (save on translation cost)
python -m src.main \
  --input_audio audio.mp3 \
  --transcribe whisper_openai_api \
  --translate NLLB

# Use local processing + GPT-4 for summary only
python -m src.main \
  --input_audio audio.mp3 \
  --transcribe whisper_medium \
  --translate NLLB \
  --summarize \
  --summarize-model gpt-4 \
  --summarize-backend openai_api
```

## Model Selection

### For Transcription
- `whisper_openai_api` - OpenAI's hosted Whisper (fast, accurate)

### For Translation
- `openai_api` - Uses GPT-4 by default (configurable in code)

### For Refinement & Summarization
- `gpt-4` - Best quality, slower, more expensive
- `gpt-4-turbo` - Fast, good quality, mid-price
- `gpt-3.5-turbo` - Fastest, cheapest, good quality

## Configuration

All backends can be configured in code:

**Translator (translator.py:238)**
```python
model="gpt-4",  # Change to "gpt-3.5-turbo" or "gpt-4-turbo"
```

**Text Refiner (text_refiner.py:225)**
```python
model=self.model_name,  # Passed via --refine-model
```

**Summarizer (summarizer.py)**
```python
model=self.model_name,  # Passed via --summarize-model
```

## Troubleshooting

### Error: OPENAI_API_KEY not found
- Make sure you added `OPENAI_API_KEY=...` to your `.env` file
- The `.env` file should be in the project root directory

### Error: Audio file too large (OpenAI Whisper)
- OpenAI Whisper API has a 25MB limit
- Use local Whisper models instead: `--transcribe whisper_medium`
- Or compress your audio file first

### Error: Rate limit exceeded
- You've hit OpenAI's rate limits
- Wait a few seconds and retry
- Consider upgrading your OpenAI plan

### Error: OpenAI library not installed
```bash
pip install openai>=1.6.0
```

## Architecture

New modules added:
- `src/summarizer.py` - New summarization module (supports Ollama + OpenAI)
- Updated `src/transcriber.py` - Added `_transcribe_with_openai_api()` method
- Updated `src/translator.py` - Implemented `_translate_with_openai()` method
- Updated `src/text_refiner.py` - Added OpenAI backend support
- Updated `src/config.py` - Added `RefineOptions` and `SummarizeOptions` enums

## What's Next

See [ROADMAP.md](PROJECT_SUMMARY.md#L162-L185) for planned features:
- v1.3: ✅ OpenAI API integration (completed!)
- v1.4: Text file ingestion improvements
- v2.0: Speaker diarisation, larger models, more languages
