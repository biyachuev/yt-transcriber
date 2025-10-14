# YouTube Transcriber - Roadmap

## Recently Completed

### ✅ Speaker Diarization (v1.5.0)
**Status:** Implemented and committed (2025-10-13)

**What was done:**
- Implemented speaker identification using pyannote.audio
- Added `_perform_speaker_diarization()` method in Transcriber
- Integrated speaker labels with TranscriptionSegment
- Updated document_writer to format speaker labels in output
- Works with both local Whisper and OpenAI Whisper API
- Graceful fallback when HF_TOKEN not available
- Added comprehensive test suite (8 tests)

**Technical details:**
- Uses pyannote/speaker-diarization-3.1 model
- Assigns speakers based on maximum overlap with speech segments
- Seamlessly integrates with existing VAD infrastructure
- Speaker labels automatically included in DOCX and Markdown outputs
- Enable with `--speakers` CLI flag

**Benefits:**
- Better readability for multi-speaker content (interviews, podcasts)
- Professional quality output with clear speaker attribution
- Foundation for future speaker name mapping

---

### ✅ VAD-based Intelligent Audio Chunking (v1.4.0)
**Status:** Implemented and committed (2025-10-12)

**What was done:**
- Implemented Voice Activity Detection (VAD) using pyannote.audio
- Smart audio splitting at natural speech boundaries for OpenAI Whisper API
- Graceful fallback to time-based splitting without HF_TOKEN
- Eliminates word loss at chunk boundaries for files >25MB

**Technical details:**
- Uses pyannote/voice-activity-detection model
- Finds optimal split points at speech gaps (≥0.5s) within ±30s window
- Only activates for `whisper_openai_api` method with large files
- Foundation for future speaker diarization integration

---

## Planned Features

### 🎯 High Priority

#### 1. Optimized Chunk Processing for OpenAI API
**Target:** v1.5.1
**Dependencies:** None

**Description:**
Improve processing efficiency when handling chunked audio files.

**Implementation plan:**
- Add configurable overlap between chunks (default 5-10 seconds)
- Implement deduplication logic for overlapping segments
- Add progress bar showing chunk processing status
- Optimize prompt context passing between chunks
- Cache chunk metadata for retry scenarios

**Benefits:**
- Even better quality at chunk boundaries
- Faster retries on failures
- Better user experience with progress visibility

---

#### 2. Batch Processing Support
**Target:** v1.6.0
**Dependencies:** None

**Description:**
Process multiple videos in a single command with parallel execution.

**Implementation plan:**
- Add `--batch` flag accepting file with URLs
- Implement parallel processing with configurable workers
- Add batch progress tracking and reporting
- Handle failures gracefully (continue on error)
- Generate summary report for batch operations

**Use cases:**
- Processing entire playlists
- Bulk transcription of lecture series
- Automated content pipelines

---

### 🔄 Medium Priority

#### 4. Advanced Prompt Engineering
**Target:** v1.6.0

**Description:**
Improve transcription accuracy through better prompt generation.

**Ideas:**
- Analyze video comments/description for context
- Use first chunk transcription to improve subsequent chunks
- Genre-specific prompt templates
- Custom vocabulary injection from user-provided glossaries

---

#### 5. Real-time Streaming Support
**Target:** v2.0.0

**Description:**
Support live stream transcription with incremental processing.

**Challenges:**
- Handle continuous audio streams
- Implement sliding window approach
- Real-time output updates

---

#### 6. Web UI
**Target:** v2.0.0

**Description:**
Optional web interface for easier usage.

**Features:**
- Drag-and-drop video URLs
- Progress visualization
- Preview transcriptions before export
- Settings management

---

### 💡 Future Ideas (No timeline)

- **Multi-language video support**: Detect and handle videos with multiple languages
- **Subtitle timing optimization**: Improve subtitle timing for better readability
- **Integration with video editors**: Export formats compatible with Premiere/DaVinci Resolve
- **Cloud deployment**: Docker container for easy deployment
- **Mobile app**: iOS/Android companion app
- **Podcast-specific features**: Chapter detection, show notes generation
- **Academic features**: Citation generation, key concept extraction

---

## Technical Debt & Maintenance

### Code Quality
- [ ] Increase test coverage to 90%+ (current: ~75%)
- [ ] Add integration tests for VAD chunking
- [ ] Refactor main.py (too large, needs modularization)
- [ ] Add type hints to remaining functions

### Documentation
- [ ] Add troubleshooting guide for common errors
- [ ] Create video tutorials for setup
- [ ] Document internal API for contributors
- [ ] Add performance benchmarks

### Performance
- [ ] Profile and optimize memory usage for large files
- [ ] Implement caching for repeated operations
- [ ] Add GPU utilization metrics
- [ ] Optimize NLLB translation speed

---

## How to Contribute

If you want to work on any of these features:

1. Check if there's an open issue for it
2. Comment on the issue to avoid duplicate work
3. Fork the repo and create a feature branch
4. Follow the coding standards in CONTRIBUTING.md
5. Submit a PR with tests and documentation

---

## Version History

- **v1.5.0** (2025-10-13): Speaker diarization
- **v1.4.0** (2025-10-12): VAD-based intelligent chunking
- **v1.3.0** (2025-10-XX): OpenAI API integration (Whisper + GPT)
- **v1.2.0** (2025-XX-XX): NLLB translation support
- **v1.1.0** (2025-XX-XX): Document output formats
- **v1.0.0** (2025-XX-XX): Initial release

---

**Last updated:** 2025-10-13
**Maintainer:** @biyachuev
