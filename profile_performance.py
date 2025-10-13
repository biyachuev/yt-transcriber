"""
Performance profiling script for youtube-transcriber

Usage:
    python profile_performance.py --test transcribe
    python profile_performance.py --test translate
    python profile_performance.py --test all
"""
import cProfile
import pstats
import io
import sys
import argparse
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def profile_transcription():
    """Profile transcription performance"""
    print("🔍 Profiling transcription...")
    from src.transcriber import Transcriber, TranscriptionSegment
    from src.config import TranscribeOptions

    # Create mock segments
    segments = [
        TranscriptionSegment(i * 5, (i + 1) * 5, f"Segment {i} text content")
        for i in range(1000)  # 1000 segments
    ]

    transcriber = Transcriber(method=TranscribeOptions.WHISPER_BASE)

    # Profile segments_to_text
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(10):  # Repeat 10 times
        text = transcriber.segments_to_text(segments)

    profiler.disable()

    # Print results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())


def profile_translation():
    """Profile translation performance"""
    print("🔍 Profiling translation...")
    from src.translator import Translator
    from src.config import TranslateOptions

    translator = Translator(method=TranslateOptions.NLLB)

    # Mock text
    text = "This is a test sentence. " * 100

    # Profile chunking
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(100):
        from src.utils import chunk_text
        chunks = chunk_text(text)

    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())


def profile_file_operations():
    """Profile file I/O operations"""
    print("🔍 Profiling file operations...")
    from src.text_reader import TextReader
    from src.document_writer import DocumentWriter
    import tempfile

    reader = TextReader()
    writer = DocumentWriter()

    profiler = cProfile.Profile()
    profiler.enable()

    # Profile text operations
    for _ in range(100):
        text = "# Heading\n\nParagraph text. " * 10
        stripped = reader._strip_markdown(text)

    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())


def profile_utils():
    """Profile utility functions"""
    print("🔍 Profiling utils...")
    from src.utils import sanitize_filename, detect_language, format_timestamp

    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(10000):
        # Profile sanitize_filename
        filename = f"Test: File? Name* {i}!@#$%.txt"
        clean = sanitize_filename(filename)

        # Profile detect_language
        text = "This is English text " * 10 + "Это русский текст " * 10
        lang = detect_language(text)

        # Profile format_timestamp
        timestamp = format_timestamp(i * 1.5)

    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())


def main():
    parser = argparse.ArgumentParser(description='Profile youtube-transcriber performance')
    parser.add_argument(
        '--test',
        choices=['transcribe', 'translate', 'files', 'utils', 'all'],
        default='all',
        help='Which component to profile'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🚀 YOUTUBE TRANSCRIBER PERFORMANCE PROFILING")
    print("=" * 70)
    print()

    if args.test in ['transcribe', 'all']:
        profile_transcription()
        print()

    if args.test in ['translate', 'all']:
        profile_translation()
        print()

    if args.test in ['files', 'all']:
        profile_file_operations()
        print()

    if args.test in ['utils', 'all']:
        profile_utils()
        print()

    print("=" * 70)
    print("✅ Profiling complete!")
    print("=" * 70)
    print()
    print("💡 Tips for optimization:")
    print("  - Look for functions with high cumulative time")
    print("  - Check for unnecessary repeated operations")
    print("  - Consider caching expensive operations")
    print("  - Use generators for large datasets")


if __name__ == '__main__':
    main()
