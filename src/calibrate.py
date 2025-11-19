"""
Calibration utility for estimating processing time of different Whisper models.
Run: python calibrate.py [whisper_base|whisper_small]
"""

import time
from pathlib import Path
import subprocess
import sys

# Make the project root importable.
script_dir = Path(__file__).parent
project_root = script_dir if (script_dir / "src").exists() else script_dir.parent
sys.path.insert(0, str(project_root))

from src.transcriber import Transcriber
from src.translator import Translator
from src.utils import detect_language
from src.config import settings, TranscribeOptions

# ============================================================================
# Settings
# ============================================================================

# Resolve model from CLI args.
if len(sys.argv) > 1:
    MODEL = sys.argv[1]
    if MODEL not in ["whisper_base", "whisper_small"]:
        print(f"❌ Unknown model: {MODEL}")
        print("💡 Use: python calibrate.py [whisper_base|whisper_small]")
        sys.exit(1)
else:
    MODEL = "whisper_base"

print("=" * 70)
print("⏱️  CALIBRATION RUN")
print("=" * 70)
print(f"🎯 Model: {MODEL}")
print()

# ============================================================================
# Locate test file
# ============================================================================

temp_dir = project_root / "temp"

if not temp_dir.exists():
    print(f"❌ temp/ directory not found: {temp_dir}")
    print("\n💡 Create a test file, for example:")
    print("   ffmpeg -i temp/your_audio.mp3 -t 30 temp/test.mp3")
    sys.exit(1)

# Search for test*.mp3 first, fallback to any mp3.
test_files = list(temp_dir.glob("test*.mp3")) or list(temp_dir.glob("*.mp3"))

if not test_files:
    print(f"❌ No mp3 files found in {temp_dir}")
    print("\n💡 Download a video or create a sample file:")
    print("   python -m src.main --url 'YOUTUBE_URL' --transcribe whisper_base")
    sys.exit(1)

audio_path = test_files[0]
print(f"✅ Found file: {audio_path.name}")
print(f"📁 Path: {audio_path}")

# ============================================================================
# Determine duration
# ============================================================================

try:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    duration = float(result.stdout.strip())
except subprocess.CalledProcessError:
    print("❌ Failed to determine audio duration")
    print("💡 Ensure ffmpeg is installed (e.g. brew install ffmpeg)")
    sys.exit(1)
except ValueError:
    print("❌ Could not parse file duration")
    sys.exit(1)

print(f"📏 Audio length: {duration:.1f} seconds ({duration/60:.1f} minutes)")
print()

# ============================================================================
# Transcription benchmark
# ============================================================================

print("=" * 70)
print(f"📝 TRANSCRIPTION TEST ({MODEL})")
print("=" * 70)
print()

print(f"Loading {MODEL} model...")
transcriber = Transcriber(method=MODEL)
print(f"Device: {transcriber.device}")

model_info = {
    "whisper_base": {
        "params": "74M parameters",
        "size": "~150 MB",
        "quality": "good",
        "speed": "fast",
    },
    "whisper_small": {
        "params": "244M parameters",
        "size": "~500 MB",
        "quality": "very good",
        "speed": "slower",
    },
}

info = model_info.get(MODEL, {})
if info:
    print(f"   Parameters: {info['params']}")
    print(f"   Size: {info['size']}")
    print(f"   Quality: {info['quality']}")
    print(f"   Speed: {info['speed']}")
print()

print("🎵 Starting transcription...")
print("   (First run may take longer due to model download)")
start_time = time.time()

try:
    segments = transcriber.transcribe(audio_path, language=None)
    transcribe_time = time.time() - start_time
except Exception as e:
    print(f"❌ Transcription failed: {e}")
    sys.exit(1)

transcribe_ratio = transcribe_time / duration

print()
print("✅ Transcription finished!")
print(f"   Processing time: {transcribe_time:.1f} seconds")
print(f"   Segments created: {len(segments)}")
print(f"   📊 Multiplier: {transcribe_ratio:.2f}x")
print()

if transcribe_ratio < 0.5:
    print("   🚀 Very fast (less than half real-time)")
elif transcribe_ratio < 1.0:
    print("   ✨ Fast (faster than real-time)")
elif transcribe_ratio < 1.5:
    print("   👍 Acceptable (around real-time)")
elif transcribe_ratio < 2.0:
    print("   🐢 Slowish (1.5–2× real-time)")
else:
    print("   🐌 Slow (more than 2× real-time)")

# ============================================================================
# Translation benchmark
# ============================================================================

print()
print("=" * 70)
print("🌍 TRANSLATION TEST")
print("=" * 70)
print()

original_text = transcriber.segments_to_text(segments)
detected_lang = detect_language(original_text)

print(f"Detected language: {detected_lang}")
print()

if detected_lang == "ru":
    print("⚠️  Text is in Russian, translation skipped")
    translate_ratio = 0.0
else:
    print("Loading NLLB model...")
    translator = Translator(method="NLLB")
    print()

    print("🔄 Starting translation...")
    start_time = time.time()

    try:
        translation_segments = translator.translate_segments(
            segments,
            source_lang=detected_lang,
            target_lang="ru",
        )
        translate_time = time.time() - start_time
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        translate_time = 0.0
        translate_ratio = 0.0
    else:
        translate_ratio = translate_time / duration

        print()
        print("✅ Translation finished!")
        print(f"   Processing time: {translate_time:.1f} seconds")
        print(f"   Segments translated: {len(translation_segments)}")
        print(f"   📊 Multiplier: {translate_ratio:.2f}x")
        print()

        if translate_ratio < 0.75:
            print("   🚀 Very fast translation")
        elif translate_ratio < 1.5:
            print("   ✨ Fast enough")
        elif translate_ratio < 2.0:
            print("   👍 Acceptable speed")
        else:
            print("   🐢 Consider using a smaller NLLB model")

# ============================================================================
# Comparison table
# ============================================================================

print()
print("=" * 70)
print("📊 MODEL COMPARISON")
print("=" * 70)
print()

comparison_table = [
    {"model": "whisper_base", "quality": "good", "speed": "fast", "multiplier": 0.06},
    {
        "model": "whisper_small",
        "quality": "very good",
        "speed": "medium",
        "multiplier": 0.19,
    },
    {
        "model": "whisper_medium",
        "quality": "excellent",
        "speed": "slow",
        "multiplier": 0.45,
    },
]

print("Model           | Quality     | Speed      | Multiplier")
print("----------------+-------------+------------+------------")
for row in comparison_table:
    current = "← current" if row["model"] == MODEL else ""
    print(
        f"{row['model']:<15} | {row['quality']:<11} | {row['speed']:<10} | "
        f"{row['multiplier']:<10.2f} {current}"
    )

print()
print("💡 Recommendations:")
print("   • whisper_base  — great for fast batch processing")
print("   • whisper_small — better quality (+5% accuracy)")
print("   • whisper_medium — use when accuracy matters most")

# ============================================================================
# Summary
# ============================================================================

print()
print("=" * 70)
print("✅ CALIBRATION SUMMARY")
print("=" * 70)
print()

print(f"Model: {MODEL}")
print(f"Device: {transcriber.device}")
print(f"Audio length: {duration:.1f} sec")
print()
print("Processing multipliers:")
print(f"  • Transcription ({MODEL}): {transcribe_ratio:.2f}x")
if detected_lang != "ru":
    print(f"  • Translation: {translate_ratio:.2f}x")

print()
print("🔧 Update suggestion:")
print("  • Consider adjusting estimate_processing_time in src/utils.py ")
print("    with the measured multipliers above.")

# ============================================================================
# Example estimates
# ============================================================================

print()
print(f"📈 SAMPLE ESTIMATES ({MODEL})")
print()


def humanize(seconds: float) -> str:
    if seconds < 60:
        return f"~{int(seconds)} sec"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 10:
        return f"~{minutes}m {secs:02d}s"
    return f"~{minutes} min"


examples = [
    ("10 minutes", 600),
    ("30 minutes", 1800),
    ("60 minutes", 3600),
    ("120 minutes", 7200),
]

for label, secs in examples:
    trans_time = transcribe_ratio * secs
    total_time = trans_time
    if detected_lang != "ru":
        total_time += translate_ratio * secs

    print(
        f"{label:>10} → transcription: {humanize(trans_time)}, "
        f"total: {humanize(total_time)}"
    )

print()
print("✅ Calibration complete!")
