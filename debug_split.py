"""Debug script to test audio splitting."""
from pathlib import Path
from src.transcriber import Transcriber

# Create transcriber
transcriber = Transcriber(method="whisper_openai_api", use_cache=False)

# Path to the audio file
import os
temp_dir = '/Users/biyachuev/Documents/Python/yt-transcriber/temp/'
files = [f for f in os.listdir(temp_dir) if 'Andrej' in f]
audio_path = Path(temp_dir) / files[0]
print(f"Audio file: {audio_path}")

# Split the file
print("Splitting audio file...")
chunks = transcriber._split_audio_file(audio_path, max_size_mb=24)

print(f"\nTotal chunks created: {len(chunks)}")
for i, (chunk_path, start, end) in enumerate(chunks):
    exists = "EXISTS" if chunk_path.exists() else "MISSING"
    print(f"Chunk {i}: {chunk_path.name} ({start:.1f}-{end:.1f}s) - {exists}")
