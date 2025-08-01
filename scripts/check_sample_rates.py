import subprocess
from pathlib import Path

AUDIO_DIR = Path("data/raw/processed_classes_filtered/birds/")
EXPECTED_SR = 48000
EXPECTED_CHANNELS = 1
EXPECTED_CODEC = "pcm_s16le"

bad_files = []

def get_audio_info(file_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip().split("\n")

for wav_file in AUDIO_DIR.rglob("*.wav"):
    try:
        codec, sr, channels = get_audio_info(wav_file)
        sr = int(sr)
        channels = int(channels)

        if codec != EXPECTED_CODEC or sr != EXPECTED_SR or channels != EXPECTED_CHANNELS:
            bad_files.append((wav_file, codec, sr, channels))

    except Exception as e:
        print(f"[ERROR] Could not process {wav_file.name}: {e}")

if not bad_files:
    print("✅ All files match expected format:")
    print(f"   Codec: {EXPECTED_CODEC}, Sample Rate: {EXPECTED_SR} Hz, Channels: {EXPECTED_CHANNELS}")
else:
    print("❌ Found files with incorrect format:")
    for file, codec, sr, ch in bad_files:
        print(f"  - {file}: codec={codec}, sr={sr}, channels={ch}")

