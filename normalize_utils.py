from __future__ import annotations

import subprocess
from pathlib import Path
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps


CACHE_DIR = Path(__file__).resolve().parent / "cache"


# This function gets an absolute path of a media file, checks if it exists,
# ensures a cache directory exists, derives an output filename from the source,
# converts
# the media with ffmpeg into a 16 kHz mono WAV, writes the normalized file to
# disk, and returns the absolute output path. On error it returns None.
def convert_and_store_normalized_audio_from_file(absolute_path: str) -> str | None:
    source_path = Path(absolute_path).expanduser()
    if not source_path.is_absolute():
        return None

    source_path = source_path.resolve()
    if not source_path.is_file():
        return None

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    output_path = CACHE_DIR / f"{source_path.stem}_normalized.wav"
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return None

    if completed.returncode != 0 or not output_path.is_file():
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None

    return str(output_path.resolve())



#this function receive a normalized audio and run silero vad, returning segments with timestamps.

def filter_with_vad(absolute_path: str,min_silence_duration_ms:int=100,threshold:float=0.1):
    model = load_silero_vad()
    wav = read_audio(absolute_path)
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        min_speech_duration_ms = 250,
        max_speech_duration_s = float('inf'),
        min_silence_duration_ms = min_silence_duration_ms,
        threshold=threshold,
        return_seconds=True,  # Return speech timestamps in seconds (default is samples)
    )
    return speech_timestamps