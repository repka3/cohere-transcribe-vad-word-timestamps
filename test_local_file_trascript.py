from __future__ import annotations

import sys
import time
from pathlib import Path

from cohere_transcript import CohereTranscript


AUDIO_FILE_PATH = Path("/mnt/SharedFolder/riunione_save_10min.mp3")
DEFAULT_LANGUAGE = "it"


def main() -> int:
    transcript_client = CohereTranscript(default_language=DEFAULT_LANGUAGE,device='cuda:1')

    try:
        start = time.perf_counter()
        transcript = transcript_client.transcribe_file(str(AUDIO_FILE_PATH))
        elapsed = time.perf_counter() - start

        print(f"Audio file: {AUDIO_FILE_PATH}")
        print(f"Language: {DEFAULT_LANGUAGE}")
        print(f"Elapsed: {elapsed:.2f}s")
        print("Transcript:")
        print(transcript)
    except Exception as exc:
        print(f"Transcription failed: {exc}", file=sys.stderr)
        return 1
    finally:
        transcript_client.unload_model()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
