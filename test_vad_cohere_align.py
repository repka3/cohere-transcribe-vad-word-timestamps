from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import soundfile as sf

from cohere_transcript import CohereTranscript
from forced_align_utils import (
    AlignedSegment,
    ForcedAlignmentResources,
    WordAlignment,
    build_aligned_segment,
    load_forced_alignment_resources,
    unload_forced_alignment_resources,
)
from normalize_utils import convert_and_store_normalized_audio_from_file, filter_with_vad


AUDIO_FILE_PATH = Path("/mnt/SharedFolder/riunione_save_10min.mp3")
DEFAULT_LANGUAGE = "it"
DEVICE = "cuda:1"
VAD_THRESHOLD = 0.1


def format_timestamp(seconds: float) -> str:
    minutes, remainder = divmod(max(seconds, 0.0), 60.0)
    hours, minutes = divmod(int(minutes), 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{remainder:05.2f}"
    return f"{minutes:02d}:{remainder:05.2f}"


def print_aligned_words(words: list[WordAlignment]) -> None:
    if not words:
        print("<no aligned words>")
        return

    for word in words:
        print(
            f"  {format_timestamp(float(word['start']))} -> "
            f"{format_timestamp(float(word['end']))} | "
            f"{word['text']} | score={float(word['score']):.3f}"
        )


def main() -> int:
    source_path = AUDIO_FILE_PATH.expanduser().resolve()
    if not source_path.is_file():
        print(f"Audio file not found: {source_path}", file=sys.stderr)
        return 1

    normalized_path = convert_and_store_normalized_audio_from_file(str(source_path))
    if normalized_path is None:
        print(f"Failed to normalize audio file: {source_path}", file=sys.stderr)
        return 1

    try:
        segments = filter_with_vad(
            normalized_path,
            threshold=VAD_THRESHOLD,
        )
    except Exception as exc:
        print(f"VAD failed for normalized audio {normalized_path}: {exc}", file=sys.stderr)
        return 1

    print(f"Audio file: {source_path}")
    print(f"Normalized file: {normalized_path}")
    print(f"Language: {DEFAULT_LANGUAGE}")
    print(f"Device: {DEVICE}")
    print(f"VAD segments: {len(segments)}")

    if not segments:
        print("No speech segments detected.")
        return 0

    try:
        normalized_audio, sample_rate = sf.read(normalized_path, dtype="float32")
    except Exception as exc:
        print(f"Failed to read normalized WAV {normalized_path}: {exc}", file=sys.stderr)
        return 1

    transcript_client: CohereTranscript | None = None
    alignment_resources: ForcedAlignmentResources | None = None

    try:
        transcript_client = CohereTranscript(default_language=DEFAULT_LANGUAGE, device=DEVICE)
        alignment_resources = load_forced_alignment_resources(device=DEVICE)
    except Exception as exc:
        if transcript_client is not None:
            transcript_client.unload_model()
        print(f"Failed to initialize transcription/alignment: {exc}", file=sys.stderr)
        return 1
    assert transcript_client is not None
    assert alignment_resources is not None

    had_failures = False
    run_start = time.perf_counter()
    aligned_segments: list[AlignedSegment] = []

    try:
        with tempfile.TemporaryDirectory(prefix="vad_cohere_segments_") as temp_dir:
            temp_dir_path = Path(temp_dir)

            for index, segment in enumerate(segments, start=1):
                start_seconds = float(segment.get("start", 0.0))
                end_seconds = float(segment.get("end", start_seconds))
                start_frame = max(int(start_seconds * sample_rate), 0)
                end_frame = min(int(end_seconds * sample_rate), len(normalized_audio))

                if end_frame <= start_frame:
                    had_failures = True
                    print(
                        f"[{index:03d}] {format_timestamp(start_seconds)} -> "
                        f"{format_timestamp(end_seconds)} | invalid segment boundaries",
                        file=sys.stderr,
                    )
                    continue

                segment_audio = normalized_audio[start_frame:end_frame]
                segment_path = temp_dir_path / f"segment_{index:03d}.wav"

                try:
                    sf.write(segment_path, segment_audio, sample_rate, subtype="PCM_16")
                    transcript = transcript_client.transcribe_file(str(segment_path))
                    aligned_segment = build_aligned_segment(
                        audio_path=str(segment_path),
                        transcript=transcript,
                        segment_start_seconds=start_seconds,
                        segment_end_seconds=end_seconds,
                        language=DEFAULT_LANGUAGE,
                        resources=alignment_resources,
                    )
                except Exception as exc:
                    had_failures = True
                    print(
                        f"[{index:03d}] {format_timestamp(start_seconds)} -> "
                        f"{format_timestamp(end_seconds)} | transcription/alignment failed: {exc}",
                        file=sys.stderr,
                    )
                    continue

                aligned_segments.append(aligned_segment)
                duration = end_seconds - start_seconds
                print(
                    f"[{index:03d}] {format_timestamp(start_seconds)} -> "
                    f"{format_timestamp(end_seconds)} ({duration:.2f}s)"
                )
                print(aligned_segment["transcript"].strip() or "<empty transcript>")
                print_aligned_words(aligned_segment["words"])
                print()
    finally:
        if transcript_client is not None:
            transcript_client.unload_model()
        unload_forced_alignment_resources(alignment_resources)

    elapsed = time.perf_counter() - run_start
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Aligned segments: {len(aligned_segments)}")
    return 1 if had_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
