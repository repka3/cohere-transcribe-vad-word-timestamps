# Cohere Transcribe + VAD + Word-Level Timestamps

Model: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026

This repository is a proof of concept that combines:

- Cohere Transcribe for speech-to-text
- Silero VAD for speech segmentation
- `ctc-forced-aligner` for word-level timestamps

The goal is simple: start from a local audio or media file, split it into speech regions with VAD, transcribe each region with Cohere, and then recover word-level timestamps aligned back to the original full-audio timeline.

This is not a production-ready package. It is a small runnable demo intended to validate the pipeline and serve as a starting point for later work such as speaker matching and diarization.

## How it works

The current pipeline is:

1. Normalize the input file to `16 kHz`, mono WAV with `ffmpeg`.
2. Run Silero VAD to detect speech segments.
3. Cut the normalized audio into one WAV file per VAD segment.
4. Transcribe each segment with `CohereLabs/cohere-transcribe-03-2026`.
5. Force-align each segment transcript at word level with `ctc-forced-aligner`.
6. Offset each aligned word back to absolute time in the original audio.

The result is segment-level transcription plus word-level timestamps for each segment.

## Repository layout

- [cohere_transcript.py](./cohere_transcript.py): small wrapper around the local Cohere Transcribe model.
- [normalize_utils.py](./normalize_utils.py): audio normalization and Silero VAD utilities.
- [forced_align_utils.py](./forced_align_utils.py): reusable forced-alignment helpers for word-level timestamps.
- [test_vad_cohere_align.py](./test_vad_cohere_align.py): current demo entrypoint for the end-to-end pipeline.
- [sst_asr_endpoint_openclaw_hermes.py](./sst_asr_endpoint_openclaw_hermes.py): FastAPI endpoint for local agentic voice assistants (see below).

## Requirements

- Python environment in the repo-local `.venv`
- Python `3.12`
- NVIDIA GPU setup using CUDA `12.8` wheels
- `ffmpeg` available in `PATH`
- Hugging Face access to the gated Cohere model
- Enough RAM / VRAM to load both the transcription model and the alignment model

This repo is currently set up and tested as a local script workflow, with GPU usage configured in the demo script.

## Installation

Create a fresh local virtual environment, then install the supported CUDA `12.8` dependency set:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-cu128.txt
```

The Cohere model is gated on Hugging Face. Before running the demo:

1. Accept the model terms on Hugging Face:
   `https://huggingface.co/CohereLabs/cohere-transcribe-03-2026`
2. Export your Hugging Face token in the same shell where you run the demo:

```bash
export HF_TOKEN=hf_...
```

`HF_TOKEN` is read from the current shell environment; the repository does not store it for you.

## Usage

The current demo script is:

```bash
.venv/bin/python test_vad_cohere_align.py
```

At the moment, the following values are hardcoded in `test_vad_cohere_align.py`:

- input audio path
- language
- device
- VAD threshold

So the expected workflow today is to edit those constants and then run the script.


## Output shape

Each aligned segment follows this structure:

```python
{
    "start": 0.40,
    "end": 124.60,
    "transcript": "Full segment transcript here",
    "words": [
        {"start": 0.78, "end": 0.88, "text": "si", "score": -0.671},
        {"start": 0.88, "end": 1.04, "text": "si", "score": -7.661},
    ],
}
```

Notes:

- `start` and `end` at segment level are absolute timestamps in the full audio.
- `words[].start` and `words[].end` are also absolute timestamps in the full audio.
- `words[].text` comes from the aligner-normalized text, not from punctuation-preserving postprocessing.

## Current limitations

- This is a proof of concept, not a finished library or service.
- The Cohere model is gated and requires Hugging Face access approval.
- The current demo is script-driven and configured via in-file constants, not a CLI.
- The forced-alignment step currently includes a local workaround for a token-index mismatch seen with the installed `ctc-forced-aligner` helper and the selected default alignment model.
- The repo currently focuses on transcription plus timing recovery only. Diarization is not implemented yet.

## Why word-level timestamps

The immediate purpose of the word-level timestamps is to support the next step: matching words to speakers once diarization is added. In other words, this repository is validating the transcription-plus-alignment part first.

## Credits and sources

- Cohere Transcribe model:
  [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- Forced aligner:
  [MahmoudAshraf97/ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner)
- Silero VAD:
  [snakers4/silero-vad](https://github.com/snakers4/silero-vad)

## Status

Current status: working local PoC for:

- VAD-based segmentation
- Cohere transcription per segment
- Word-level timestamp recovery per segment
- Absolute timestamp remapping to the original full audio
- FastAPI endpoint for local agentic voice assistants (OpenClaw, Hermes)

Next logical step: add diarization and use the aligned word timestamps for speaker attribution.

## Local voice assistant endpoint

This repository also provides a standalone FastAPI endpoint for using Cohere Transcribe as the speech-to-text backend in local agentic voice assistants such as OpenClaw and Hermes.

The endpoint receives base64-encoded audio and a language code, loads the model, transcribes the audio entirely in-memory (no temp files), and unloads the model after each request to free GPU memory for other agent tasks.

### Running the endpoint

```bash
.venv/bin/python -m uvicorn sst_asr_endpoint_openclaw_hermes:app --host 0.0.0.0 --port 8000
```

### Example request

```bash
AUDIO_B64=$(base64 -w0 /path/to/audio.wav)
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d "{\"audio_base64\": \"$AUDIO_B64\", \"language\": \"en\"}"
```

### Example response

```json
{"text": "Hello, this is the transcribed text."}
```

The endpoint accepts any audio format that `soundfile` can read (WAV, FLAC, OGG, etc.), automatically converts to mono 16 kHz, and supports all languages available in the Cohere Transcribe model.
