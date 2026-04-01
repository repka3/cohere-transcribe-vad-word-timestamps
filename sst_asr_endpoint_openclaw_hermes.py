from __future__ import annotations

import base64
import io

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from cohere_transcript import CohereTranscript

app = FastAPI()

TARGET_SR = 16000


class TranscribeRequest(BaseModel):
    audio_base64: str
    language: str = "en"


class TranscribeResponse(BaseModel):
    text: str


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest):
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data")

    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode audio (unsupported format?)")

    # Convert to mono if multi-channel
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample to 16 kHz if needed
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    client = CohereTranscript(default_language=req.language)
    try:
        text = client.transcribe_array(audio, language=req.language)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    finally:
        client.unload_model()

    return TranscribeResponse(text=text)
