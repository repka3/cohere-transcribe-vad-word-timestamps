from __future__ import annotations

import gc
from pathlib import Path

import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio


class CohereTranscript:
    """Small wrapper for local file-based transcription with Cohere Transcribe."""

    def __init__(
        self,
        model_id: str = "CohereLabs/cohere-transcribe-03-2026",
        default_language: str = "en",
        device: str | None = None,
        punctuation: bool = True,
        max_new_tokens: int = 256,
    ) -> None:
        self.model_id = model_id
        self.default_language = default_language
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.punctuation = punctuation
        self.max_new_tokens = max_new_tokens
        self.processor = None
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        if self._is_loaded:
            return

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = CohereAsrForConditionalGeneration.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            self.unload_model()
            raise RuntimeError(self._format_load_error(exc)) from exc

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    def transcribe_file(
        self,
        audio_path: str,
        language: str | None = None,
        punctuation: bool | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        self.load_model()

        path = Path(audio_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        resolved_language = language or self.default_language
        resolved_punctuation = self.punctuation if punctuation is None else punctuation
        resolved_max_new_tokens = (
            self.max_new_tokens if max_new_tokens is None else max_new_tokens
        )

        audio = load_audio(str(path), sampling_rate=16000)
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            language=resolved_language,
            punctuation=resolved_punctuation,
        )
        audio_chunk_index = inputs.get("audio_chunk_index")
        inputs = inputs.to(self._model_device, dtype=self.model.dtype)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=resolved_max_new_tokens)

        outputs = outputs.cpu()
        decode_kwargs = {"skip_special_tokens": True}
        if audio_chunk_index is not None:
            decode_kwargs["audio_chunk_index"] = audio_chunk_index
            decode_kwargs["language"] = resolved_language

        transcript = self.processor.decode(outputs, **decode_kwargs)
        if isinstance(transcript, list):
            if len(transcript) == 1:
                return transcript[0]
            return "\n".join(transcript)

        return transcript

    @property
    def _is_loaded(self) -> bool:
        return self.processor is not None and self.model is not None

    @property
    def _model_device(self) -> torch.device:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        return next(self.model.parameters()).device

    def _format_load_error(self, exc: Exception) -> str:
        message = str(exc)
        lowered = message.lower()
        if "gated" in lowered or "403" in lowered or "access to model" in lowered:
            return (
                "Failed to load Cohere Transcribe. This model is gated on Hugging Face, "
                "so make sure you have accepted the access terms and authenticated "
                "locally, for example with `hf auth login`."
            )
        return f"Failed to load Cohere Transcribe: {message}"

    def __del__(self) -> None:
        try:
            self.unload_model()
        except Exception:
            pass
