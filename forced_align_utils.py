from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import torch
from ctc_forced_aligner import (
    forced_align,
    generate_emissions,
    get_spans,
    load_alignment_model,
    load_audio,
    merge_repeats,
    postprocess_results,
    preprocess_text,
)


DEFAULT_ALIGNMENT_MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"
LANGUAGE_ALIASES = {
    "de": "deu",
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "it": "ita",
    "pt": "por",
}


@dataclass
class ForcedAlignmentResources:
    model: torch.nn.Module
    tokenizer: object
    device: str
    compute_dtype: torch.dtype
    alignment_model: str
    romanize: bool
    split_size: str
    star_frequency: str
    window_size: int
    context_size: int
    batch_size: int


class WordAlignment(TypedDict):
    start: float
    end: float
    text: str
    score: float


class AlignedSegment(TypedDict):
    start: float
    end: float
    transcript: str
    words: list[WordAlignment]


def load_forced_alignment_resources(
    device: str,
    alignment_model: str = DEFAULT_ALIGNMENT_MODEL,
    compute_dtype: torch.dtype | None = None,
    romanize: bool = True,
    split_size: str = "word",
    star_frequency: str = "edges",
    window_size: int = 30,
    context_size: int = 2,
    batch_size: int = 4,
) -> ForcedAlignmentResources:
    resolved_dtype = compute_dtype or _default_compute_dtype(device)
    model, tokenizer = load_alignment_model(
        device=device,
        model_path=alignment_model,
        dtype=resolved_dtype,
    )
    return ForcedAlignmentResources(
        model=model,
        tokenizer=tokenizer,
        device=device,
        compute_dtype=resolved_dtype,
        alignment_model=alignment_model,
        romanize=romanize,
        split_size=split_size,
        star_frequency=star_frequency,
        window_size=window_size,
        context_size=context_size,
        batch_size=batch_size,
    )


def unload_forced_alignment_resources(resources: ForcedAlignmentResources | None) -> None:
    if resources is None:
        return

    if getattr(resources, "model", None) is not None:
        del resources.model
        resources.model = None

    if getattr(resources, "tokenizer", None) is not None:
        del resources.tokenizer
        resources.tokenizer = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def align_segment_words(
    audio_path: str,
    transcript: str,
    segment_start_seconds: float,
    segment_end_seconds: float,
    language: str,
    resources: ForcedAlignmentResources,
    merge_threshold: float = 0.0,
) -> list[WordAlignment]:
    stripped_transcript = transcript.strip()
    if not stripped_transcript:
        return []

    resolved_language = resolve_alignment_language(language)
    tokens_starred, text_starred = preprocess_text(
        stripped_transcript,
        resources.romanize,
        resolved_language,
        split_size=resources.split_size,
        star_frequency=resources.star_frequency,
    )
    if not _has_alignable_text(text_starred):
        return []

    audio_waveform = load_audio(audio_path, resources.compute_dtype, resources.device)
    emissions, stride = generate_emissions(
        resources.model,
        audio_waveform,
        window_length=resources.window_size,
        context_length=resources.context_size,
        batch_size=resources.batch_size,
    )

    segments, scores, blank = _get_alignments_safe(
        emissions,
        tokens_starred,
        resources.tokenizer,
    )
    spans = get_spans(tokens_starred, segments, blank)
    results = postprocess_results(
        text_starred,
        spans,
        stride,
        scores,
        merge_threshold=merge_threshold,
    )

    del emissions
    del audio_waveform

    words: list[WordAlignment] = []
    last_end = max(float(segment_start_seconds), 0.0)

    for result in results:
        absolute_start = _clamp(
            float(segment_start_seconds) + float(result["start"]),
            lower=float(segment_start_seconds),
            upper=float(segment_end_seconds),
        )
        absolute_start = max(absolute_start, last_end)

        absolute_end = _clamp(
            float(segment_start_seconds) + float(result["end"]),
            lower=absolute_start,
            upper=float(segment_end_seconds),
        )

        words.append(
            {
                "start": absolute_start,
                "end": absolute_end,
                "text": str(result["text"]),
                "score": float(result["score"]),
            }
        )
        last_end = absolute_end

    return words


def build_aligned_segment(
    audio_path: str,
    transcript: str,
    segment_start_seconds: float,
    segment_end_seconds: float,
    language: str,
    resources: ForcedAlignmentResources,
    merge_threshold: float = 0.0,
) -> AlignedSegment:
    words = align_segment_words(
        audio_path=audio_path,
        transcript=transcript,
        segment_start_seconds=segment_start_seconds,
        segment_end_seconds=segment_end_seconds,
        language=language,
        resources=resources,
        merge_threshold=merge_threshold,
    )
    return {
        "start": float(segment_start_seconds),
        "end": float(segment_end_seconds),
        "transcript": transcript,
        "words": words,
    }


def resolve_alignment_language(language: str) -> str:
    normalized = language.strip().lower()
    if len(normalized) == 3:
        return normalized
    if normalized in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[normalized]
    raise ValueError(
        f"Unsupported alignment language '{language}'. Use an ISO 639-3 code or add an alias."
    )


def _default_compute_dtype(device: str) -> torch.dtype:
    return torch.float16 if device.startswith("cuda") else torch.float32


def _has_alignable_text(text_starred: list[str]) -> bool:
    return any(chunk.strip() for chunk in text_starred if chunk != "<star>")


def _get_alignments_safe(
    emissions: torch.Tensor,
    tokens: list[str],
    tokenizer: object,
) -> tuple[list[object], np.ndarray, str]:
    if not tokens:
        raise ValueError("Empty transcript.")

    dictionary = tokenizer.get_vocab()
    dictionary = {key.lower(): value for key, value in dictionary.items()}
    star_index = emissions.size(-1) - 1
    dictionary["<star>"] = star_index

    token_indices: list[int] = []
    for token in " ".join(tokens).split(" "):
        if token not in dictionary:
            continue

        token_index = int(dictionary[token])
        if token != "<star>" and token_index >= star_index:
            continue
        token_indices.append(token_index)

    if not token_indices:
        raise ValueError("No valid alignment tokens remained after normalization.")

    blank_id = int(dictionary.get("<blank>", tokenizer.pad_token_id))

    emissions_cpu = emissions.cpu() if not emissions.is_cpu else emissions
    targets = np.asarray([token_indices], dtype=np.int64)
    path, scores = forced_align(
        emissions_cpu.unsqueeze(0).float().numpy(),
        targets,
        blank=blank_id,
    )
    path = path.squeeze().tolist()

    idx_to_token_map = {value: key for key, value in dictionary.items()}
    idx_to_token_map[star_index] = "<star>"
    segments = merge_repeats(path, idx_to_token_map)
    return segments, scores, idx_to_token_map[blank_id]


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
