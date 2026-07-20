"""Transcription via faster-whisper, with word-level timestamps."""
from __future__ import annotations

import numpy as np

from .types import Word


def transcribe(
    audio: np.ndarray,
    sr: int,
    model_size: str = "base",
    language: str | None = None,
) -> list[Word]:
    """Transcribe mono 16 kHz audio to a list of timestamped Words."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Transcription requires faster-whisper: "
            "pip install 'audioparser[transcribe]' (or run with --no-transcript)"
        ) from exc

    if sr != 16_000:
        raise ValueError("faster-whisper expects 16 kHz audio")

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments, _info = model.transcribe(
        audio,
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )

    words: list[Word] = []
    for segment in segments:
        for w in segment.words or []:
            words.append(
                Word(
                    start=float(w.start),
                    end=float(w.end),
                    text=w.word.strip(),
                    probability=float(w.probability),
                )
            )
    return words
