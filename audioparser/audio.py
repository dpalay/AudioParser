"""Audio loading and per-speaker splitting.

Uses soundfile + numpy only. WAV/FLAC/OGG load natively; other formats
(mp3, m4a, ...) are converted through ffmpeg when it is on PATH.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .types import SpeakerTurn

TARGET_SR = 16_000

_NATIVE_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 at ``target_sr``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() not in _NATIVE_SUFFIXES:
        return _load_via_ffmpeg(path, target_sr), target_sr

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != target_sr:
        mono = _resample(mono, sr, target_sr)
    return mono, target_sr


def _load_via_ffmpeg(path: Path, target_sr: int) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"Cannot read {path.suffix} without ffmpeg. Install ffmpeg or "
            "convert the file to WAV/FLAC first."
        )
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(path),
                "-ac", "1", "-ar", str(target_sr),
                "-loglevel", "error", tmp.name,
            ],
            check=True,
        )
        data, _ = sf.read(tmp.name, dtype="float32", always_2d=True)
    return data.mean(axis=1)


def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Linear-interpolation resample; adequate for speech pipelines."""
    if sr == target_sr:
        return audio
    n_out = int(round(len(audio) * target_sr / sr))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def split_by_speaker(
    audio: np.ndarray,
    sr: int,
    turns: list[SpeakerTurn],
    out_dir: str | Path,
    mode: str = "silence",
) -> dict[str, Path]:
    """Write one WAV per speaker.

    mode="silence": full-length track with everyone else muted, so
    timestamps still line up with the original recording.
    mode="concat": only that speaker's segments, back to back (shorter
    files, timing not preserved).
    """
    if mode not in ("silence", "concat"):
        raise ValueError(f"unknown split mode: {mode}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    speakers = sorted({t.speaker for t in turns})
    written: dict[str, Path] = {}
    for speaker in speakers:
        spans = [
            (max(0, int(t.start * sr)), min(len(audio), int(t.end * sr)))
            for t in turns
            if t.speaker == speaker
        ]
        if mode == "silence":
            track = np.zeros_like(audio)
            for lo, hi in spans:
                track[lo:hi] = audio[lo:hi]
        else:
            pieces = [audio[lo:hi] for lo, hi in spans if hi > lo]
            track = np.concatenate(pieces) if pieces else np.zeros(0, dtype=audio.dtype)

        out_path = out_dir / f"{speaker}.wav"
        sf.write(out_path, track, sr)
        written[speaker] = out_path
    return written
