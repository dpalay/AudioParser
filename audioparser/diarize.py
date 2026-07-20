"""Speaker diarization backends.

Two backends, one interface — each returns (turns, embeddings) where
``embeddings`` maps a speaker label to a voice-fingerprint vector used for
cross-recording speaker identification:

* ``builtin`` — energy VAD + MFCC features + agglomerative clustering.
  numpy/scipy/scikit-learn only; runs anywhere, decent on clean audio.
* ``pyannote`` — pyannote.audio pipeline (state of the art; needs torch
  and a Hugging Face token with the gated models accepted).
"""
from __future__ import annotations

import numpy as np

from .types import SpeakerTurn

Embeddings = dict[str, np.ndarray]

FRAME_S = 0.03
HOP_S = 0.01
N_MFCC = 20


# --------------------------------------------------------------------------
# builtin backend
# --------------------------------------------------------------------------

def _frame_signal(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    frame = int(FRAME_S * sr)
    hop = int(HOP_S * sr)
    n = 1 + max(0, (len(audio) - frame)) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return audio[idx] * np.hanning(frame), hop


def _energy_vad(
    frames: np.ndarray, percentile: float = 35.0, max_gap_frames: int = 30
) -> np.ndarray:
    """Boolean speech mask per frame from smoothed log-energy thresholding."""
    energy = np.log10(np.sum(frames**2, axis=1) + 1e-10)
    # smooth over ~0.25s so amplitude modulation within a word doesn't
    # flicker the mask on and off
    kernel = np.ones(25) / 25
    energy = np.convolve(energy, kernel, mode="same")
    threshold = np.percentile(energy, percentile) + 0.3 * (
        np.percentile(energy, 95) - np.percentile(energy, percentile)
    )
    mask = energy > threshold
    # close short gaps (~0.3s) so pauses between words stay in one region
    gap_start = None
    for i in range(len(mask)):
        if not mask[i]:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None and 0 < gap_start and i - gap_start <= max_gap_frames:
                mask[gap_start:i] = True
            gap_start = None
    return mask


def _mfcc(frames: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Minimal MFCC implementation (mel filterbank + DCT) over frames."""
    from scipy.fft import dct, rfft

    n_fft = frames.shape[1]
    spectrum = np.abs(rfft(frames, axis=1)) ** 2

    n_mels = 40
    f_max = sr / 2
    mel_max = 2595 * np.log10(1 + f_max / 700)
    mel_pts = np.linspace(0, mel_max, n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bins = np.clip(bins, 0, spectrum.shape[1] - 1)

    fbank = np.zeros((n_mels, spectrum.shape[1]))
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center > left:
            fbank[m - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fbank[m - 1, center:right] = (right - np.arange(center, right)) / (right - center)

    mel_energy = np.log10(spectrum @ fbank.T + 1e-10)
    return dct(mel_energy, type=2, axis=1, norm="ortho")[:, :n_mfcc]


def _speech_regions(mask: np.ndarray, hop_s: float, min_dur: float = 0.25) -> list[tuple[int, int]]:
    """Contiguous True runs in the mask, as frame-index spans."""
    regions: list[tuple[int, int]] = []
    start = None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(mask)))
    return [(a, b) for a, b in regions if (b - a) * hop_s >= min_dur]


def _diarize_builtin(
    audio: np.ndarray, sr: int, num_speakers: int | None
) -> tuple[list[SpeakerTurn], Embeddings]:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    frames, hop = _frame_signal(audio, sr)
    hop_s = hop / sr
    mask = _energy_vad(frames)
    regions = _speech_regions(mask, hop_s)
    if not regions:
        return [], {}

    feats = _mfcc(frames, sr)

    # one feature vector per region: MFCC mean + std, a crude but
    # serviceable voice fingerprint
    vectors = []
    for a, b in regions:
        seg = feats[a:b]
        vectors.append(np.concatenate([seg.mean(axis=0), seg.std(axis=0)]))
    X = np.array(vectors)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

    if len(regions) == 1:
        labels = np.array([1])
    else:
        Z = linkage(pdist(X, metric="cosine"), method="average")
        if num_speakers is not None:
            labels = fcluster(Z, t=max(1, num_speakers), criterion="maxclust")
        else:
            labels = fcluster(Z, t=0.7, criterion="distance")

    # stable labels ordered by first appearance
    order: dict[int, str] = {}
    turns: list[SpeakerTurn] = []
    for (a, b), lab in zip(regions, labels):
        if lab not in order:
            order[lab] = f"SPEAKER_{len(order):02d}"
        turns.append(SpeakerTurn(start=a * hop_s, end=b * hop_s, speaker=order[lab]))

    embeddings: Embeddings = {}
    for lab, name in order.items():
        member_vecs = X[labels == lab]
        embeddings[name] = member_vecs.mean(axis=0)

    return _merge_adjacent(turns), embeddings


def _merge_adjacent(turns: list[SpeakerTurn], gap: float = 0.4) -> list[SpeakerTurn]:
    merged: list[SpeakerTurn] = []
    for t in sorted(turns, key=lambda t: t.start):
        if merged and merged[-1].speaker == t.speaker and t.start - merged[-1].end <= gap:
            merged[-1] = SpeakerTurn(merged[-1].start, t.end, t.speaker)
        else:
            merged.append(t)
    return merged


# --------------------------------------------------------------------------
# pyannote backend
# --------------------------------------------------------------------------

def _diarize_pyannote(
    audio: np.ndarray, sr: int, num_speakers: int | None, hf_token: str | None
) -> tuple[list[SpeakerTurn], Embeddings]:
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote backend requires extras: pip install 'audioparser[pyannote]'"
        ) from exc

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    if pipeline is None:
        raise RuntimeError(
            "Could not load pyannote/speaker-diarization-3.1. Pass --hf-token and "
            "accept the model terms at https://huggingface.co/pyannote/speaker-diarization-3.1"
        )

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))

    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    result = pipeline({"waveform": waveform, "sample_rate": sr}, **kwargs)

    turns = [
        SpeakerTurn(start=seg.start, end=seg.end, speaker=label)
        for seg, _, label in result.itertracks(yield_label=True)
    ]

    # fingerprint each speaker with the builtin MFCC embedding over their
    # own audio, so the voice registry works identically across backends
    embeddings: Embeddings = {}
    for speaker in sorted({t.speaker for t in turns}):
        spans = [t for t in turns if t.speaker == speaker]
        pieces = [audio[int(t.start * sr): int(t.end * sr)] for t in spans]
        clip = np.concatenate([p for p in pieces if len(p)]) if pieces else np.zeros(1)
        embeddings[speaker] = embed_clip(clip, sr)

    return turns, embeddings


def embed_clip(audio: np.ndarray, sr: int) -> np.ndarray:
    """MFCC mean+std fingerprint of a clip (same space as builtin backend)."""
    if len(audio) < int(FRAME_S * sr) + 1:
        return np.zeros(2 * N_MFCC)
    frames, _ = _frame_signal(audio, sr)
    feats = _mfcc(frames, sr)
    return np.concatenate([feats.mean(axis=0), feats.std(axis=0)])


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------

def diarize(
    audio: np.ndarray,
    sr: int,
    backend: str = "builtin",
    num_speakers: int | None = None,
    hf_token: str | None = None,
) -> tuple[list[SpeakerTurn], Embeddings]:
    if backend == "builtin":
        return _diarize_builtin(audio, sr, num_speakers)
    if backend == "pyannote":
        return _diarize_pyannote(audio, sr, num_speakers, hf_token)
    raise ValueError(f"unknown diarization backend: {backend}")
