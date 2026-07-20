"""Per-utterance speaker-attribution confidence.

Each utterance's audio is re-fingerprinted and compared against every
detected speaker's fingerprint; a softmax over the cosine similarities
gives the probability (among detected speakers) that the utterance really
belongs to the speaker it was assigned to. Low values flag likely
diarization mistakes — crosstalk, brief interjections, boundary spill.
"""
from __future__ import annotations

import numpy as np

from .diarize import embed_clip
from .output import LOW_CONFIDENCE  # noqa: F401  (canonical threshold lives with the writers)
from .types import Utterance
from .voices import cosine_similarity

# Softmax temperature: cosine similarities of MFCC-stat fingerprints live
# in a narrow band, so a small temperature is needed to spread them out.
TEMPERATURE = 0.07
MIN_SCORABLE_S = 0.4


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = x / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def score_utterances(
    utterances: list[Utterance],
    audio: np.ndarray,
    sr: int,
    speaker_embeddings: dict[str, np.ndarray],
    temperature: float = TEMPERATURE,
    min_duration: float = MIN_SCORABLE_S,
) -> None:
    """Set ``confidence`` on each utterance, in place.

    ``speaker_embeddings`` must be keyed by the same labels the utterances
    use (i.e. score before renaming speakers to people).
    Utterances shorter than ``min_duration`` are left as None — their
    fingerprints are too noisy to mean anything.
    """
    labels = list(speaker_embeddings)
    if not labels:
        return
    refs = np.array([speaker_embeddings[label] for label in labels])

    for u in utterances:
        if u.end - u.start < min_duration or u.speaker not in speaker_embeddings:
            continue
        clip = audio[int(u.start * sr): int(u.end * sr)]
        emb = embed_clip(clip, sr)
        sims = np.array([cosine_similarity(emb, ref) for ref in refs])
        probs = _softmax(sims, temperature)
        u.confidence = float(probs[labels.index(u.speaker)])
