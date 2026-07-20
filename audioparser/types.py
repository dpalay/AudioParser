"""Core data types shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class SpeakerTurn:
    """A contiguous span of time attributed to one speaker."""

    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlap(self, start: float, end: float) -> float:
        """Seconds of overlap between this turn and [start, end]."""
        return max(0.0, min(self.end, end) - max(self.start, start))


@dataclass(frozen=True)
class Word:
    """A single transcribed word with timing."""

    start: float
    end: float
    text: str
    probability: float = 1.0


@dataclass
class Utterance:
    """A run of consecutive words spoken by one speaker."""

    start: float
    end: float
    speaker: str
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "speaker": self.speaker,
            "text": self.text,
            "words": [asdict(w) for w in self.words],
        }
