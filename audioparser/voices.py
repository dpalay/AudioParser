"""Persistent voice registry: remember who a voice belongs to across runs.

Stores one or more fingerprint vectors per person in a JSON file
(default ~/.audioparser/voices.json). New recordings are matched by
cosine similarity; unknown voices can be enrolled interactively.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_DB = Path.home() / ".audioparser" / "voices.json"
MATCH_THRESHOLD = 0.80


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class VoiceRegistry:
    def __init__(self, path: str | Path = DEFAULT_DB, threshold: float = MATCH_THRESHOLD):
        self.path = Path(path)
        self.threshold = threshold
        self._people: dict[str, list[np.ndarray]] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self.path.read_text())
        for name, vectors in data.get("people", {}).items():
            self._people[name] = [np.array(v, dtype=np.float64) for v in vectors]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "people": {
                name: [np.round(v, 6).tolist() for v in vectors]
                for name, vectors in self._people.items()
            }
        }
        self.path.write_text(json.dumps(data, indent=2))

    @property
    def names(self) -> list[str]:
        return sorted(self._people)

    def enroll(self, name: str, embedding: np.ndarray, max_samples: int = 5) -> None:
        """Add a fingerprint sample for a person (keeps the newest few)."""
        samples = self._people.setdefault(name, [])
        samples.append(np.asarray(embedding, dtype=np.float64))
        del samples[:-max_samples]

    def match(self, embedding: np.ndarray) -> tuple[str, float] | None:
        """Best-scoring known person, or None below the threshold."""
        best: tuple[str, float] | None = None
        for name, samples in self._people.items():
            score = max(cosine_similarity(embedding, s) for s in samples)
            if best is None or score > best[1]:
                best = (name, score)
        if best is not None and best[1] >= self.threshold:
            return best
        return None


def identify_speakers(
    registry: VoiceRegistry,
    embeddings: dict[str, np.ndarray],
) -> tuple[dict[str, str], list[str]]:
    """Match each diarized speaker label against the registry.

    Returns (label -> person name for matches, list of unmatched labels).
    Two labels never map to the same person; the better score wins.
    """
    scored: list[tuple[float, str, str]] = []
    for label, emb in embeddings.items():
        hit = registry.match(emb)
        if hit is not None:
            scored.append((hit[1], label, hit[0]))

    mapping: dict[str, str] = {}
    taken: set[str] = set()
    for _, label, name in sorted(scored, reverse=True):
        if name not in taken:
            mapping[label] = name
            taken.add(name)

    unknown = [label for label in embeddings if label not in mapping]
    return mapping, unknown
