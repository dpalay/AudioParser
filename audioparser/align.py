"""Merge word-level transcription with speaker turns.

Pure logic, no ML dependencies: given a list of Words (from transcription)
and a list of SpeakerTurns (from diarization), attribute each word to a
speaker and group consecutive same-speaker words into Utterances.
"""
from __future__ import annotations

from .types import SpeakerTurn, Utterance, Word

# A word whose midpoint is farther than this from any turn is attached to
# the nearest turn anyway; there is no "unknown" bucket because diarizers
# routinely clip a little audio off turn boundaries.
MERGE_GAP = 1.0  # seconds of silence that still continues an utterance


def assign_speaker(word: Word, turns: list[SpeakerTurn]) -> str:
    """Pick the speaker for a word.

    Prefers the turn with the largest temporal overlap; falls back to the
    turn whose boundary is closest to the word's midpoint.
    """
    if not turns:
        return "SPEAKER_00"

    best_turn = max(turns, key=lambda t: t.overlap(word.start, word.end))
    if best_turn.overlap(word.start, word.end) > 0:
        return best_turn.speaker

    mid = (word.start + word.end) / 2.0

    def distance(turn: SpeakerTurn) -> float:
        if turn.start <= mid <= turn.end:
            return 0.0
        return min(abs(turn.start - mid), abs(turn.end - mid))

    return min(turns, key=distance).speaker


def build_utterances(
    words: list[Word],
    turns: list[SpeakerTurn],
    merge_gap: float = MERGE_GAP,
) -> list[Utterance]:
    """Group words into speaker-attributed utterances.

    A new utterance starts when the speaker changes or when the gap since
    the previous word exceeds ``merge_gap`` seconds.
    """
    utterances: list[Utterance] = []
    for word in sorted(words, key=lambda w: w.start):
        speaker = assign_speaker(word, turns)
        current = utterances[-1] if utterances else None
        if (
            current is not None
            and current.speaker == speaker
            and word.start - current.end <= merge_gap
        ):
            current.words.append(word)
            current.end = max(current.end, word.end)
        else:
            utterances.append(
                Utterance(start=word.start, end=word.end, speaker=speaker, words=[word])
            )
    return utterances


def turns_only_utterances(turns: list[SpeakerTurn]) -> list[Utterance]:
    """Utterances with empty text, for when transcription is disabled."""
    return [
        Utterance(start=t.start, end=t.end, speaker=t.speaker)
        for t in sorted(turns, key=lambda t: t.start)
    ]
