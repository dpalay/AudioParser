"""Transcript writers: LLM-ready markdown, plain text, JSON, SRT."""
from __future__ import annotations

import json
from pathlib import Path

from .types import Utterance


def _ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _srt_ts(seconds: float) -> str:
    ms = int(round((seconds % 1) * 1000))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def speaker_stats(utterances: list[Utterance]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for u in utterances:
        entry = stats.setdefault(u.speaker, {"talk_time": 0.0, "turns": 0, "words": 0})
        entry["talk_time"] += u.end - u.start
        entry["turns"] += 1
        entry["words"] += len(u.words)
    return stats


def write_markdown(utterances: list[Utterance], path: Path, source_name: str) -> None:
    """Markdown transcript designed to be pasted into an LLM prompt:
    a speaker summary up top, then compact timestamped turns."""
    stats = speaker_stats(utterances)
    total = sum(s["talk_time"] for s in stats.values()) or 1.0

    lines = [f"# Meeting transcript: {source_name}", "", "## Speakers", ""]
    for speaker in sorted(stats, key=lambda s: -stats[s]["talk_time"]):
        s = stats[speaker]
        lines.append(
            f"- **{speaker}** — {_ts(s['talk_time'])} talk time "
            f"({100 * s['talk_time'] / total:.0f}%), {s['turns']} turns"
        )
    lines += ["", "## Transcript", ""]
    for u in utterances:
        text = u.text or "(no transcription)"
        lines.append(f"**{u.speaker}** [{_ts(u.start)}]: {text}")
        lines.append("")
    path.write_text("\n".join(lines))


def write_text(utterances: list[Utterance], path: Path) -> None:
    lines = [
        f"[{_ts(u.start)} - {_ts(u.end)}] {u.speaker}: {u.text or '(speech)'}"
        for u in utterances
    ]
    path.write_text("\n".join(lines) + "\n")


def write_json(utterances: list[Utterance], path: Path, source_name: str) -> None:
    payload = {
        "source": source_name,
        "speakers": speaker_stats(utterances),
        "utterances": [u.to_dict() for u in utterances],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def write_srt(utterances: list[Utterance], path: Path) -> None:
    blocks = []
    for i, u in enumerate(utterances, start=1):
        text = u.text or "(speech)"
        blocks.append(f"{i}\n{_srt_ts(u.start)} --> {_srt_ts(u.end)}\n{u.speaker}: {text}\n")
    path.write_text("\n".join(blocks))
