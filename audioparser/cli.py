"""audioparser CLI: diarize + transcribe a meeting recording.

Pipeline:
  1. load audio            (audio.py)
  2. diarize into turns    (diarize.py)  -> who spoke when + voice fingerprints
  3. identify speakers     (voices.py)   -> match fingerprints to known people,
                                           prompt for unknowns, remember them
  4. transcribe            (transcribe.py, optional)
  5. align words to turns  (align.py)
  6. write outputs         (output.py, audio.py)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from . import align, audio as audio_mod, diarize as diarize_mod, output, voices
from .types import SpeakerTurn


def _relabel(turns: list[SpeakerTurn], mapping: dict[str, str]) -> list[SpeakerTurn]:
    return [
        SpeakerTurn(t.start, t.end, mapping.get(t.speaker, t.speaker)) for t in turns
    ]


def _sample_clip(
    audio: np.ndarray, sr: int, turns: list[SpeakerTurn], label: str, path: Path,
    max_s: float = 10.0,
) -> None:
    """Write a short clip of one speaker's longest turn, for listening."""
    own = [t for t in turns if t.speaker == label]
    if not own:
        return
    longest = max(own, key=lambda t: t.duration)
    lo = int(longest.start * sr)
    hi = min(int(longest.end * sr), lo + int(max_s * sr))
    sf.write(path, audio[lo:hi], sr)


def _prompt_unknowns(
    unknown: list[str],
    registry: voices.VoiceRegistry,
    embeddings: dict[str, np.ndarray],
    utterances_by_label: dict[str, list[str]],
    audio: np.ndarray,
    sr: int,
    turns: list[SpeakerTurn],
    out_dir: Path,
) -> dict[str, str]:
    """Interactively name unknown voices; enroll answers in the registry."""
    mapping: dict[str, str] = {}
    for label in unknown:
        clip_path = out_dir / f"sample_{label}.wav"
        _sample_clip(audio, sr, turns, label, clip_path)

        print(f"\nUnknown voice: {label}")
        if clip_path.exists():
            print(f"  Sample clip: {clip_path}")
        for line in utterances_by_label.get(label, [])[:3]:
            print(f'  Says: "{line}"')
        if registry.names:
            print(f"  Known people: {', '.join(registry.names)}")
        try:
            name = input(f"  Who is this? (name, or Enter to keep '{label}'): ").strip()
        except EOFError:
            name = ""
        if name:
            registry.enroll(name, embeddings[label])
            mapping[label] = name
    return mapping


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audioparser",
        description="Split a meeting recording by voice and build a "
        "speaker-attributed transcript ready for LLM analysis.",
    )
    p.add_argument("input", help="audio file (wav/flac natively; others need ffmpeg)")
    p.add_argument("-o", "--out-dir", default=None,
                   help="output directory (default: <input>_parsed)")
    p.add_argument("-n", "--num-speakers", type=int, default=None,
                   help="number of speakers, if known (improves clustering)")
    p.add_argument("--backend", choices=["builtin", "pyannote"], default="builtin",
                   help="diarization backend (default: builtin)")
    p.add_argument("--hf-token", default=None,
                   help="Hugging Face token for the pyannote backend")
    p.add_argument("--whisper-model", default="base",
                   help="faster-whisper model size (tiny/base/small/medium/large-v3)")
    p.add_argument("--language", default=None, help="spoken language hint, e.g. en")
    p.add_argument("--no-transcript", action="store_true",
                   help="skip transcription; still diarizes and splits audio")
    p.add_argument("--no-split", action="store_true",
                   help="skip writing per-speaker audio files")
    p.add_argument("--split-mode", choices=["silence", "concat"], default="silence",
                   help="per-speaker audio: full-length with others muted, or "
                   "segments concatenated (default: silence)")
    p.add_argument("--voices-db", default=str(voices.DEFAULT_DB),
                   help="voice registry file (default: ~/.audioparser/voices.json)")
    p.add_argument("--no-identify", action="store_true",
                   help="skip voice identification against the registry")
    p.add_argument("--non-interactive", action="store_true",
                   help="never prompt for unknown voices")
    p.add_argument("--match-threshold", type=float, default=voices.MATCH_THRESHOLD,
                   help="cosine similarity needed to auto-match a known voice")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    in_path = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else in_path.with_name(in_path.stem + "_parsed")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {in_path} ...")
    samples, sr = audio_mod.load_audio(in_path)
    print(f"  {len(samples) / sr:.1f}s of audio at {sr} Hz")

    print(f"Diarizing ({args.backend}) ...")
    turns, embeddings = diarize_mod.diarize(
        samples, sr,
        backend=args.backend,
        num_speakers=args.num_speakers,
        hf_token=args.hf_token,
    )
    if not turns:
        print("No speech detected.", file=sys.stderr)
        return 1
    print(f"  {len({t.speaker for t in turns})} speakers, {len(turns)} turns")

    words = []
    if not args.no_transcript:
        print(f"Transcribing (whisper {args.whisper_model}) ...")
        from . import transcribe as transcribe_mod
        words = transcribe_mod.transcribe(
            samples, sr, model_size=args.whisper_model, language=args.language
        )
        print(f"  {len(words)} words")

    utterances = (
        align.build_utterances(words, turns) if words
        else align.turns_only_utterances(turns)
    )

    # --- speaker identification against the persistent voice registry ---
    if not args.no_identify:
        registry = voices.VoiceRegistry(args.voices_db, threshold=args.match_threshold)
        mapping, unknown = voices.identify_speakers(registry, embeddings)
        for label, name in mapping.items():
            print(f"Recognized {label} as {name}")

        if unknown and not args.non_interactive and sys.stdin.isatty():
            texts = {
                lab: [u.text for u in utterances if u.speaker == lab and u.text]
                for lab in unknown
            }
            mapping.update(_prompt_unknowns(
                unknown, registry, embeddings, texts, samples, sr, turns, out_dir
            ))
            registry.save()
        elif unknown:
            print(f"Unknown voices kept as generic labels: {', '.join(unknown)}")

        if mapping:
            turns = _relabel(turns, mapping)
            for u in utterances:
                u.speaker = mapping.get(u.speaker, u.speaker)

    # --- outputs ---
    output.write_markdown(utterances, out_dir / "transcript.md", in_path.name)
    output.write_text(utterances, out_dir / "transcript.txt")
    output.write_json(utterances, out_dir / "transcript.json", in_path.name)
    output.write_srt(utterances, out_dir / "transcript.srt")

    if not args.no_split:
        print("Splitting audio by speaker ...")
        written = audio_mod.split_by_speaker(
            samples, sr, turns, out_dir / "speakers", mode=args.split_mode
        )
        for speaker, path in written.items():
            print(f"  {speaker}: {path}")

    print(f"\nDone. Transcript for LLM analysis: {out_dir / 'transcript.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
