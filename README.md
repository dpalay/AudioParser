# AudioParser

Split a meeting recording by voice, identify who is speaking, and build a
speaker-attributed transcript ready to hand to an LLM for meeting analysis.

What it does with one command:

1. **Diarization** — figures out *who spoke when* (no training needed).
2. **Speaker identification** — fingerprints each voice, matches it against a
   local voice library, and **prompts you to name voices it doesn't know**.
   Named voices are remembered for every future recording.
3. **Transcription** — word-level timestamps via Whisper (runs locally).
4. **Alignment** — merges the two into `Dave [03:12]: let's ship it` style turns.
5. **Outputs** — LLM-ready markdown, plain text, JSON, SRT subtitles, and one
   WAV per speaker.

## Install

Requires Python 3.10+. Core install (diarization + splitting, no ML downloads):

```bash
pip install -e .
```

Add local transcription (recommended — this is what makes the transcript):

```bash
pip install -e '.[transcribe]'
```

Optional: higher-quality diarization with pyannote (needs PyTorch and a free
Hugging Face token with the
[pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
terms accepted):

```bash
pip install -e '.[pyannote]'
```

For mp3/m4a inputs install [ffmpeg](https://ffmpeg.org); WAV/FLAC/OGG work
without it.

## Usage

```bash
audioparser meeting.wav
```

First run, it will ask about voices it doesn't recognize (it writes a short
sample clip you can listen to first):

```
Unknown voice: SPEAKER_00
  Sample clip: meeting_parsed/sample_SPEAKER_00.wav
  Says: "I think we should move the launch to Thursday"
  Who is this? (name, or Enter to keep 'SPEAKER_00'): Dave
```

Next run, Dave is recognized automatically. Voices live in
`~/.audioparser/voices.json`; each person keeps up to 5 fingerprint samples so
recognition improves as you confirm them across meetings.

Common options:

```bash
audioparser meeting.wav -n 4                  # you know there were 4 speakers
audioparser meeting.wav --backend pyannote --hf-token hf_xxx   # best accuracy
audioparser meeting.wav --whisper-model small # bigger = more accurate, slower
audioparser meeting.wav --no-transcript       # just split voices, no Whisper
audioparser meeting.wav --split-mode concat   # per-speaker files without gaps
audioparser meeting.wav --non-interactive     # never prompt (CI/scripts)
```

## Output

`meeting_parsed/` contains:

| File | What it's for |
| --- | --- |
| `transcript.md` | **Paste this into an LLM.** Speaker stats header + timestamped turns. |
| `transcript.json` | Structured data (word timings, per-speaker stats) for pipelines. |
| `transcript.txt` | Plain readable transcript. |
| `transcript.srt` | Subtitles with speaker labels. |
| `speakers/<name>.wav` | One track per speaker (others muted, timing preserved). |

Example `transcript.md`:

```markdown
# Meeting transcript: standup.wav

## Speakers

- **Dave** — 04:12 talk time (55%), 23 turns
- **Sarah** — 03:26 talk time (45%), 19 turns

## Transcript

**Dave** [00:03]: Morning everyone, let's get started with the launch plan.

**Sarah** [00:11]: I looked at the numbers last night and we're on track.
```

## Feeding it to an LLM

`transcript.md` is designed to drop straight into a prompt:

> Here is a meeting transcript. Summarize the decisions made, list action
> items with owners (use the speaker names), and flag any disagreements.

For programmatic use, `transcript.json` has per-utterance timing and
per-speaker talk-time stats.

## How speaker identification works

Each diarized speaker gets a voice fingerprint (MFCC statistics over their
speech). Fingerprints are cosine-matched against your voice library; matches
above the threshold (`--match-threshold`, default 0.80) are auto-named, and a
given person is never assigned to two different voices in one recording. The
same fingerprint space is used by both diarization backends, so a library
built with the builtin backend still works if you switch to pyannote.

Accuracy notes: the builtin backend is lightweight and does well on clean
recordings with distinct voices; for overlapping speech, far-field mics, or
many similar voices, use `--backend pyannote`.

## Development

```bash
pip install -e '.[dev]'
pytest
```

Tests cover the word/turn alignment logic, the voice registry, and an
end-to-end run of the builtin diarizer on synthetic two-voice audio — no model
downloads needed.
