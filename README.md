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

## Setting up the pyannote backend (recommended for real meetings)

The builtin backend struggles with crosstalk and far-field audio. pyannote
handles overlapping speech and is the accuracy upgrade. One-time setup:

1. Install the extras (pulls PyTorch, ~2 GB):

   ```bash
   pip install -e '.[pyannote]'
   ```

2. Create a free account at [huggingface.co](https://huggingface.co) and
   accept the terms on **both** gated model pages (the pipeline uses the
   second internally):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

3. Create a **read** token at https://huggingface.co/settings/tokens and
   export it:

   ```bash
   export HF_TOKEN=hf_xxx        # or pass --hf-token per run
   ```

4. Run:

   ```bash
   audioparser meeting.wav --backend pyannote
   ```

The first run downloads the models (a few hundred MB, cached in
`~/.cache/huggingface` — offline afterwards). A CUDA GPU or Apple Silicon
(MPS) is used automatically when available; on CPU expect roughly
10-30 minutes for an hour of audio, a few minutes with acceleration.

### What the token is for, and what leaves your machine

The Hugging Face token is a **download credential only**. The pyannote
authors gate their weights behind an accept-terms form; the token proves
your account accepted so the one-time download is authorized. **Your audio
is never uploaded anywhere** — diarization and transcription both run
entirely on your machine. After the first download, no network is needed.

To *guarantee* zero outbound traffic at runtime (skips even cache-freshness
checks):

```bash
export HF_HUB_OFFLINE=1
```

### Fully air-gapped setup (no Hugging Face contact from this machine)

If the processing machine can't touch the internet at all, fetch the models
elsewhere and carry them over:

1. On any internet-connected machine:

   ```bash
   pip install huggingface_hub
   hf download pyannote/speaker-diarization-3.1 --local-dir models/diarization
   hf download pyannote/segmentation-3.0        --local-dir models/segmentation
   hf download pyannote/wespeaker-voxceleb-resnet34-LM --local-dir models/embedding
   hf download Systran/faster-whisper-base      --local-dir models/whisper-base
   ```

   (Requires accepting the terms + a token once, on that machine.)

2. In `models/diarization/config.yaml`, replace the two hub references with
   the local paths you copied them to:

   ```yaml
   segmentation: /path/to/models/segmentation/pytorch_model.bin
   embedding: /path/to/models/embedding/pytorch_model.bin
   ```

3. Copy `models/` to the offline machine and run:

   ```bash
   audioparser meeting.wav --backend pyannote \
       --pyannote-model /path/to/models/diarization/config.yaml \
       --whisper-model  /path/to/models/whisper-base
   ```

No token, no network, nothing outbound. The `analyze` step is the **only**
part of this tool that sends data anywhere, and it only runs when you
explicitly invoke it.

During overlapping speech pyannote emits overlapping turns for both
speakers; each transcribed word is attributed to whichever turn it overlaps
most, so brief interjections land with the interjector.

## Feeding it to an LLM

`transcript.md` is designed to drop straight into a prompt:

> Here is a meeting transcript. Summarize the decisions made, list action
> items with owners (use the speaker names), and flag any disagreements.

For programmatic use, `transcript.json` has per-utterance timing and
per-speaker talk-time stats.

### `audioparser analyze` — send it to your LLM API automatically

Works with any OpenAI-compatible endpoint, including org LLM proxies:

```bash
export AUDIOPARSER_LLM_BASE_URL=https://your-llm-proxy.example.com/v1
export AUDIOPARSER_LLM_API_KEY=sk-...
export AUDIOPARSER_LLM_MODEL=gpt-5.4      # whatever your proxy calls it

audioparser analyze meeting_parsed/transcript.md
```

The default prompt asks for a summary, decisions, action items with owners,
disagreements, and follow-ups; it writes `analysis.md` next to the
transcript. Use `--prompt "..."` (or `--prompt prompt.txt`) to ask something
else, and `--base-url/--model` to override the env vars per run.
(`OPENAI_BASE_URL`/`OPENAI_API_KEY` are honored as fallbacks.)

Notes for shared/org proxies: a one-hour transcript is roughly 10-15k
tokens, so single-meeting analysis costs pennies per run on GPT-class
models. Remember that transcripts contain your colleagues' names and
words — check your org's data-handling rules (PII/PHI, regional
restrictions) before sending, same as you would for any meeting notes.

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

## Confidence scores

Two confidences are reported:

- **Identification similarity** (per speaker) — how well the voice matched
  the registry entry it was named from. Printed at run time
  (`Recognized SPEAKER_00 as Dave (similarity 0.91)`) and stored under
  `speakers.<name>.identification_similarity` in `transcript.json`.
- **Attribution confidence** (per utterance) — each turn's audio is
  re-fingerprinted and compared against every detected speaker; a softmax
  over the similarities gives the probability that the turn belongs to the
  speaker it was assigned to. Stored as `confidence` on every utterance in
  `transcript.json`; turns below 0.6 are marked `[?]` in `transcript.md`
  (with a legend telling the LLM to treat those speaker names as
  uncertain). Turns shorter than 0.4s are left unscored — too little
  audio to fingerprint meaningfully.

Low attribution confidence clusters where diarization is hardest:
crosstalk, brief interjections, and turn boundaries. Note it is a
*relative* measure among the speakers detected in the recording — it
cannot flag a person the diarizer never separated out.

## Development

```bash
pip install -e '.[dev]'
pytest
```

Tests cover the word/turn alignment logic, the voice registry, and an
end-to-end run of the builtin diarizer on synthetic two-voice audio — no model
downloads needed.
