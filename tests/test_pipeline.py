"""End-to-end tests of the builtin diarizer and audio splitting on
synthetic two-voice audio (no ML models needed)."""
import numpy as np
import pytest
import soundfile as sf

from audioparser.audio import load_audio, split_by_speaker
from audioparser.diarize import diarize, embed_clip
from audioparser.types import SpeakerTurn

SR = 16_000


def _voice(freqs: list[float], duration: float, rng) -> np.ndarray:
    """A crude synthetic 'voice': harmonic stack with amplitude jitter."""
    t = np.arange(int(duration * SR)) / SR
    sig = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate(freqs))
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3.1 * t + rng.uniform(0, 6))
    return (0.3 * sig * envelope).astype(np.float32)


def _two_speaker_audio() -> np.ndarray:
    """low voice 0-3s, silence, high voice 4-7s, silence, low voice 8-11s."""
    rng = np.random.default_rng(0)
    low = [110, 220, 330, 440]
    high = [280, 560, 840, 1120]
    silence = np.zeros(SR, dtype=np.float32)
    parts = [
        _voice(low, 3.0, rng), silence,
        _voice(high, 3.0, rng), silence,
        _voice(low, 3.0, rng),
    ]
    audio = np.concatenate(parts)
    return audio + rng.normal(0, 0.003, len(audio)).astype(np.float32)


@pytest.fixture(scope="module")
def audio():
    return _two_speaker_audio()


def test_builtin_diarize_finds_two_speakers(audio):
    turns, embeddings = diarize(audio, SR, backend="builtin", num_speakers=2)
    speakers = {t.speaker for t in turns}
    assert speakers == {"SPEAKER_00", "SPEAKER_01"}
    assert set(embeddings) == speakers

    # first and last blocks are the same voice, middle block is the other
    def speaker_at(t: float) -> str:
        hits = [x.speaker for x in turns if x.start <= t <= x.end]
        assert hits, f"no turn covers t={t}"
        return hits[0]

    assert speaker_at(1.5) == speaker_at(9.5)
    assert speaker_at(1.5) != speaker_at(5.5)


def test_embeddings_separate_voices(audio):
    turns, embeddings = diarize(audio, SR, backend="builtin", num_speakers=2)
    from audioparser.voices import cosine_similarity

    # a fresh clip of the low voice should be closer to the low-voice
    # cluster's embedding than to the high one's
    rng = np.random.default_rng(7)
    clip = _voice([110, 220, 330, 440], 2.0, rng)
    clip_emb = embed_clip(clip, SR)

    low_label = next(
        x.speaker for x in turns if x.start <= 1.5 <= x.end
    )
    other = next(s for s in embeddings if s != low_label)
    assert cosine_similarity(clip_emb, embeddings[low_label]) > cosine_similarity(
        clip_emb, embeddings[other]
    )


def test_split_by_speaker_silence_mode(audio, tmp_path):
    turns = [
        SpeakerTurn(0.0, 3.0, "A"),
        SpeakerTurn(4.0, 7.0, "B"),
        SpeakerTurn(8.0, 11.0, "A"),
    ]
    written = split_by_speaker(audio, SR, turns, tmp_path, mode="silence")
    assert set(written) == {"A", "B"}

    a, _ = sf.read(written["A"])
    b, _ = sf.read(written["B"])
    assert len(a) == len(audio) and len(b) == len(audio)
    # B's track is silent where A speaks and loud where B speaks
    assert np.abs(b[: 3 * SR]).max() == 0
    assert np.abs(b[int(4.5 * SR): int(6.5 * SR)]).max() > 0.01
    assert np.abs(a[int(4.5 * SR): int(6.5 * SR)]).max() == 0


def test_split_by_speaker_concat_mode(audio, tmp_path):
    turns = [SpeakerTurn(0.0, 3.0, "A"), SpeakerTurn(8.0, 11.0, "A")]
    written = split_by_speaker(audio, SR, turns, tmp_path, mode="concat")
    a, _ = sf.read(written["A"])
    assert abs(len(a) - 6 * SR) < 10


def test_load_audio_resamples(tmp_path):
    path = tmp_path / "x.wav"
    sf.write(path, np.zeros(44100, dtype=np.float32), 44100)
    data, sr = load_audio(path)
    assert sr == SR
    assert abs(len(data) - SR) < 5
