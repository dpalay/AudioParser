import numpy as np

from audioparser.voices import VoiceRegistry, identify_speakers


def _vec(seed: int, dim: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=dim)


def test_enroll_match_roundtrip(tmp_path):
    db = tmp_path / "voices.json"
    reg = VoiceRegistry(db)
    alice = _vec(1)
    reg.enroll("Alice", alice)
    reg.save()

    reg2 = VoiceRegistry(db)
    hit = reg2.match(alice + 0.01 * _vec(99))
    assert hit is not None
    name, score = hit
    assert name == "Alice"
    assert score > 0.99


def test_no_match_below_threshold(tmp_path):
    reg = VoiceRegistry(tmp_path / "v.json")
    reg.enroll("Alice", _vec(1))
    assert reg.match(_vec(2)) is None  # independent random vectors ~ orthogonal


def test_identify_speakers_unique_assignment(tmp_path):
    reg = VoiceRegistry(tmp_path / "v.json")
    alice, bob = _vec(1), _vec(2)
    reg.enroll("Alice", alice)
    reg.enroll("Bob", bob)

    embeddings = {
        "SPEAKER_00": bob + 0.01 * _vec(50),
        "SPEAKER_01": alice + 0.01 * _vec(51),
        "SPEAKER_02": _vec(77),  # a stranger
    }
    mapping, unknown = identify_speakers(reg, embeddings)
    assert mapping == {"SPEAKER_00": "Bob", "SPEAKER_01": "Alice"}
    assert unknown == ["SPEAKER_02"]


def test_same_person_not_assigned_twice(tmp_path):
    reg = VoiceRegistry(tmp_path / "v.json")
    alice = _vec(1)
    reg.enroll("Alice", alice)
    embeddings = {
        "SPEAKER_00": alice + 0.001 * _vec(50),
        "SPEAKER_01": alice + 0.2 * _vec(51),
    }
    mapping, unknown = identify_speakers(reg, embeddings)
    assert mapping["SPEAKER_00"] == "Alice"
    assert "SPEAKER_01" in unknown


def test_enroll_caps_samples(tmp_path):
    reg = VoiceRegistry(tmp_path / "v.json")
    for i in range(10):
        reg.enroll("Alice", _vec(i), max_samples=5)
    assert len(reg._people["Alice"]) == 5
