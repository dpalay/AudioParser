import numpy as np

from audioparser.confidence import _softmax, score_utterances
from audioparser.diarize import diarize
from audioparser.types import Utterance

from test_pipeline import SR, _two_speaker_audio


def test_softmax_is_a_distribution():
    probs = _softmax(np.array([0.9, 0.5, 0.1]), temperature=0.07)
    assert abs(probs.sum() - 1.0) < 1e-9
    assert probs[0] > probs[1] > probs[2]


def _diarized():
    audio = _two_speaker_audio()
    turns, embeddings = diarize(audio, SR, backend="builtin", num_speakers=2)
    low_label = next(t.speaker for t in turns if t.start <= 1.5 <= t.end)
    high_label = next(s for s in embeddings if s != low_label)
    return audio, embeddings, low_label, high_label


def test_correct_attribution_scores_high():
    audio, embeddings, low_label, high_label = _diarized()
    utts = [
        Utterance(0.3, 2.7, low_label),     # low voice, correctly labeled
        Utterance(4.3, 6.7, high_label),    # high voice, correctly labeled
    ]
    score_utterances(utts, audio, SR, embeddings)
    assert all(u.confidence is not None and u.confidence > 0.5 for u in utts)


def test_wrong_attribution_scores_lower():
    audio, embeddings, low_label, high_label = _diarized()
    right = Utterance(0.3, 2.7, low_label)
    wrong = Utterance(0.3, 2.7, high_label)  # low voice mislabeled as high
    score_utterances([right, wrong], audio, SR, embeddings)
    assert right.confidence > wrong.confidence
    assert wrong.confidence < 0.5


def test_short_utterance_left_unscored():
    audio, embeddings, low_label, _ = _diarized()
    u = Utterance(1.0, 1.2, low_label)
    score_utterances([u], audio, SR, embeddings)
    assert u.confidence is None


def test_unknown_label_left_unscored():
    audio, embeddings, _, _ = _diarized()
    u = Utterance(0.3, 2.7, "SOMEONE_ELSE")
    score_utterances([u], audio, SR, embeddings)
    assert u.confidence is None


def test_to_dict_includes_confidence():
    u = Utterance(0.0, 1.0, "A", confidence=0.87654)
    assert u.to_dict()["confidence"] == 0.877
    assert Utterance(0.0, 1.0, "A").to_dict()["confidence"] is None
