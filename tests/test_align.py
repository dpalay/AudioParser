from audioparser.align import assign_speaker, build_utterances
from audioparser.types import SpeakerTurn, Word

TURNS = [
    SpeakerTurn(0.0, 5.0, "alice"),
    SpeakerTurn(5.5, 10.0, "bob"),
    SpeakerTurn(10.5, 15.0, "alice"),
]


def test_word_inside_turn():
    assert assign_speaker(Word(1.0, 1.5, "hi"), TURNS) == "alice"
    assert assign_speaker(Word(6.0, 6.3, "yo"), TURNS) == "bob"


def test_word_in_gap_goes_to_nearest_turn():
    # 5.1-5.2 sits in the gap; bob's turn boundary (5.5) is nearer than alice's (5.0)?
    # midpoint 5.15 -> alice at distance 0.15, bob at 0.35
    assert assign_speaker(Word(5.1, 5.2, "um"), TURNS) == "alice"
    assert assign_speaker(Word(5.4, 5.45, "so"), TURNS) == "bob"


def test_word_overlapping_two_turns_prefers_larger_overlap():
    # 4.8-5.7: 0.2s in alice, 0.2s in bob's... alice 5.0-4.8=0.2, bob 5.7-5.5=0.2
    # use a clearly asymmetric word instead
    assert assign_speaker(Word(4.9, 5.6, "and"), TURNS) == "alice"  # 0.1 vs 0.1 -> tie: max picks first
    assert assign_speaker(Word(4.95, 5.9, "then"), TURNS) == "bob"


def test_empty_turns_defaults():
    assert assign_speaker(Word(1.0, 2.0, "x"), []) == "SPEAKER_00"


def test_build_utterances_groups_by_speaker():
    words = [
        Word(0.5, 0.8, "hello"),
        Word(0.9, 1.2, "there"),
        Word(6.0, 6.4, "hi"),
        Word(11.0, 11.5, "back"),
    ]
    utts = build_utterances(words, TURNS)
    assert [(u.speaker, u.text) for u in utts] == [
        ("alice", "hello there"),
        ("bob", "hi"),
        ("alice", "back"),
    ]
    assert utts[0].start == 0.5 and utts[0].end == 1.2


def test_long_pause_splits_same_speaker():
    words = [Word(0.5, 0.8, "one"), Word(3.5, 3.8, "two")]
    utts = build_utterances(words, TURNS, merge_gap=1.0)
    assert len(utts) == 2
    assert all(u.speaker == "alice" for u in utts)
