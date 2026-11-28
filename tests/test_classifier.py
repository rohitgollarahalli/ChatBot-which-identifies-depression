import dataclasses

import pytest

from moodcheck.research import preprocessing
from moodcheck.research.classifier import Prediction, TrainedModel, predict_text, train
from moodcheck.research.config import DEPRESSED_LABEL, NOT_DEPRESSED_LABEL, ModelConfig
from moodcheck.research.units import split_units
from moodcheck.storage import TRANSCRIPT_HEADER

NB = ModelConfig("multinomial_nb", "tfidf", "unigram")


@pytest.fixture(scope="module")
def trained() -> TrainedModel:
    return train(NB)


def test_train_returns_model_and_does_not_print(capsys):
    result = train(NB)
    assert capsys.readouterr().out == ""
    assert isinstance(result, TrainedModel)
    assert result.config == NB
    assert "precision" in result.report


@pytest.mark.skipif(
    not preprocessing.has_full_stop_words(), reason="needs the full NLTK stop-word list"
)
def test_multinomial_nb_accuracy_is_unchanged(trained):
    # Pinned to the bundled data; re-pin deliberately when the corpora change
    # (0.833 before the neutral GoEmotions comments and the 27800 fix).
    assert f"{trained.accuracy:.3f}" == "0.741"


def test_training_is_deterministic(trained):
    again = train(NB)
    assert again.accuracy == trained.accuracy
    assert again.report == trained.report


def test_predict_text_labels(trained):
    assert (
        predict_text("I feel hopeless and empty and worthless.", trained).label == DEPRESSED_LABEL
    )
    assert predict_text("", trained) == Prediction(NOT_DEPRESSED_LABEL, 0, 0)
    assert predict_text("the of and", trained) == Prediction(NOT_DEPRESSED_LABEL, 0, 0)


def test_tie_prefers_not_depressed(trained):
    class Fixed:
        def __init__(self, labels):
            self.labels = labels

        def predict(self, _features):
            return self.labels

    trained.model, original = Fixed([DEPRESSED_LABEL, NOT_DEPRESSED_LABEL]), trained.model
    try:
        assert predict_text("sad day. glad day.", trained) == Prediction(NOT_DEPRESSED_LABEL, 1, 2)
    finally:
        trained.model = original


class _Fixed:
    def __init__(self, labels):
        self.labels = labels
        self.seen = 0

    def predict(self, features):
        self.seen = features.shape[0]
        return self.labels[: self.seen]


def test_each_line_is_its_own_unit_even_without_punctuation(trained):
    fixed = _Fixed([DEPRESSED_LABEL] * 3)
    trained.model, original = fixed, trained.model
    try:
        result = predict_text(f"{TRANSCRIPT_HEADER}\nsad day\nglad day\nbad day\n", trained)
    finally:
        trained.model = original
    assert fixed.seen == 3
    assert result == Prediction(DEPRESSED_LABEL, 3, 3)


def test_lines_are_split_into_sentences_too(trained):
    fixed = _Fixed([DEPRESSED_LABEL] * 4)
    trained.model, original = fixed, trained.model
    try:
        result = predict_text("sad day. bad day.\n\nglad day! good day?", trained)
    finally:
        trained.model = original
    assert result.total_votes == 4


def test_old_single_line_transcript_still_works(trained):
    assert predict_text("sad day. glad day.", trained).total_votes == 2


def test_crlf_and_blank_lines_are_ignored(trained):
    text = f"{TRANSCRIPT_HEADER}\r\nsad day\r\n\r\n   \r\nglad day\r\n"
    assert predict_text(text, trained).total_votes == 2


def test_wrapped_prose_is_split_on_sentences_not_lines():
    assert split_units("I felt bad. Then I\nwent home. It was\nfine.") == [
        "I felt bad.",
        "Then I went home.",
        "It was fine.",
    ]


def test_transcript_lines_split_only_on_real_newlines():
    text = f"{TRANSCRIPT_HEADER}\nfine\u2028really\nok\rnext\r\nlast\n"
    assert split_units(text) == ["fine really", "ok", "next", "last"]


def test_header_line_is_not_a_unit():
    assert split_units(f"{TRANSCRIPT_HEADER}\n") == []


def test_prediction_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Prediction(NOT_DEPRESSED_LABEL, 0, 0).label = DEPRESSED_LABEL  # type: ignore[misc]


def test_gaussian_nb_uses_dense_features():
    result = train(ModelConfig("gaussian_nb", "count", "unigram"))
    assert predict_text("I want to cry all the time.", result).label in {
        DEPRESSED_LABEL,
        NOT_DEPRESSED_LABEL,
    }


def test_config_rejects_unknown_names():
    with pytest.raises(ValueError):
        ModelConfig("nope")
