import pytest

from moodcheck.research import preprocessing
from moodcheck.research.classifier import TrainedModel, predict_text, train
from moodcheck.research.config import DEPRESSED_LABEL, NOT_DEPRESSED_LABEL, ModelConfig

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
    assert f"{trained.accuracy:.3f}" == "0.833"


def test_training_is_deterministic(trained):
    again = train(NB)
    assert again.accuracy == trained.accuracy
    assert again.report == trained.report


def test_predict_text_labels(trained):
    assert predict_text("I feel hopeless and empty and worthless.", trained) == DEPRESSED_LABEL
    assert predict_text("", trained) == NOT_DEPRESSED_LABEL
    assert predict_text("the of and", trained) == NOT_DEPRESSED_LABEL


def test_tie_prefers_not_depressed(trained):
    class Fixed:
        def __init__(self, labels):
            self.labels = labels

        def predict(self, _features):
            return self.labels

    trained.model, original = Fixed([DEPRESSED_LABEL, NOT_DEPRESSED_LABEL]), trained.model
    try:
        assert predict_text("sad day. glad day.", trained) == NOT_DEPRESSED_LABEL
    finally:
        trained.model = original


def test_gaussian_nb_uses_dense_features():
    result = train(ModelConfig("gaussian_nb", "count", "unigram"))
    assert predict_text("I want to cry all the time.", result) in {
        DEPRESSED_LABEL,
        NOT_DEPRESSED_LABEL,
    }


def test_config_rejects_unknown_names():
    with pytest.raises(ValueError):
        ModelConfig("nope")
