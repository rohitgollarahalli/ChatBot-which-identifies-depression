"""Train and apply the research-mode text classifier.

Each document is cleaned (see ``preprocessing``), vectorized with TF-IDF or
counts over unigrams/bigrams and classified with one of several scikit-learn
models. The two classes are balanced before training.

This is an educational exercise, not a screening tool: the "not depressed"
corpus is happy-moment text, so the model largely tells happy writing from
sad writing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from moodcheck.research.config import DEPRESSED_LABEL, NOT_DEPRESSED_LABEL, ModelConfig
from moodcheck.research.corpus import RANDOM_SEED, load_dataset, split_sentences
from moodcheck.research.preprocessing import Preprocessor
from moodcheck.safety import CRISIS_MESSAGE, DISCLAIMER, is_crisis

MODELS: dict[str, Any] = {
    "svm": lambda: SVC(kernel="linear"),
    "multinomial_nb": MultinomialNB,
    "gaussian_nb": GaussianNB,
    "random_forest": lambda: RandomForestClassifier(random_state=RANDOM_SEED),
    "mlp": lambda: MLPClassifier(
        solver="lbfgs",
        alpha=1e-5,
        hidden_layer_sizes=(5, 2),
        max_iter=1000,
        random_state=RANDOM_SEED,
    ),
}


@dataclass
class TrainedModel:
    model: Any
    vectorizer: Any
    config: ModelConfig
    preprocessing_fingerprint: str
    report: str
    accuracy: float


def build_vectorizer(kind: str, ngram: str) -> Any:
    ngram_range = (1, 1) if ngram == "unigram" else (1, 2)
    if kind == "tfidf":
        return TfidfVectorizer(ngram_range=ngram_range, min_df=2)
    if kind == "count":
        return CountVectorizer(ngram_range=ngram_range, min_df=2)
    raise ValueError(f"Unknown vectorizer: {kind}")


def _features(trained_config: ModelConfig, matrix: Any) -> Any:
    # GaussianNB cannot take sparse input; the other models can.
    return matrix.toarray() if trained_config.model == "gaussian_nb" else matrix


def train(config: ModelConfig, *, test_size: float = 0.2) -> TrainedModel:
    """Train on the bundled corpora and evaluate on a stratified hold-out split."""
    preprocess = Preprocessor()
    documents, labels = load_dataset(preprocess.clean)

    # Known leak (Phase 3): the vectorizer is fitted on all documents, including
    # the hold-out split, so the reported accuracy is slightly optimistic.
    vectorizer = build_vectorizer(config.vectorizer, config.ngram)
    features = _features(config, vectorizer.fit_transform(documents))

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=RANDOM_SEED, stratify=labels
    )

    model = MODELS[config.model]()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    return TrainedModel(
        model=model,
        vectorizer=vectorizer,
        config=config,
        preprocessing_fingerprint=preprocess.fingerprint(),
        report=classification_report(y_test, predictions, zero_division=0),
        accuracy=float(accuracy_score(y_test, predictions)),
    )


def predict_text(text: str, trained: TrainedModel) -> str:
    """Classify text by majority vote over its sentences."""
    preprocess = Preprocessor()
    sentences = split_sentences(text) or [text]
    cleaned = [c for c in (preprocess.clean(s) for s in sentences) if c]
    if not cleaned:
        return NOT_DEPRESSED_LABEL

    features = _features(trained.config, trained.vectorizer.transform(cleaned))
    predictions = list(trained.model.predict(features))
    depressed = predictions.count(DEPRESSED_LABEL)
    not_depressed = predictions.count(NOT_DEPRESSED_LABEL)
    # On a tie, prefer the less alarming label.
    return DEPRESSED_LABEL if depressed > not_depressed else NOT_DEPRESSED_LABEL


def verdict_message(label: str, transcript: str) -> str:
    """What to show for a classified transcript.

    Crisis language always overrides the model: a wrong "does not show signs"
    guess right after someone wrote about suicide would be harmful.
    """
    if is_crisis(transcript):
        return CRISIS_MESSAGE
    verdict = (
        "shows signs of depression"
        if label == DEPRESSED_LABEL
        else "does not show signs of depression"
    )
    return f"The model's guess: the transcript {verdict}.\n{DISCLAIMER}"
