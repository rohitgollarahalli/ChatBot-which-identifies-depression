"""Option names and the model configuration.

Kept free of heavy imports so the CLI can list choices without loading scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass

MODEL_NAMES = ("svm", "multinomial_nb", "gaussian_nb", "random_forest", "mlp")
VECTORIZER_NAMES = ("tfidf", "count")
NGRAM_NAMES = ("unigram", "bigram")

DEPRESSED_LABEL = "depressed"
NOT_DEPRESSED_LABEL = "not_depressed"


@dataclass(frozen=True)
class ModelConfig:
    model: str = "svm"
    vectorizer: str = "tfidf"
    ngram: str = "unigram"

    def __post_init__(self) -> None:
        if self.model not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {self.model}")
        if self.vectorizer not in VECTORIZER_NAMES:
            raise ValueError(f"Unknown vectorizer: {self.vectorizer}")
        if self.ngram not in NGRAM_NAMES:
            raise ValueError(f"Unknown ngram: {self.ngram}")
