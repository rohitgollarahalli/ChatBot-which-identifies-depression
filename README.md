# Depression Detector

A two-part natural language processing project that holds a short, empathetic
conversation with a user and then classifies the resulting transcript as
**depressed** or **not depressed** using classical machine-learning models.

[![CI](https://github.com/rohitgollarahalli/ChatBot-which-identifies-depression/actions/workflows/ci.yml/badge.svg)](https://github.com/rohitgollarahalli/ChatBot-which-identifies-depression/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Built with scikit-learn](https://img.shields.io/badge/built%20with-scikit--learn-orange.svg)](https://scikit-learn.org/)

> ⚠️ **Disclaimer:** This is an academic NLP project, **not** a medical device.
> It cannot diagnose depression or any other condition and must not be used as a
> substitute for professional care. If you or someone you know is struggling,
> please reach out to a qualified mental-health professional or a local helpline.

---

## How it works

The system has two parts:

1. **Reflective chatbot** (`moodcheck chat`) — An ELIZA-style bot, inspired by
   Weizenbaum's original, that uses regular-expression pattern matching to ask
   open-ended, empathetic questions and reflect the user's statements back to
   them. Every message goes through a crisis-language screen first. Nothing is
   saved unless you pass `--save PATH`; then, once you have shared enough
   (≥ 31 words) and type `good-bye`, your messages are written to that file.

2. **Text classifier** (`moodcheck research ...`) — An educational exercise, not
   a screening tool. Text is cleaned, tokenized, stop-word filtered, and
   Porter-stemmed, then vectorized (TF-IDF or counts, over unigrams or bigrams)
   and scored by one of five scikit-learn classifiers. A majority vote over the
   predicted sentences yields the final label. The trained model is saved in
   your cache directory and reused.

```
 ┌───────────────┐   --save PATH    ┌─────────────────────┐
 │ moodcheck chat│ ───────────────► │ moodcheck research  │
 │  (gather)     │   transcript     │ predict (preprocess │
 └───────────────┘   (optional)     │ → vectorize →       │
                                    │ classify → vote)    │
                                    └─────────────────────┘
```

## Project structure

```
.
├── src/moodcheck/
│   ├── safety.py            # crisis-language screen and help resources
│   ├── conversation/        # engine interface, ELIZA engine, chat loop
│   ├── research/            # corpora (data/), preprocessing, classifier, saved models
│   ├── storage.py           # opt-in transcript saving
│   └── cli.py               # the `moodcheck` command
├── tests/                   # pytest suite
├── plans/                   # improvement plan
├── pyproject.toml           # packaging + ruff/mypy/pytest config
├── requirements-dev.txt     # dev tools (pinned)
├── CONTRIBUTING.md · SECURITY.md · CODE_OF_CONDUCT.md · CHANGELOG.md
└── LICENSE
```

## Getting started

### 1. Install

```bash
pip install .
```

The first classifier run downloads the small NLTK `stopwords` resource
automatically (a built-in shorter list is used if that fails).

### 2. Have a conversation

```bash
moodcheck chat                    # saves nothing
moodcheck chat --save talk.txt    # saves what you type when you finish
```

Chat with the bot, then type `good-bye` to finish.

### 3. Try research mode

```bash
moodcheck research train --model svm --vectorizer tfidf --ngram unigram
moodcheck research predict talk.txt
```

`predict` loads the model saved by `train` (or trains and saves one the first time).

## Command-line options

`moodcheck chat`:

| Option   | Description                                                    | Default  |
| -------- | -------------------------------------------------------------- | -------- |
| `--save` | Save your messages to this new file when you finish (never overwrites) | *(none)* |

`moodcheck research train` and `moodcheck research predict FILE`:

| Option         | Choices                                                          | Default   |
| -------------- | ---------------------------------------------------------------- | --------- |
| `--model`      | `svm`, `multinomial_nb`, `gaussian_nb`, `random_forest`, `mlp`   | `svm`     |
| `--vectorizer` | `tfidf`, `count`                                                 | `tfidf`   |
| `--ngram`      | `unigram`, `bigram`                                              | `unigram` |

## Results

The depressed and not-depressed corpora come in different formats (free text
vs. HappyDB CSV), so the loader parses each correctly — extracting only the
moment text from the CSV — and the two classes are **balanced** before training
to avoid bias toward the larger class. Trained on the balanced set with an
80/20 stratified split:

| Model           | Accuracy |
| --------------- | -------- |
| Multinomial NB  | ~0.87    |
| MLP (neural net)| ~0.82    |
| Linear SVM      | ~0.80    |
| Random Forest   | ~0.80    |

Exact numbers vary with the random split and on whether the full NLTK
stop-word list or the lightweight offline fallback is used. The dataset is small, so
these figures should be read as a comparison of approaches rather than a
benchmark.

## Project report

The original project report is available as a [PDF](https://github.com/rohitgollarahalli/ChatBot-which-identifies-depression/blob/f00e128b2bf75694236be5dc146c80f27d3bbd46/report.pdf)
(kept in git history rather than in the current tree).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, linting, type-checking and test commands.

## Tech stack

Python · NLTK · scikit-learn · joblib

## License

Released under the [MIT License](LICENSE).
