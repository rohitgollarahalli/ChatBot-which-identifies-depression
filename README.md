# moodcheck

A small, local, open-source chat companion with a crisis-language safety
screen, built from a 2018 student NLP project. Its research mode includes a
text classifier that was meant to "identify depression". **It does not**, and
this README explains why.

[![CI](https://github.com/rohitgollarahalli/ChatBot-which-identifies-depression/actions/workflows/ci.yml/badge.svg)](https://github.com/rohitgollarahalli/ChatBot-which-identifies-depression/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **If you are struggling right now:** in the US, call or text **988** (Suicide &
> Crisis Lifeline, free, 24/7). Elsewhere, find a free helpline at
> https://findahelpline.com. If you are in immediate danger, call your local
> emergency number. This program is not a crisis service.

## What it is

- A reflective, ELIZA-style chatbot (`moodcheck chat`) that runs entirely on
  your computer. It sends nothing over the network and saves nothing unless you
  ask it to. (Research mode's only network use is the one-time NLTK
  stop-word download described below.)
- A crisis-language screen that checks every message before anything else
  happens and points you to real help.
- A research mode (`moodcheck research`) that shows a classical NLP pipeline
  (cleaning, stemming, TF-IDF, five scikit-learn models) and, by accident, a
  textbook example of a dataset confound.

## What it is not

- **Not a medical device, a diagnosis, or a screening tool.** It cannot tell
  whether anyone has depression.
- **Not a therapist.** ELIZA (Weizenbaum, 1966) reflects your words back with
  pattern matching. It does not understand them.
- **Not a crisis service.** The safety screen is a keyword check. It catches
  clearly worded crisis language and misses indirect statements.
- **Not a dataset.** The training text is unlicensed third-party writing; see
  [the data card](src/moodcheck/research/data/README.md).

## Quickstart

Requires Python 3.10 or newer.

```bash
git clone https://github.com/rohitgollarahalli/ChatBot-which-identifies-depression.git
cd ChatBot-which-identifies-depression
python3 -m venv .venv && . .venv/bin/activate
pip install .

moodcheck chat                      # talk; type good-bye to finish. Saves nothing.
moodcheck chat --save ~/talk.txt    # also save what you typed to a new file
```

With `--save`, the chat keeps going until you have typed at least 31 words, and
then `good-bye` (or Ctrl-D) saves and finishes. Ctrl-C leaves without saving. The
file must not exist yet.

Research mode:

```bash
moodcheck research train --model svm --vectorizer tfidf --ngram unigram
moodcheck research predict ~/talk.txt   # any UTF-8 text file works
```

`train` saves the model in your cache directory; `predict` reuses it (or trains
one the first time). The first run downloads NLTK's small `stopwords` list; if
that fails, a built-in shorter list is used.

## Safety

- **Every message is screened first**, in the chat loop and again inside the
  conversation engine, which engine classes cannot override. A match replaces
  the bot's reply with crisis resources (988, https://findahelpline.com, local
  emergency number). In `research predict`, crisis language in the file shows
  the same resources instead of a model verdict.
- **It errs towards false positives.** Showing help to someone who did not need
  it costs little; missing someone who did costs a lot.
- **Known gaps.** Indirect statements ("nobody would notice if I was gone") need a
  semantic model. They are tracked as strict expected failures in
  `tests/test_safety.py`, so closing a gap is a visible change.
- **Privacy.** `moodcheck chat` writes nothing to disk without `--save`, and saves
  nothing if you typed fewer than 31 words. A saved file is never overwritten and
  is created readable only by you. Research mode stores trained models (not your
  text) in your cache directory.

Found a way to make the screen miss clear crisis language? Report it privately;
see [SECURITY.md](SECURITY.md).

## How it works

```
                       every message
 you ──► moodcheck chat ─────────────► safety screen ──crisis──► help resources
            │                               │
            │                            no match
            │                               ▼
            │                     ELIZA engine (regex reflection) ──► reply
            │
            └─ --save PATH (opt-in) ──► transcript file
                                              │
 moodcheck research predict FILE ◄────────────┘
            │
            ▼
   safety screen ──crisis──► help resources (no verdict)
            │ no match
            ▼
   clean → tokenize → stop words → Porter stem → TF-IDF/counts
   → one classifier chosen with --model (SVM, multinomial/Gaussian NB, random forest, MLP)
   → majority vote over the sentences → "the model's guess"
```

| Module | Job |
| --- | --- |
| `src/moodcheck/safety.py` | Crisis-language screen and help resources |
| `src/moodcheck/conversation/` | Engine interface (enforces the screen), ELIZA engine, chat loop |
| `src/moodcheck/research/` | Corpus loading, preprocessing, classifier, saved models |
| `src/moodcheck/research/data/` | The two corpora and their [data card](src/moodcheck/research/data/README.md) |
| `src/moodcheck/storage.py` | Opt-in transcript saving |
| `src/moodcheck/cli.py` | The `moodcheck` command |

## What the classifier actually learns

The "depressed" examples are posts from the Reddit forum r/depression. The "not
depressed" examples are all *happy moments* from the HappyDB corpus. Nothing in
the training data is ordinary, neutral text, so the model learns "happy moment
versus anything else":

```bash
echo "My manager scheduled a meeting about the quarterly budget on Tuesday." > neutral.txt
moodcheck research predict neutral.txt
# The model's guess: the transcript shows signs of depression.
```

Earlier versions of this README showed an accuracy table (about 0.8 to 0.87).
Those numbers came from a single split of 30 test documents per class and
measured how well the model tells the two *sources* apart, not depression. They
have been removed. A fair evaluation, with neutral control text, k-fold
cross-validation, precision and recall, and a probe set of neutral sentences, is
Phase 3 of the [improvement plan](plans/pending/improvement-plan.md).

Full details, sources, licensing and known biases:
[`src/moodcheck/research/data/README.md`](src/moodcheck/research/data/README.md).

### Command-line options

`moodcheck chat`:

| Option   | Description                                                             | Default  |
| -------- | ----------------------------------------------------------------------- | -------- |
| `--save` | Save your messages to this new file when you finish (never overwrites)  | *(none)* |

`moodcheck research train` and `moodcheck research predict FILE`:

| Option         | Choices                                                          | Default   |
| -------------- | ---------------------------------------------------------------- | --------- |
| `--model`      | `svm`, `multinomial_nb`, `gaussian_nb`, `random_forest`, `mlp`   | `svm`     |
| `--vectorizer` | `tfidf`, `count`                                                 | `tfidf`   |
| `--ngram`      | `unigram`, `bigram`                                              | `unigram` |

## Roadmap

The project is moving towards a safe check-in companion. See the
[improvement plan](plans/pending/improvement-plan.md) for details.

1. A validated questionnaire (PHQ-9) for the actual check-in result, with
   plain-language guidance and crisis resources.
2. An honest research mode, as described above.
3. Optional better conversation through a self-hosted LLM (Ollama on
   `localhost`). No cloud APIs.
4. A local web interface (Streamlit).

## Data and licensing

The code is released under the [MIT License](LICENSE). **The corpora in
`src/moodcheck/research/data/` are not**: one is verbatim Reddit posts used
without their authors' consent, and the other is a slice of HappyDB, which states
no license. For this reason the package is marked `Private :: Do Not Upload` and
is not published to PyPI. See the [data card](src/moodcheck/research/data/README.md).

## Project history

Started in 2018 as a university NLP project. The original report is in git
history:
[`report.pdf`](https://github.com/rohitgollarahalli/ChatBot-which-identifies-depression/blob/f00e128b2bf75694236be5dc146c80f27d3bbd46/report.pdf).
Its accuracy claims predate the data card and should be read with the caveats
above.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, linting, type-checking and
tests, and [CHANGELOG.md](CHANGELOG.md) for changes.
