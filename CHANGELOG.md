# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Neutral control text: `neutral_goemotions.tsv`, 300 comments from GoEmotions
  (Apache-2.0, `LICENSE-GoEmotions.txt`) labelled only neutral, reproducible with
  `scripts/build_neutral_subset.py`. They are labelled `not_depressed`, which removes
  the "happy moments versus everything else" shortcut; labels are still by source.
- Leakage groups for every document (`dep:N` per blank-line block of the Reddit text,
  `happy:ID`, `neutral:ID`) so train and test can be split by group.
- Bring-your-own-data: `load_dataset(..., data_path=...)` reads a user CSV
  (`text`, `label`, optional `group`) with strict validation (`UserDataError`);
  `user_data_sha256()` hashes the file for model versioning. The data card warns
  that models trained on gated data must not be shared.
- Crisis-language safety screen and disclaimer.
- Project tooling: `pyproject.toml`, ruff, mypy, pre-commit, GitHub Actions CI,
  contributing guide, security policy, code of conduct, issue and PR templates.
- Data card for the research corpora (`src/moodcheck/research/data/README.md`):
  sources, missing licenses and consent, how labels were assigned, known biases,
  and the happy-moments confound. A test fails if a corpus changes without the card.
- Neutral probe set (`src/moodcheck/research/data/probe_neutral.txt`, 60 everyday
  sentences written for this project, MIT) and `load_probe_sentences()`, for a
  later evaluation of how the model treats ordinary text.

### Changed

- **Breaking:** `moodcheck chat --save` now writes a header line
  (`# moodcheck transcript: one message per line`) and then one message per line,
  instead of joining all messages with spaces.
- **Breaking:** the verdict now shows the vote, for example "The model's guess: the
  transcript shows signs of depression (4 of 6 parts).", says when a tie counted
  as no, and adds a line saying this is a research demo, not a mental-health
  assessment.
- `moodcheck research predict` votes on each message of a saved transcript (and
  each sentence within it), so chat messages without final punctuation are no
  longer merged. Any other file, including old transcripts, is read as prose and
  split on sentence punctuation only, so hard-wrapped lines are not cut apart.
  `predict_text` now returns a `Prediction` (label and votes) instead of a string.
- The crisis screen treats any run of whitespace, including line breaks and
  Unicode line separators, as a single space.
- **Breaking:** `load_dataset(clean, balance=True, data_path=None)` returns a
  `Dataset` (`documents`, `labels`, `groups`, `sources`) instead of a
  `(documents, labels)` tuple. `corpus_sha256()` now also covers the neutral file, so
  saved models trained before this change are no longer matched to the corpus.
- The Reddit text is split into sentences per blank-line block, so a sentence no
  longer spans a block boundary: 424 documents instead of 423.
- `not_depressed.txt` now loads 154 happy moments (was 150), see Fixed. The bundled
  training set is 846 balanced documents (was 300).
- README rewritten: what the project is and is not, quickstart, safety section,
  architecture diagram, and what the classifier actually learns. The
  benchmark-style accuracy table is removed; it measured source separation, not
  depression.
- Classifier verdict wording.
- Crisis language now overrides the classifier verdict.
- **Breaking:** the project is now the installable `moodcheck` package
  (`pip install .`) with a `moodcheck` command. `python -m src.chatbot` and
  `python -m src.classifier` are replaced by `moodcheck chat` and
  `moodcheck research train|predict FILE`.
- **Breaking:** saving the conversation is now opt-in (`moodcheck chat --save PATH`).
  Nothing is written to disk by default, and an existing file is never overwritten.
  The file is created readable only by you. A path that cannot be written is
  rejected before the chat starts, Ctrl-D now saves, and an exit that saves nothing
  says so. Without `--save`, `good-bye` exits at once.
- Every message goes through the crisis screen twice: in the chat loop and in
  `Engine.respond`, which engine classes may not override or replace via a mixin.
- Research mode trains once and saves the model in your cache directory
  (`$XDG_CACHE_HOME` or `~/.cache`, `%LOCALAPPDATA%` on Windows) with a version
  stamp; `research predict` loads it and retrains if it is missing or stale.
  Text containing crisis language is checked before any model work.
- Tokenizing no longer needs the NLTK `punkt` download; the token output is unchanged.
- The corpora now ship inside the package.

### Fixed

- Malformed HappyDB row 27800 in `not_depressed.txt` (unclosed quote) swallowed
  moments 27844 to 27847 into one fragment; replaced with the correctly quoted
  upstream record.
- `research predict` gives no verdict on an empty file.
- A model cache that other users own or can write to is never loaded (loading a model
  can run code).
- Chat input in an unexpected encoding no longer crashes the session.
- `research predict` refuses files that are not UTF-8. Undecodable bytes used to be
  dropped, so crisis language in a UTF-16 file skipped the safety screen.

### Removed

- `requirements.txt` (dependencies are declared in `pyproject.toml`), the `src.*`
  module paths, and the `--output` and `--predict` options.
- Presentation binaries and PDFs from the tree (still available in git history).
