# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Crisis-language safety screen and disclaimer.
- Project tooling: `pyproject.toml`, ruff, mypy, pre-commit, GitHub Actions CI,
  contributing guide, security policy, code of conduct, issue and PR templates.
- Data card for the research corpora (`src/moodcheck/research/data/README.md`):
  sources, missing licenses and consent, how labels were assigned, known biases,
  and the happy-moments confound. A test fails if a corpus changes without the card.

### Changed

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
