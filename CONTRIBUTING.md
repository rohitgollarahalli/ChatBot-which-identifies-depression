# Contributing

Thanks for helping out. This project is an academic NLP chatbot, not a medical
tool, so changes that touch crisis handling get extra care (see below).

## Development setup

```bash
git clone https://github.com/rohitgollarahalli/ChatBot-which-identifies-depression.git
cd ChatBot-which-identifies-depression
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt   # installs the package in editable mode plus dev tools
```

Try it with `moodcheck chat`.

```bash
pre-commit install   # optional: run ruff on every commit
```

Python 3.10 or newer is required.

## Checks

Run these before opening a PR; CI runs the same commands.

```bash
ruff check .             # lint
ruff format --check .    # formatting (use `ruff format .` to fix)
mypy                     # type check
pytest -q                # tests
```

Code lives in `src/moodcheck/`: `conversation/` (engines and the chat loop),
`research/` (corpora, preprocessing, classifier, saved models), `safety.py`,
`storage.py` and `cli.py`. A new engine implements `Engine._generate`; never
define `respond`, which runs the crisis screen first.

## Commits and pull requests

- Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`,
  `docs:`, `chore:`, `build:`, `test:`, `refactor:`).
- Open pull requests against `master`, from a topic branch.
- Fill in the PR template and keep PRs focused.
- Update `CHANGELOG.md` under `[Unreleased]` for user-visible changes.

## Safety-related changes

Anything that touches crisis detection (`src/moodcheck/safety.py`) or how the bot responds to
crisis language must add or update tests in `tests/test_safety.py`, including
phrasings that should and should not trigger the screen.

## Conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
