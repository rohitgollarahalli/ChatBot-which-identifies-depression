"""Command-line wiring. Heavy modules (scikit-learn, nltk) are imported lazily."""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

from moodcheck import __version__
from moodcheck.research.config import MODEL_NAMES, NGRAM_NAMES, VECTORIZER_NAMES, ModelConfig
from moodcheck.safety import CRISIS_MESSAGE, CRISIS_REMINDER, is_crisis

RESEARCH_NOTE = (
    "Research mode is an educational exercise in text classification. It is not a "
    "screening tool and cannot tell you whether anyone is depressed."
)


def _err(message: str) -> None:
    print(f"moodcheck: {message}", file=sys.stderr)


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=MODEL_NAMES, default="svm")
    parser.add_argument("--vectorizer", choices=VECTORIZER_NAMES, default="tfidf")
    parser.add_argument("--ngram", choices=NGRAM_NAMES, default="unigram")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moodcheck",
        description="A reflective chatbot with a crisis-language safety screen. "
        "Not a medical device.",
    )
    parser.add_argument("--version", action="version", version=f"moodcheck {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    chat = commands.add_parser("chat", help="Have a conversation with the chatbot.")
    chat.add_argument(
        "--save",
        type=Path,
        metavar="PATH",
        help="Save what you type to PATH when you finish. Off by default: without this "
        "option nothing is written to disk. Refuses to overwrite an existing file.",
    )

    research = commands.add_parser(
        "research", help="Research mode (educational).", description=RESEARCH_NOTE
    )
    research_commands = research.add_subparsers(dest="research_command", required=True)

    train = research_commands.add_parser(
        "train",
        help="Train the classifier, save it and print its evaluation.",
        description=RESEARCH_NOTE,
    )
    _add_model_options(train)

    predict = research_commands.add_parser(
        "predict",
        help="Classify a text file with the saved model (trains one first if needed).",
        description=RESEARCH_NOTE,
    )
    predict.add_argument("file", type=Path, help="Text file to classify.")
    _add_model_options(predict)
    return parser


def _save_problem(save: Path) -> str | None:
    """Why a transcript could not be saved to ``save``, checked before the chat starts."""
    try:
        if save.exists() or save.is_symlink():
            return f"{save} already exists; choose a new file name (nothing was overwritten)."
        parent = save.resolve().parent
        if not parent.is_dir():
            return f"the directory for {save} does not exist."
        if not os.access(parent, os.W_OK | os.X_OK):
            return f"cannot write to the directory for {save}."
        # Reject a name the filesystem will not accept (too long, NUL byte).
        os.lstat(save)
    except FileNotFoundError:
        return None
    except (OSError, RuntimeError, ValueError) as exc:
        return f"cannot use {str(save)!r}: {exc}"
    return None


def _chat(save: Path | None) -> int:
    if sys.stdin is None:
        _err(f"there is no input to read from. {CRISIS_REMINDER}")
        return 2

    from moodcheck.conversation.eliza import ElizaEngine
    from moodcheck.conversation.session import run
    from moodcheck.storage import TranscriptSaveError

    if save is not None:
        problem = _save_problem(save)
        if problem:
            _err(problem)
            return 2
    # A byte the terminal encoding cannot represent must not end the conversation.
    for stream in (sys.stdin, sys.stdout):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="replace")
    try:
        run(ElizaEngine(), save_path=save)
    except TranscriptSaveError as exc:
        _err(f"could not save to {save}: {exc}. Nothing was saved.")
        return 1
    return 0


def _train(config: ModelConfig) -> int:
    from moodcheck.research.classifier import train
    from moodcheck.research.model_store import save_model

    trained = train(config)
    print(f"Accuracy: {trained.accuracy:.3f}")
    print(trained.report)
    try:
        path = save_model(trained)
    except OSError as exc:
        _err(f"could not save the model: {exc}")
        return 1
    print(f"Model saved to {path}")
    return 0


def _predict(file: Path, config: ModelConfig) -> int:
    try:
        text = file.read_bytes().decode("utf-8-sig")
        if "\0" in text:  # UTF-16/32 without a BOM can decode as UTF-8 with NULs
            raise UnicodeError("contains NUL characters")
    except OSError as exc:
        _err(f"cannot read {file}: {exc.strerror or exc}")
        return 2
    except UnicodeError:
        # Dropping bytes could hide crisis language, so refuse instead of guessing.
        _err(f"{file} is not UTF-8 text, so it cannot be checked safely. {CRISIS_REMINDER}")
        return 2

    if not text.strip():
        _err(f"{file} contains no text to classify.")
        return 2

    # The crisis screen comes before any model work and is never overridden.
    if is_crisis(text):
        print(CRISIS_MESSAGE)
        return 0

    from moodcheck.research.classifier import predict_text, train, verdict_message
    from moodcheck.research.model_store import ModelUnavailable, load_model, save_model

    try:
        trained = load_model(config)
    except ModelUnavailable as exc:
        _err(f"{exc}; training a model now (this happens once).")
        trained = train(config)
        try:
            save_model(trained)
        except OSError as save_exc:
            _err(f"warning: could not save the model for next time: {save_exc}")

    print(verdict_message(predict_text(text, trained), text))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "chat":
        return _chat(args.save)
    config = ModelConfig(args.model, args.vectorizer, args.ngram)
    if args.research_command == "train":
        return _train(config)
    return _predict(args.file, config)
