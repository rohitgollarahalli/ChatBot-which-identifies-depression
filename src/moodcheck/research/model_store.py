"""Save and load a trained model, so prediction does not retrain every time.

A model is a joblib file plus a sidecar JSON recording what produced it. Loading
validates the sidecar and the file's hash *before* ``joblib.load`` runs, because
unpickling executes code: a version stamp inside the pickle would be read too late.

The hash detects corruption and stale or mismatched files. It is not a security
boundary: someone who can write to the cache directory can rewrite the sidecar
too. So on POSIX a directory or file that other users own or can write to is
refused, and the CLI never loads a model from an arbitrary path.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any

import joblib
import nltk
import sklearn

from moodcheck import __version__
from moodcheck.research.classifier import TrainedModel
from moodcheck.research.config import ModelConfig
from moodcheck.research.corpus import corpus_sha256
from moodcheck.research.preprocessing import Preprocessor

FORMAT_VERSION = 1


class ModelUnavailable(Exception):
    """No usable saved model; the message says why."""


def default_cache_dir() -> Path:
    """``$XDG_CACHE_HOME`` or ``~/.cache`` (``%LOCALAPPDATA%`` on Windows), plus ``moodcheck``.

    A relative value is ignored, as the XDG spec requires: it would make the
    current directory decide which files get unpickled. Raises ``OSError`` if
    there is no home directory to fall back to.
    """
    if sys.platform == "win32":
        variable, fallback = "LOCALAPPDATA", ("AppData", "Local")
    else:
        variable, fallback = "XDG_CACHE_HOME", (".cache",)
    base = Path(os.environ.get(variable, ""))
    if not base.is_absolute():
        try:
            base = Path.home().joinpath(*fallback)
        except RuntimeError as exc:
            raise OSError(str(exc)) from exc
    return base / "moodcheck"


def model_paths(config: ModelConfig, directory: Path | None = None) -> tuple[Path, Path]:
    """Return (joblib file, sidecar file) for a config; one pair per config."""
    stem = f"{config.model}-{config.vectorizer}-{config.ngram}"
    base = directory or default_cache_dir()
    return base / f"{stem}.joblib", base / f"{stem}.json"


def _check_private(*paths: Path) -> None:
    """Raise ``OSError`` if another user owns or can write to any of ``paths`` (POSIX)."""
    if os.name != "posix":
        return
    for path in paths:
        info = path.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise OSError(f"{path} is owned by or writable by other users")


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def training_code_sha256() -> str:
    """Hash of the code that decides what a model learns, so editing it retrains."""
    digest = hashlib.sha256()
    for name in ("classifier.py", "corpus.py", "preprocessing.py"):
        digest.update((resources.files("moodcheck.research") / name).read_bytes())
    return digest.hexdigest()


def _expected_stamp(config: ModelConfig) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "moodcheck_version": __version__,
        "scikit_learn_version": sklearn.__version__,
        "nltk_version": nltk.__version__,
        "python_version": _python_version(),
        "config": {"model": config.model, "vectorizer": config.vectorizer, "ngram": config.ngram},
        "preprocessing_fingerprint": Preprocessor().fingerprint(),
        "corpus_sha256": corpus_sha256(),
        "training_code_sha256": training_code_sha256(),
    }


def _atomic_write(path: Path, write: Any) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_model(trained: TrainedModel, directory: Path | None = None) -> Path:
    """Write the model and its sidecar atomically; returns the joblib path."""
    model_file, sidecar_file = model_paths(trained.config, directory)
    model_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _check_private(model_file.parent)

    buffer = io.BytesIO()
    joblib.dump(
        {
            "model": trained.model,
            "vectorizer": trained.vectorizer,
            "report": trained.report,
            "accuracy": trained.accuracy,
        },
        buffer,
    )
    data = buffer.getvalue()

    stamp = _expected_stamp(trained.config)
    stamp["preprocessing_fingerprint"] = trained.preprocessing_fingerprint
    stamp["model_sha256"] = hashlib.sha256(data).hexdigest()

    # Model file first: a sidecar is only ever written for a complete model file.
    _atomic_write(model_file, lambda fh: fh.write(data))
    _atomic_write(sidecar_file, lambda fh: fh.write(json.dumps(stamp, indent=2).encode("utf-8")))
    return model_file


def load_model(config: ModelConfig, directory: Path | None = None) -> TrainedModel:
    """Load the saved model for ``config`` or raise ``ModelUnavailable`` with the reason."""
    try:
        model_file, sidecar_file = model_paths(config, directory)
        sidecar_file.stat()  # a missing sidecar is the normal first run
        _check_private(sidecar_file.parent, sidecar_file)
        stamp = json.loads(sidecar_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ModelUnavailable("no saved model yet") from None
    except (OSError, ValueError, RecursionError) as exc:
        raise ModelUnavailable(f"saved model metadata is unreadable ({exc})") from None
    if not isinstance(stamp, dict):
        raise ModelUnavailable("saved model metadata is corrupt")

    for key, expected in _expected_stamp(config).items():
        if stamp.get(key) != expected:
            raise ModelUnavailable(f"saved model is out of date or does not match ({key})")

    try:
        _check_private(model_file)
        data = model_file.read_bytes()
    except OSError as exc:
        raise ModelUnavailable(f"saved model file is unreadable ({exc})") from None
    if hashlib.sha256(data).hexdigest() != stamp.get("model_sha256"):
        raise ModelUnavailable("saved model file is corrupt (hash mismatch)")

    try:
        # Load the bytes that were hashed, not the path, so the file cannot change in between.
        payload = joblib.load(io.BytesIO(data))
        return TrainedModel(
            model=payload["model"],
            vectorizer=payload["vectorizer"],
            config=config,
            preprocessing_fingerprint=stamp["preprocessing_fingerprint"],
            report=payload["report"],
            accuracy=payload["accuracy"],
        )
    except Exception as exc:
        raise ModelUnavailable(f"saved model could not be loaded ({exc})") from None
