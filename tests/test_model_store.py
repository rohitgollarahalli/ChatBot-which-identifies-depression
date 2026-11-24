import json
import stat
import sys

import joblib
import pytest
import sklearn

from moodcheck.research import model_store
from moodcheck.research.classifier import TrainedModel, predict_text, train
from moodcheck.research.config import ModelConfig
from moodcheck.research.model_store import (
    ModelUnavailable,
    default_cache_dir,
    load_model,
    model_paths,
    save_model,
)

CONFIG = ModelConfig("multinomial_nb", "count", "unigram")


@pytest.fixture(scope="module")
def trained() -> TrainedModel:
    return train(CONFIG)


@pytest.fixture
def saved(trained, tmp_path):
    save_model(trained, tmp_path)
    return tmp_path


@pytest.fixture
def load_spy(monkeypatch):
    calls = []

    def spy(*args, **kwargs):
        calls.append(args)
        raise AssertionError("joblib.load must not be called")

    monkeypatch.setattr(joblib, "load", spy)
    return calls


def _edit_sidecar(directory, **changes):
    _, sidecar = model_paths(CONFIG, directory)
    data = json.loads(sidecar.read_text())
    data.update(changes)
    sidecar.write_text(json.dumps(data))


def test_roundtrip(trained, saved):
    loaded = load_model(CONFIG, saved)
    assert loaded.config == CONFIG
    assert loaded.accuracy == trained.accuracy
    assert loaded.report == trained.report
    text = "I feel hopeless and empty."
    assert predict_text(text, loaded) == predict_text(text, trained)


def test_sidecar_contents(saved):
    _, sidecar = model_paths(CONFIG, saved)
    data = json.loads(sidecar.read_text())
    assert data["format_version"] == model_store.FORMAT_VERSION
    assert data["scikit_learn_version"] == sklearn.__version__
    assert data["config"] == {"model": "multinomial_nb", "vectorizer": "count", "ngram": "unigram"}
    for key in ("moodcheck_version", "python_version", "preprocessing_fingerprint"):
        assert data[key]
    assert len(data["corpus_sha256"]) == len(data["model_sha256"]) == 64


def test_no_leftover_temp_files(saved):
    assert sorted(p.suffix for p in saved.iterdir()) == [".joblib", ".json"]


def test_missing_sidecar(saved, load_spy):
    model_paths(CONFIG, saved)[1].unlink()
    with pytest.raises(ModelUnavailable, match="no saved model"):
        load_model(CONFIG, saved)
    assert load_spy == []


def test_nothing_saved_yet(tmp_path):
    with pytest.raises(ModelUnavailable):
        load_model(CONFIG, tmp_path / "empty")


@pytest.mark.parametrize("content", ["{not json", "[]", "null", ""])
def test_corrupt_sidecar(saved, load_spy, content):
    model_paths(CONFIG, saved)[1].write_text(content)
    with pytest.raises(ModelUnavailable):
        load_model(CONFIG, saved)
    assert load_spy == []


def test_tampered_model_file(saved, load_spy):
    model_file = model_paths(CONFIG, saved)[0]
    model_file.write_bytes(model_file.read_bytes() + b"x")
    with pytest.raises(ModelUnavailable, match="hash"):
        load_model(CONFIG, saved)
    assert load_spy == []


def test_missing_model_file(saved, load_spy):
    model_paths(CONFIG, saved)[0].unlink()
    with pytest.raises(ModelUnavailable):
        load_model(CONFIG, saved)
    assert load_spy == []


@pytest.mark.parametrize(
    "field, value",
    [
        ("scikit_learn_version", "0.0.1"),
        ("python_version", "2.7"),
        ("moodcheck_version", "0.0.0"),
        ("format_version", 99),
        ("preprocessing_fingerprint", "abc"),
        ("corpus_sha256", "abc"),
        ("config", {"model": "svm", "vectorizer": "count", "ngram": "unigram"}),
    ],
)
def test_stamp_mismatch_never_unpickles(saved, load_spy, field, value):
    _edit_sidecar(saved, **{field: value})
    with pytest.raises(ModelUnavailable):
        load_model(CONFIG, saved)
    assert load_spy == []


def test_wrong_sklearn_version_reason(saved):
    _edit_sidecar(saved, scikit_learn_version="0.0.1")
    with pytest.raises(ModelUnavailable, match="scikit_learn_version"):
        load_model(CONFIG, saved)


def test_unloadable_payload_is_unavailable(saved, monkeypatch):
    monkeypatch.setattr(joblib, "load", lambda *a, **k: {"model": 1})
    with pytest.raises(ModelUnavailable):
        load_model(CONFIG, saved)


def test_one_file_per_config(tmp_path):
    other = ModelConfig("svm", "tfidf", "bigram")
    assert model_paths(CONFIG, tmp_path)[0] != model_paths(other, tmp_path)[0]


def test_default_cache_dir_uses_xdg(monkeypatch, tmp_path):
    monkeypatch.setattr(model_store.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "x"))
    assert default_cache_dir() == tmp_path / "x" / "moodcheck"


def test_default_cache_dir_falls_back_to_home(monkeypatch, tmp_path):
    monkeypatch.setattr(model_store.sys, "platform", "linux")
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert default_cache_dir() == tmp_path / "home" / ".cache" / "moodcheck"


def test_default_cache_dir_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(model_store.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert default_cache_dir() == tmp_path / "local" / "moodcheck"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_cache_dir_is_private(trained, tmp_path):
    target = tmp_path / "fresh" / "moodcheck"
    save_model(trained, target)
    assert stat.S_IMODE(target.stat().st_mode) & 0o077 == 0
    for path in target.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0


def test_relative_xdg_cache_home_is_ignored(monkeypatch, tmp_path):
    # The XDG spec says relative paths are invalid. Honouring one would unpickle
    # files from whatever directory the user happens to be in.
    monkeypatch.setattr(model_store.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", "rel")
    assert default_cache_dir() == tmp_path / "home" / ".cache" / "moodcheck"


@pytest.mark.parametrize("name", ["training_code_sha256", "nltk_version"])
def test_training_code_and_nltk_version_are_stamped(saved, load_spy, name):
    _edit_sidecar(saved, **{name: "other"})
    with pytest.raises(ModelUnavailable, match=name):
        load_model(CONFIG, saved)
    assert load_spy == []


def test_changed_training_code_invalidates_the_model(saved, load_spy, monkeypatch):
    monkeypatch.setattr(model_store, "training_code_sha256", lambda: "changed")
    with pytest.raises(ModelUnavailable, match="training_code"):
        load_model(CONFIG, saved)
    assert load_spy == []
