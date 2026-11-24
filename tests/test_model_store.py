import io
import json
import os
import stat
import sys

import joblib
import pytest
import sklearn

from moodcheck.research import corpus, model_store
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


def test_loads_the_hashed_bytes_not_the_path(saved, monkeypatch):
    # Loading by path would let the file change between the hash check and the load.
    sources = []
    real = joblib.load

    def spy(source, *args, **kwargs):
        sources.append(source)
        return real(source, *args, **kwargs)

    monkeypatch.setattr(joblib, "load", spy)
    load_model(CONFIG, saved)
    assert [type(source) for source in sources] == [io.BytesIO]


@pytest.mark.parametrize("name", ["depressed.txt", "not_depressed.txt"])
def test_corpus_hash_covers_each_file(monkeypatch, name):
    before = corpus.corpus_sha256()
    real = corpus._read_bytes
    monkeypatch.setattr(corpus, "_read_bytes", lambda n: real(n) + (b"x" if n == name else b""))
    assert corpus.corpus_sha256() != before


def test_deeply_nested_sidecar_is_unavailable(saved, load_spy):
    _, sidecar = model_paths(CONFIG, saved)
    sidecar.write_text("[" * 200_000)
    with pytest.raises(ModelUnavailable):
        load_model(CONFIG, saved)
    assert load_spy == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
@pytest.mark.parametrize("which", ["directory", "model", "sidecar"])
def test_files_others_can_write_are_never_loaded(saved, load_spy, which):
    model_file, sidecar = model_paths(CONFIG, saved)
    {"directory": saved, "model": model_file, "sidecar": sidecar}[which].chmod(0o777)
    with pytest.raises(ModelUnavailable, match="other users"):
        load_model(CONFIG, saved)
    assert load_spy == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership")
def test_files_owned_by_someone_else_are_never_loaded(saved, load_spy, monkeypatch):
    uid = os.getuid()
    monkeypatch.setattr(model_store.os, "getuid", lambda: uid + 1)
    with pytest.raises(ModelUnavailable, match="other users"):
        load_model(CONFIG, saved)
    assert load_spy == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_refuses_to_save_into_a_shared_directory(trained, tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    with pytest.raises(OSError, match="other users"):
        save_model(trained, shared)
    assert list(shared.iterdir()) == []
