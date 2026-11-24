import io
import os
import subprocess
import sys

import pytest

from moodcheck import cli
from moodcheck.research import classifier
from moodcheck.safety import CRISIS_MESSAGE, CRISIS_REMINDER, DISCLAIMER


def _feed(monkeypatch, lines):
    monkeypatch.setattr(sys, "stdin", io.StringIO("".join(f"{line}\n" for line in lines)))


def test_chat_help_does_not_load_sklearn():
    code = (
        "import sys\nfrom moodcheck.cli import main\n"
        "try:\n    main(['chat', '--help'])\nexcept SystemExit:\n    pass\n"
        "assert 'sklearn' not in sys.modules\nassert 'joblib' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_python_dash_m_runs_help():
    result = subprocess.run(
        [sys.executable, "-m", "moodcheck", "--version"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert result.stdout.startswith("moodcheck ")


def test_research_help_says_educational(capsys):
    for argv in (["research", "--help"], ["research", "train", "--help"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
        assert "not a screening tool" in " ".join(capsys.readouterr().out.split())


def test_chat_default_saves_nothing(tmp_path, monkeypatch, capsys):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    _feed(monkeypatch, ["hello", "good-bye"])
    assert cli.main(["chat"]) == 0
    assert DISCLAIMER in capsys.readouterr().out
    assert list(work.iterdir()) == []


def test_chat_save_writes_file(tmp_path, monkeypatch):
    path = tmp_path / "t.txt"
    _feed(monkeypatch, ["word " * 40, "good-bye"])
    assert cli.main(["chat", "--save", str(path)]) == 0
    assert path.read_text(encoding="utf-8").split() == ["word"] * 40


def test_chat_save_existing_file_fails_before_conversation(tmp_path, monkeypatch, capsys):
    path = tmp_path / "t.txt"
    path.write_text("keep")
    _feed(monkeypatch, [])
    assert cli.main(["chat", "--save", str(path)]) == 2
    captured = capsys.readouterr()
    assert "already exists" in captured.err
    assert DISCLAIMER not in captured.out
    assert path.read_text() == "keep"


def test_chat_save_missing_directory(tmp_path, capsys):
    assert cli.main(["chat", "--save", str(tmp_path / "nope" / "t.txt")]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_chat_save_error_at_end_is_clean(tmp_path, monkeypatch, capsys):
    path = tmp_path / "t.txt"
    _feed(monkeypatch, ["I want to die", "word " * 40, "good-bye"])

    def boom(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("moodcheck.storage.os.open", boom)
    assert cli.main(["chat", "--save", str(path)]) == 1
    captured = capsys.readouterr()
    assert "could not save" in captured.err
    assert captured.out.rstrip().endswith(CRISIS_REMINDER)


def test_predict_crisis_text_never_trains(tmp_path, monkeypatch, capsys):
    path = tmp_path / "t.txt"
    path.write_text("Work is fine. I want to kill myself.")

    def fail(*args, **kwargs):
        raise AssertionError("must not train or load")

    monkeypatch.setattr(classifier, "train", fail)
    monkeypatch.setattr("moodcheck.research.model_store.load_model", fail)
    assert cli.main(["research", "predict", str(path)]) == 0
    assert capsys.readouterr().out.strip() == CRISIS_MESSAGE


def test_predict_unreadable_file(tmp_path, capsys):
    assert cli.main(["research", "predict", str(tmp_path / "missing.txt")]) == 2
    assert "cannot read" in capsys.readouterr().err
    assert cli.main(["research", "predict", str(tmp_path)]) == 2


def test_predict_trains_once_then_reuses(tmp_path, capsys, isolated_cache):
    path = tmp_path / "t.txt"
    path.write_text("I had a lovely day with my friends.")
    argv = ["research", "predict", str(path), "--model", "multinomial_nb"]

    assert cli.main(argv) == 0
    first = capsys.readouterr()
    assert "training a model now" in first.err
    assert "The model's guess" in first.out
    assert DISCLAIMER in first.out
    assert (isolated_cache / "moodcheck" / "multinomial_nb-tfidf-unigram.joblib").exists()

    assert cli.main(argv) == 0
    second = capsys.readouterr()
    assert second.err == ""
    assert second.out == first.out


def test_predict_cache_write_failure_only_warns(tmp_path, monkeypatch, capsys):
    path = tmp_path / "t.txt"
    path.write_text("I had a lovely day with my friends.")

    def boom(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("moodcheck.research.model_store.save_model", boom)
    assert cli.main(["research", "predict", str(path), "--model", "multinomial_nb"]) == 0
    captured = capsys.readouterr()
    assert "could not save the model" in captured.err
    assert "The model's guess" in captured.out


def test_predict_retrains_when_model_is_stale(tmp_path, capsys, isolated_cache):
    path = tmp_path / "t.txt"
    path.write_text("I had a lovely day with my friends.")
    argv = ["research", "predict", str(path), "--model", "multinomial_nb"]
    cli.main(argv)
    capsys.readouterr()
    model_file = isolated_cache / "moodcheck" / "multinomial_nb-tfidf-unigram.joblib"
    model_file.write_bytes(b"tampered")
    assert cli.main(argv) == 0
    assert "hash mismatch" in capsys.readouterr().err


def test_train_prints_report_and_saves(capsys, isolated_cache):
    assert cli.main(["research", "train", "--model", "multinomial_nb"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Accuracy: ")
    assert "precision" in out
    assert (isolated_cache / "moodcheck" / "multinomial_nb-tfidf-unigram.json").exists()


def test_no_model_path_option():
    with pytest.raises(SystemExit):
        cli.main(["research", "predict", "x.txt", "--model-file", "evil.joblib"])


def test_tests_do_not_touch_real_cache(isolated_cache):
    assert "cache-home" in os.environ["XDG_CACHE_HOME"]


def test_io_error_without_save_is_not_reported_as_a_save_failure(monkeypatch, capsys):
    class BrokenStdin(io.StringIO):
        def readline(self, *_args):
            raise OSError("terminal went away")

    monkeypatch.setattr(sys, "stdin", BrokenStdin())
    with pytest.raises(OSError, match="terminal went away"):
        cli.main(["chat"])
    assert "could not save" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "data",
    [
        "I want to die".encode("utf-16"),
        "I want to die".encode("utf-16-le"),
        "I want to die".encode("utf-16-be"),
        "I want to die".encode("utf-32"),
        b"I don\x92t feel great",
    ],
)
def test_predict_refuses_text_it_cannot_screen(tmp_path, monkeypatch, capsys, data):
    # Undecodable text could hide crisis language, so there must be no verdict.
    path = tmp_path / "t.txt"
    path.write_bytes(data)
    monkeypatch.setattr(classifier, "train", lambda *_a, **_k: pytest.fail("trained"))
    assert cli.main(["research", "predict", str(path)]) == 2
    captured = capsys.readouterr()
    assert "UTF-8" in captured.err
    assert "findahelpline.com" in captured.err
    assert captured.out == ""


def test_predict_screens_utf8_with_bom(tmp_path, capsys):
    path = tmp_path / "t.txt"
    path.write_bytes("I want to die".encode("utf-8-sig"))
    assert cli.main(["research", "predict", str(path)]) == 0
    assert capsys.readouterr().out.strip() == CRISIS_MESSAGE
