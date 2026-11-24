import os
import stat
import sys

import pytest

from moodcheck.storage import TranscriptSaveError, save_transcript


def test_saves_utf8(tmp_path):
    path = tmp_path / "t.txt"
    save_transcript(path, "café ☕")
    assert path.read_bytes() == "café ☕".encode()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_file_is_private(tmp_path):
    path = tmp_path / "t.txt"
    old = os.umask(0)
    try:
        save_transcript(path, "x")
    finally:
        os.umask(old)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_refuses_to_overwrite(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("keep me")
    with pytest.raises(TranscriptSaveError):
        save_transcript(path, "new")
    assert path.read_text() == "keep me"


def test_missing_parent_directory(tmp_path):
    with pytest.raises(TranscriptSaveError):
        save_transcript(tmp_path / "nope" / "t.txt", "x")


def test_failed_write_leaves_no_partial_file(tmp_path, monkeypatch):
    real_fdopen = os.fdopen

    def broken_fdopen(fd, *args, **kwargs):
        fh = real_fdopen(fd, *args, **kwargs)

        class Broken:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                fh.close()

            def write(self, _text):
                fh.write("partial")
                fh.flush()
                raise OSError("disk full")

        return Broken()

    monkeypatch.setattr("moodcheck.storage.os.fdopen", broken_fdopen)
    target = tmp_path / "t.txt"
    with pytest.raises(TranscriptSaveError):
        save_transcript(target, "hello")
    assert not target.exists()
