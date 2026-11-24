"""moodcheck: a reflective chatbot with a crisis-language safety screen."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("moodcheck")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0+unknown"
