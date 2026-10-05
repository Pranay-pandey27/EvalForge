"""EvalForge package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("evalforge")
except PackageNotFoundError:  # pragma: no cover - source tree import
    __version__ = "1.0.0"
