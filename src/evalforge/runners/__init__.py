"""Execution backends for verifier commands."""

from evalforge.runners.docker import DockerRunner
from evalforge.runners.local import LocalRunner

__all__ = ["DockerRunner", "LocalRunner"]
