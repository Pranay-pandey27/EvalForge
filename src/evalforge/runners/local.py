"""Repository-contained local execution backend."""

from evalforge.tools.shell import ControlledShell


class LocalRunner(ControlledShell):
    """Named local backend used by experiment configuration."""
