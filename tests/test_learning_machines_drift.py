"""Tests for overall drift package."""
from learning_machines_drift import __version__


def test_version() -> None:
    """Tests whether version is expected value."""
    assert __version__ == "0.1.0"
