"""Exceptions module."""


class ReferenceDatasetMissing(Exception):
    """Raised when no reference dataset logged."""

    def __str__(self) -> str:
        return "No reference dataset found."
