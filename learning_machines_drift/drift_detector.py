from dataclasses import dataclass
from pydantic import BaseModel
from pygments import highlight, lexers, formatters


class FeatureSummary(BaseModel):
    n_rows: int
    n_features: int


class LabelSummary(BaseModel):
    n_rows: int
    n_labels: int


class ShapeSummary(BaseModel):

    features: FeatureSummary
    labels: LabelSummary


class BaselineSummary(BaseModel):

    shapes: ShapeSummary

    def __str__(self) -> str:
        output = self.json(indent=2)
        return highlight(output, lexers.JsonLexer(), formatters.TerminalFormatter())


class DriftDetector:
    def __init__(self, tag: str):

        pass

    def register_ref_dataset(self, features, labels) -> None:

        pass

    def ref_summary(self) -> BaselineSummary:

        return BaselineSummary(
            shapes=ShapeSummary(
                features=FeatureSummary(n_rows=10, n_features=3),
                labels=LabelSummary(n_rows=10, n_labels=2),
            )
        )
