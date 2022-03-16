from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSummary:
    n_rows: int
    n_features: int


@dataclass(frozen=True)
class LabelSummary:
    n_rows: int
    n_labels: int


@dataclass(frozen=True)
class ShapeSummary:

    features: FeatureSummary
    labels: LabelSummary


@dataclass(frozen=True)
class BaselineSummary:

    shapes: ShapeSummary


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
