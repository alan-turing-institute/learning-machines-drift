from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

# from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel
from pygments import formatters, highlight, lexers

from learning_machines_drift.exceptions import ReferenceDatasetMissing


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
        return str(
            highlight(output, lexers.JsonLexer(), formatters.TerminalFormatter())
        )


@dataclass
class DataSet:
    features: npt.NDArray[Any]
    labels: npt.NDArray[np.int_]


class DriftDetector:
    def __init__(self, tag: str):

        self.ref_dataset: Optional[DataSet] = None

    def register_ref_dataset(
        self, features: npt.ArrayLike, labels: npt.ArrayLike
    ) -> None:

        self.ref_dataset = DataSet(features=np.array(features), labels=np.array(labels))

    def ref_summary(self) -> BaselineSummary:

        if self.ref_dataset is None:

            raise ReferenceDatasetMissing

        feature_n_rows, feature_n_features = self.ref_dataset.features.shape
        label_n_row = self.ref_dataset.labels.shape[0]

        return BaselineSummary(
            shapes=ShapeSummary(
                features=FeatureSummary(
                    n_rows=feature_n_rows, n_features=feature_n_features
                ),
                labels=LabelSummary(n_rows=label_n_row, n_labels=2),
            )
        )
