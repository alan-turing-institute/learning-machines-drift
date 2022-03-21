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
    def __init__(
        self,
        tag: str,
        expect_features: bool = True,
        expect_labels: bool = True,
        expect_latent: bool = False,
    ):

        self.tag = tag
        self.ref_dataset: Optional[DataSet] = None

        self.expect_features = expect_features
        self.expect_labels = expect_labels
        self.expect_latent = expect_latent

        # Registrations
        self.registered_features: Optional[npt.NDArray[Any]] = None
        self.registered_labels: Optional[npt.NDArray[np.int_]] = None
        self.registered_latent: Optional[npt.NDArray[Any]] = None

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

    def log_features(self, features: npt.ArrayLike) -> None:

        self.registered_features = np.array(features)

    def log_labels(self, labels: npt.ArrayLike) -> None:

        self.registered_labels = np.array(labels)

    def log_latent(self, latent: npt.ArrayLike) -> None:

        self.registered_latent = np.array(latent)

    def all_registered(self):

        if self.expect_features and self.registered_features is None:

            return False

        if self.expect_labels and self.registered_labels is None:
            return False

        if self.expect_latent and self.registered_latent is None:
            return False

        return True

    def __enter__(self) -> "DriftDetector":

        return self

    def __exit__(self, exc_type, exc_value, traceback):

        pass
