from dataclasses import dataclass
from typing import Any, Optional, List, Dict

import pandas as pd
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
class Features:
    values: Dict[str, npt.NDArray[Any]]


@dataclass
class Labels:
    values: npt.NDArray[np.int_]
    label_name: str


@dataclass
class Dataset:

    features: pd.DataFrame
    labels: pd.Series


class DriftDetector:
    def __init__(
        self,
        tag: str,
        expect_features: bool = True,
        expect_labels: bool = True,
        expect_latent: bool = False,
    ):

        self.tag = tag
        self.ref_dataset: Optional[Dataset] = None

        self.expect_features = expect_features
        self.expect_labels = expect_labels
        self.expect_latent = expect_latent

        # Registrations
        self.registered_features: Optional[pd.DataFrame] = None
        self.registered_labels: Optional[pd.Series] = None
        self.registered_latent: Optional[pd.DataFrame] = None

    def register_ref_dataset(
        self, features: pd.DataFrame, labels: pd.DataFrame
    ) -> None:

        self.ref_dataset = Dataset(features=features, labels=labels)

    def ref_summary(self) -> BaselineSummary:

        if self.ref_dataset is None:

            raise ReferenceDatasetMissing

        feature_n_rows = self.ref_dataset.features.shape[0]
        feature_n_features = self.ref_dataset.features.shape[1]
        label_n_row = self.ref_dataset.labels.shape[0]

        return BaselineSummary(
            shapes=ShapeSummary(
                features=FeatureSummary(
                    n_rows=feature_n_rows, n_features=feature_n_features
                ),
                labels=LabelSummary(n_rows=label_n_row, n_labels=2),
            )
        )

    def log_features(self, features: pd.DataFrame) -> None:

        self.registered_features = features

    def log_labels(self, labels: pd.Series) -> None:

        self.registered_labels = labels

    def log_latent(self, latent: pd.DataFrame) -> None:

        self.registered_latent = latent

    def all_registered(self):

        if self.expect_features and self.registered_features is None:

            return False

        if self.expect_labels and self.registered_labels is None:
            return False

        if self.expect_latent and self.registered_latent is None:
            return False

        return True

    # @property
    # def hypothesis_tests(self):

    #     return HypothesisTests(self.ref_dataset)

    def __enter__(self) -> "DriftDetector":

        return self

    def __exit__(self, exc_type, exc_value, traceback):

        pass


# class HypothesisTests:
#     def __init__(
#         self, reference_dataset: Dataset, new_dataset: Dict[str, npt.NDArray[Any]],
#     ):
#         self.reference_dataset = reference_dataset
#         self.new_dataset = new_dataset

#     def kolmogorov_smirnov(self,) -> None:

#         pass
