from atexit import register
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

# from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel
from pygments import formatters, highlight, lexers
from scipy import stats

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

    @property
    def feature_names(self) -> List[str]:

        return list(self.features.columns)


@dataclass
class DatasetLatent:

    dataset: Dataset
    latent: Optional[pd.DataFrame]

    @staticmethod
    def from_dataset(dataset: Dataset) -> "DatasetLatent":

        return DatasetLatent(dataset=dataset, latent=None)


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

    def all_registered(self) -> bool:

        if self.expect_features and self.registered_features is None:
            return False

        if self.expect_labels and self.registered_labels is None:
            return False

        if self.expect_latent and self.registered_latent is None:
            return False

        return True

    @property
    def registered_dataset(self) -> Dataset:

        # ToDo: Mypy prob will make a fuss
        # This should check these two things are not None

        return Dataset(self.registered_features, self.registered_labels)

    @property
    def hypothesis_tests(self) -> HypothesisTests:

        if self.ref_dataset is None:
            raise ValueError("A reference dataset has not been registered")

        if self.registered_dataset is None:
            raise ValueError("A reference dataset is registered but not a new datasets")

        return HypothesisTests(self.ref_dataset, self.registered_dataset)

    def __enter__(self) -> "DriftDetector":

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:

        pass


class HypothesisTests:
    def __init__(self, reference_dataset: Dataset, registered_dataset: Dataset):
        self.reference_dataset = reference_dataset
        self.registered_dataset = registered_dataset

    def _calc(self, f: Callable[[npt.ArrayLike, npt.ArrayLike], Any]) -> Any:

        results = {}
        for feature in self.reference_dataset.feature_names:
            ref_col = self.reference_dataset.features[feature]
            reg_col = self.registered_dataset.features[feature]
            results[feature] = f(ref_col, reg_col)

        return results

    def kolmogorov_smirnov(self) -> Any:

        return self._calc(stats.ks_2samp)

    @staticmethod
    def _chi_square(data1: npt.ArrayLike, data2: npt.ArrayLike) -> Any:

        d1_unique, d1_counts = np.unique(data1, return_counts=True)
        d2_unique, d2_counts = np.unique(data1, return_counts=True)

        return stats.chisquare(d1_counts, d2_counts)
