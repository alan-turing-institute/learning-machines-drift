import os
from atexit import register
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

# from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel
from pygments import formatters, highlight, lexers
from scipy import stats

from learning_machines_drift.backends import Backend, FileBackend
from learning_machines_drift.exceptions import ReferenceDatasetMissing
from learning_machines_drift.hypothesis_tests import HypothesisTests
from learning_machines_drift.types import (
    BaselineSummary,
    Dataset,
    FeatureSummary,
    LabelSummary,
    ShapeSummary,
)


class DriftDetector:
    def __init__(
        self,
        tag: str,
        expect_features: bool = True,
        expect_labels: bool = True,
        expect_latent: bool = False,
        backend: Optional[Backend] = None,
    ):

        self.backend: Backend = (
            backend
            if backend is not None
            else FileBackend(Path(os.getcwd()).joinpath("lm-drift-data"))
        )
        self.tag: str = tag
        self.ref_dataset: Optional[Dataset] = None

        self.expect_features: bool = expect_features
        self.expect_labels: bool = expect_labels
        self.expect_latent: bool = expect_latent

        # Registrations
        self.registered_features: Optional[pd.DataFrame] = None
        self.registered_labels: Optional[pd.Series] = None
        self.registered_latent: Optional[pd.DataFrame] = None

    def register_ref_dataset(
        self, features: pd.DataFrame, labels: pd.DataFrame
    ) -> None:

        self.ref_dataset = Dataset(features=features, labels=labels)

        self.backend.save_reference_dataset(self.tag, self.ref_dataset)

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
        self.backend.save_logged_features(self.tag, self.registered_features)

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
            raise ReferenceDatasetMissing

        if self.registered_dataset is None:
            raise ValueError("A reference dataset is registered but not a new datasets")

        return HypothesisTests(self.ref_dataset, self.registered_dataset)

    def __enter__(self) -> "DriftDetector":

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:

        pass
