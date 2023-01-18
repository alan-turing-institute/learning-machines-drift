"""TODO PEP 257"""
import os
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

import pandas as pd

from learning_machines_drift.backends import Backend, FileBackend
from learning_machines_drift.exceptions import ReferenceDatasetMissing
from learning_machines_drift.types import (
    BaselineSummary,
    Dataset,
    FeatureSummary,
    LabelSummary,
    LatentSummary,
    ShapeSummary,
)


class Registry:
    """TODO PEP 257"""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        tag: str,
        expect_features: bool = True,
        expect_labels: bool = True,
        expect_latent: bool = False,
        backend: Optional[Backend] = None,
        clear_logged: bool = False,
        clear_reference: bool = False,
    ):
        """TODO PEP 257"""
        # pylint: disable=too-many-instance-attributes

        if backend:
            self.backend: Backend = backend
        else:
            self.backend = FileBackend(Path(os.getcwd()).joinpath("lm-drift-data"))

        self.tag: str = tag

        if clear_logged:
            self.backend.clear_logged_dataset(self.tag)
        if clear_reference:
            self.backend.clear_reference_dataset(self.tag)

        self._identifier: Optional[
            UUID
        ] = None  # A unique identifier used to match logged features and labels
        self.ref_dataset: Optional[Dataset] = None

        self.expect_features: bool = expect_features
        self.expect_labels: bool = expect_labels
        self.expect_latent: bool = expect_latent

        # Registrations
        self.registered_features: Optional[pd.DataFrame] = None
        self.registered_labels: Optional[pd.Series] = None
        self.registered_latent: Optional[pd.DataFrame] = None

    @property
    def identifier(self) -> UUID:
        """TODO PEP 257"""

        if self._identifier is None:
            raise ValueError("DriftDetector must be used in a context manager")

        return self._identifier

    def register_ref_dataset(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        latents: Optional[pd.DataFrame] = None,
    ) -> None:
        """TODO PEP 257"""

        self.ref_dataset = Dataset(features=features, labels=labels, latents=latents)

        self.backend.save_reference_dataset(self.tag, self.ref_dataset)

    def ref_summary(self) -> BaselineSummary:
        """Return a json describing shape of dataset features and labels"""

        if self.ref_dataset is None:
            raise ReferenceDatasetMissing

        feature_n_rows = self.ref_dataset.features.shape[0]
        feature_n_features = self.ref_dataset.features.shape[1]
        label_n_row = self.ref_dataset.labels.shape[0]
        if self.ref_dataset.latents is not None:
            latent_n_row = self.ref_dataset.latents.shape[0]
            latent_n_latents = self.ref_dataset.latents.shape[1]
            latents = LatentSummary(n_rows=latent_n_row, n_latents=latent_n_latents)
        else:
            latents = None

        return BaselineSummary(
            shapes=ShapeSummary(
                features=FeatureSummary(
                    n_rows=feature_n_rows, n_features=feature_n_features
                ),
                labels=LabelSummary(n_rows=label_n_row, n_labels=2),
                latents=latents,
            )
        )

    def log_features(self, features: pd.DataFrame) -> None:
        """Log dataset features"""

        self.registered_features = features
        self.backend.save_logged_features(
            self.tag, self.identifier, self.registered_features
        )

    def log_labels(self, labels: pd.Series) -> None:
        """Log dataset labels"""

        self.registered_labels = labels
        self.backend.save_logged_labels(
            self.tag, self.identifier, self.registered_labels
        )

    def log_latents(self, latent: pd.DataFrame) -> None:
        """TODO PEP 257"""

        self.registered_latent = latent
        self.backend.save_logged_latents(
            self.tag, self.identifier, self.registered_latent
        )

    def all_registered(self) -> bool:
        """TODO PEP 257"""

        if self.expect_features and self.registered_features is None:
            return False

        if self.expect_labels and self.registered_labels is None:
            return False

        if self.expect_latent and self.registered_latent is None:
            return False

        return True

    @property
    def registered_dataset(self) -> Dataset:
        """TODO PEP 257"""

        # This should check these two things are not None

        return Dataset(
            self.registered_features, self.registered_labels, self.registered_latent
        )

    def __enter__(self) -> "Registry":
        """TODO PEP 257"""

        self._identifier = uuid4()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore
        """TODO PEP 257"""

        self._identifier = None


#        pass
