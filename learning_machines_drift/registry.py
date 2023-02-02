"""Module for registry handling storage and logging of datasets."""
from __future__ import annotations

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
    """Class for registry for logging datasets.

    Attributes:
        backend (Optional[Backend]): Optional backend for data.
        tag (str): Tag identifying dataset.
        ref_dataset (Optional[Dataset]): Optional reference dataset.
        registered_features (Optional[pd.DataFrame]): Optional registered
            features.
        registered_labels (Optional[pd.Series]): Optional registered labels.
        registered_latents (Optional[pd.Series]): Optional registered latents.
        expect_features (bool): Whether features are expected in registry.
        expect_labels (bool): Whether a labels series is expected in registry.
        expect_latent (bool): Whether latents are expected in registry.

    """

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
        """Initializes a registry.


        Args:
            tag (str): A tag to be used with in backend for the registry.
            expect_features (bool): Should features be present.
            expect_labels (bool): Should labels be present.
            expect_latent (bool): Should latents be present.
            backend (Optional[Backend]): An optional backend to be used for
                storage.
            clear_logged (bool): Whether any existing registered data should at
                `tag` in `backend` should be cleared.
            clear_reference (bool): Whether any existing reference data at
                `tag` in `backend` should be cleared.

        """

        if backend:
            self.backend: Backend = backend
        else:
            self.backend = FileBackend(Path(os.getcwd()).joinpath("lm-drift-data"))

        self.tag: str = tag

        if clear_logged:
            self.backend.clear_logged_dataset(self.tag)
        if clear_reference:
            self.backend.clear_reference_dataset(self.tag)

        # A unique identifier used to match logged features and labels
        self._identifier: Optional[UUID] = None
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
        """Gets the identifier of the registry.

        Returns:
            UUID: The identifier.

        """

        if self._identifier is None:
            raise ValueError("DriftDetector must be used in a context manager")

        return self._identifier

    def register_ref_dataset(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        latents: Optional[pd.DataFrame] = None,
    ) -> None:
        """Registers passed reference data.

        Args:
            features (pd.DataFrame): Reference features to be stored.
            labels (pd.Series): Reference labels to be stored.
            latents (Optional[pd.DataFrame]): Reference latents to be stored.

        """

        self.ref_dataset = Dataset(features=features, labels=labels, latents=latents)

        self.backend.save_reference_dataset(self.tag, self.ref_dataset)

    # TODO: add test # pylint: disable=fixme
    def save_reference_dataset(self, dataset: Dataset) -> None:
        """Registers passed reference data.

        Args:
            dataset (Dataset): Reference dataset to be stored.
        """

        self.ref_dataset = dataset

        self.backend.save_reference_dataset(self.tag, self.ref_dataset)

    def ref_summary(self) -> BaselineSummary:
        """Return a JSON describing shape of dataset feature, labels and
            latents.

        Returns:
            BaselineSummary: Summary of the dataset shapes.

        """

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
        """Logs dataset features in registered data.

        Args:
            features (pd.DataFrame): Features dataframe to be registered.

        """

        self.registered_features = features
        self.backend.save_logged_features(
            self.tag, self.identifier, self.registered_features
        )

    # TODO: add test # pylint: disable=fixme
    def log_dataset(self, dataset: Dataset) -> None:
        """Logs dataset features in registered data.

        Args:
            dataset (Dataset): New dataset to be logged.

        """
        self.log_features(dataset.features)
        self.log_labels(dataset.labels)
        self.log_latents(dataset.latents)

    def log_labels(self, labels: pd.Series) -> None:
        """Logs dataset labels in registered data.

        Args:
            labels (pd.Series): Labels series to be registered.

        """

        self.registered_labels = labels
        self.backend.save_logged_labels(
            self.tag, self.identifier, self.registered_labels
        )

    def log_latents(self, latent: pd.DataFrame) -> None:
        """Logs dataset latents in registered data.

        Args:
            latents (pd.DataFrame): Latents dataframe to be registered.

        """

        self.registered_latent = latent
        self.backend.save_logged_latents(
            self.tag, self.identifier, self.registered_latent
        )

    def all_registered(self) -> bool:
        """Checks whether all expected datastes are registered.

        Returns:
            bool: True if all expected registered, False otherwise.

        """

        if self.expect_features and self.registered_features is None:
            return False

        if self.expect_labels and self.registered_labels is None:
            return False

        if self.expect_latent and self.registered_latent is None:
            return False

        return True

    @property
    def registered_dataset(self) -> Dataset:
        """Gets the registered dataset.

        Returns:
            Dataset: The registered dataset.

        """

        # This should check these two things are not None

        return Dataset(
            self.registered_features, self.registered_labels, self.registered_latent
        )

    def __enter__(self) -> Registry:
        """Assigns a new UUID upon context.

        Returns:
            Registry: With new UUID assigned to `_identifier`.

        """

        self._identifier = uuid4()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore
        """Sets `_identifier` as `None` upon leaving context.

        Returns:
            Registry: With new UUID assigned to `_identifier`.

        """
        self._identifier = None
