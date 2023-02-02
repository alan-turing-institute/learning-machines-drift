"""Monitor class for interacting with data and scoring drift."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from learning_machines_drift.backends import Backend, FileBackend
from learning_machines_drift.exceptions import ReferenceDatasetMissing
from learning_machines_drift.filter import Filter
from learning_machines_drift.metrics import Metrics
from learning_machines_drift.types import Dataset


class Monitor:
    """A class for monitoring data with data loading from backend and scoring
    drift scoring with metrics class.


    Attributes:
        tag (str): The tag where data for monitoring is located within backend.
        ref_dataset (Optional[Dataset]): The reference dataset.
        registered_dataset (Optional[Dataset]): The logged, registered dataset
            for drift comparison to reference dataset.

    """

    def __init__(
        self,
        tag: str,
        backend: Optional[Backend] = None,
    ) -> None:
        """Initialize monitor class.

        Args:
            tag (str): tag where data is located within backend.
            backend (Optional[Backend]): optional backend where data is stored.

        """

        if backend:
            self.backend: Backend = backend
        else:
            self.backend = FileBackend(Path(os.getcwd()).joinpath("lm-drift-data"))

        self.tag = tag
        self.ref_dataset: Optional[Dataset] = None
        self.registered_dataset: Optional[Dataset] = None

    def load_data(self, drift_filter: Optional[Filter] = None) -> Monitor:
        """Load data from backend into monitor.

        Args:
            drift_filter (Filter, optional): An optional filter with conditions
                applied to both reference and registered loaded data.

        Returns:
            Monitor: The calling `Monitor` instance with (optionally) filtered
                datasets loaded.

        """

        self.ref_dataset = self.backend.load_reference_dataset(self.tag)
        self.registered_dataset = self.backend.load_logged_dataset(self.tag)

        # Apply filter if passed
        if drift_filter is not None:
            self.ref_dataset = drift_filter.transform(self.ref_dataset)
            self.registered_dataset = drift_filter.transform(self.registered_dataset)

        return self

    @property
    def metrics(self) -> Metrics:
        """Drift metrics.

        Raises:
            ReferenceDatasetMissing: The reference dataset is `None`.
            ValueError: There is no additional registered data.

        """

        if self.ref_dataset is None:
            raise ReferenceDatasetMissing

        if self.registered_dataset is None:
            raise ValueError("A reference dataset is registered but not a new datasets")

        return Metrics(self.ref_dataset, self.registered_dataset)
