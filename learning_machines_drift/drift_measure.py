"""TODO PEP 257"""
import os
from pathlib import Path
from typing import Optional

from learning_machines_drift.backends import Backend, FileBackend
from learning_machines_drift.drift_filter import Filter
from learning_machines_drift.exceptions import ReferenceDatasetMissing
from learning_machines_drift.hypothesis_tests import HypothesisTests
from learning_machines_drift.types import Dataset


class Monitor:
    """TODO PEP 257"""

    def __init__(
        self,
        tag: str,
        backend: Optional[Backend] = None,
    ) -> None:
        """TODO PEP 257"""

        if backend:
            self.backend: Backend = backend
        else:
            self.backend = FileBackend(Path(os.getcwd()).joinpath("lm-drift-data"))

        self.tag = tag
        self.ref_dataset: Optional[Dataset] = None
        self.registered_dataset: Optional[Dataset] = None

    def load_data(self, drift_filter: Optional[Filter] = None) -> Dataset:
        """TODO PEP 257"""

        self.ref_dataset = self.backend.load_reference_dataset(self.tag)
        self.registered_dataset = self.backend.load_logged_dataset(self.tag)

        # Apply filter if passed
        if drift_filter is not None:
            self.ref_dataset = drift_filter.transform(self.ref_dataset)
            self.registered_dataset = drift_filter.transform(self.registered_dataset)

        return self.registered_dataset

    @property
    def hypothesis_tests(self) -> HypothesisTests:
        """TODO PEP 257"""

        if self.ref_dataset is None:
            raise ReferenceDatasetMissing

        if self.registered_dataset is None:
            raise ValueError("A reference dataset is registered but not a new datasets")

        return HypothesisTests(self.ref_dataset, self.registered_dataset)
