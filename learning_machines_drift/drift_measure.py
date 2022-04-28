import os
from pathlib import Path
from typing import Optional

from learning_machines_drift.backends import Backend, FileBackend
from learning_machines_drift.exceptions import ReferenceDatasetMissing
from learning_machines_drift.hypothesis_tests import HypothesisTests
from learning_machines_drift.types import Dataset


class DriftMeasure:
    def __init__(
        self,
        tag: str,
        backend: Optional[Backend] = None,
    ) -> None:

        if backend:
            self.backend: Backend = backend
        else:
            self.backend = FileBackend(Path(os.getcwd()).joinpath("lm-drift-data"))

        self.tag = tag
        self.ref_dataset: Optional[Dataset] = None

    def load_data(self) -> Dataset:

        self.ref_dataset = self.backend.load_reference_dataset(self.tag)
        loaded_dataset = self.backend.load_logged_dataset(self.tag)

        self.registered_dataset = loaded_dataset

        return loaded_dataset

    @property
    def hypothesis_tests(self) -> HypothesisTests:

        if self.ref_dataset is None:
            raise ReferenceDatasetMissing

        if self.registered_dataset is None:
            raise ValueError("A reference dataset is registered but not a new datasets")

        return HypothesisTests(self.ref_dataset, self.registered_dataset)
