from typing import Optional
from learning_machines_drift.backends import Backend, FileBackend
from learning_machines_drift.types import Dataset
from pathlib import Path
import os


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

    def load_data(self) -> Dataset:

        return self.backend.load_logged_dataset(self.tag)
