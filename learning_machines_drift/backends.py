from typing import Protocol
from learning_machines_drift.types import Dataset
from pathlib import Path
import pandas as pd


class Backend(Protocol):
    def save_reference_dataset(self, tag: str, dataset: Dataset) -> None:
        pass

    def load_reference_dataset(self, tag: str) -> Dataset:
        pass

    def save_logged_dataset(self, dataset: Dataset) -> None:
        pass

    def load_logged_dataset(self) -> Dataset:
        pass


class FileBackend:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    def _get_reference_path(self, tag: str) -> Path:
        """Get the directory for storing reference data,
        creating directories if required

        Args:
            tag (str): Tag identifying dataset

        Returns:
            Path: Directory storing reference datasets
        """

        # Make a directory for the current tag
        tag_dir = self.root_dir.joinpath(tag)
        if not tag_dir.exists():

            tag_dir.mkdir()

        # Make a directory for the reference data
        reference_dir = tag_dir.joinpath("reference")
        if not reference_dir.exists():
            reference_dir.mkdir()

        return reference_dir

    def save_reference_dataset(self, tag: str, dataset: Dataset) -> None:

        reference_dir = self._get_reference_path(tag)
        dataset.features.to_csv(reference_dir.joinpath("features.csv"))
        dataset.labels.to_csv(reference_dir.joinpath("labels.csv"))

    def load_reference_dataset(self, tag: str) -> Dataset:
        reference_dir = self._get_reference_path(tag)

        features_df = pd.read_csv(reference_dir.joinpath("features.csv"))
        labels_df = pd.read_csv(reference_dir.joinpath("labels.csv"))

        return Dataset(features_df, labels_df)

    # def save_logged_dataset(self, dataset: Dataset) -> None:
    #     raise NotImplementedError

    # def load_logged_dataset(self) -> Dataset:
    #     raise NotImplementedError
