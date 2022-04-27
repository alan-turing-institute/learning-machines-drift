import glob
import uuid
from pathlib import Path
from typing import Protocol

import pandas as pd

from learning_machines_drift.types import Dataset


class Backend(Protocol):
    def save_reference_dataset(self, tag: str, dataset: Dataset) -> None:
        pass

    def load_reference_dataset(self, tag: str) -> Dataset:
        pass

    def save_logged_features(self, tag: str, dataframe: pd.DataFrame) -> None:
        pass

    # def save_logged_labels(self, dataframe: pd.DataFrame) -> None:
    #     pass

    def load_logged_features(self, tag: str) -> pd.DataFrame:
        pass

    # def load_logged_labels(self) -> pd.DataFrame:
    #     pass


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

    def _get_logged_path(self, tag: str) -> Path:

        # Make a directory for the current tag
        tag_dir = self.root_dir.joinpath(tag)
        if not tag_dir.exists():
            tag_dir.mkdir()

        # Make a directory for the reference data
        logged_dir = tag_dir.joinpath("logged")
        if not logged_dir.exists():
            logged_dir.mkdir()

        return logged_dir

    def save_reference_dataset(self, tag: str, dataset: Dataset) -> None:

        reference_dir = self._get_reference_path(tag)
        dataset.features.to_csv(reference_dir.joinpath("features.csv"))
        dataset.labels.to_csv(reference_dir.joinpath("labels.csv"))

    def load_reference_dataset(self, tag: str) -> Dataset:
        reference_dir = self._get_reference_path(tag)

        features_df = pd.read_csv(reference_dir.joinpath("features.csv"))
        labels_df = pd.read_csv(reference_dir.joinpath("labels.csv"))

        return Dataset(features_df, labels_df)

    def save_logged_features(self, tag: str, dataframe: pd.DataFrame) -> None:

        logged_dir = self._get_logged_path(tag)
        name = str(uuid.uuid4())
        dataframe.to_csv(logged_dir.joinpath(f"{name}.csv"), index=False)

    def load_logged_features(self, tag: str) -> pd.DataFrame:

        files = glob.glob(f"{self._get_logged_path(tag)}/*")
        all_df = []
        for f in files:
            all_df.append(pd.read_csv(f, index_col=False))
        return pd.concat(all_df).reset_index(drop=True)
