"""TODO PEP 257"""
import glob
import re

# import uuid
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Union
from uuid import UUID

import pandas as pd

from learning_machines_drift.types import Dataset

UUIDHex4 = re.compile(
    "^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I
)

RE_LABEL = re.compile("(labels)", re.I)
RE_FEATURES = re.compile("(features)", re.I)
RE_LATENTS = re.compile("(latents)", re.I)


def get_identifier(path_object: Union[str, Path]) -> Optional[UUID]:
    """TODO PEP 257"""

    a_match = UUIDHex4.match(Path(path_object).stem)

    if a_match is None:
        return None

    return UUID(a_match.groups()[0])


class Backend(Protocol):
    """TODO PEP 257"""

    def save_reference_dataset(self, tag: str, dataset: Dataset) -> None:
        """TODO PEP 257"""
        # pass

    def load_reference_dataset(self, tag: str) -> Dataset:
        """TODO PEP 257"""
        # pass

    def save_logged_features(
        self, tag: str, identifier: UUID, dataframe: pd.DataFrame
    ) -> None:
        """TODO PEP 257"""
        # pass

    def save_logged_labels(
        self, tag: str, identifier: UUID, dataframe: pd.DataFrame
    ) -> None:
        """TODO PEP 257"""
        # pass

    def save_logged_latents(
        self, tag: str, identifier: UUID, dataframe: pd.DataFrame
    ) -> None:
        """TODO PEP 257"""

    def load_logged_dataset(self, tag: str) -> Dataset:
        """Return a Dataset consisting of two pd.DataFrames.
        The dataframes must have the same index"""
        # pass


class FileBackend:
    """Implements the Backend protocol. Writes files to the filesystem"""

    def __init__(self, root_dir: Union[str, Path]) -> None:
        """TODO PEP 257"""
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(exist_ok=True)

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
        """TODO PEP 257"""

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
        """TODO PEP 257"""

        reference_dir = self._get_reference_path(tag)
        dataset.features.to_csv(reference_dir.joinpath("features.csv"), index=False)
        dataset.labels.to_csv(reference_dir.joinpath("labels.csv"), index=False)
        if dataset.latents is not None:
            dataset.latents.to_csv(reference_dir.joinpath("latents.csv"), index=False)

    def load_reference_dataset(self, tag: str) -> Dataset:
        """TODO PEP 257"""
        reference_dir = self._get_reference_path(tag)

        features_df: pd.DataFrame = pd.read_csv(reference_dir.joinpath("features.csv"))

        labels_df: pd.DataFrame = pd.read_csv(reference_dir.joinpath("labels.csv"))
        assert len(labels_df.columns) == 1
        labels_series: pd.Series = labels_df.iloc[:, 0]

        latents_df: Optional[pd.DataFrame] = None
        if reference_dir.joinpath("latents.csv").exists():
            latents_df = pd.read_csv(reference_dir.joinpath("latents.csv"))

        return Dataset(features_df, labels_series, latents_df)

    def save_logged_features(
        self, tag: str, identifier: UUID, dataframe: pd.DataFrame
    ) -> None:
        """TODO PEP 257"""

        logged_dir = self._get_logged_path(tag)
        dataframe.to_csv(logged_dir.joinpath(f"{identifier}_features.csv"), index=False)

    def save_logged_labels(
        self, tag: str, identifier: UUID, dataframe: pd.DataFrame
    ) -> None:
        """TODO PEP 257"""

        logged_dir = self._get_logged_path(tag)

        dataframe.to_csv(logged_dir.joinpath(f"{identifier}_labels.csv"), index=False)

    def save_logged_latents(
        self, tag: str, identifier: UUID, dataframe: Optional[pd.DataFrame]
    ) -> None:
        """TODO PEP 257"""
        if dataframe is not None:
            logged_dir = self._get_logged_path(tag)
            dataframe.to_csv(
                logged_dir.joinpath(f"{identifier}_latents.csv"), index=False
            )

    def load_logged_dataset(self, tag: str) -> Dataset:
        """Return a Dataset consisting of three (optional one) pd.DataFrames.
        The dataframes must have the same index"""

        files = [Path(f) for f in glob.glob(f"{self._get_logged_path(tag)}/*")]
        loaded_file_dict: Dict[UUID, List[Path]] = {}

        # Add each identifier to file lists
        for file in files:
            # Get its identifier
            key = get_identifier(Path(file))
            if key is None:
                raise IOError("File name does not start with a UUID")

            loaded_file_dict.setdefault(key, []).append(file)

        all_feature_dfs: List[pd.DataFrame] = []
        all_label_series: List[pd.Series] = []
        all_latent_dfs: List[pd.DataFrame] = []

        for key, value in loaded_file_dict.items():
            # Must have at least features and labels, optional latents
            assert len(value) >= 2

            # Loop over files in list of files for each identifier
            for fname in value:
                if RE_LABEL.search(fname.stem) is not None:
                    all_label_df: pd.DataFrame = pd.read_csv(fname)
                    assert len(all_label_df.columns) == 1
                    all_label_series.append(all_label_df.iloc[:, 0])

                if RE_FEATURES.search(fname.stem) is not None:
                    all_feature_dfs.append(pd.read_csv(fname))

                if RE_LATENTS.search(fname.stem) is not None:
                    all_latent_dfs.append(pd.read_csv(fname))

        # Assert that the features and labels are non-empty
        assert all_feature_dfs and all_label_series

        # If latents found, return with latents, otherwise no latents
        features: pd.DataFrame = pd.concat(all_feature_dfs).reset_index(drop=True)
        labels: pd.Series = pd.concat(all_label_series).reset_index(drop=True)
        latents: Optional[pd.DataFrame] = (
            pd.concat(all_latent_dfs).reset_index(drop=True) if all_latent_dfs else None
        )

        return Dataset(features=features, labels=labels, latents=latents)
