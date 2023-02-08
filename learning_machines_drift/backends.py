"""Backend module.
"""
import glob
import os
import re
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

DEFAULT_LABELS_NAME = "predicted-labels"


def get_identifier(path_object: Union[str, Path]) -> Optional[UUID]:
    """Extract the UUID from the filename. The filename should have the
    format UUID + some other text and a file extension. The UUID should match
    the regex in the pattern variable `UUIDHex4`.

    Args:
        path_obejct (Union[str, Path]):

    Returns:
        Optional[UUID]: Optional universally unique identifier (UUID) from
            `path_object`.

    """

    a_match = UUIDHex4.match(Path(path_object).stem)

    if a_match is None:
        return None

    return UUID(a_match.groups()[0])


class Backend(Protocol):
    """A protocol class for a Backend."""

    def save_reference_dataset(self, tag: str, dataset: Dataset) -> None:
        """Saves passed `dataset` to backend under `tag`.

        Args:
            tag (str): A tag for locating the dataset within the backend.
            dataset (Dataset): Reference dataset to be saved.

        """

    def load_reference_dataset(self, tag: str) -> Dataset:
        """Load reference dataset from reference path.

        Args:
            tag (str): Tag identifying dataset.

        """

    def save_logged_features(
        self, tag: str, identifier: UUID, dataframe: pd.DataFrame
    ) -> None:
        """Save logged features using tag as the path with UUID prepended to
        filename.

        Args:
            tag (str): Tag identifying dataset.
            identifier (UUID): A unique identifier for the logged dataset.
            dataframe (pd.DataFrame): The dataframe that needs saving.

        """

    def save_logged_labels(self, tag: str, identifier: UUID, labels: pd.Series) -> None:
        """Save logged labels using tag as the path with UUID prepended to
        filename.

        Args:
            tag (str): Tag identifying dataset.
            identifier (UUID): A unique identifier for the labels of the
                dataset.
            labels (pd.Series): The dataframe that needs saving.

        """

    def save_logged_latents(
        self, tag: str, identifier: UUID, dataframe: pd.DataFrame
    ) -> None:
        """Save optionally passed latents `dataframe` using tag as the path
        with UUID prepended to filename.

        Args:
            tag (str): Tag identifying dataset.
            identifier (UUID): A unique identifier for the labels of the
                dataset.
            dataframe (pd.DataFrame): The dataframe of latents to be saved.

        """

    def load_logged_dataset(self, tag: str) -> Dataset:
        """Return a Dataset from the union of logged data.

        Args:
            tag (str): Tag identifying dataset.

        """

    def clear_reference_dataset(self, tag: str) -> bool:
        """Delete directory containing reference files.

        Args:
            tag (str): Path to reference directory.

         Return:
            True: if `tag/reference` path exists.
            False: if `tag/reference` path does not exist.
        """

    def clear_logged_dataset(self, tag: str) -> bool:
        """Delete directory containing logged files.

        Args:
            tag (str): Path to logged directory.

        Return:
            True: if `tag/logged` path exists.
            False: if `tag/logged` path does not exist.
        """


class FileBackend:
    """Implements the Backend protocol for writing files to the filesystem."""

    def __init__(self, root_dir: Union[str, Path]) -> None:
        """Creates root directory for output
        Args:
            root_dir (Union[str, Path]): Absolute path to where outputs will be
                saved.

        Returns:
            None
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(exist_ok=True)

    def _get_reference_path(self, tag: str) -> Path:
        """Get the directory for storing reference data, creating directories
        if required.

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
        """Get the directory for storing logged data, creating directories if
        required.

        Args:
            tag (str): Tag identifying dataset.

        Returns:
            Path: Directory storing logged datasets.
        """

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
        """Saves passed `dataset` to backend under `tag`.

        Args:
            tag (str): A tag for locating the dataset within the backend.
            dataset (Dataset): Reference dataset to be saved.

        """
        reference_dir = self._get_reference_path(tag)
        dataset.features.to_csv(reference_dir.joinpath("features.csv"), index=False)
        if dataset.labels.name is None:
            dataset.labels.name = DEFAULT_LABELS_NAME
        labels_dataframe = pd.DataFrame(dataset.labels)
        labels_dataframe.to_csv(reference_dir.joinpath("labels.csv"), index=False)
        if dataset.latents is not None:
            dataset.latents.to_csv(reference_dir.joinpath("latents.csv"), index=False)

    def load_reference_dataset(self, tag: str) -> Dataset:
        """Load reference dataset from reference path.

        Args:
            tag (str): Tag identifying dataset.

        """
        reference_dir = self._get_reference_path(tag)

        features_df: pd.DataFrame = pd.read_csv(reference_dir.joinpath("features.csv"))
        labels_df: pd.DataFrame = pd.read_csv(reference_dir.joinpath("labels.csv"))
        labels_series: pd.Series = labels_df.iloc[:, 0]
        latents_df: Optional[pd.DataFrame] = None
        if reference_dir.joinpath("latents.csv").exists():
            latents_df = pd.read_csv(reference_dir.joinpath("latents.csv"))

        return Dataset(features_df, labels_series, latents_df)

    def save_logged_features(
        self, tag: str, identifier: UUID, dataframe: pd.DataFrame
    ) -> None:
        """Save logged features using tag as the path with UUID prepended to
        filename.

        Args:
            tag (str): Tag identifying dataset.
            identifier (UUID): A unique identifier for the logged dataset.
            dataframe (pd.DataFrame): The dataframe that needs saving.

        """
        logged_dir = self._get_logged_path(tag)
        dataframe.to_csv(logged_dir.joinpath(f"{identifier}_features.csv"), index=False)

    def save_logged_labels(self, tag: str, identifier: UUID, labels: pd.Series) -> None:
        """Save logged labels using tag as the path with UUID prepended to
        filename.

        Args:
            tag (str): Tag identifying dataset.
            identifier (UUID): A unique identifier for the labels of the
                dataset.
            labels (pd.Series): The dataframe that needs saving.

        """
        logged_dir = self._get_logged_path(tag)
        if labels.name is None:
            labels.name = DEFAULT_LABELS_NAME
        labels_dataframe = pd.DataFrame(labels)

        labels_dataframe.to_csv(
            logged_dir.joinpath(f"{identifier}_labels.csv"), index=False
        )

    def save_logged_latents(
        self, tag: str, identifier: UUID, dataframe: Optional[pd.DataFrame]
    ) -> None:
        """Save optionally passed latents `dataframe` using tag as the path
        with UUID prepended to filename.

        Args:
            tag (str): Tag identifying dataset.
            identifier (UUID): A unique identifier for the labels of the
                dataset.
            dataframe (pd.DataFrame): The dataframe of latents to be saved.

        """
        if dataframe is not None:
            logged_dir = self._get_logged_path(tag)
            dataframe.to_csv(
                logged_dir.joinpath(f"{identifier}_latents.csv"), index=False
            )

    def load_logged_dataset(self, tag: str) -> Dataset:
        """Return a Dataset from the union of logged data.

        Args:
            tag (str): Tag identifying dataset.

        """
        files = [Path(f) for f in glob.glob(f"{self._get_logged_path(tag)}/*")]
        loaded_file_dict: Dict[UUID, List[Path]] = {}

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
                    all_label_df: pd.DataFrame = pd.read_csv(fname, header=0)
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

    def clear_reference_dataset(self, tag: str) -> bool:
        """Delete directory containing reference files.

        Args:
            tag (str): Path to reference directory.

         Return:
            True: if `tag/reference` path exists.
            False: if `tag/reference` path does not exist.
        """
        reference_dir = self.root_dir.joinpath(tag).joinpath("reference")
        if not reference_dir.exists():
            return False
        if len(os.listdir(reference_dir)) == 0:
            return True

        for root, _, files in os.walk(reference_dir):
            for file in files:
                os.remove(os.path.join(root, file))
        return True

    def clear_logged_dataset(self, tag: str) -> bool:
        """Delete directory containing logged files.

        Args:
            tag (str): Path to logged directory.

        Return:
            True: if `tag/logged` path exists.
            False: if `tag/logged` path does not exist.
        """
        logged_dir = self.root_dir.joinpath(tag).joinpath("logged")
        if not logged_dir.exists():
            return False
        if len(os.listdir(logged_dir)) == 0:
            return True

        for root, _, files in os.walk(logged_dir):
            for file in files:
                os.remove(os.path.join(root, file))
        return True
