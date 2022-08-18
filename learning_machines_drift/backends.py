"""TODO PEP 257"""
import glob
import re

# import uuid
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple, Union
from uuid import UUID

import pandas as pd

from learning_machines_drift.types import Dataset

UUIDHex4 = re.compile(
    "^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I
)

RE_LABEL = re.compile("(labels)", re.I)


def get_identifier(path_object: Union[str, Path]) -> Optional[UUID]:
    """Extract the UUID from the filename. The filename should have the format UUID + some other text
    and a file extension. The UUID should match the regex in the Pattern variable UUIDHex4."""

    a_match = UUIDHex4.match(Path(path_object).stem)

    if a_match is None:
        return None

    return UUID(a_match.groups()[0])


class Backend(Protocol):
    """Appears to be a placeholder for something - unknown at the time"""

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

    def load_logged_dataset(self, tag: str) -> Dataset:
        """Return a Dataset consisting of two pd.DataFrames.
        The dataframes must have the same index"""
        # pass


class FileBackend:
    """Implements the Backend protocol. Writes files to the filesystem"""

    def __init__(self, root_dir: Union[str, Path]) -> None:
        """Creates root directory for output
        Args:
            root_dir (Union[str, Path]): Absolute path to where outputs will be saved.

        Returns:
            None
        """
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
        """
        Get the directory for storing logged data,
        creating directories if required

        Args:
            tag (str): Tag identifying dataset

        Returns:
            Path: Directory storing logged datasets
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
        """
        Save dataset to reference path.

        Args:
            tag (str): Tag identifying dataset
            dataset (Dataset): Dataset that needs saving

        Returns:
            None

        """

        reference_dir = self._get_reference_path(tag)
        dataset.features.to_csv(reference_dir.joinpath("features.csv"), index=False)
        dataset.labels.to_csv(reference_dir.joinpath("labels.csv"), index=False)

    def load_reference_dataset(self, tag: str) -> Dataset:
        """
        Load reference dataset from reference path.

        Args:
            tag (str): Tag identifying dataset

        Returns:
            Dataset


        """
        reference_dir = self._get_reference_path(tag)

        features_df = pd.read_csv(reference_dir.joinpath("features.csv"))
        labels_df = pd.read_csv(reference_dir.joinpath("labels.csv"))

        return Dataset(features_df, labels_df)

    def save_logged_features(
        self, tag: str, identifier: UUID, dataframe: pd.DataFrame
    ) -> None:
        """
        Save logged features using tag as the path with UUID prepended to filename.

        Args:
            tag (str): Tag identifying dataset
            identifier (UUID): A unique identifier for the logged dataset
            dataframe (pd.DataFrame): The dataframe that needs saving

        Returns:
            None
        """

        logged_dir = self._get_logged_path(tag)
        dataframe.to_csv(logged_dir.joinpath(f"{identifier}_features.csv"), index=False)

    def save_logged_labels(
        self, tag: str, identifier: UUID, dataframe: pd.DataFrame
    ) -> None:
        """
         Save logged labels using tag as the path with UUID prepended to filename.

         Args:
             tag (str): Tag identifying dataset
             identifier (UUID): A unique identifier for the labels of the dataset
             dataframe (pd.DataFrame): The dataframe that needs saving

         Returns:
             None
        """
        logged_dir = self._get_logged_path(tag)

        dataframe.to_csv(logged_dir.joinpath(f"{identifier}_labels.csv"), index=False)

    def load_logged_dataset(self, tag: str) -> Dataset:
        """
        Loops through files in tag subdirectory to create a Dataset class consisting of two
        concatenated dataframes of logged features and logged labels. Labels and features
        are paired based on the UUID in the filename.

        Args:
            tag (str): Tag identifying dataset

        Return:
            Dataset

        """

        files = [Path(f) for f in glob.glob(f"{self._get_logged_path(tag)}/*")]
        file_pairs: List[Tuple[Path, Path]] = []
        matcher: Dict[UUID, Path] = {}

        for file in files:

            key = get_identifier(Path(file))
            if key is None:
                raise IOError("File name does not start with a UUID")

            value = matcher.pop(key, None)

            if value is not None:
                file_pairs.append((value, file))
            else:
                matcher[key] = file

        # Check which is the label

        all_feature_dfs = []
        all_label_dfs = []
        for pair in file_pairs:

            if RE_LABEL.search(pair[0].stem) is not None:

                all_feature_dfs.append(pd.read_csv(pair[1]))
                all_label_dfs.append(pd.read_csv(pair[0]))
            else:
                all_feature_dfs.append(pd.read_csv(pair[0]))
                all_label_dfs.append(pd.read_csv(pair[1]))

        return Dataset(
            features=pd.concat(all_feature_dfs), labels=pd.concat(all_label_dfs)
        )
