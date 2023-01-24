# pylint: disable=no-member
"""Module of drift types."""
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel
from pygments import formatters, highlight, lexers


class FeatureSummary(BaseModel):
    """Provides a summary of a features dataframe."""

    #: int: Number of samples (rows).
    n_rows: int
    #: int: Number of features (columns).
    n_features: int


class LabelSummary(BaseModel):
    """Provides a summary of a labels series."""

    #: int: Number of samples (rows).
    n_rows: int
    #: int: Number of distinct labels. For example, for binary data, this would
    #: be equal to 2.
    n_labels: int


class LatentSummary(BaseModel):
    """Provides a summary of a latents dataframe."""

    #: int: Number of samples (rows).
    n_rows: int
    #: int: Number of latent features (columns).
    n_latents: int


class ShapeSummary(BaseModel):
    """Provides a summary of the object shapes in a dataset of features,
    labels and latents.
    """

    #: FeatureSummary: Features shape summary.
    features: FeatureSummary
    #: LabelSummary: Labels shape summary.
    labels: LabelSummary
    #: Optional[LatentSummary]: Optional latents shape summary.
    latents: Optional[LatentSummary]


class BaselineSummary(BaseModel):
    """Class for storing a shape summary with JSON string representation."""

    #: ShapeSummary: A shape summary instance of a dataset.
    shapes: ShapeSummary

    def __str__(self) -> str:
        """Generates a JSON string for shape summary of dataset."""
        output = self.json(indent=2)
        return str(
            highlight(output, lexers.JsonLexer(), formatters.TerminalFormatter())
        )


@dataclass
class Dataset:
    """Class for representing a drift dataset."""

    #: pd.DataFrame: A combined dataframe of input features and ground truth labels.
    features: pd.DataFrame
    #: pd.Series: A series of predicted labels from a model.
    labels: pd.Series
    #: Optional[pd.DataFrame]: An optional dataframe of latent variables per sample.
    latents: Optional[pd.DataFrame] = None

    @property
    def feature_names(self) -> List[str]:
        """Returns a list of features dataframe columns.

        Returns:
            List[str]: A list of feature column names as strings.

        """
        return list(self.features.columns)

    def unify(self) -> pd.DataFrame:
        """Returns a column-wise concatenated dataframe of features, labels and
        latents.

        Returns:
            pd.DataFrame: Column-wise concatenated dataframe of features,
                labels and latents.

        """
        # if self.latents is None:
        # return pd.concat([self.features, pd.DataFrame(self.labels, columns=["labels"])], axis=1)
        # else:
        return pd.concat(
            [
                self.features,
                pd.DataFrame(self.labels, columns=["labels"]),
                self.latents,
            ],
            axis=1,
        )


@dataclass
class DatasetLatent:
    """TODO PEP 257"""

    dataset: Dataset
    latent: Optional[pd.DataFrame]

    @staticmethod
    def from_dataset(dataset: Dataset) -> "DatasetLatent":
        """TODO PEP 257"""
        return DatasetLatent(dataset=dataset, latent=None)


@dataclass
class StructuredResult:
    """TODO PEP 257"""

    method_name: str
    results: Dict[str, Dict[str, float]]
