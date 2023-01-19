# pylint: disable=no-member
"""Module of drift types."""
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel
from pygments import formatters, highlight, lexers


class FeatureSummary(BaseModel):
    """Provides a summary of a features dataframe.

    Attributes:
        n_rows (int): Number of samples (rows).
        n_features (int): Number of features (columns).

    """

    n_rows: int
    n_features: int


class LabelSummary(BaseModel):
    """Provides a summary of a labels series.

    Attributes:
        n_rows (int): Number of samples (rows).
        n_labels (int): Number of distinct labels. For example, for binary
            data, this would be equal to 2.

    """

    n_rows: int
    n_labels: int


class LatentSummary(BaseModel):
    """Provides a summary of a latents dataframe.

    Attributes:
        n_rows (int): Number of samples (rows).
        n_latents (int): Number of latent features (columns).

    """

    n_rows: int
    n_latents: int


class ShapeSummary(BaseModel):
    """Provides a summary of the object shapes in a dataset of features,
    labels and latents.

    Attributes:
        features (FeatureSummary): Features shape summary.
        labels (LabelSummary): Labels shape summary.
        latents (Optional[LatentSummary]): Optional latents shape summary.

    """

    features: FeatureSummary
    labels: LabelSummary
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
    """Class for representing a drift dataset.

    Attributes:
        features (pd.DataFrame): A combined dataframe of input features and
            ground truth labels.
        labels (pd.Series): A series of predicted labels from a model.
        latents (Optional[pd.DataFrame]): An optional dataframe of latent
            variables per sample.

    """

    features: pd.DataFrame
    labels: pd.Series
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
        return pd.concat([self.features, self.labels, self.latents], axis=1)
