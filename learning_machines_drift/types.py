# pylint: disable=no-member
"""Module of drift types."""
from dataclasses import dataclass
from io import StringIO
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
        return pd.concat(
            [
                self.features,
                pd.DataFrame(self.labels),
                self.latents,
            ],
            axis=1,
        )


@dataclass
class StructuredResult:
    """A type for representing a result from the hypothesis tests module."""

    #: str: Name of the scoring method used.
    method_name: str
    #: Dict[str, Dict[str, float]]: Dictionary of results with keys as
    #: `feature_name` or, if for a unified dataset, "single_value".
    #: Values are a dictionary containing the result statistic and p-value (if
    #: available) for a given `method_name`.
    results: Dict[str, Dict[str, float]]

    def __repr__(self) -> str:
        output = StringIO()
        print(f"Method: {self.method_name}", file=output)
        for result, result_dict in self.results.items():
            print(f"  {result}", file=output)
            for (result_key, result_value) in result_dict.items():
                print(f"{result_key: >15}: {result_value:>10.2e}", file=output)
            print(file=output)
        return output.getvalue()
