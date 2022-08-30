# pylint: disable=no-member
"""TODO PEP 257"""
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel
from pygments import formatters, highlight, lexers


class FeatureSummary(BaseModel):
    """TODO PEP 257"""

    n_rows: int
    n_features: int


class LabelSummary(BaseModel):
    """TODO PEP 257"""

    n_rows: int
    n_labels: int


class LatentSummary(BaseModel):
    """TODO PEP 257"""

    n_rows: int
    n_latents: int


class ShapeSummary(BaseModel):
    """TODO PEP 257"""

    features: FeatureSummary
    labels: LabelSummary
    latents: Optional[LatentSummary]


class BaselineSummary(BaseModel):
    """TODO PEP 257"""

    shapes: ShapeSummary

    def __str__(self) -> str:
        """TODO PEP 257"""
        output = self.json(indent=2)
        return str(
            highlight(output, lexers.JsonLexer(), formatters.TerminalFormatter())
        )


@dataclass
class Dataset:
    """TODO PEP 257"""

    features: pd.DataFrame
    labels: pd.Series
    latents: Optional[pd.DataFrame] = None

    @property
    def feature_names(self) -> List[str]:
        """TODO PEP 257"""
        return list(self.features.columns)

    def unify(self) -> pd.DataFrame:
        """TODO PEP 257"""
        return pd.concat([self.features, self.labels, self.latents], axis=1)


@dataclass
class DatasetLatent:
    """TODO PEP 257"""

    dataset: Dataset
    latent: Optional[pd.DataFrame]

    @staticmethod
    def from_dataset(dataset: Dataset) -> "DatasetLatent":
        """TODO PEP 257"""
        return DatasetLatent(dataset=dataset, latent=None)
