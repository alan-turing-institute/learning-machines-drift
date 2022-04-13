from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from pydantic import BaseModel
from pygments import formatters, highlight, lexers


class FeatureSummary(BaseModel):
    n_rows: int
    n_features: int


class LabelSummary(BaseModel):
    n_rows: int
    n_labels: int


class ShapeSummary(BaseModel):

    features: FeatureSummary
    labels: LabelSummary


class BaselineSummary(BaseModel):

    shapes: ShapeSummary

    def __str__(self) -> str:
        output = self.json(indent=2)
        return str(
            highlight(output, lexers.JsonLexer(), formatters.TerminalFormatter())
        )


@dataclass
class Features:
    values: Dict[str, npt.NDArray[Any]]


@dataclass
class Labels:
    values: npt.NDArray[np.int_]
    label_name: str


@dataclass
class Dataset:

    features: pd.DataFrame
    labels: pd.Series

    @property
    def feature_names(self) -> List[str]:

        return list(self.features.columns)


@dataclass
class DatasetLatent:

    dataset: Dataset
    latent: Optional[pd.DataFrame]

    @staticmethod
    def from_dataset(dataset: Dataset) -> "DatasetLatent":

        return DatasetLatent(dataset=dataset, latent=None)
