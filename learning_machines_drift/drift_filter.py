# pylint: disable=C0103
# pylint: disable=W0621
# pylint: disable=R0913

"""Module to filter a dataset."""
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from learning_machines_drift.types import Dataset


class Filter:  # pylint: disable=too-few-public-methods
    """Filter class.

    Filters a given dataset through an AND operation applied across all passed
    conditions.

    Args:
        conditions (dict[str, object]): Dict with key (variable) and value
            as a list of (condition, value) to be used for filtering.

    Attributes:
        conditions (dict[str, object]): Dict with key (variable) and value
            as a list of (condition, value) to be used for filtering.
    """

    conditions: Optional[dict[str, List[Tuple[str, Any]]]]

    def __init__(self, conditions: Optional[dict[str, List[Tuple[str, Any]]]]):
        """Initialize a dict with variable keys and (condition, value) values."""
        self.conditions = conditions

    def _filter_df(
        self, df: pd.DataFrame, variable: str, condition: str, value: Any
    ) -> pd.DataFrame:
        """Subset a dataframe given condition and value."""
        if condition == "less":
            return df.loc[df[variable] < value]
        if condition == "greater":
            return df.loc[df[variable] > value]
        if condition == "equal":
            return df.loc[df[variable] == value]
        raise ValueError(
            f"'{condition}' is not implemented. "
            "Please choose one of 'equal', 'less', 'greater'."
        )

    def _filter_series(
        self, series: pd.Series, condition: str, value: Any
    ) -> pd.Series:
        """Subset a series given condition and value."""
        if condition == "less":
            return series[series < value]
        if condition == "greater":
            return series[series > value]
        if condition == "equal":
            return series[series == value]
        raise ValueError(
            f"'{condition}' is not implemented. "
            "Please choose one of 'equal', 'less', 'greater'."
        )

    def transform(self, dataset: Dataset) -> Dataset:
        """Transform the passed dataset given filter.

        Args:
            dataset (Dataset): the dataset to be filtered.

        Returns:
            Dataset: transformed dataset given filters
        """
        if self.conditions is None:
            return dataset

        # Filter the dataframe or series of the variables
        for (variable, list_of_conditions) in self.conditions.items():
            for (condition, value) in list_of_conditions:
                if variable in dataset.features.columns:
                    dataset.features = self._filter_df(
                        dataset.features, variable, condition, value
                    )
                elif variable == dataset.labels.name:
                    dataset.labels = self._filter_series(
                        dataset.labels, condition, value
                    )
                elif dataset.latents is not None:
                    if variable in dataset.latents.columns:
                        dataset.latents = self._filter_df(
                            dataset.latents, variable, condition, value
                        )

        # Select only the common idx
        common_idx = np.intersect1d(
            dataset.features.index.to_numpy(),
            dataset.labels.index.to_numpy(),
        )
        if dataset.latents is not None:
            common_idx = np.intersect1d(
                common_idx,
                dataset.latents.index.to_numpy(),
            )

        # Update dataset with common idx
        dataset.features = dataset.features.loc[common_idx]
        dataset.labels = dataset.labels.loc[common_idx]
        if dataset.latents is not None:
            dataset.latents = dataset.latents.loc[common_idx]

        # Return filtered dataset
        return dataset
